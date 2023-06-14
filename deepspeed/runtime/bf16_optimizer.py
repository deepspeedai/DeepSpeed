# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import OrderedDict
import torch
import sys
import os
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed import comm as dist
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.runtime import ZeROOptimizer
from packaging import version as pkg_version

from deepspeed.git_version_info import version
from deepspeed.runtime.utils import (get_global_norm_of_tensors, clip_tensors_by_global_norm, DummyOptim,
                                     align_dense_tensors, all_gather_dp_groups, bwc_tensor_model_parallel_rank,
                                     is_model_parallel_parameter, see_memory_usage)

from deepspeed.utils import link_hp_params, fragment_address
from deepspeed.checkpoint import enable_universal_checkpoint
from deepspeed.checkpoint.constants import (DS_VERSION, PARTITION_COUNT, BASE_OPTIMIZER_STATE,
                                            SINGLE_PARTITION_OF_FP32_GROUPS, CLIP_GRAD, GROUP_PADDINGS,
                                            PARAM_SLICE_MAPPINGS)

setattr(sys.modules[__name__], 'fragment_address', fragment_address)


class BF16_Optimizer(ZeROOptimizer):

    def __init__(self,
                 init_optimizer,
                 param_names,
                 mpu=None,
                 clip_grad=0.0,
                 norm_type=2,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 timers=None):
        super().__init__()
        see_memory_usage('begin bf16_optimizer', force=True)
        self.timers = timers
        self.optimizer = init_optimizer
        self.param_names = param_names
        self.using_real_optimizer = not isinstance(self.optimizer, DummyOptim)

        self.clip_grad = clip_grad
        self.norm_type = norm_type
        self.mpu = mpu
        self.allgather_bucket_size = int(allgather_bucket_size)
        self.dp_process_group = dp_process_group
        self.dp_rank = dist.get_rank(group=self.dp_process_group)
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        #align nccl all-gather send buffers to 4-bye boundary
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        # Build BF16/FP32 groups
        self.bf16_groups = []
        self.bf16_groups_flat = []
        self.bf16_partitioned_groups = []

        self.fp32_groups_flat_partition = []

        # Maintain different fp32 gradients views for convenience
        self.fp32_groups_gradients = []
        self.fp32_groups_gradient_dict = {}
        self.fp32_groups_gradients_flat = []
        self.fp32_groups_actual_gradients_flat = []
        self.fp32_groups_gradient_flat_partition = []
        self.fp32_groups_has_gradients = []

        self.step_count = 0
        self.group_paddings = []

        if self.using_real_optimizer:
            self._setup_for_real_optimizer()

        see_memory_usage('end bf16_optimizer', force=True)

    def _setup_for_real_optimizer(self):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        self.partition_count = [dp_world_size for i in range(len(self.optimizer.param_groups))]

        for i, param_group in enumerate(self.optimizer.param_groups):
            see_memory_usage(f'before initializing group {i}', force=True)

            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # grab the original list
            trainable_parameters = [param for param in param_group['params'] if param.requires_grad]
            self.bf16_groups.append(trainable_parameters)

            # create flat bf16 params
            self.bf16_groups_flat.append(
                self._flatten_dense_tensors_aligned(self.bf16_groups[i],
                                                    self.nccl_start_alignment_factor * dp_world_size))

            # Make bf16 params point to flat tensor storage
            self._update_storage_to_flattened_tensor(tensor_list=self.bf16_groups[i],
                                                     flat_tensor=self.bf16_groups_flat[i])

            # divide flat weights into equal sized partitions
            partition_size = self.bf16_groups_flat[i].numel() // dp_world_size
            bf16_dp_partitions = [
                self.bf16_groups_flat[i].narrow(0, dp_index * partition_size, partition_size)
                for dp_index in range(dp_world_size)
            ]
            self.bf16_partitioned_groups.append(bf16_dp_partitions)

            # create fp32 params partition
            self.fp32_groups_flat_partition.append(bf16_dp_partitions[partition_id].clone().float().detach())
            self.fp32_groups_flat_partition[i].requires_grad = True

            num_elem_list = [t.numel() for t in self.bf16_groups[i]]

            # create fp32 gradients
            self.fp32_groups_gradients_flat.append(torch.zeros_like(self.bf16_groups_flat[i], dtype=torch.float32))

            # track individual fp32 gradients for entire model
            fp32_gradients = self._split_flat_tensor(flat_tensor=self.fp32_groups_gradients_flat[i],
                                                     num_elem_list=num_elem_list)
            self.fp32_groups_gradients.append(fp32_gradients)
            self.fp32_groups_gradient_dict[i] = fp32_gradients

            # flat tensor corresponding to actual fp32 gradients (i.e., minus alignment padding)
            length_without_padding = sum(num_elem_list)
            self.fp32_groups_actual_gradients_flat.append(
                torch.narrow(self.fp32_groups_gradients_flat[i], 0, 0, length_without_padding))

            # flat tensor corresponding to gradient partition
            self.fp32_groups_gradient_flat_partition.append(
                torch.narrow(self.fp32_groups_gradients_flat[i], 0, partition_id * partition_size, partition_size))

            # track fp32 gradient updates
            self.fp32_groups_has_gradients.append([False] * len(self.bf16_groups[i]))

            # Record padding required for alignment
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                padding = self.bf16_groups_flat[i].numel() - length_without_padding
            else:
                padding = 0

            self.group_paddings.append(padding)

            # update optimizer param groups to reference fp32 params partition
            param_group['params'] = [self.fp32_groups_flat_partition[i]]

            see_memory_usage(f'after initializing group {i}', force=True)

        see_memory_usage('before initialize_optimizer', force=True)
        self.initialize_optimizer_states()
        see_memory_usage('end initialize_optimizer', force=True)

        # Need optimizer states initialized before linking lp to optimizer state
        self._link_all_hp_params()
        self._enable_universal_checkpoint()
        self._param_slice_mappings = self._create_param_mapping()

    def _enable_universal_checkpoint(self):
        for lp_param_group in self.bf16_groups:
            enable_universal_checkpoint(param_list=lp_param_group)

    def _create_param_mapping(self):
        param_mapping = []
        for i, _ in enumerate(self.optimizer.param_groups):
            param_mapping_per_group = OrderedDict()
            for lp in self.bf16_groups[i]:
                if lp._hp_mapping is not None:
                    lp_name = self.param_names[lp]
                    param_mapping_per_group[lp_name] = lp._hp_mapping.get_hp_fragment_address()
            param_mapping.append(param_mapping_per_group)

        return param_mapping

    def _link_all_hp_params(self):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        for i, _ in enumerate(self.optimizer.param_groups):
            # Link bf16 and fp32 params in partition
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            partition_size = self.bf16_groups_flat[i].numel() // dp_world_size
            flat_hp_partition = self.fp32_groups_flat_partition[i]
            link_hp_params(lp_param_list=self.bf16_groups[i],
                           flat_hp_partition=flat_hp_partition,
                           gradient_dict=self.fp32_groups_gradient_dict,
                           offload_gradient_dict=None,
                           use_offload=False,
                           param_group_index=i,
                           partition_start=partition_id * partition_size,
                           partition_size=partition_size,
                           partition_optimizer_state=self.optimizer.state[flat_hp_partition],
                           dp_group=self.real_dp_process_group[i])

    def initialize_optimizer_states(self):
        """Take an optimizer step with zero-valued gradients to allocate internal
        optimizer state.

        This helps prevent memory fragmentation by allocating optimizer state at the
        beginning of training instead of after activations have been allocated.
        """
        for param_partition, grad_partition in zip(self.fp32_groups_flat_partition,
                                                   self.fp32_groups_gradient_flat_partition):
            param_partition.grad = grad_partition

        self.optimizer.step()

        self.clear_hp_grads()

    def _split_flat_tensor(self, flat_tensor, num_elem_list):
        assert sum(num_elem_list) <= flat_tensor.numel()
        tensor_list = []
        offset = 0
        for num_elem in num_elem_list:
            dense_tensor = torch.narrow(flat_tensor, 0, offset, num_elem)
            tensor_list.append(dense_tensor)
            offset += num_elem

        return tensor_list

    def _update_storage_to_flattened_tensor(self, tensor_list, flat_tensor):
        updated_params = self.unflatten(flat_tensor, tensor_list)
        for p, q in zip(tensor_list, updated_params):
            p.data = q.data

    def _flatten_dense_tensors_aligned(self, tensor_list, alignment):
        return self.flatten(align_dense_tensors(tensor_list, alignment))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError(f'{self.__class__} does not support closure.')

        all_groups_norm = get_global_norm_of_tensors(input_tensors=self.get_grads_for_norm(),
                                                     mpu=self.mpu,
                                                     norm_type=self.norm_type)
        self._global_grad_norm = all_groups_norm

        assert all_groups_norm > 0.
        if self.clip_grad > 0.:
            clip_tensors_by_global_norm(input_tensors=self.get_grads_for_norm(for_clipping=True),
                                        max_norm=self.clip_grad,
                                        global_norm=all_groups_norm,
                                        mpu=self.mpu)

        self.optimizer.step()

        self.update_lp_params()

        self.clear_hp_grads()
        self.step_count += 1

    def backward(self, loss, update_hp_grads=True, clear_lp_grads=False, **bwd_kwargs):
        """Perform a backward pass and copy the low-precision gradients to the
        high-precision copy.

        We copy/accumulate to the high-precision grads now to prevent accumulating in the
        bf16 grads after successive backward() calls (i.e., grad accumulation steps > 1)

        The low-precision grads are deallocated during this procedure.
        """
        self.clear_lp_grads()
        loss.backward(**bwd_kwargs)

        if update_hp_grads:
            self.update_hp_grads(clear_lp_grads=clear_lp_grads)

    @torch.no_grad()
    def update_hp_grads(self, clear_lp_grads=False):
        for i, group in enumerate(self.bf16_groups):
            for j, lp in enumerate(group):
                if lp.grad is None:
                    continue

                hp_grad = self.fp32_groups_gradients[i][j]
                assert hp_grad is not None, \
                    f'high precision param has no gradient, lp param_id = {id(lp)} group_info = [{i}][{j}]'

                hp_grad.data.add_(lp.grad.data.to(hp_grad.dtype).view(hp_grad.shape))
                lp._hp_grad = hp_grad
                self.fp32_groups_has_gradients[i][j] = True

                # clear gradients
                if clear_lp_grads:
                    lp.grad = None

    @torch.no_grad()
    def get_grads_for_reduction(self):
        return self.fp32_groups_gradients_flat

    @torch.no_grad()
    def get_grads_for_norm(self, for_clipping=False):
        grads = []
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        for i, group in enumerate(self.bf16_groups):
            for j, lp in enumerate(group):
                if not for_clipping:
                    if hasattr(lp, PIPE_REPLICATED) and lp.ds_pipe_replicated:
                        continue

                    if not (tensor_mp_rank == 0 or is_model_parallel_parameter(lp)):
                        continue

                if not self.fp32_groups_has_gradients[i][j]:
                    continue

                grads.append(self.fp32_groups_gradients[i][j])

        return grads

    @torch.no_grad()
    def update_lp_params(self):
        for i, (bf16_partitions,
                fp32_partition) in enumerate(zip(self.bf16_partitioned_groups, self.fp32_groups_flat_partition)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            bf16_partitions[partition_id].data.copy_(fp32_partition.data)
            # print_rank_0(f'update_lp_params {i=} {partition_id=}', force=True)
            # if i == 0:
            #     print_rank_0(f'{fp32_partition[:10]=}', force=True)

        all_gather_dp_groups(partitioned_param_groups=self.bf16_partitioned_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)

    def clear_hp_grads(self):
        for flat_gradients in self.fp32_groups_gradients_flat:
            flat_gradients.zero_()

        for i, group in enumerate(self.fp32_groups_gradients):
            self.fp32_groups_has_gradients[i] = [False] * len(group)

    def clear_lp_grads(self):
        for group in self.bf16_groups:
            for param in group:
                param.grad = None

    def state_dict(self):
        state_dict = {}
        state_dict[CLIP_GRAD] = self.clip_grad
        state_dict[BASE_OPTIMIZER_STATE] = self.optimizer.state_dict()
        state_dict[SINGLE_PARTITION_OF_FP32_GROUPS] = self.fp32_groups_flat_partition
        state_dict[GROUP_PADDINGS] = self.group_paddings
        state_dict[PARTITION_COUNT] = self.partition_count
        state_dict[DS_VERSION] = version
        state_dict[PARAM_SLICE_MAPPINGS] = self._param_slice_mappings

        return state_dict

    # Restore base optimizer fp32 weights bfloat16 weights
    def _restore_from_bit16_weights(self):
        for i, group in enumerate(self.bf16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            for bf16_partitions, fp32_partition in zip(self.bf16_partitioned_groups, self.fp32_groups_flat_partition):
                fp32_partition.data.copy_(bf16_partitions[partition_id].data)

    def refresh_fp32_params(self):
        self._restore_from_bit16_weights()

    def load_state_dict(self,
                        state_dict_list,
                        checkpoint_folder,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False):
        if checkpoint_folder:
            self._load_universal_checkpoint(checkpoint_folder, load_optimizer_states, load_from_fp32_weights)
        else:
            self._load_legacy_checkpoint(state_dict_list, load_optimizer_states, load_from_fp32_weights)

    def _load_legacy_checkpoint(self, state_dict_list, load_optimizer_states=True, load_from_fp32_weights=False):

        dp_rank = dist.get_rank(group=self.dp_process_group)
        current_rank_sd = state_dict_list[dp_rank]

        ckpt_version = current_rank_sd.get(DS_VERSION, False)
        assert ckpt_version, f"Empty ds_version in checkpoint, not clear how to proceed"
        ckpt_version = pkg_version.parse(ckpt_version)

        self.clip_grad = current_rank_sd.get(CLIP_GRAD, self.clip_grad)

        if load_optimizer_states:
            self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])

        if load_from_fp32_weights:
            for current, saved in zip(self.fp32_groups_flat_partition,
                                      current_rank_sd[SINGLE_PARTITION_OF_FP32_GROUPS]):
                src_tensor = _get_padded_tensor(saved, current.numel())
                current.data.copy_(src_tensor.data)

        if load_optimizer_states:
            self._link_all_hp_params()

    def _load_universal_checkpoint(self, checkpoint_folder, load_optimizer_states, load_from_fp32_weights):
        self._load_hp_checkpoint_state(checkpoint_folder)

    @property
    def param_groups(self):
        """Forward the wrapped optimizer's parameters."""
        return self.optimizer.param_groups

    def _load_hp_checkpoint_state(self, checkpoint_dir):
        checkpoint_dir = os.path.join(checkpoint_dir, "zero")
        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        tp_world_size = self.mpu.get_slice_parallel_world_size()

        for i, _ in enumerate(self.optimizer.param_groups):
            for lp in self.bf16_groups[i]:
                if lp._hp_mapping is not None:
                    #print(f"Loading {self.param_names[lp]} {tp_rank=} {tp_world_size=}")
                    lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, self.param_names[lp]), tp_rank,
                                                tp_world_size)


def _get_padded_tensor(src_tensor, size):
    if src_tensor.numel() >= size:
        return src_tensor
    padded_tensor = torch.zeros(size, dtype=src_tensor.dtype, device=src_tensor.device)
    slice_tensor = torch.narrow(padded_tensor, 0, 0, src_tensor.numel())
    slice_tensor.data.copy_(src_tensor.data)
    return padded_tensor
