# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import OrderedDict
import torch
import sys
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed import comm as dist
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.runtime.base_optimizer import ZeROOptimizer
from packaging import version as pkg_version
from deepspeed.git_version_info import version
from deepspeed.runtime.utils import (get_global_norm_of_tensors, clip_tensors_by_global_norm, DummyOptim,
                                     align_dense_tensors, all_gather_dp_groups, is_model_parallel_parameter,
                                     see_memory_usage, graph_process, get_norm_with_moe_layers)
from deepspeed.utils import link_hp_params, lazy_init_hp_params_optimizer_state, fragment_address, groups
from deepspeed.moe.utils import is_moe_param, is_moe_param_group
from deepspeed.utils.bwc import bwc_tensor_model_parallel_rank
from deepspeed.utils.torch import register_grad_hook
from deepspeed.checkpoint import enable_universal_checkpoint
from deepspeed.checkpoint.constants import (DS_VERSION, PARTITION_COUNT, BASE_OPTIMIZER_STATE,
                                            SINGLE_PARTITION_OF_FP32_GROUPS, CLIP_GRAD, GROUP_PADDINGS,
                                            PARAM_SLICE_MAPPINGS)

setattr(sys.modules[__name__], 'fragment_address', fragment_address)


def print_rank_0(message, debug=False, force=False):
    if dist.get_rank() == 0 and (debug or force):
        print(message)


def should_preserve_dtype(param):
    """Check if parameter should preserve its original dtype.
    
    This function allows specific parameters to opt-out of bf16 conversion
    by checking for the presence of a 'preserve_dtype' attribute.
    
    Args:
        param (torch.nn.Parameter): The parameter to check
        
    Returns:
        bool: True if parameter should preserve its dtype, False otherwise
    """
    return hasattr(param, 'preserve_dtype') and param.preserve_dtype


class BF16_Optimizer(ZeROOptimizer):

    def __init__(self,
                 init_optimizer,
                 param_names,
                 bfloat16_config,
                 mpu=None,
                 clip_grad=0.0,
                 norm_type=2,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 timers=None,
                 grad_acc_dtype=None,
                 graph_harvesting=False,
                 has_moe_layers=False):
        super().__init__()
        see_memory_usage('begin bf16_optimizer', force=True)
        self.timers = timers
        self.optimizer = init_optimizer
        self.param_names = param_names
        self.using_real_optimizer = not isinstance(self.optimizer, DummyOptim)

        assert bfloat16_config.enabled, "BF16Optimizer: requires bfloat16 to be enabled"
        assert grad_acc_dtype in [torch.float32, torch.bfloat16
                                  ], f"BF16Optimizer: Unsupported gradient accumulation data type: {grad_acc_dtype}"
        self.grad_acc_dtype = grad_acc_dtype

        self.immediate_grad_update = bfloat16_config.immediate_grad_update

        self.clip_grad = clip_grad
        self.norm_type = norm_type
        self.mpu = mpu
        self.allgather_bucket_size = int(allgather_bucket_size)
        self.dp_process_group = dp_process_group
        self.dp_rank = dist.get_rank(group=self.dp_process_group)
        self.has_moe_layers = has_moe_layers
        self.non_expert_gradients = []
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        if self.has_moe_layers:
            self._configure_moe_settings()

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

        self.group_paddings = []
        self.graph_harvesting = graph_harvesting
        if self.using_real_optimizer:
            self._setup_for_real_optimizer()

        see_memory_usage('end bf16_ optimizer', force=True)

    def destroy(self):
        for i, _ in enumerate(self.optimizer.param_groups):
            for p in self.bf16_groups[i]:
                if getattr(p, '_hp_mapping', None):
                    p._hp_mapping = None
        for hook in self._grad_acc_hooks:
            hook.remove()
        print_rank_0("Removed grad acc hooks")

    def _configure_moe_settings(self):
        assert any(
            [is_moe_param_group(group) for group in self.optimizer.param_groups]
        ), "The model has moe layers, but None of the param groups are marked as MoE. Create a param group with 'moe' key set to True before creating optimizer"

        for i, group in enumerate(self.optimizer.param_groups):
            if is_moe_param_group(group):
                assert all([is_moe_param(param)
                            for param in group['params']]), "All params in MoE group must be MoE params"
                self.real_dp_process_group[i] = groups._get_expert_data_parallel_group(group['name'])
        self.expert_gradients = {}
        if self.has_moe_layers:
            for key in groups._get_expert_data_parallel_group_dict().keys():
                self.expert_gradients[key] = []

    def _setup_for_real_optimizer(self):
        self.partition_count = [dist.get_world_size(group=pg) for pg in self.real_dp_process_group]

        for i, param_group in enumerate(self.optimizer.param_groups):
            real_dp_world_size = dist.get_world_size(group=self.real_dp_process_group[i])
            see_memory_usage(f'before initializing group {i}', force=True)

            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # grab the original list
            trainable_parameters = [param for param in param_group['params'] if param.requires_grad]
            self.bf16_groups.append(trainable_parameters)

            # create flat bf16 params
            self.bf16_groups_flat.append(
                self._flatten_dense_tensors_aligned(self.bf16_groups[i],
                                                    self.nccl_start_alignment_factor * real_dp_world_size))
            # Make bf16 params point to flat tensor storage
            self._update_storage_to_flattened_tensor(tensor_list=self.bf16_groups[i],
                                                     flat_tensor=self.bf16_groups_flat[i])

            # divide flat weights into equal sized partitions
            partition_size = self.bf16_groups_flat[i].numel() // real_dp_world_size
            bf16_dp_partitions = [
                self.bf16_groups_flat[i].narrow(0, dp_index * partition_size, partition_size)
                for dp_index in range(real_dp_world_size)
            ]
            self.bf16_partitioned_groups.append(bf16_dp_partitions)

            # create fp32 params partition - preserve original dtype for parameters that require it
            if any(should_preserve_dtype(param) for param in self.bf16_groups[i]):
                # For parameters that should preserve dtype, we need special handling
                preserved_params_fp32_partition = []
                for param in self.bf16_groups[i]:
                    if should_preserve_dtype(param):
                        # Keep original dtype for preserved parameters
                        param_data = param.data
                        if param_data.dtype != torch.float32:
                            param_data = param_data.float()
                        preserved_params_fp32_partition.append(param_data.detach().clone())
                    else:
                        # Convert to float32 for non-preserved parameters
                        preserved_params_fp32_partition.append(param.data.float().detach())
                
                # Create flat tensor from mixed dtype parameters
                flat_preserved = self.flatten(preserved_params_fp32_partition)
                self.fp32_groups_flat_partition.append(flat_preserved)
            else:
                # Original behavior for all parameters
                self.fp32_groups_flat_partition.append(bf16_dp_partitions[partition_id].clone().float().detach())
            
            self.fp32_groups_flat_partition[i].requires_grad = True

            num_elem_list = [t.numel() for t in self.bf16_groups[i]]

            # create fp32 gradients
            fp32_flat_buffer = torch.zeros_like(self.bf16_groups_flat[i], dtype=self.grad_acc_dtype)
            self.fp32_groups_gradients_flat.append(fp32_flat_buffer)
            if self.has_moe_layers and is_moe_param_group(param_group):
                self.expert_gradients[param_group['name']].append(fp32_flat_buffer)
            else:
                self.non_expert_gradients.append(fp32_flat_buffer)

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

    def _flatten_dense_tensors_aligned(self, tensor_list, alignment):
        return self.flatten(align_dense_tensors(tensor_list, alignment))

    def _update_storage_to_flattened_tensor(self, tensor_list, flat_tensor):
        """Update tensor storage to point to flattened tensor, preserving dtype for specific parameters."""
        flattened_tensors = self.unflatten(flat_tensor, tensor_list)
        
        for i, (param, flattened) in enumerate(zip(tensor_list, flattened_tensors)):
            if should_preserve_dtype(param):
                # For parameters that should preserve dtype, convert back to original dtype
                original_dtype = param.data.dtype
                if flattened.dtype != original_dtype:
                    flattened = flattened.to(original_dtype)
                param.data = flattened
            else:
                # Original behavior for other parameters
                param.data = flattened

    def _split_flat_tensor(self, flat_tensor, num_elem_list):
        """Split flat tensor into list of tensors with specified number of elements."""
        split_tensors = []
        start = 0
        for num_elem in num_elem_list:
            end = start + num_elem
            split_tensors.append(flat_tensor[start:end])
            start = end
        return split_tensors

    # ... rest of the class methods remain unchanged

