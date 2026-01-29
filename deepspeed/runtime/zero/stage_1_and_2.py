# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from packaging import version as pkg_version
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed.runtime.zenflow import zenflow_utils

import gc
from typing import Container
from deepspeed.runtime.zero.offload_states import offload_optimizer_states, reload_optimizer_states
from deepspeed.runtime.base_optimizer import ZeROOptimizer
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler
from deepspeed.runtime.torch_autocast import get_autocast_dtype, get_all_comm_dtypes, is_autocast_initialized, sort_dtypes
from deepspeed.runtime.utils import (empty_cache, see_memory_usage, inf, is_model_parallel_parameter,
                                     align_dense_tensors, all_gather_dp_groups, mask_nan_or_inf_with_val_inplace,
                                     count_used_parameters_in_backward)
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils import logger
from deepspeed.utils.torch import register_grad_hook
from deepspeed.utils.bwc import bwc_tensor_model_parallel_rank
from deepspeed.moe.utils import is_moe_param
from deepspeed.git_version_info import version

from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.muon.original_muon import muon_update
from deepspeed.checkpoint.constants import (DS_VERSION, GROUP_PADDINGS, PARTITION_COUNT, LOSS_SCALER,
                                            SINGLE_PARTITION_OF_FP32_GROUPS, BASE_OPTIMIZER_STATE,
                                            BASE_OPTIMIZER_STATE_STEP, CLIP_GRAD, ZERO_STAGE, PARAM_SLICE_MAPPINGS)
from deepspeed.utils import link_hp_params, lazy_init_hp_params_optimizer_state
from deepspeed.checkpoint import enable_universal_checkpoint

from deepspeed.utils import groups
from deepspeed.utils.debug import debug_param2name
# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False

OPTIMIZER_ALLGATHER_TIMER = 'optimizer_allgather'
OPTIMIZER_GRADIENTS_TIMER = 'optimizer_gradients'
OPTIMIZER_STEP_TIMER = 'optimizer_step'
OPTIMIZER_TIMERS = [OPTIMIZER_ALLGATHER_TIMER, OPTIMIZER_GRADIENTS_TIMER, OPTIMIZER_STEP_TIMER]
INITIAL_MICRO_STEP_ID = -1


def input(msg):
    return


def split_half_float_double(tensors):
    device_type = get_accelerator().device_name()
    dtypes = [
        "torch.{}.HalfTensor".format(device_type), "torch.{}.FloatTensor".format(device_type),
        "torch.{}.DoubleTensor".format(device_type), "torch.{}.BFloat16Tensor".format(device_type)
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def isclose(a, b, rtol=1e-09, atol=0.0):
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def lcm(x, y):
    from fractions import gcd  # or can import gcd from `math` in Python 3
    return x * y // gcd(x, y)


def get_alignment_padding(tensor_list, alignment):
    num_elements = sum([tensor.numel() for tensor in tensor_list])
    remainder = num_elements % alignment
    return (alignment - remainder) if remainder else remainder


def print_rank_msg(msg):
    print(f"rank {dist.get_rank()} - {msg}")


def _get_padded_tensor(src_tensor, size):
    if src_tensor.numel() >= size:
        return src_tensor
    padded_tensor = torch.zeros(size, dtype=src_tensor.dtype, device=src_tensor.device)
    slice_tensor = torch.narrow(padded_tensor, 0, 0, src_tensor.numel())
    slice_tensor.data.copy_(src_tensor.data)
    return padded_tensor


def _pad_tensor_by_size(src_tensor, pad_size, dtype, device):
    padded_tensor = torch.zeros(src_tensor.numel() + pad_size, dtype=dtype, device=device)
    padded_tensor.data[:src_tensor.numel()].copy_(src_tensor.data)
    return padded_tensor


@dataclass
class IPGBucket:
    buffer: List[torch.Tensor] = field(default_factory=list)
    params: List[torch.Tensor] = field(default_factory=list)
    grads: List[torch.Tensor] = field(default_factory=list)
    elements: int = 0
    index: int = 0
    has_moe_params: bool = False

    def clear(self):
        self.params.clear()
        self.grads.clear()
        self.elements = 0
        self.has_moe_params = False


class DeepSpeedZeroOptimizer(ZeROOptimizer):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """

    def __init__(self,
                 init_optimizer,
                 param_names,
                 timers,
                 optimizer_params,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 use_multi_rank_bucket_allreduce=True,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 expert_parallel_group=None,
                 expert_data_parallel_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 zenflow_config=None,
                 mpu=None,
                 clip_grad=0.0,
                 gradient_accumulation_dtype=torch.float32,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 ignore_unused_parameters=True,
                 partition_grads=True,
                 round_robin_gradients=False,
                 has_moe_layers=False,
                 fp16_master_weights_and_gradients=False,
                 bf16_master_weights_and_gradients=False,
                 bf16_optimizer_states=False,
                 elastic_checkpoint=False,
                 check_grad_overflow=True):

        super().__init__()

        if offload_optimizer_config is not None and offload_optimizer_config.device != OffloadDeviceEnum.none:
            self.cpu_offload = True
            self.cpu_offload_pin_memory = offload_optimizer_config.pin_memory
        else:
            self.cpu_offload = False
            self.cpu_offload_pin_memory = False

        # TODO: Remove zenflow-specific call from vanilla ZeroOptimizer, try to isolate zenflow-specific code into sub-class zenflow_zero_optimizer
        self.zenflow = True if zenflow_config is not None else False

        if dist.get_rank() == 0:
            logger.info(f"Reduce bucket size {reduce_bucket_size}")
            logger.info(f"Allgather bucket size {allgather_bucket_size}")
            logger.info(f"CPU Offload: {self.cpu_offload}")
            logger.info(f'Round robin gradient partitioning: {round_robin_gradients}')
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        self.elastic_checkpoint = elastic_checkpoint
        self.check_grad_overflow = check_grad_overflow
        self.param_names = param_names
        self.mpu = mpu
        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
        if not get_accelerator().is_available():
            raise SystemError("Accelerator is not detected, cannot perform low precision training (e.g., fp16, bf16).")
        self.optimizer = init_optimizer

        # Use torch or zenflow (un)flatten ops
        self.flatten = _flatten_dense_tensors if not self.zenflow else zenflow_utils._flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors if not self.zenflow else zenflow_utils._unflatten_dense_tensors

        # ZeRO stage 1 (False) or 2 (True)
        self.partition_gradients = partition_grads
        self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"

        self.timers = timers

        self.reduce_scatter = reduce_scatter

        self.overlap_comm = overlap_comm

        self.deepspeed_adam_offload = self.cpu_offload

        self.device = get_accelerator().current_device_name() if not self.cpu_offload else 'cpu'

        self.dp_process_group = dp_process_group
        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        #expert parallel group
        self.ep_process_group = expert_parallel_group

        #data parallel group for experts
        self.expert_dp_process_group = expert_data_parallel_group

        #data parallel size for non-experts
        dp_size = dist.get_world_size(group=self.dp_process_group)

        #For MoE models this maybe different for different param group
        #It will be modified during MoE setup later in the init
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]

        self.is_gradient_accumulation_boundary = True

        # CPU-Offload requires contiguous gradients
        self.contiguous_gradients = contiguous_gradients or self.cpu_offload

        self.has_moe_layers = has_moe_layers
        if self.has_moe_layers:
            self._configure_moe_settings()
        self._global_grad_norm = 0.

        if mpu is None or hasattr(mpu, 'initialize_sequence_parallel'):
            self.model_parallel_group = None
            self.model_parallel_world_size = 1
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_world_size = mpu.get_model_parallel_world_size()
            self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)

        self.overflow = False
        self.clip_grad = clip_grad
        self.communication_data_type = communication_data_type
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = INITIAL_MICRO_STEP_ID
        self.ignore_unused_parameters = ignore_unused_parameters
        self.round_robin_gradients = round_robin_gradients

        self.extra_large_param_to_reduce: Dict[int, torch.Tensor] = {}

        def _enforce_cpu_offload():
            assert self.cpu_offload and type(self.optimizer) in [DeepSpeedCPUAdam], \
                f"Master weights feature requires {self.zero_stage_string} Offload with DeepSpeedCPUAdam. " \
                f"Current ZeRO-Offload:{self.cpu_offload} optimizer type {type(self.optimizer)}."

        self.master_weights_and_grads_dtype = self._configure_master_weights(
            fp16_master_weights_and_gradients=fp16_master_weights_and_gradients,
            bf16_master_weights_and_gradients=bf16_master_weights_and_gradients,
            bf16_optimizer_states=bf16_optimizer_states,
            fp16_offload_validator=_enforce_cpu_offload,
            bf16_fp32_offload_validator=_enforce_cpu_offload)

        self.low_precision_master_weights_and_grads = self.master_weights_and_grads_dtype != torch.float32

        if self.reduce_scatter and self.partition_gradients:
            valid_reduce_scatter_dtypes = (torch.float16, torch.bfloat16, torch.float32)
            assert self.communication_data_type in valid_reduce_scatter_dtypes, f"{self.zero_stage_string} supports {valid_reduce_scatter_dtypes} communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, f"gradient_predivide_factor != 1.0 is not yet supported with {self.zero_stage_string} with reduce scatter enabled"
            assert self.postscale_gradients, f"pre-scale gradients is not yet supported with {self.zero_stage_string} with reduce scatter enabled"

        # param flattened by groups
        self.bit16_groups = []
        self.bit16_groups_flat = []

        # param partitioned by data parallel degree
        # this will contain a list of equal sized tensors
        # each of which will be updated by a different process
        self.parallel_partitioned_bit16_groups = []

        # a single 32-bit partition of the parallel partitioned parameters
        # that this process will update
        self.single_partition_of_fp32_groups = []

        # a 16-bit CPU param buffer for cpu offload
        if self.cpu_offload:
            self.param_buffer_of_bit16_for_cpu_offload_groups = []

        # param partition info

        # These are the parameters in each group that will not be updated by this process directly
        self.params_not_in_partition = []

        # These are the parameters that will be updated by this process directly
        self.params_in_partition = []

        # Offset from the first parameter in the self.params_in_partition
        # the parameter boundaries may not align with partition boundaries
        # so we need to keep track of the offset
        self.first_offset = []

        # number of elements per partition in each group
        self.partition_size = []

        # align nccl all-gather send buffers to 4-byte boundary
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        assert (
            allgather_bucket_size % self.nccl_start_alignment_factor == 0
        ), f"allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor} "

        self.all_reduce_print = False
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        self.gradient_accumulation_dtype = gradient_accumulation_dtype

        if self.dtype != self.gradient_accumulation_dtype:
            self.use_separate_grad_accum = True
        else:
            self.use_separate_grad_accum = False
        if self.use_separate_grad_accum and not self.partition_gradients:
            self.use_grad_accum_attribute = True
        else:
            self.use_grad_accum_attribute = False

        self.round_robin_bit16_groups = []
        self.round_robin_bit16_indices = []
        self.round_robin_bit16_meta = []

        # Use different parallel to do all_to_all_reduce related things
        # padding on each partition for alignment purposes
        self.groups_padding = []
        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # push this group to list before modify
            # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
            trainable_parameters = []
            for param in param_group['params']:
                if param.requires_grad:
                    param.grad_accum = None
                    param.param_idx_in_group = len(trainable_parameters)
                    trainable_parameters.append(param)
            self.bit16_groups.append(trainable_parameters)

            # not sure why apex was cloning the weights before flattening
            # removing cloning here

            see_memory_usage(f"Before moving param group {i} to CPU")
            # move all the parameters to cpu to free up GPU space for creating flat buffer

            # Create temp CPU param copies, free accelerator tensors
            orig_group_numel = 0
            for param in self.bit16_groups[i]:
                orig_group_numel += param.numel()
                param.cpu_data = param.data.cpu()
                param.data = torch.empty(1).to(param.device)

            empty_cache()
            see_memory_usage(f"After moving param group {i} to CPU", force=False)

            # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
            # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
            # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
            # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
            if self.round_robin_gradients:
                round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                    self.bit16_groups[i], dist.get_world_size(group=self.real_dp_process_group[i]))
            else:
                round_robin_tensors = self.bit16_groups[i]
                round_robin_indices = list(range(len(self.bit16_groups[i])))

            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)

            # Create meta tensors list, ordered according to round_robin_tensors
            meta_tensors = []
            for param in round_robin_tensors:
                meta_tensors.append(torch.zeros_like(param.cpu_data, device="meta"))
            self.round_robin_bit16_meta.append(meta_tensors)

            # create flat buffer in CPU
            flattened_buffer = self.flatten_dense_tensors_aligned(
                self.round_robin_bit16_groups[i],
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]),
                use_cpu_data=True)

            # free temp CPU params
            for param in self.bit16_groups[i]:
                del param.cpu_data

            # Move CPU flat tensor to the accelerator memory.
            self.bit16_groups_flat.append(flattened_buffer.to(get_accelerator().current_device_name()))
            del flattened_buffer

            see_memory_usage(f"After flattening and moving param group {i} to GPU", force=False)

            if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
                see_memory_usage(f"After Flattening and after emptying param group {i} cache", force=False)

            # set model bit16 weight to slices of flattened buffer
            self._update_model_bit16_weights(i)

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

            # Record padding required for alignment
            left_boundary = sum([t.numel() for t in data_parallel_partitions[:partition_id]])
            curr_partition_size = data_parallel_partitions[partition_id].numel()

            if orig_group_numel <= left_boundary:
                padding = curr_partition_size
            elif orig_group_numel < left_boundary + curr_partition_size:
                padding = left_boundary + curr_partition_size - orig_group_numel
            else:
                padding = 0
            self.groups_padding.append(padding)

            # verify that data partition start locations are 4-byte aligned
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0)

            # A partition of the fp32 master weights that will be updated by this process.
            # Note that the params in single_partition_of_fp32_groups is cloned and detached
            # from the origin params of the model.
            weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id].detach().clone().to(
                device=self.device, dtype=self.master_weights_and_grads_dtype)

            if self.cpu_offload:
                if self.cpu_offload_pin_memory:
                    weights_partition = get_accelerator().pin_memory(weights_partition)
                temp_dtype = self.parallel_partitioned_bit16_groups[i][partition_id].dtype
                temp_buffer_bit16 = torch.full(weights_partition.shape,
                                               fill_value=0.0,
                                               dtype=temp_dtype,
                                               device=weights_partition.device)
                if self.cpu_offload_pin_memory:
                    temp_pinned = get_accelerator().pin_memory(temp_buffer_bit16)
                    self.param_buffer_of_bit16_for_cpu_offload_groups.append(temp_pinned)
                else:
                    self.param_buffer_of_bit16_for_cpu_offload_groups.append(temp_buffer_bit16)

            self.single_partition_of_fp32_groups.append(weights_partition)

            # Set local optimizer to have flat params of its own partition.
            # After this, the local optimizer will only contain its own partition of params.
            # In that case, the local optimizer only saves the states(momentum, variance, etc.) related to its partition's params(zero stage1).
            self.single_partition_of_fp32_groups[
                i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]

            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
                self.round_robin_bit16_groups[i], partition_size, partition_id)

            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)

        self.reduce_bucket_size = int(reduce_bucket_size)
        self.use_multi_rank_bucket_allreduce = use_multi_rank_bucket_allreduce
        self.allgather_bucket_size = int(allgather_bucket_size)

        self.reduction_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream()
        #self.copy_grad_stream = get_accelerator().Stream()
        self.callback_queued = False

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.torch_autocast_gradscaler = None
        if is_autocast_initialized():
            comm_dtypes = get_all_comm_dtypes([p for params in self.bit16_groups for p in params])
            if get_autocast_dtype() == torch.float16:
                self.torch_autocast_gradscaler = torch.amp.GradScaler(device=get_accelerator().device_name())
        else:
            comm_dtypes = {self.communication_data_type}

        self.ipg_buckets: Dict[torch.dtype, IPGBucket] = {dtype: IPGBucket() for dtype in comm_dtypes}

        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads: Dict[int, List[torch.Tensor]] = defaultdict(list)

        # simplified param id
        self.param_id = {}

        #interesting code: unique ids being assigned to individual parameters
        largest_param_numel = 0
        count = 0
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1

        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True

        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False

        if self.cpu_offload:
            self.accumulated_grads_in_cpu = {}
            self.norm_for_param_grads = {}
            self.local_overflow = False
            self.grad_position = {}
            self.temp_grad_buffer_for_cpu_offload = torch.zeros(largest_param_numel,
                                                                device=self.device,
                                                                dtype=self.dtype)
            if self.cpu_offload_pin_memory:
                self.temp_grad_buffer_for_cpu_offload = get_accelerator().pin_memory(
                    self.temp_grad_buffer_for_cpu_offload)
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel,
                                                                device=get_accelerator().current_device_name(),
                                                                dtype=self.dtype)
            for i, params_group in enumerate(self.bit16_groups):
                self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])

        # mapping from parameter to partition that it belongs to
        self.param_to_partition_ids = {}

        # stores if a partition has been reduced in this step
        self.is_partition_reduced = {}

        # number of grads in partition that still need to be computed
        self.remaining_grads_in_partition = {}

        # total number of grads in partition
        self.total_grads_in_partition = {}

        # stores if a grad in a partition has been computed or not
        self.is_grad_computed = {}

        # stores the offset at which a parameter gradient needs to be inserted in a partition
        self.grad_partition_insertion_offset = {}

        # the offset in the gradient at which it must be inserted at the beginning of the partition
        self.grad_start_offset = {}

        # will store the averaged gradients required by this partition
        self.averaged_gradients = {}
        self.all_grad_tensors = {}
        # For cpu_offload, will store the averaged gradients required by this partition
        self.offload_gradient_dict = {}

        # store index of first parameter in each partition
        self.first_param_index_in_partition = {}

        # initializes all data structures for implementing gradient partitioning
        self.initialize_gradient_partitioning_data_structures()

        # resets the data structure value for the next backward propagation
        self.reset_partition_gradient_structures()

        # creates backward hooks for the following special handling of gradients
        # 1. upcasting for fp32 gradient accumulation
        # 2. gradient partitioning
        # 3. overlapping backward and reduction
        self._grad_acc_hooks = []

        if (self.partition_gradients or self.overlap_comm or self.use_grad_accum_attribute
                or self.contiguous_gradients):
            self.create_gradient_handling_hooks()

        self.ready_for_gradients = False
        self.custom_loss_scaler = False
        self.external_loss_scale = None

        # we may have a way of fusing dynamic scale. Do not support for now
        self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_scaling=dynamic_loss_scale,
                                            dynamic_loss_args=dynamic_loss_args)
        self.dynamic_loss_scale = self.loss_scaler.dynamic

        if self.dtype != torch.float16:
            # Only fp16 should use dynamic loss scaling
            assert self.loss_scaler.cur_scale == 1.0
            assert not self.dynamic_loss_scale

        see_memory_usage("Before initializing optimizer states", force=True)
        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=True)

        if dist.get_rank() == 0:
            logger.info("optimizer state initialized")

        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage("After initializing ZeRO optimizer", force=True)

        self._link_all_hp_params()
        self._hp_optimizer_states_linked = False

        self._enable_universal_checkpoint()
        self._param_slice_mappings = self._create_param_mapping()
        if self.cpu_offload:
            self._create_optimizer_mapping()

        self.offloaded_states: Set[OffloadStateTypeEnum] = set()

    def destroy(self):
        for i, _ in enumerate(self.optimizer.param_groups):
            for p in self.bit16_groups[i]:
                if getattr(p, '_hp_mapping', None):
                    p._hp_mapping = None
        for hook in self._grad_acc_hooks:
            hook.remove()
        self.print_rank_0("Removed grad acc hooks")

    def _enable_universal_checkpoint(self):
        for lp_param_group in self.bit16_groups:
            enable_universal_checkpoint(param_list=lp_param_group)

    def _create_param_mapping(self):
        param_mapping = []
        for i, _ in enumerate(self.optimizer.param_groups):
            param_mapping_per_group = OrderedDict()
            for lp in self.bit16_groups[i]:
                if lp._hp_mapping is not None:
                    lp_name = self.param_names[lp]
                    param_mapping_per_group[lp_name] = lp._hp_mapping.get_hp_fragment_address()
            param_mapping.append(param_mapping_per_group)

        return param_mapping

    def _create_optimizer_mapping(self):
        for i, _ in enumerate(self.optimizer.param_groups):
            for lp in self.bit16_groups[i]:
                if lp._hp_mapping is not None:
                    lp._zero_optimizer = self

    def _link_all_hp_params(self):
        if self.cpu_offload:
            self._get_offload_gradient_dict()

        for i, _ in enumerate(self.optimizer.param_groups):
            # Link bit16 and fp32 params in partition
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            partition_size = self.bit16_groups_flat[i].numel() // dist.get_world_size(
                group=self.real_dp_process_group[i])
            flat_hp_partition = self.single_partition_of_fp32_groups[i]
            link_hp_params(lp_param_list=self.bit16_groups[i],
                           flat_hp_partition=flat_hp_partition,
                           gradient_dict=self.averaged_gradients,
                           offload_gradient_dict=self.offload_gradient_dict,
                           use_offload=self.cpu_offload,
                           param_group_index=i,
                           partition_start=partition_id * partition_size,
                           partition_size=partition_size,
                           dp_group=self.real_dp_process_group[i])

    def _lazy_init_hp_params_optimizer_state(self):
        if not self._hp_optimizer_states_linked:
            for i, _ in enumerate(self.optimizer.param_groups):
                lazy_init_hp_params_optimizer_state(self.bit16_groups[i], self.single_partition_of_fp32_groups[i],
                                                    self.optimizer.state)
            self._hp_optimizer_states_linked = True

    def is_moe_group(self, group):
        return 'moe' in group and group['moe']

    def _configure_moe_settings(self):
        # if we're using ZeRO stage 2, ensure contiguous gradients are used
        if self.partition_gradients:
            assert self.contiguous_gradients, "Contiguous Gradients in ZeRO Stage 2 must be set to True for MoE. Other code paths are not tested with MoE"
        # NOTE: To run ZeRO stage 1 with MoE, we need to set self.contiguous_gradients to True or ignore the assertion
        if not self.partition_gradients and not self.contiguous_gradients:
            logger.warning(
                "ZeRO Stage 1 has not been thoroughly tested with MoE. This configuration is still experimental.")
        assert self.reduce_scatter, "Reduce Scatter in ZeRO Stage 2 must be set to True for MoE. Other code paths are not tested with MoE"

        assert any(
            [self.is_moe_group(group) for group in self.optimizer.param_groups]
        ), "The model has moe layers, but None of the param groups are marked as MoE. Create a param group with 'moe' key set to True before creating optimizer"
        self.is_moe_param_group = []
        for i, group in enumerate(self.optimizer.param_groups):
            if self.is_moe_group(group):
                assert all([is_moe_param(param)
                            for param in group['params']]), "All params in MoE group must be MoE params"
                self.real_dp_process_group[i] = self.expert_dp_process_group[group['name']]
                self.partition_count[i] = dist.get_world_size(group=self.expert_dp_process_group[group['name']])
                self.is_moe_param_group.append(True)
            else:
                self.is_moe_param_group.append(False)

        assert self.expert_dp_process_group is not None, "Expert data parallel group should be configured with MoE"
        assert self.ep_process_group is not None, "Expert parallel group should be configured with MoE"

    def _update_model_bit16_weights(self, group_index):
        updated_params = self.unflatten(self.bit16_groups_flat[group_index], self.round_robin_bit16_meta[group_index])
        for p, q in zip(self.round_robin_bit16_groups[group_index], updated_params):
            p.data = q.data

        # set model fp16 weight to slices of reordered flattened buffer
        for param_index, param in enumerate(self.bit16_groups[group_index]):
            new_index = self.round_robin_bit16_indices[group_index][param_index]
            param.data = self.round_robin_bit16_groups[group_index][new_index].data

    def _round_robin_reorder(self, tensor_list, num_partitions):

        # disable round robin if need to debug something
        # return tensor_list, list(range(len(tensor_list)))

        partition_tensors = {}

        for i, tensor in enumerate(tensor_list):
            j = i % num_partitions
            if j not in partition_tensors:
                partition_tensors[j] = []
            partition_tensors[j].append((i, tensor))

        reordered_tensors = []
        reordered_indices = {}

        for partition_index in partition_tensors.keys():
            for i, (original_index, tensor) in enumerate(partition_tensors[partition_index]):
                reordered_indices[original_index] = len(reordered_tensors)
                reordered_tensors.append(tensor)

        return reordered_tensors, reordered_indices

    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            for bucket in self.ipg_buckets.values():
                bucket.buffer.clear()

            self.grads_in_partition = None
            self.grads_in_partition_offset = 0
        self.ready_for_gradients = False

    def initialize_optimizer_states(self):

        for i, group in enumerate(self.bit16_groups):
            single_grad_partition = torch.zeros(int(self.partition_size[i]),
                                                dtype=self.single_partition_of_fp32_groups[i].dtype,
                                                device=self.device)
            self.single_partition_of_fp32_groups[i].grad = get_accelerator().pin_memory(
                single_grad_partition) if self.cpu_offload_pin_memory else single_grad_partition

        # Initialize the optimizer states with the flattened fp32 partition.
        # State initialization for the Adagrad optimizer occurs at construction as opposed to other optimizers
        # which do lazy initialization of the state at the first call to step.
        if isinstance(self.optimizer, torch.optim.Adagrad):
            self.optimizer = torch.optim.Adagrad(self.single_partition_of_fp32_groups, **self.optimizer.defaults)

        if not self.cpu_offload:
            for group in self.single_partition_of_fp32_groups:
                group.grad = None  #class init

        return

    #########################################################################
    #################### ZeRO Stage 1 - reduce gradients ####################
    #########################################################################
    def reduce_gradients(self, pipeline_parallel=False):
        world_size = dist.get_world_size(self.dp_process_group)
        my_rank = dist.get_rank(self.dp_process_group)

        # with PP we must create ipg buffer, since backward is handled outside zero
        if pipeline_parallel and self.contiguous_gradients:
            for dtype, bucket in self.ipg_buckets.items():
                bucket.buffer.append(
                    torch.empty(int(self.reduce_bucket_size),
                                dtype=dtype,
                                device=get_accelerator().current_device_name()))
                bucket.index = 0

        if not self.overlap_comm:
            for i, group in enumerate(self.bit16_groups):
                for param in group:
                    grad_reduc = self.get_gradient_for_reduction(param)
                    if grad_reduc is not None:
                        self.reduce_ready_partitions_and_remove_grads(param, i)
        # reduce any pending grads in either hook/non-hook case
        self.overlapping_partition_gradients_reduce_epilogue()

    #########################################################################
    #########################ZeRO Partition Gradients########################
    #########################################################################

    def get_first_param_index(self, group_id, param_group, partition_id):
        for index, param in enumerate(param_group):
            param_id = self.get_param_id(param)
            if group_id in self.param_to_partition_ids and param_id in self.param_to_partition_ids[group_id]:
                if partition_id in self.param_to_partition_ids[group_id][param_id]:
                    return index
        return None

    def initialize_gradient_partitioning_data_structures(self):

        for i, param_group in enumerate(self.round_robin_bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            self.param_to_partition_ids[i] = {}
            self.is_partition_reduced[i] = {}
            self.total_grads_in_partition[i] = {}
            self.remaining_grads_in_partition[i] = {}
            self.is_grad_computed[i] = {}
            self.grad_partition_insertion_offset[i] = {}
            self.grad_start_offset[i] = {}
            self.first_param_index_in_partition[i] = {}

            for partition_id in range(total_partitions):
                self.is_grad_computed[i][partition_id] = {}
                self.grad_partition_insertion_offset[i][partition_id] = {}
                self.grad_start_offset[i][partition_id] = {}
                self.total_grads_in_partition[i][partition_id] = 0
                self.initialize_gradient_partition(i, param_group, partition_id)
                self.is_partition_reduced[i][partition_id] = False
                self.first_param_index_in_partition[i][partition_id] = self.get_first_param_index(
                    i, param_group, partition_id)

    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage("In ipg_epilogue before reduce_ipg_grads", 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage("In ipg_epilogue after reduce_ipg_grads", 0)

        # if dist.get_rank() == 0:
        #    logger.info("Params already reduced %s", self.params_already_reduced)
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False

        if self.overlap_comm:
            if not get_accelerator().resolves_data_dependency():
                get_accelerator().synchronize()
            # It is safe to clear previously reduced grads of other partitions
            self._clear_previous_reduced_grads()

        if self.cpu_offload is False:
            for i, _ in enumerate(self.bit16_groups):
                if i not in self.all_grad_tensors or self.all_grad_tensors[i] is None:
                    self.all_grad_tensors[i] = self.get_all_grad_tensors(self.params_in_partition[i],
                                                                         dtype=self.gradient_accumulation_dtype)
                else:
                    avg_new = self.get_all_grad_tensors(self.params_in_partition[i],
                                                        dtype=self.gradient_accumulation_dtype)
                    for accumulated_grad, new_avg_grad in zip(self.all_grad_tensors[i], avg_new):
                        accumulated_grad.add_(new_avg_grad)
                if self.is_gradient_accumulation_boundary:
                    self.averaged_gradients[i] = self.get_flat_partition(
                        self.params_in_partition[i],
                        self.first_offset[i],
                        self.partition_size[i],
                        dtype=self.gradient_accumulation_dtype,
                        device=get_accelerator().current_device_name(),
                        param_group_idx=i,
                        return_tensor_list=True)
                    # Clear all_grad_tensors after use. With reentrant checkpointing,
                    # the epilogue may run multiple times per backward pass. Each time,
                    # we read the cumulative grad_accum (which PyTorch naturally accumulates)
                    # and the final phase will have all gradients.
                    self.all_grad_tensors[i] = None

        self._release_ipg_buffers()

        # Clear param.grad so safe_get_full_grad() goes through the proper _hp_mapping
        # path (which does all_reduce for ZeRO-2). Keep grad_accum intact for reentrant
        # checkpointing where gradients need to accumulate across multiple phases.
        # grad_accum is cleared in clear_backward_seen_flag() at the start of next forward.
        self._clear_param_grad_only()
        self._epilogue_ran_this_backward = True

        see_memory_usage("End ipg_epilogue")

    def clear_backward_seen_flag(self):
        """Clear the backward seen flag and do deferred cleanup.

        With reentrant gradient checkpointing, the epilogue may run multiple times
        per backward pass (once per phase). We defer clearing grad_accum until here
        (called at the start of the next forward) to ensure all phases have completed.

        Note: param.grad is cleared in the epilogue via _clear_param_grad_only() to
        ensure safe_get_full_grad() works correctly. Only grad_accum is deferred.
        """
        if self._epilogue_ran_this_backward:
            # Clear grad_accum for next step. param.grad is already cleared in epilogue.
            for group in self.bit16_groups:
                for p in group:
                    p.grad_accum = None

        super().clear_backward_seen_flag()

    # resets all partition to no reduced
    # sets remaining grads to the total number of grads in each partition
    # set is grad computed to false for all grads in partition
    def reset_partition_gradient_structures(self):
        for i, _ in enumerate(self.bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            for partition_id in range(total_partitions):
                self.is_partition_reduced[i][partition_id] = False
                self.remaining_grads_in_partition[i][partition_id] = self.total_grads_in_partition[i][partition_id]

                for param_id in self.is_grad_computed[i][partition_id]:
                    self.is_grad_computed[i][partition_id][param_id] = False

    def initialize_gradient_partition(self, i, param_group, partition_id):

        def set_key_value_list(dictionary, key, value):
            if key in dictionary:
                dictionary[key].append(value)
            else:
                dictionary[key] = [value]

        def increment_value(dictionary, key):
            if key in dictionary:
                dictionary[key] += 1
            else:
                dictionary[key] = 1

        partition_size = self.partition_size[i]

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for param in param_group:

            param_size = param.numel()
            param_id = self.get_param_id(param)

            if start_index <= current_index < end_index:
                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0

            elif current_index < start_index < (current_index + param_size):
                assert (first_offset == 0
                        ), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset

            current_index = current_index + param_size

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()

    def _fill_param_grad_accum_attribute(self, param):
        if param.grad is not None:
            if param.grad_accum is None:
                param.grad_accum = param.grad.to(self.gradient_accumulation_dtype)
            else:
                param.grad_accum.add_(param.grad.to(self.gradient_accumulation_dtype).view(param.grad_accum.shape))
            param.grad = None

    def fill_grad_accum_attribute(self):
        for group in self.bit16_groups:
            for param in group:
                self._fill_param_grad_accum_attribute(param)

    def get_gradient_for_reduction(self, param):
        if self.use_grad_accum_attribute:
            return param.grad_accum.to(self.dtype) if param.grad_accum is not None else None
        else:
            return param.grad

    def get_param_gradient_attribute(self, param):
        return param.grad_accum if self.use_grad_accum_attribute else param.grad

    # Clear the tensor the reduction gradient attribute is pointing to
    def clear_grad_attribute(self, param):
        if self.use_grad_accum_attribute:
            param.grad_accum = None
        else:
            param.grad = None

    def create_gradient_handling_hooks(self):
        all_params_requiring_grad = []

        for i, param_group in enumerate(self.bit16_groups):
            for param in param_group:
                if param.requires_grad:
                    all_params_requiring_grad.append(param)

        for i, param_group in enumerate(self.bit16_groups):
            for param in param_group:
                if param.requires_grad:

                    def wrapper(param, i):

                        def grad_handling_hook(*notneeded):
                            self.reenter_backward_if_needed()
                            self.process_gradients(param, i)
                            current_expected = count_used_parameters_in_backward(all_params_requiring_grad)
                            self.update_hook_state_and_maybe_run_epilogue(current_expected)

                        self._grad_acc_hooks.append(register_grad_hook(param, grad_handling_hook))

                    wrapper(param, i)

        self._remaining_grad_acc_hooks = 0

    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    # create a flat tensor aligned at the alignment boundary
    def flatten_dense_tensors_aligned(self, tensor_list, alignment, use_cpu_data=False):
        tensor_list = [param.cpu_data for param in tensor_list] if use_cpu_data else tensor_list
        return self.flatten(align_dense_tensors(tensor_list, alignment))

    ############### Independent Partition Gradient ########################
    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):

        grad_reduc = self.get_gradient_for_reduction(param)
        comm_dtype = self.get_param_comm_dtype(param)
        bucket = self.ipg_buckets[comm_dtype]
        if bucket.elements + param.numel() > self.reduce_bucket_size:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.numel())
            self.reduce_ipg_grads(comm_dtype=comm_dtype)
            if self.contiguous_gradients and self.overlap_comm:
                # Swap index between 0 and 1
                bucket.index = 1 - bucket.index
            self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads", param.numel())

        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, \
            f"The parameter {debug_param2name(param)} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reductions are currently not supported"

        if self.contiguous_gradients:
            if param.numel() > self.reduce_bucket_size:
                self.extra_large_param_to_reduce[comm_dtype] = param
            else:
                # keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening
                new_grad_tensor = bucket.buffer[bucket.index].narrow(0, bucket.elements, param.numel())
                new_grad_tensor.copy_(
                    grad_reduc.view(-1) if not self.zenflow else grad_reduc.permute(
                        *reversed(range(grad_reduc.ndim))).contiguous().view(-1))
                grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc) if (
                    not self.zenflow or grad_reduc.dim() == 1) else new_grad_tensor.data.view_as(
                        grad_reduc.transpose(0, 1))

        bucket.elements += param.numel()

        assert grad_reduc is not None, f"rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient"

        # deal with a use-case of transient grads that will be generated in a loop for the same computation involving some model params - e.g. when performing a tiled memory calculation that shards the normal single sub-module call into a loop over a shards.
        if getattr(param, "ds_grad_is_ready", True):
            bucket.grads.append(grad_reduc)
            bucket.params.append((i, param.param_idx_in_group, param_id))

        #make sure the average tensor function knows how to average the gradients
        if is_moe_param(param):
            bucket.has_moe_params = True

        self.report_ipg_memory_usage("End ipg_remove_grads", 0)

    def print_rank_0(self, message):
        if dist.get_rank() == 0:
            logger.info(message)

    def gradient_reduction_w_predivide(self, tensor, communication_data_type: torch.dtype):
        if tensor.size().numel() == 0:
            return tensor

        dp_world_size = dist.get_world_size(group=self.dp_process_group)

        tensor_to_allreduce = tensor

        if communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(communication_data_type)

        if self.postscale_gradients:
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)

            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

            if self.gradient_predivide_factor != dp_world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor /
                                         (dp_world_size / float(self.sequence_parallel_size)))
        else:
            tensor_to_allreduce.div_(dp_world_size / float(self.sequence_parallel_size))
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy_with_multiple_ranks(self,
                                               small_bucket,
                                               communication_data_type: torch.dtype,
                                               log=None,
                                               divide=True,
                                               process_group=None,
                                               bucket_ranks=None):
        process_group = self.dp_process_group if process_group is None else process_group
        allreduced = self.allreduce_bucket(small_bucket,
                                           communication_data_type,
                                           log=log,
                                           divide=divide,
                                           process_group=process_group)
        for buf, synced, bucket_rank in zip(small_bucket, self.unflatten(allreduced, small_bucket), bucket_ranks):
            if dist.get_rank(group=process_group) == bucket_rank:
                buf.copy_(synced)

    def allreduce_and_scatter(self,
                              bucket,
                              communication_data_type: torch.dtype,
                              numel_per_bucket=500000000,
                              log=None,
                              divide=True,
                              process_group=None):
        small_bucket = []
        small_bucket_ranks = []
        numel = 0
        allreduce_sizes = []

        for i, bucket_elem in enumerate(bucket):
            rank, tensor = bucket_elem
            small_bucket.append(tensor)
            small_bucket_ranks.append(rank)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy_with_multiple_ranks(small_bucket,
                                                            communication_data_type,
                                                            log=None,
                                                            divide=divide,
                                                            process_group=process_group,
                                                            bucket_ranks=small_bucket_ranks)
                small_bucket = []
                small_bucket_ranks = []
                numel = 0

        if len(small_bucket) > 0:
            self.allreduce_and_copy_with_multiple_ranks(small_bucket,
                                                        communication_data_type,
                                                        log=None,
                                                        divide=divide,
                                                        process_group=process_group,
                                                        bucket_ranks=small_bucket_ranks)

    def average_tensor(self, tensor: torch.Tensor, communication_data_type: torch.dtype):
        if self.overlap_comm:
            stream = self.reduction_stream
            if not get_accelerator().resolves_data_dependency():
                stream.wait_stream(get_accelerator().current_stream())
                get_accelerator().current_stream().wait_stream(stream)
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            if not self.reduce_scatter:
                self.gradient_reduction_w_predivide(tensor, communication_data_type)
                return

            # Accumulate destination ranks and bucket offsets for each gradient slice.
            # Note: potential future optimization, record access pattern of parameters
            # in backward pass and partition gradients w.r.t. access pattern so that our
            # bucket is guaranteed to be contiguous w.r.t. ranks
            rank_and_offsets = []
            real_dp_process_group = []
            curr_size = 0
            prev_id, prev_process_group = -1, None

            process_group = self.dp_process_group
            # count = 0
            bucket = self.ipg_buckets[communication_data_type]
            for i, param_idx_in_group, param_id in bucket.params:
                param = self.bit16_groups[i][param_idx_in_group]

                process_group = self.dp_process_group

                if bucket.has_moe_params:
                    process_group = self.expert_dp_process_group[param.group_name] if is_moe_param(
                        param) else self.dp_process_group

                partition_ids = self.param_to_partition_ids[i][param_id]
                assert all([p_id < dist.get_world_size(group=process_group) for p_id in partition_ids
                            ]), f"world size {dist.get_world_size(group=process_group)} and p_ids: {partition_ids}"
                partition_size = self.partition_size[i]
                # Get all partition ids + their offsets
                partition_ids_w_offsets = []
                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])

                # Calculate rank and offsets for grad slices
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]

                    # if dist.get_rank() == 0 and count < 100:
                    #     print(f"Rank {dist.get_rank()} rank offset id {idx} calculated dp size {dist.get_world_size(group=process_group)} real dp size {dist.get_world_size(self.real_dp_process_group[i])} and dst: {partition_id}")
                    # count += 1

                    # Calculate numel for grad slice depending on partition location
                    if idx == len(partition_ids_w_offsets) - 1:
                        # Last partition_id uses its own offset
                        numel = param.numel() - offset
                    else:
                        # Set numel to next partition's offset
                        numel = partition_ids_w_offsets[idx + 1][1] - offset

                    # Merge bucket ranges if they belong to the same rank
                    if partition_id == prev_id and process_group == prev_process_group:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                        real_dp_process_group.append(process_group)
                    curr_size += numel
                    prev_id, prev_process_group = partition_id, process_group

            tensor.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))

            buckets = {}
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
                bucket_key = real_dp_process_group[i] if self.use_multi_rank_bucket_allreduce else (
                    dst, real_dp_process_group[i])
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                if self.use_multi_rank_bucket_allreduce:
                    buckets[bucket_key].append((dst, grad_slice))
                else:
                    buckets[bucket_key].append(grad_slice)

            for bucket_key in buckets:
                if self.use_multi_rank_bucket_allreduce:
                    self.allreduce_and_scatter(buckets[bucket_key],
                                               communication_data_type,
                                               numel_per_bucket=self.reduce_bucket_size,
                                               divide=False,
                                               process_group=bucket_key)
                else:
                    dst, process_group = bucket_key
                    self.allreduce_no_retain(buckets[bucket_key],
                                             communication_data_type,
                                             numel_per_bucket=self.reduce_bucket_size,
                                             rank=dst,
                                             divide=False,
                                             process_group=process_group)

    ##############################################################################
    ############################# CPU Offload Methods#############################
    ##############################################################################
    def get_grad_position(self, group_id, tensor_list, first_offset, partition_size):
        current_offset = 0

        for i, tensor in enumerate(tensor_list):
            param_id = self.get_param_id(tensor)
            param_start_offset = 0

            num_elements = tensor.numel()

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset
                param_start_offset = first_offset

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_offset):
                num_elements = partition_size - current_offset

            self.grad_position[param_id] = [
                int(group_id), int(param_start_offset),
                int(current_offset), int(num_elements)
            ]
            current_offset += num_elements

    def update_offload_overflow_tracker(self, grad):
        if grad is not None and self._has_inf_or_nan(grad.data):
            self.local_overflow = True

    def update_offload_overflow_tracker_for_param_grad(self, param):
        grad_accum = self.get_param_gradient_attribute(param)
        self.update_offload_overflow_tracker(grad_accum)

    def _get_offload_gradient_dict(self):
        for param_group_index, _ in enumerate(self.optimizer.param_groups):
            self.offload_gradient_dict[param_group_index] = []
            for lp_param in self.params_in_partition[param_group_index]:
                param_id = self.get_param_id(lp_param)
                [_, _, dest_offset, num_elements] = self.grad_position[param_id]
                dest_tensor = self.single_partition_of_fp32_groups[param_group_index].grad.view(-1).narrow(
                    0, dest_offset, num_elements)
                self.offload_gradient_dict[param_group_index].append(dest_tensor)

    def async_accumulate_grad_in_cpu_via_gpu(self, param):
        param_id = self.get_param_id(param)

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        # copy to a preexisiting buffer to avoid memory allocation penalty
        dest_buffer = self.temp_grad_buffer_for_gpu_offload.view(-1).narrow(0, 0, param.numel())

        #buffer for storing gradients for this parameter in CPU
        def buffer_to_accumulate_to_in_cpu():
            if not self.low_precision_master_weights_and_grads:
                buffer = torch.zeros(param.numel(), dtype=param.dtype, device=self.device)
                return get_accelerator().pin_memory(buffer) if self.cpu_offload_pin_memory else buffer
            else:
                return self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

        #accumulate gradients into param.grad_accum or parts of it that belongs to this partition
        def accumulate_gradients():
            grad_accum = self.get_param_gradient_attribute(param)
            if not self.low_precision_master_weights_and_grads:
                dest_buffer.copy_(self.accumulated_grads_in_cpu[param_id].view(-1), non_blocking=True)
                grad_accum.data.view(-1).add_(dest_buffer)
            else:
                dest_buffer.narrow(0, source_offset,
                                   num_elements).copy_(self.accumulated_grads_in_cpu[param_id].view(-1),
                                                       non_blocking=True)
                grad_accum.data.view(-1).narrow(0, source_offset,
                                                num_elements).add_(dest_buffer.narrow(0, source_offset, num_elements))

        #move accumulated gradients back to CPU
        def copy_gradients_to_cpu():
            grad_accum = self.get_param_gradient_attribute(param)
            if not self.low_precision_master_weights_and_grads:
                self.accumulated_grads_in_cpu[param_id].data.copy_(grad_accum.data.view(-1), non_blocking=True)
            else:
                self.accumulated_grads_in_cpu[param_id].data.copy_(grad_accum.data.view(-1).narrow(
                    0, source_offset, num_elements),
                                                                   non_blocking=True)

        if param_id not in self.accumulated_grads_in_cpu:
            self.accumulated_grads_in_cpu[param_id] = buffer_to_accumulate_to_in_cpu()

        if self.micro_step_id > 0:
            accumulate_gradients()
        else:
            copy_gradients_to_cpu()

    def set_norm_for_param_grad(self, param):
        param_id = self.get_param_id(param)
        grad_accum = self.get_param_gradient_attribute(param)
        accumulated_grad = self.accumulated_grads_in_cpu[
            param_id] if self.gradient_accumulation_steps > 1 else grad_accum

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    def set_norm_for_param_grad_in_gpu(self, param):
        param_id = self.get_param_id(param)
        grad_accum = self.get_param_gradient_attribute(param)
        if grad_accum is None:
            accumulated_grad = param.grad
        else:
            accumulated_grad = grad_accum

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param):
        param_id = self.get_param_id(param)

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        dest_tensor = self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

        grad_accum = self.get_param_gradient_attribute(param)
        assert grad_accum is not None

        src_tensor = grad_accum.view(-1).narrow(0, source_offset, num_elements)
        if src_tensor.dtype != self.master_weights_and_grads_dtype:
            src_tensor = src_tensor.to(self.master_weights_and_grads_dtype)

        dest_tensor.copy_(src_tensor, non_blocking=True)
        self.clear_grad_attribute(param)  #offload only

    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        total_norm = 0.0
        norm_type = 2.0
        for p in params:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                param_id = self.get_param_id(p)
                # as some model have trainable parameters but skipped in training,
                # their backward hooks in self.create_gradient_handling_hooks() will not run,
                # so they have no norm_for_param_grads
                if param_id in self.norm_for_param_grads:
                    param_norm = self.norm_for_param_grads[param_id]
                    total_norm += param_norm.item()**2
                else:
                    # As unused parameters in modules may not be expected sometimes,
                    # add an explicit error msg when it occurred and an option to
                    # avoid the error
                    assert self.ignore_unused_parameters, """
                        This assert indicates that your module has parameters that
                        were not used in producing loss.
                        You can avoid this assert by
                        (1) enable ignore_unused_parameters option in zero_optimization config;
                        (2) making sure all trainable parameters and `forward` function
                            outputs participate in calculating loss.
                    """

        # Sum across all model parallel GPUs.
        total_dev_norm = get_accelerator().FloatTensor([float(total_norm)])
        dist.all_reduce(total_dev_norm, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_dev_norm, op=dist.ReduceOp.SUM)

        total_norm = total_dev_norm[0].item()**(1. / norm_type)

        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1.0

        return torch.tensor(total_norm, device=self.device, dtype=torch.float)

    ############################################################################################
    def copy_grads_in_partition(self, param):
        if self.cpu_offload:

            if self.gradient_accumulation_steps > 1:
                self.async_accumulate_grad_in_cpu_via_gpu(param)

            if self.is_gradient_accumulation_boundary:
                self.set_norm_for_param_grad_in_gpu(param)

                self.update_offload_overflow_tracker_for_param_grad(param)

                self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)

            return
        #print(f"ID {self.get_param_id(param)} grad norm {param.grad.norm()}")
        if self.grads_in_partition is None:
            self.grads_in_partition_offset = 0
            total_size = 0
            for group in self.params_in_partition:
                for param_in_partition in group:
                    total_size += param_in_partition.numel()

            see_memory_usage(f"before copying {total_size} gradients into partition")
            self.grads_in_partition = torch.empty(int(total_size),
                                                  dtype=self.dtype,
                                                  device=get_accelerator().current_device_name())
            see_memory_usage(f"after copying {total_size} gradients into partition")

        grad_reduc = self.get_gradient_for_reduction(param)
        # The allreduce buffer will be rewritten. Copy the gradients in partition to a new buffer
        new_grad_tensor = self.grads_in_partition.view(-1).narrow(0, self.grads_in_partition_offset, param.numel())
        new_grad_tensor.copy_(grad_reduc.view(-1))
        grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)
        #print(f"Grad norm after copy to contiguous_buffer {param.grad.data.norm()}")
        self.grads_in_partition_offset += param.numel()

    def reduce_ipg_grads(self, comm_dtype=None):
        dtypes = sort_dtypes(self.ipg_buckets.keys())
        if comm_dtype is not None:
            dtypes = [comm_dtype]
        for comm_dtype in dtypes:
            bucket = self.ipg_buckets[comm_dtype]

            if self.contiguous_gradients:
                if comm_dtype in self.extra_large_param_to_reduce:
                    assert len(bucket.params) == 1, "more than 1 param in ipg bucket, this shouldn't happen"
                    _, _, param_id = bucket.params[0]
                    assert self.get_param_id(self.extra_large_param_to_reduce[comm_dtype]
                                             ) == param_id, "param in ipg bucket does not match extra-large param"
                    extra_large_grad_reduc = self.get_gradient_for_reduction(
                        self.extra_large_param_to_reduce[comm_dtype])

                    extra_large_grad_reduc_for_average = extra_large_grad_reduc.view(-1) if not self.zenflow \
                        else extra_large_grad_reduc.permute(*reversed(range(extra_large_grad_reduc.ndim))).contiguous().view(-1)
                    extra_large_grad_reduc.data = extra_large_grad_reduc_for_average.data.view_as(extra_large_grad_reduc) if (not self.zenflow or self.extra_large_param_to_reduce[comm_dtype].dim() == 1) \
                        else extra_large_grad_reduc_for_average.data.view_as(extra_large_grad_reduc.transpose(0, 1))

                    self.average_tensor(extra_large_grad_reduc_for_average, comm_dtype)
                    del self.extra_large_param_to_reduce[comm_dtype]
                else:
                    self.average_tensor(bucket.buffer[bucket.index].narrow(0, 0, bucket.elements), comm_dtype)
            else:
                self.buffered_reduce_fallback(None, bucket.grads, comm_dtype, elements_per_buffer=bucket.elements)

        if self.overlap_comm:
            stream = self.reduction_stream
        elif self.cpu_offload:
            # TODO: copy_grad_stream is disabled because of race with reduce. This hurts perf and should be fixed.
            #            get_accelerator().synchronize()
            #            stream = self.copy_grad_stream
            stream = get_accelerator().current_stream()
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            for comm_dtype in dtypes:
                bucket = self.ipg_buckets[comm_dtype]

                for group_idx, param_idx_in_group, param_id in bucket.params:
                    param = self.bit16_groups[group_idx][param_idx_in_group]

                    assert self.params_already_reduced[param_id] == False, \
                        f"The parameter {param_id} has already been reduced. \
                        Gradient computed twice for this partition. \
                        Multiple gradient reduction is currently not supported"

                    self.params_already_reduced[param_id] = True
                    if self.partition_gradients:
                        if not self.is_param_in_current_partition[param_id]:
                            if self.overlap_comm and self.contiguous_gradients is False:
                                # Clear grads of other partitions during the next reduction
                                # to avoid clearing them before the reduction is complete.
                                self.previous_reduced_grads[comm_dtype].append(param)
                            else:
                                self.clear_grad_attribute(param)
                        elif self.contiguous_gradients:
                            self.copy_grads_in_partition(param)
                    else:  # zero stage 1 - partition only optimizer state
                        if self.contiguous_gradients and self.is_param_in_current_partition[param_id]:
                            self.copy_grads_in_partition(param)
                bucket.clear()
        #####################################################################

    def process_gradients(self, param, i):
        self.setup_buckets()
        if self.use_grad_accum_attribute:
            self._fill_param_grad_accum_attribute(param)
        if self.partition_gradients or self.overlap_comm:
            self.reduce_ready_partitions_and_remove_grads(param, i)

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        if self.partition_gradients or self.is_gradient_accumulation_boundary or self.zenflow:
            self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def zero_reduced_gradients(self, partition_id, i):

        def are_all_related_partitions_reduced(params_id):
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            return True

        for params_id in self.is_grad_computed[i][partition_id]:
            if are_all_related_partitions_reduced(params_id):
                self.param_dict[params_id].grad = None  # dead code

    def flatten_and_print(self, message, tensors, start=0, n=5):
        flatten_tensor = self.flatten(tensors)

        def print_func():
            logger.info(flatten_tensor.contiguous().view(-1).narrow(0, start, n))

        self.sequential_execution(print_func, message)

    def get_grads_to_reduce(self, i, partition_id):

        def get_reducible_portion(key):
            grad = self.param_dict[key].grad
            total_elements = grad.numel()
            start = self.grad_start_offset[i][partition_id][key]
            num_elements = min(total_elements - start,
                               self.partition_size[i] - self.grad_partition_insertion_offset[i][partition_id][key])
            if not pg_correctness_test:
                if num_elements == total_elements:
                    return grad
                else:
                    return grad.contiguous().view(-1).narrow(0, int(start), int(num_elements))
            else:
                if num_elements == total_elements:
                    return grad.clone()
                else:
                    return grad.clone().contiguous().view(-1).narrow(0, int(start), int(num_elements))

        grads_to_reduce = []
        for key in self.is_grad_computed[i][partition_id]:
            grad = get_reducible_portion(key)
            grads_to_reduce.append(grad)
        return grads_to_reduce

    def sequential_execution(self, function, message, group=None):
        if group is None:
            group = self.dp_process_group
        if dist.get_rank(group=group) == 0:
            logger.info(message)
        for id in range(dist.get_world_size(group=group)):
            if id == dist.get_rank(group=group):
                function()
            dist.barrier(group=group)

    def set_none_gradients_to_zero(self, i, partition_id):
        for param_id in self.is_grad_computed[i][partition_id]:
            param = self.param_dict[param_id]
            if param.grad is None:
                param.grad = torch.zeros_like(param)

    ######################Reduction Related Methods##############################
    def allreduce_bucket(self,
                         bucket,
                         communication_data_type: torch.dtype,
                         rank=None,
                         log=None,
                         divide=True,
                         process_group=None):

        tensor = self.flatten(bucket)

        process_group = self.dp_process_group if process_group is None else process_group

        tensor_to_allreduce = tensor

        if pg_correctness_test or self.sequence_parallel_size > 1:
            communication_data_type = torch.float32

        if communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(communication_data_type)

        if divide:
            tensor_to_allreduce.div_(dist.get_world_size(group=process_group) / float(self.sequence_parallel_size))

        if rank is None:
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=process_group)
        else:
            global_rank = dist.get_global_rank(process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=process_group)

        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=process_group):
                tensor.copy_(tensor_to_allreduce)

        return tensor

    def _clear_previous_reduced_grads(self):
        for dtype in self.previous_reduced_grads:
            for param in self.previous_reduced_grads[dtype]:
                self.clear_grad_attribute(param)
            self.previous_reduced_grads[dtype].clear()

    # if rank is specified do a reduction instead of an allreduce
    def allreduce_and_copy(self,
                           small_bucket,
                           communication_data_type: torch.dtype,
                           rank=None,
                           log=None,
                           divide=True,
                           process_group=None):
        process_group = self.dp_process_group if process_group is None else process_group
        if self.overlap_comm:
            if not get_accelerator().resolves_data_dependency():
                get_accelerator().synchronize()
            # It is safe to clear the previously reduced grads of other partitions
            self._clear_previous_reduced_grads()
            stream = self.reduction_stream
        else:
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            allreduced = self.allreduce_bucket(
                small_bucket,
                communication_data_type,
                rank=rank,
                log=log,
                divide=divide,
                process_group=process_group,
            )
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(
        self,
        bucket,
        communication_data_type: torch.dtype,
        numel_per_bucket=500000000,
        rank=None,
        log=None,
        divide=True,
        process_group=None,
    ):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket,
                                        communication_data_type,
                                        rank=rank,
                                        log=None,
                                        divide=divide,
                                        process_group=process_group)
                small_bucket = []
                numel = 0

        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket,
                                    communication_data_type,
                                    rank=rank,
                                    log=log,
                                    divide=divide,
                                    process_group=process_group)

    # allows using reduction of gradients instead of using all_reduce

    def buffered_reduce_fallback(self,
                                 rank,
                                 grads,
                                 communication_data_type: torch.dtype,
                                 elements_per_buffer=500000000,
                                 log=None):
        split_buckets = split_half_float_double(grads)

        for i, bucket in enumerate(split_buckets):
            self.allreduce_no_retain(bucket,
                                     communication_data_type,
                                     numel_per_bucket=elements_per_buffer,
                                     rank=rank,
                                     log=log)

    #############################################################################
    #############################################################################
    #############################################################################

    # views the tensor as multiple partitions and returns
    # those partitions
    def get_data_parallel_partitions(self, tensor, group_id):
        partitions = []

        dp = dist.get_world_size(group=self.real_dp_process_group[group_id])
        # dp_id = dist.get_rank(group=self.real_dp_process_group[group_id])

        total_num_elements = tensor.numel()

        base_size = total_num_elements // dp
        remaining = total_num_elements % dp

        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for tensor in tensor_list:

            tensor_size = tensor.numel()

            if start_index <= current_index < end_index:
                params_in_partition.append(tensor)

            elif current_index < start_index < (current_index + tensor_size):
                params_in_partition.append(tensor)

                assert (first_offset == 0
                        ), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                params_not_in_partition.append(tensor)

            current_index = current_index + tensor_size

        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_to_none=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        # zero all pointers to grad tensors
        for group in self.bit16_groups:
            for p in group:
                if set_to_none:
                    p.grad = None  # epilogue and in step
                    p.grad_accum = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def _clear_param_grad_only(self):
        """Clear only param.grad but keep grad_accum intact.

        This is used at the end of the epilogue to ensure safe_get_full_grad() goes
        through the proper _hp_mapping path (which does all_reduce for ZeRO-2), while
        preserving grad_accum for reentrant checkpointing where gradients need to
        accumulate across multiple backward phases.
        """
        for group in self.bit16_groups:
            for p in group:
                p.grad = None

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        if self.model_parallel_group is None or self.model_parallel_world_size == 1:
            pass
        else:
            dist.all_reduce(tensor=tensor, op=op, group=self.model_parallel_group)

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        all_norms = []
        if norm_type == inf:
            for g in gradients:
                all_norms.append(g.data.abs().max().float())
            total_norm = torch.stack(all_norms).max()
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.MAX)
        else:
            # if dist.get_rank() == 0:
            #    logger.info(f"Total Norm beginning {total_norm}")
            for g, p in zip(gradients, params):
                # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                    continue
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    all_norms.append(
                        torch.linalg.vector_norm(g.data.double().detach(),
                                                 ord=norm_type).to(get_accelerator().current_device_name()))
            if len(all_norms) > 0:
                total_norm = torch.stack(all_norms).square().sum().float()
            else:
                total_norm = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            # Sum across all model parallel Device.
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)

            total_norm = total_norm.pow(1. / norm_type)

        mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)

        return total_norm

    def get_all_grad_tensors(self, tensor_list, dtype):
        all_grad_tensors = []
        for i, tensor in enumerate(tensor_list):
            grad_accum = self.get_param_gradient_attribute(tensor)
            if grad_accum is None:
                grad_accum = torch.zeros_like(tensor, dtype=dtype)
            all_grad_tensors.append(grad_accum)
        return all_grad_tensors

    # creates a flat fused tensor from the tensor list starting at the first_offset
    # in the first tensor of the list. If there are not enough elements in the tensor
    # list then the flat tensor will be padded with zeros
    def get_flat_partition(self,
                           tensor_list,
                           first_offset,
                           partition_size,
                           dtype,
                           device,
                           param_group_idx,
                           return_tensor_list=False):
        if len(tensor_list) == 0:
            # This condition can fire when we have small parameteters and many ranks.
            zero_buffer = torch.zeros(int(partition_size), dtype=dtype, device=device)
            if return_tensor_list:
                return [zero_buffer]
            return zero_buffer

        flat_tensor_list = []
        current_size = 0
        # find the flatten copy in the optimizer's state
        flatten_copy = self.optimizer.param_groups[param_group_idx]['params'][0]
        if (not self.optimizer.state[flatten_copy]) and getattr(
                tensor_list[0], 'use_muon', False) and 'muon' in self.optimizer.__class__.__name__.lower():
            self.optimizer.state[flatten_copy] = {}
        if "momentum_buffer" not in self.optimizer.state[flatten_copy] and getattr(
                tensor_list[0], 'use_muon', False) and 'muon' in self.optimizer.__class__.__name__.lower():
            # need to check the total # of elements in the parameters in this group and this partition
            total_size = sum([t.numel() for t in tensor_list])
            flatten_bf_list = [torch.zeros([total_size], dtype=dtype, device=device)]
            self.optimizer.state[flatten_copy]["momentum_buffer"] = self.flatten(flatten_bf_list)

        buffer_idx = 0
        for i, tensor in enumerate(tensor_list):
            grad_accum = self.all_grad_tensors[param_group_idx][i]
            if getattr(tensor, 'use_muon', False) and 'muon' in self.optimizer.__class__.__name__.lower():
                assert tensor.ndim > 1, f"if use muon, then tensor dim > 1, got {tensor.size()}"
                buffer = torch.narrow(self.optimizer.state[flatten_copy]["momentum_buffer"], 0, buffer_idx,
                                      tensor.numel()).view(tensor.size())
                grad_accum = muon_update(grad_accum, buffer, self.optimizer.param_groups[param_group_idx]['momentum'])
            tensor = grad_accum
            num_elements = tensor.numel()
            buffer_idx += num_elements
            tensor_offset = 0

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_size):
                num_elements = partition_size - current_size

            # we need a narrow view of the tensor based on the tensor offset and number of elements that
            # we need from this tensor
            if tensor_offset > 0 or num_elements < tensor.numel():
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(0, int(tensor_offset), int(num_elements)))
            else:
                flat_tensor_list.append(tensor)

            current_size = current_size + num_elements

        # this means its the last partition and does not align with the dp boundary. We need to pad before flattening
        if current_size < partition_size:
            flat_tensor_list.append(torch.zeros(int(partition_size - current_size), dtype=dtype, device=device))

        if return_tensor_list:
            return flat_tensor_list

        return self.flatten(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None  # in step
            p.grad_accum = None

    def reset_cpu_buffers(self):
        self.norm_for_param_grads = {}
        self.local_overflow = False

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def override_loss_scale(self, loss_scale):
        if loss_scale != self.external_loss_scale:
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def scaled_global_norm(self, norm_type=2):
        assert norm_type == 2, "only L2 norm supported"
        norm_groups = []
        for i, group in enumerate(self.bit16_groups):
            if self.cpu_offload:
                norm = self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i])
                norm_groups.append(norm)
            else:
                norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))

        if self.has_moe_layers:
            self._average_expert_grad_norms(norm_groups)

        # calculating L2 norm
        return torch.linalg.vector_norm(torch.stack(norm_groups), ord=norm_type)

    def get_bit16_param_group(self, group_no):
        bit16_partitions = self.parallel_partitioned_bit16_groups[group_no]
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_no])
        return [bit16_partitions[dist.get_rank(group=self.real_dp_process_group[group_no])]]

    def _optimizer_step(self, group_no):
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [original_param_groups[group_no]]
        # Disabling this as the C++ side copy & synchronize is not working correctly
        #from deepspeed.ops.adam import DeepSpeedCPUAdam
        #if type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half:
        #    self.optimizer.step(fp16_param_groups=[self.get_bit16_param_group(group_no)])
        #else:
        #    self.optimizer.step()
        if self.torch_autocast_gradscaler:
            self.torch_autocast_gradscaler.step(self.optimizer)
            self.torch_autocast_gradscaler.update()
        # TODO: Remove zenflow-specific call from vanilla ZeroOptimizer
        elif self.zenflow:
            self.zenflow_cpu_optimizer_step(group_no)
        else:
            self.optimizer.step()
        self.optimizer.param_groups = original_param_groups

        # We need to link optimizer state after the first step() call
        self._lazy_init_hp_params_optimizer_state()

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = INITIAL_MICRO_STEP_ID

        see_memory_usage("In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        if self.check_grad_overflow:
            self.check_overflow(partition_gradients=self.partition_gradients)

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                for k in self.averaged_gradients.keys():
                    self.averaged_gradients[k] = None
                    self.all_grad_tensors[k] = None

            see_memory_usage('After overflow after clearing gradients')

            for timer in OPTIMIZER_TIMERS:
                self.timers(timer).start()
                self.timers(timer).stop()
            return

        # Step 1:- Calculate gradient norm using bit-16 grads
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale
        see_memory_usage('After norm before optimizer')

        # Step 2:- run optimizer and upscaling simultaneously
        for i, group in enumerate(self.bit16_groups):
            self.timers(OPTIMIZER_GRADIENTS_TIMER).start()
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()
                self.timers(OPTIMIZER_STEP_TIMER).start()
                self._optimizer_step(i)

                # Disabled, this is not currently working
                #from deepspeed.ops.adam import DeepSpeedCPUAdam
                #if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
                #    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                #    fp32_partition = self.single_partition_of_fp32_groups[i]
                #    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partition_buffer = self.param_buffer_of_bit16_for_cpu_offload_groups[i]
                bit16_partition_buffer.data.copy_(fp32_partition.data)
                bit16_partitions[partition_id].data.copy_(bit16_partition_buffer.data, non_blocking=True)

                self.timers(OPTIMIZER_STEP_TIMER).stop()
            else:
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])).to(self.single_partition_of_fp32_groups[i].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                        self.single_partition_of_fp32_groups[i].dtype)
                assert single_grad_partition.numel() == self.partition_size[i], \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])

                self.averaged_gradients[i] = None
                self.all_grad_tensors[i] = None
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()

                # Step 3:- run the optimizer if no offloading
                self.timers(OPTIMIZER_STEP_TIMER).start()
                self._optimizer_step(i)
                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                self.single_partition_of_fp32_groups[i].grad = None
                del single_grad_partition
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.timers(OPTIMIZER_STEP_TIMER).stop()

        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                             partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)
        self.timers(OPTIMIZER_ALLGATHER_TIMER).stop()

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.timers.log(OPTIMIZER_TIMERS)
        see_memory_usage('After zero_optimizer step')

        return

    @torch.no_grad()
    def update_lp_params(self):
        for i, (bit16_partitions, fp32_partition) in enumerate(
                zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)

        all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                             partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)

    def _average_expert_grad_norms(self, norm_groups):
        for i, norm in enumerate(norm_groups):
            if self.is_moe_param_group[i]:
                scaled_norm_tensor = norm * 1.0 / dist.get_world_size(group=self.real_dp_process_group[i])
                if self.device == 'cpu':
                    scaled_norm_tensor = scaled_norm_tensor.to(get_accelerator().current_device_name())
                dist.all_reduce(scaled_norm_tensor, group=self.real_dp_process_group[i])
                norm_groups[i] = scaled_norm_tensor.to(self.device)

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            clip = torch.clamp(clip, min=1.0)
            combined_scale = clip * self.loss_scale

        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1. / combined_scale)
            else:
                grad.data.mul_(1. / combined_scale)

    def _check_overflow(self, partition_gradients=True):
        self.overflow = self.has_overflow(partition_gradients)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        invalid_grad_count = torch.zeros([1], dtype=torch.float, device=get_accelerator().current_device_name())
        for p in params:
            if p.grad is not None:
                invalid_grad_count += self._has_inf_or_nan(p.grad)
        return invalid_grad_count.bool()

    def has_overflow_partitioned_grads_serial(self):
        invalid_grad_count = torch.zeros([1], dtype=torch.float, device=get_accelerator().current_device_name())
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None:
                    invalid_grad_count += self._has_inf_or_nan(grad)
        return invalid_grad_count.bool()

    def has_overflow(self, partition_gradients=True):
        overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()
        overflow_gpu = get_accelerator().ByteTensor([overflow]) if self.cpu_offload else overflow.byte().to(
            get_accelerator().current_device_name())

        dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, j=None):
        float_x = x.float()
        nan = float_x.isnan()
        inf = float_x.isinf()
        inf_or_nan = nan.logical_or(inf)
        return inf_or_nan.float().max()

    def setup_buckets(self):
        if not self.ready_for_gradients:
            self.micro_step_id += 1

            if self.contiguous_gradients:
                for _, bucket in self.ipg_buckets.items():
                    bucket.buffer.clear()

                    # Buffer's dtype is the same as the dtype of optimizer, not dtype for autocast
                    buf_0 = torch.empty(int(self.reduce_bucket_size),
                                        dtype=self.dtype,
                                        device=get_accelerator().current_device_name())
                    bucket.buffer.append(buf_0)
                    bucket.index = 0

                # Use double buffers to avoid data access conflict when overlap_comm is enabled.
                if self.overlap_comm:
                    for _, bucket in self.ipg_buckets.items():
                        buf_1 = torch.empty(int(self.reduce_bucket_size),
                                            dtype=self.dtype,
                                            device=get_accelerator().current_device_name())
                        bucket.buffer.append(buf_1)

            self.ready_for_gradients = True

    def backward_epilogue(self, *args, **kwargs):
        # Only for Stage 1, Mode 2
        if self.use_grad_accum_attribute:
            self.fill_grad_accum_attribute()

    def check_overflow(self, partition_gradients=True):
        self._check_overflow(partition_gradients)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        if self.custom_loss_scaler:
            return self.external_loss_scale
        else:
            return self.loss_scaler.cur_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    # Return group tensor after removing paddings that are added for alignment to DP world size.
    # This method works on the assumption that each group contains a single flattened tensor.
    def _get_groups_without_padding(self, groups_with_padding):
        groups_without_padding = []
        for i, group in enumerate(groups_with_padding):
            lean_length = group.numel() - self.groups_padding[i]
            groups_without_padding.append(group[:lean_length])

        return groups_without_padding

    # Return optimizer state after removing paddings that are added for alignment.
    def _get_state_without_padding(self, state_with_padding, padding):
        lean_state = {}
        for key, value in state_with_padding.items():
            if torch.is_tensor(value) and value.dim() > 0:
                lean_length = value.numel() - padding
                lean_state[key] = value[:lean_length]
            else:
                lean_state[key] = value

        return lean_state

    # Return base optimizer states.
    # This method assumes that each param group contains a single flattened tensor.
    def _get_base_optimizer_state(self):
        optimizer_groups_state = []
        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            lean_optimizer_state = self._get_state_without_padding(self.optimizer.state[p], self.groups_padding[i])
            optimizer_groups_state.append(lean_optimizer_state)

        return optimizer_groups_state

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict[LOSS_SCALER] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict[CLIP_GRAD] = self.clip_grad

        if self.elastic_checkpoint:
            state_dict[BASE_OPTIMIZER_STATE] = self._get_base_optimizer_state()

            if "step" in self.optimizer.param_groups[0]:
                # Assuming "step" is the only item that changes through training iterations
                assert all(group["step"] == self.optimizer.param_groups[0]["step"]
                           for group in self.optimizer.param_groups), "All param groups must have the same step value"
                state_dict[BASE_OPTIMIZER_STATE_STEP] = self.optimizer.param_groups[0]["step"]
        else:
            state_dict[BASE_OPTIMIZER_STATE] = self.optimizer.state_dict()

        # Remove paddings for DP alignment to enable loading for other alignment values
        fp32_groups_without_padding = self._get_groups_without_padding(self.single_partition_of_fp32_groups)
        state_dict[SINGLE_PARTITION_OF_FP32_GROUPS] = fp32_groups_without_padding

        state_dict[
            ZERO_STAGE] = ZeroStageEnum.gradients if self.partition_gradients else ZeroStageEnum.optimizer_states
        state_dict[GROUP_PADDINGS] = self.groups_padding
        state_dict[PARTITION_COUNT] = self.partition_count

        state_dict[DS_VERSION] = version
        state_dict[PARAM_SLICE_MAPPINGS] = self._param_slice_mappings

        return state_dict

    # Restore base optimizer fp32 weights from elastic checkpoint by:
    # 1) Merging fp32 weights from checkpoints of all partitions
    # 2) Extracting fp32 weights for current partition from merged weights
    # 3) Using extracted weights to update base optimizer weights directly.
    def _restore_from_elastic_fp32_weights(self, all_state_dict):
        merged_single_partition_of_fp32_groups = []

        for i in range(len(self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            merged_partitions = [sd[SINGLE_PARTITION_OF_FP32_GROUPS][i] for sd in all_state_dict]
            if self.is_moe_group(self.optimizer.param_groups[i]):
                ranks = self.get_ep_ranks(group_name=self.optimizer.param_groups[i]['name'])
                merged_partitions = [merged_partitions[i] for i in ranks]
            flat_merged_partitions = self.flatten_dense_tensors_aligned(
                merged_partitions,
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]))
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, i)
            merged_single_partition_of_fp32_groups.append(dp_partitions[partition_id])

        for current, saved in zip(self.single_partition_of_fp32_groups, merged_single_partition_of_fp32_groups):
            current.data.copy_(saved.data)

    # Restore base optimizer fp32 weights from ZeRO fp16 or bfloat16 weights
    def _restore_from_bit16_weights(self):
        for group_id, (bit16_partitions, fp32_partition) in enumerate(
                zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
            fp32_partition.data.copy_(bit16_partitions[partition_id].data)

    # Refresh the fp32 master params from the fp16 or bfloat16 copies.
    def refresh_fp32_params(self):
        self._restore_from_bit16_weights()

    # Extract optimizer state for current partition from merged states of all partitions
    def _partition_base_optimizer_state(self, state_key, all_partition_states, group_id):
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
        alignment = self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[group_id])
        if torch.is_tensor(all_partition_states[0]):
            flat_merged_partitions = self.flatten_dense_tensors_aligned(all_partition_states, alignment)
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, group_id)
            return dp_partitions[partition_id]
        else:
            # Assume non-tensor states are not partitioned and equal across ranks, so return first one
            return all_partition_states[0]

    def _restore_step_from_elastic_checkpoint(self, all_state_dict):
        assert BASE_OPTIMIZER_STATE_STEP in all_state_dict[0]
        assert all(sd[BASE_OPTIMIZER_STATE_STEP] == all_state_dict[0][BASE_OPTIMIZER_STATE_STEP]
                   for sd in all_state_dict), "State dicts of all partitions must have the same step value"
        return all_state_dict[0][BASE_OPTIMIZER_STATE_STEP]

    def _restore_base_optimizer_state(self, base_optimizer_group_states, base_optimizer_state_step, group_paddings):
        if type(base_optimizer_group_states) == dict:
            base_optimizer_group_states = base_optimizer_group_states['state']

        saved_keys = base_optimizer_group_states[0].keys()

        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            padding = 0 if group_paddings is None else group_paddings[i]
            for key in saved_keys:
                saved = base_optimizer_group_states[i][key]

                if torch.is_tensor(saved):
                    if key in self.optimizer.state[p]:
                        dst_tensor = self.optimizer.state[p][key]
                        src_tensor = _get_padded_tensor(saved, dst_tensor.numel())
                        self.optimizer.state[p][key].data.copy_(src_tensor.data)
                    else:
                        self.optimizer.state[p][key] = _pad_tensor_by_size(
                            saved, padding, torch.float32,
                            torch.device('cpu') if self.cpu_offload else self.device)
                else:
                    self.optimizer.state[p][key] = saved

        for param_group in self.optimizer.param_groups:
            param_group['step'] = base_optimizer_state_step

    def get_ep_ranks(self, rank=0, group_name=None):
        from deepspeed.utils import groups
        expert_parallel_size_ = groups._get_expert_parallel_world_size(group_name)
        world_size = groups._get_data_parallel_world_size()
        rank = groups._get_expert_parallel_rank(group_name)
        ranks = range(rank, world_size, expert_parallel_size_)
        return list(ranks)

    # Restore base optimizer state from elastic checkpoint by
    # 1) Merging optimizer state from checkpoints of all partitions
    # 2) Extracting optimizer state for current partition from the merged state
    # 3) Using the extracted value to directly update the base optimizer.
    def _restore_elastic_base_optimizer_state(self, all_state_dict):
        base_optimizer_group_states = []
        for i in range(len(self.optimizer.param_groups)):
            partition_states = {}
            all_partition_group_states = [sd[BASE_OPTIMIZER_STATE][i] for sd in all_state_dict]

            if self.is_moe_group(self.optimizer.param_groups[i]):
                ranks = self.get_ep_ranks(group_name=self.optimizer.param_groups[i]['name'])
                all_partition_group_states = [all_partition_group_states[i] for i in ranks]

            for key in all_partition_group_states[0].keys():
                all_partition_states = [all_states[key] for all_states in all_partition_group_states]
                partition_states[key] = self._partition_base_optimizer_state(key, all_partition_states, i)
            base_optimizer_group_states.append(partition_states)

        self._restore_base_optimizer_state(base_optimizer_group_states,
                                           self._restore_step_from_elastic_checkpoint(all_state_dict), None)

    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None,
                        load_serial=None,
                        param_shapes=None):
        if checkpoint_folder:
            self._load_universal_checkpoint(checkpoint_folder, load_optimizer_states, load_from_fp32_weights)
        else:
            self._load_legacy_checkpoint(state_dict_list, load_optimizer_states, load_from_fp32_weights)

    def _load_universal_checkpoint(self, checkpoint_folder, load_optimizer_states, load_from_fp32_weights):
        self.load_hp_checkpoint_state_from_checkpoint_dir("bit16_groups", checkpoint_folder)

    def _load_global_state(self, sd):
        self.loss_scaler = sd.get(LOSS_SCALER, self.loss_scaler)
        self.dynamic_loss_scale = sd.get('dynamic_loss_scale', self.dynamic_loss_scale)
        self.overflow = sd.get('overflow', self.overflow)
        self.clip_grad = sd.get(CLIP_GRAD, self.clip_grad)

        ckpt_version = sd.get(DS_VERSION, False)
        assert ckpt_version, "Empty ds_version in checkpoint, not clear how to proceed"
        ckpt_version = pkg_version.parse(ckpt_version)

        # zero stage 1 mode
        if not self.partition_gradients:
            required_version = pkg_version.parse("0.3.17")
            error_str = f"ZeRO stage 1 changed in {required_version} and is not backwards compatible " \
                "with older stage 1 checkpoints. If you'd like to load an old ZeRO-1 checkpoint " \
                "please use an older version of DeepSpeed (<= 0.5.8) and set 'legacy_stage1': true in your zero config json."
            assert required_version <= ckpt_version, f"Old version: {ckpt_version} {error_str}"

    def _load_legacy_checkpoint(self, state_dict_list, load_optimizer_states=True, load_from_fp32_weights=False):
        r"""Loading ZeRO checkpoint

        Arguments:
            state_dict_list: List of all saved ZeRO checkpoints, one for each saved partition.
                Note that the number of saved partitions may differ from number of loading partitions to support
                changing GPU count, specifically DP world size, between saving and loading checkpoints.
            load_optimizer_states: Boolean indicating whether or not to load base optimizer states
            load_from_fp32_weights: Boolean indicating whether to initialize fp32 master weights from fp32
            copies in checkpoints (no precision loss) or from model's fp16 copies (with precision loss).
        """
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).to(get_accelerator().device_name()).half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """

        # I think it should actually be ok to reload the optimizer before the model.
        dp_rank = dist.get_rank(group=self.dp_process_group)
        current_rank_sd = state_dict_list[dp_rank]
        self._load_global_state(current_rank_sd)

        ckpt_is_rigid = isinstance(current_rank_sd[BASE_OPTIMIZER_STATE], dict)

        # padding is always at the last rank/partition
        # if DP=1024 and param-group elems=16 -> padding will be 1024-16 across all but one rank
        # scenario-1 (shrink): saving w. 4 gpus -> loading w. 2 gpus
        # scenario-2 (expand): saving w. 2 gpus -> loading w. 4 gpus
        # if load_optimizer_states:
        #     if new_dp_size:
        #         self.strip_padding()
        #         self.add_padding_w_new_dp_size()
        #     self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])

        if load_optimizer_states:
            if ckpt_is_rigid:
                # loading rigid ckpt into either rigid or elastic exec
                self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])
            else:
                if self.elastic_checkpoint:
                    # loading elastic into elastic exec
                    self._restore_elastic_base_optimizer_state(state_dict_list)
                else:
                    # loading an elastic checkpoint into rigid exec
                    self._restore_base_optimizer_state(current_rank_sd[BASE_OPTIMIZER_STATE],
                                                       current_rank_sd[BASE_OPTIMIZER_STATE_STEP],
                                                       current_rank_sd[GROUP_PADDINGS])

        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 1 if changing DP degree and option 2 otherwise.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.

        if load_from_fp32_weights:
            # option 2 from above
            if self.elastic_checkpoint and not ckpt_is_rigid:
                self._restore_from_elastic_fp32_weights(state_dict_list)
            else:
                # For non-elastic checkpoint, simply copying from saved weights of current rank is sufficient.
                for current, saved in zip(self.single_partition_of_fp32_groups,
                                          current_rank_sd[SINGLE_PARTITION_OF_FP32_GROUPS]):
                    src_tensor = _get_padded_tensor(saved, current.numel())
                    current.data.copy_(src_tensor.data)
        else:
            # option 1 from above
            self._restore_from_bit16_weights()

        if load_optimizer_states:
            self._link_all_hp_params()

    def _clear_hp_buffer_references(self):
        """
        Clear all references that might prevent GPU memory release when offloading HP params.
        This includes gradient views, HP mapping fragments, and optimizer state fragments.
        """
        # Clear gradient references in offload_gradient_dict
        if hasattr(self, 'offload_gradient_dict'):
            for param_group_index in self.offload_gradient_dict:
                if self.offload_gradient_dict[param_group_index] is not None:
                    self.offload_gradient_dict[param_group_index].clear()

        # Clear gradient buffers attached to HP params
        for i, buf in enumerate(self.single_partition_of_fp32_groups):
            if hasattr(buf, 'grad') and buf.grad is not None:
                buf.grad = None

        # Clear HP mapping references in model parameters
        for i, param_group in enumerate(self.bit16_groups):
            for param in param_group:
                if hasattr(param, '_hp_mapping') and param._hp_mapping is not None:
                    # Clear the fragment references that point to GPU buffers
                    if hasattr(param._hp_mapping, 'hp_fragment'):
                        param._hp_mapping.hp_fragment = None
                    if hasattr(param._hp_mapping, 'optim_fragment') and param._hp_mapping.optim_fragment is not None:
                        param._hp_mapping.optim_fragment.clear()

        # Force garbage collection to release references
        gc.collect()

    def _clear_lp_params_references(self):
        """
        Clear all references that might prevent GPU memory release when offloading LP params.
        This includes HP mapping lp_fragment references and completely nullifying _hp_mapping.
        """
        # Completely clear HP mapping to break all references to GPU tensors
        for i, param_group in enumerate(self.bit16_groups):
            for param in param_group:
                if hasattr(param, '_hp_mapping') and param._hp_mapping is not None:
                    # Completely nullify _hp_mapping to break all references
                    param._hp_mapping = None

        # Force garbage collection to release references
        gc.collect()

    def offload_states(self,
                       include: Container[OffloadStateTypeEnum] = None,
                       device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                       pin_memory: bool = True,
                       non_blocking: bool = False):
        """
        Offload optimizer states from GPU to the specified device (typically CPU).

        Args:
            include (Container[OffloadStateTypeEnum], optional):
                Collection of state types to offload. If None, offloads all supported states.
                Defaults to None.
            device (OffloadDeviceEnum, optional):
                Target device for offloading. Defaults to OffloadDeviceEnum.cpu.
            pin_memory (bool, optional):
                If True, pins data in memory before moving to CPU.
                This can accelerate subsequent CPU-to-GPU transfers. Defaults to True.
            non_blocking (bool, optional):
                If True, attempts to perform offload operations asynchronously. Defaults to False.
        """
        device = device.value

        def needs_offload(target):
            return target not in self.offloaded_states and (include is None or target in include)

        # Offload FP32 Master Parameters (HP Params)
        if needs_offload(OffloadStateTypeEnum.hp_params):
            self._clear_hp_buffer_references()
            if pin_memory:
                if not hasattr(self, "hp_params_pin_buffers"):
                    self.hp_params_pin_buffers = [
                        torch.empty_like(t, device=device).pin_memory() for t in self.single_partition_of_fp32_groups
                    ]
                for src_tensor, dest_buf in zip(self.single_partition_of_fp32_groups, self.hp_params_pin_buffers):
                    dest_buf.copy_(src_tensor, non_blocking=non_blocking)
                    src_tensor.data = dest_buf
            else:
                for buf in self.single_partition_of_fp32_groups:
                    buf.data = buf.data.to(device, non_blocking=non_blocking)

            self.offloaded_states.add(OffloadStateTypeEnum.hp_params)

        # Offload FP16/BF16 Model Parameters (LP Params)
        if needs_offload(OffloadStateTypeEnum.lp_params):
            self._clear_lp_params_references()
            for group in self.bit16_groups:
                for param in group:
                    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            for group_partitions in self.parallel_partitioned_bit16_groups:
                group_partitions.clear()

            if pin_memory:
                if not hasattr(self, "lp_params_pin_buffers"):
                    self.lp_params_pin_buffers = [
                        torch.empty_like(t, device=device).pin_memory() for t in self.bit16_groups_flat
                    ]
                for src_tensor, dest_buf in zip(self.bit16_groups_flat, self.lp_params_pin_buffers):
                    dest_buf.copy_(src_tensor, non_blocking=non_blocking)
                    src_tensor.data = dest_buf
            else:
                for buf in self.bit16_groups_flat:
                    buf.data = buf.data.to(device, non_blocking=non_blocking)
            for i in range(len(self.bit16_groups)):
                self._update_model_bit16_weights(i)

            self.offloaded_states.add(OffloadStateTypeEnum.lp_params)

        # Offload Partitioned Gradients (LP Grads)
        if needs_offload(OffloadStateTypeEnum.lp_grads):
            for group_idx in self.averaged_gradients:
                grad_list = self.averaged_gradients.get(group_idx)
                if grad_list is not None:
                    for grad_tensor in grad_list:
                        if grad_tensor is not None and grad_tensor.device.type != device:
                            # Key insight: We only move the underlying data storage (`.data`) to the target device.
                            # The Python tensor object and the dictionary structure (`self.averaged_gradients`)
                            # remain intact, preserving the references needed for reloading.
                            grad_tensor.data = grad_tensor.data.to(device, non_blocking=non_blocking)

            self.offloaded_states.add(OffloadStateTypeEnum.lp_grads)

        # Offload Optimizer States
        if needs_offload(OffloadStateTypeEnum.optim_states):
            offload_optimizer_states(self.optimizer, device, pin_memory=pin_memory, non_blocking=non_blocking)
            self.offloaded_states.add(OffloadStateTypeEnum.optim_states)

        if not non_blocking:
            if get_accelerator().is_available():
                get_accelerator().synchronize()

        gc.collect()
        if get_accelerator().is_available():
            get_accelerator().empty_cache()

    def reload_states(self, non_blocking: bool = False):
        """
        Reload previously offloaded optimizer states from CPU back to GPU.

        Args:
            non_blocking (bool, optional):
                If True, attempts to perform reload operations asynchronously. Defaults to False.
        """
        device = get_accelerator().current_device_name()

        # Reload FP32 Master Parameters (HP Params)
        if OffloadStateTypeEnum.hp_params in self.offloaded_states:
            for buf in self.single_partition_of_fp32_groups:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
            if hasattr(self, "hp_params_pin_buffers"):
                del self.hp_params_pin_buffers
            self._link_all_hp_params()
            self.offloaded_states.remove(OffloadStateTypeEnum.hp_params)

        # Reload FP16/BF16 Model Parameters (LP Params)
        if OffloadStateTypeEnum.lp_params in self.offloaded_states:
            for buf in self.bit16_groups_flat:
                buf.data = buf.data.to(device, non_blocking=non_blocking)

            # Reconstruct the parallel partitions now that the flat buffer is back on GPU.
            self.parallel_partitioned_bit16_groups.clear()
            for i, flat_group in enumerate(self.bit16_groups_flat):
                data_parallel_partitions = self.get_data_parallel_partitions(flat_group, i)
                self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

            for i in range(len(self.bit16_groups)):
                self._update_model_bit16_weights(i)

            if hasattr(self, "lp_params_pin_buffers"):
                del self.lp_params_pin_buffers
            self._link_all_hp_params()
            self.offloaded_states.remove(OffloadStateTypeEnum.lp_params)

        # Reload Partitioned Gradients (LP Grads)
        if OffloadStateTypeEnum.lp_grads in self.offloaded_states:
            # Since we preserved the `self.averaged_gradients` structure during offload,
            # we can now iterate through it again. The tensors within currently point to CPU data.
            for group_idx in self.averaged_gradients:
                grad_list = self.averaged_gradients.get(group_idx)
                if grad_list is not None:
                    for grad_tensor in grad_list:
                        if grad_tensor is not None and grad_tensor.device.type != device:
                            grad_tensor.data = grad_tensor.data.to(device, non_blocking=non_blocking)

            self.offloaded_states.remove(OffloadStateTypeEnum.lp_grads)

        # Reload Optimizer States
        if OffloadStateTypeEnum.optim_states in self.offloaded_states:
            reload_optimizer_states(self.optimizer, device, non_blocking=non_blocking)
            self.offloaded_states.remove(OffloadStateTypeEnum.optim_states)

        if non_blocking:
            get_accelerator().synchronize()


def _handle_overflow(cpu_sum, x, i):
    import math
    rank = dist.get_rank()
    if rank == 0:
        t_i = -1
        for v_i, v in enumerate(x.data.contiguous().view(-1)):
            if not math.isfinite(float(v)):
                t_i = v_i
                break
        logger.info(f"rank {rank} detected overflow {cpu_sum} in tensor {i}:{t_i} shape {x.shape}")


def estimate_zero2_model_states_mem_needs(total_params,
                                          num_gpus_per_node=1,
                                          num_nodes=1,
                                          cpu_offload=True,
                                          additional_buffer_factor=1.5):

    total_gpus = num_nodes * num_gpus_per_node

    if cpu_offload:
        gpu_mem = 2 * total_params
        cpu_mem = total_params * max(4 * total_gpus, 16) * additional_buffer_factor
    else:
        # GPU's total_params multipliers: 2 = params_16bit,
        # 18 = 2_grads_16bit + 4_grads_32bit + 4_params_32bit + 8_optimizer_states_32bit(momentum and variance)
        gpu_mem = 2 * total_params + int(18 * total_params / total_gpus)
        cpu_mem = total_params * 4 * num_gpus_per_node * additional_buffer_factor

    return int(cpu_mem), int(gpu_mem)


def model_to_params(model):
    # shared params calculated only once
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    return total_params


def estimate_zero2_model_states_mem_needs_all_live(model,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """
    Print out estimates on memory usage requirements for ZeRO 2 params, optim states and gradients
    for a given ``model`` and hardware setup.

    If you have an actual model object, use this function and everything will be derived
    automatically.

    If it's a hypothetical model, use ``estimate_zero2_model_states_mem_needs_all_cold`` where you have to pass
    the ``total_params`` explicitly.

    Args:
        - ``model``: ``nn.Module`` object
        - ``num_gpus_per_node``: how many gpus per node (defaults to 1)
        - ``num_nodes``: how many nodes (defaults to 1),
        - ``additional_buffer_factor``: estimation factor (defaults to 1.5):

    """

    total_params = model_to_params(model)

    estimate_zero2_model_states_mem_needs_all_cold(total_params=total_params,
                                                   num_gpus_per_node=num_gpus_per_node,
                                                   num_nodes=num_nodes,
                                                   additional_buffer_factor=additional_buffer_factor)


def estimate_zero2_model_states_mem_needs_all_cold(total_params,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """
    Print out estimates on memory usage requirements for ZeRO 2 params, optim states and gradients
    for a given ``model`` and hardware setup.

    If it's a hypothetical model, use this function where you have to pass
    the ``total_params`` and ``largest_layer_params`` explicitly.

    If you have an actual model object, use ``estimate_zero2_model_states_mem_needs_all_live`` and everything
    will be derived automatically.

    Args:
        - ``total_params``: total  model params
        - ``num_gpus_per_node``: how many gpus per node (defaults to 1)
        - ``num_nodes``: how many nodes (defaults to 1),
        - ``additional_buffer_factor``: estimation factor (defaults to 1.5):

    """

    def format_options(cpu_offload):
        enabled = []
        device = f'{OffloadDeviceEnum.cpu:4}' if cpu_offload else "none"
        enabled.append(f"offload_optimizer={device}")
        return ", ".join(enabled)

    nodes_str = "nodes" if num_nodes > 1 else "node"
    gpus_str = "GPUs" if num_gpus_per_node > 1 else "GPU"
    print("Estimated memory needed for params, optim states and gradients for a:\n"
          f"HW: Setup with {num_nodes} {nodes_str}, {num_gpus_per_node} {gpus_str} per node.\n"
          f"SW: Model with {int(total_params/1e6)}M total params.")
    print("  per CPU  |  per GPU |   Options")
    for cpu_offload in [True, False]:
        cpu_mem, gpu_mem = estimate_zero2_model_states_mem_needs(total_params=total_params,
                                                                 num_gpus_per_node=num_gpus_per_node,
                                                                 num_nodes=num_nodes,
                                                                 cpu_offload=cpu_offload,
                                                                 additional_buffer_factor=additional_buffer_factor)

        options_str = format_options(cpu_offload=cpu_offload)
        print(f" {cpu_mem/2**30:7.2f}GB | {gpu_mem/2**30:6.2f}GB | {options_str}")
