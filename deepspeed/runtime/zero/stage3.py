# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import gc
import collections
from typing import Deque, Dict, Tuple
from deepspeed import comm as dist
from deepspeed.utils import groups

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed.runtime import ZeROOptimizer
from deepspeed.utils import logger
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler
from deepspeed.runtime.comm.coalesced_collectives import reduce_scatter_coalesced, all_to_all_quant_reduce
from deepspeed.runtime.utils import inf, get_global_norm, is_model_parallel_parameter, get_only_unique_item
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.utils import apply_to_tensors_only
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus
from deepspeed.runtime.swap_tensor.optimizer_utils import OptimizerSwapper
from deepspeed.runtime.swap_tensor.partitioned_optimizer_swapper import PartitionedOptimizerSwapper
from deepspeed.runtime.swap_tensor.pipelined_optimizer_swapper import PipelinedOptimizerSwapper
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, FP32_FLAT_GROUPS, PARTITION_COUNT, ZERO_STAGE, LOSS_SCALER
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import z3_leaf_parameter

# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False

OPTIMIZER_SWAP_IN_STATE_TIMER = 'optimizer_swap_in_state'
INIT_OPTIMIZER_TIMER = 'init_optimizer_state'
OPTIMIZER_SWAP_OUT_STATE_TIMER = 'optimizer_swap_out_state'
OPTIMIZER_STEP_TIMER = 'optimizer_step'


def print_rank_0(message, debug=False, force=False):
    rank = dist.get_rank()
    if rank == 0 and (debug or force):
        logger.info(message)
    # other variations
    # - print for all ranks w/o interleaving
    # printflock(f"[{rank}] {message}")
    # - print to log file per rank
    # log_rank_file(rank, message)


def input(msg):
    return


def isclose(a, b, rtol=1e-09, atol=0.0):
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def lcm(x, y):
    from fractions import gcd  # or can import gcd from `math` in Python 3
    return x * y // gcd(x, y)


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()


INITIAL_MICRO_STEP_ID = -1


class DeepSpeedZeroOptimizer_Stage3(ZeROOptimizer):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """

    def __init__(
        self,
        module,
        init_optimizer,
        timers,
        ds_config,
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        dynamic_loss_args=None,
        verbose=True,
        contiguous_gradients=True,
        reduce_bucket_size=500000000,
        prefetch_bucket_size=50000000,
        max_reuse_distance=1000000000,
        max_live_parameters=1000000000,
        param_persistence_threshold=100000,
        model_persistence_threshold=sys.maxsize,
        dp_process_group=None,
        reduce_scatter=True,
        overlap_comm=False,
        offload_optimizer_config=None,
        offload_param_config=None,
        sub_group_size=1000000000000,
        offload_ratio=0.0,
        mpu=None,
        clip_grad=0.0,
        gradient_accumulation_dtype=torch.float32,
        communication_data_type=torch.float16,
        postscale_gradients=True,
        gradient_predivide_factor=1.0,
        gradient_accumulation_steps=1,
        elastic_checkpoint=False,
        aio_config=None,
        all2all_process_group=None,
        zero_hpz_partition_size=1,
        zero_quantized_weights=False,
        zero_quantized_nontrainable_weights=False,
    ):
        see_memory_usage("Stage 3 initialize beginning", force=True)

        print_rank_0(f"initialized {__class__.__name__} with args: {locals()}", force=False)

        if dist.get_rank() == 0:
            logger.info(f"Reduce bucket size {reduce_bucket_size}")
            logger.info(f"Prefetch bucket size {prefetch_bucket_size}")
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
        if not get_accelerator().is_available():
            raise SystemError("Cannot use fp16 without accelerator.")

        self.optimizer = init_optimizer

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        self.gradient_accumulation_dtype = gradient_accumulation_dtype
        self._global_grad_norm = 0.

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        self.optimizer_swapper = None
        self.swap_optimizer = False

        self.offload_optimizer = False
        self.offload_optimizer_pin_memory = False
        self.offload_optimizer_fast_init = False
        self.offload_param = False
        self.offload_param_pin_memory = False
        self.params_in_nvme_and_cpu = False
        self.max_params_in_cpu = 0
        self.partial_offload = offload_ratio

        #num of ranks in a ZeRO param partitioning group
        self.zero_hpz_partition_size = zero_hpz_partition_size

        zero_param_parallel_group = groups._get_zero_param_intra_parallel_group()
        print_rank_0(
            f"ZeRO Stage 3 param partitioning group {self.zero_hpz_partition_size} {zero_param_parallel_group}",
            force=False)
        if self.zero_hpz_partition_size > 1 and zero_param_parallel_group is None:
            self._set_zero_group_parallelism()
            zero_param_parallel_group = groups._get_zero_param_intra_parallel_group()

        self.parameter_offload = self.initialize_ds_offload(
            module=module,
            timers=timers,
            ds_config=ds_config,
            overlap_comm=overlap_comm,
            prefetch_bucket_size=prefetch_bucket_size,
            max_reuse_distance=max_reuse_distance,
            max_live_parameters=max_live_parameters,
            param_persistence_threshold=param_persistence_threshold,
            model_persistence_threshold=model_persistence_threshold,
            dp_process_group=dp_process_group,
            offload_param_config=offload_param_config,
            mpu=mpu,
            zero_param_parallel_group=zero_param_parallel_group,
            zero_quantized_weights=zero_quantized_weights,
            zero_quantized_nontrainable_weights=zero_quantized_nontrainable_weights)

        self.persistent_parameters = self.parameter_offload.persistent_parameters
        self._configure_offloading(offload_optimizer_config, offload_param_config)

        # backup fused_adam optimizer init
        if self.offload_optimizer and self.partial_offload != 1.0:
            backup_gpu_tensor = torch.randn(1, device=get_accelerator().device_name()).to(self.dtype)
            backup_gpu_param = torch.nn.Parameter(backup_gpu_tensor)
            assert type(init_optimizer) == DeepSpeedCPUAdam, 'Hybrid Optimizer Only Supports DeepSpeedCPUAdam'
            self.backup_optimizer = torch.optim.AdamW([backup_gpu_param],
                                                      lr=self.optimizer.param_groups[0]["lr"],
                                                      betas=self.optimizer.param_groups[0]["betas"],
                                                      eps=self.optimizer.param_groups[0]["eps"],
                                                      weight_decay=self.optimizer.param_groups[0]["weight_decay"],
                                                      amsgrad=self.optimizer.param_groups[0]["amsgrad"])
            # Multiple param_groups configs for back-up optimizer
            if len(self.optimizer.param_groups) > 1:
                for i in range(1, len(self.optimizer.param_groups)):
                    self.backup_optimizer.add_param_group(self.optimizer.param_groups[i])

        self.module = module
        self.elastic_checkpoint = elastic_checkpoint

        self.inf_or_nan_tracker: Tensor = torch.zeros(1,
                                                      dtype=torch.bool,
                                                      device=get_accelerator().current_device_name(),
                                                      requires_grad=False)

        self.deepspeed_adam_offload = (self.offload_optimizer and type(init_optimizer) == DeepSpeedCPUAdam)

        self.device = get_accelerator().current_device_name() if not self.offload_optimizer else OffloadDeviceEnum.cpu
        ### streams used for overlapping computation with communication
        self.reduce_and_partition_stream = None if get_accelerator().is_synchronized_device() else get_accelerator(
        ).Stream() if overlap_comm else get_accelerator().default_stream()

        ############################################################################

        self.n_caching_allocator_flushes = 0

        #-------------Stage 3 Setup-------------------#

        self.timers = timers

        self.all2all_process_group = all2all_process_group

        self.reduce_scatter = reduce_scatter

        self.dp_process_group = self.parameter_offload.dp_process_group
        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()

        self.all2all_process_group = all2all_process_group

        self.zero_quantized_nontrainable_weights = zero_quantized_nontrainable_weights

        self.partition_count = dist.get_world_size(group=self.dp_process_group)

        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_rank = mpu.get_model_parallel_rank()

        self.overflow = False
        self.clip_grad = clip_grad
        self.communication_data_type = communication_data_type
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = 0
        self.reduce_bucket_size = int(reduce_bucket_size)

        if self.all2all_process_group is not None:
            assert self.all2all_process_group is not None and self.reduce_scatter == True, "when enable all_to_all_reduce, reduce_scatter should also be enabled for data type checks."

        if self.reduce_scatter:
            valid_reduce_scatter_dtypes = (torch.float16, torch.bfloat16, torch.float32)
            assert self.communication_data_type in valid_reduce_scatter_dtypes, f"ZeRO-3 supports {valid_reduce_scatter_dtypes} communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with ZeRO-3 with reduce scatter enabled"
            assert self.postscale_gradients, "pre-scale gradients is not yet supported with ZeRO-3 with reduce scatter enabled"

        # Holds the mode parameter
        # The param.data may not hold any meaningful data
        # when param's status is NOT_AVAILABLE or IN_FLGHT
        self.fp16_groups = []

        # Hold partitioned parameters
        self.fp16_partitioned_groups = []

        # Holds a fused and flattened copy of the parameters
        self.fp16_partitioned_groups_flat = []
        self.fp16_partitioned_groups_flat_numel = []
        self.fp16_partitioned_groups_flat_id = []

        #defragmented pinned memory
        self.param_groups_fp16_flat_cpu_memory = []

        #a single 32-bit partition of the parallel partitioned parameters
        #that this process will update
        self.fp32_partitioned_groups_flat = []
        self.next_swappable_fp32_partitioned_groups = []

        # number of elements per partition in each group
        self.partition_size = []

        self.all_reduce_print = False

        self.prefetch_elements = int(prefetch_bucket_size)

        self.contiguous_gradients = contiguous_gradients

        # padding on each partition for alignment purposes
        self.groups_padding = []

        self.sub_group_size = sub_group_size

        self.sub_group_to_group_id = {}

        # Trainable parameters
        self.trainable_param_groups = self._get_trainable_parameter_groups()

        see_memory_usage("Before creating fp16 partitions", force=True)
        self._create_fp16_partitions_with_defragmentation(self.trainable_param_groups)
        num_fp16_subgroups = len(self.fp16_partitioned_groups_flat)
        see_memory_usage(f"After creating fp16 partitions: {num_fp16_subgroups}", force=True)

        # Optimizer tensor swapping
        if self.swap_optimizer:
            self._configure_tensor_swapping(offload_optimizer_config, aio_config)

        self.is_gradient_accumulation_boundary: bool = True

        self.param_reduce_events: Deque[get_accelerator().Event] = collections.deque()
        # TODO. make this configurable via JSON
        self.max_param_reduce_events: int = 2

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.extra_large_param_to_reduce = None
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []

        self.params_already_reduced = {}
        self.is_gradient_accumulation_boundary = True
        self._release_ipg_buffers()
        self.previous_reduced_grads = None

        # model parameter traversal-based param id that's stable across runs
        for params_group in self.fp16_groups:
            for param in params_group:
                param_id = self.get_param_id(param)
                self.param_dict[param_id] = param
                self.params_already_reduced[param_id] = False

        #Largest partitioned param
        largest_partitioned_param_numel = 0
        for fp16_partitioned_group in self.fp16_partitioned_groups:
            if len(fp16_partitioned_group) > 0:
                largest_partitioned_param_numel = max(
                    largest_partitioned_param_numel,
                    max([max(tensor.numel(), tensor.ds_numel) for tensor in fp16_partitioned_group]))

        print_rank_0(f'Largest partitioned param numel = {largest_partitioned_param_numel}', force=False)

        self._setup_for_real_optimizer()
        self.grad_position = {}
        self.set_grad_positions()

        if self.offload_optimizer:
            self.norm_for_param_grads = {}

        # stores if a partition has been reduced in this step
        self.is_partition_reduced = {}

        # stores if a grad in a partition has been computed or not
        self.is_grad_computed = {}

        # will store the averaged gradients required by this partition
        self.averaged_gradients = {}

        #creates backward hooks for gradient partitioning
        ###Calls all gather param
        self._grad_acc_hooks = []
        self._leaf_module_hooks = []
        self.create_reduce_and_remove_grad_hooks()

        #exit(0)

        # we may have a way of fusing dynamic scale. Do not support for now
        self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_scaling=dynamic_loss_scale,
                                            dynamic_loss_args=dynamic_loss_args)
        self.dynamic_loss_scale = self.loss_scaler.dynamic

        self.debug_fp16_grads = [{} for _ in self.fp16_groups]

        self._link_all_hp_params()

        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)

    def destroy(self):
        self.parameter_offload.destroy()
        for hook in self._grad_acc_hooks:
            hook.remove()
        for hook in self._leaf_module_hooks:
            hook.remove()
        print_rank_0("Removed grad acc hooks", force=False)
        del self.__ipg_bucket_flat_buffer

    def initialize_ds_offload(
        self,
        module,
        timers,
        ds_config,
        overlap_comm,
        prefetch_bucket_size,
        max_reuse_distance,
        max_live_parameters,
        param_persistence_threshold,
        model_persistence_threshold,
        dp_process_group,
        offload_param_config,
        mpu,
        zero_param_parallel_group,
        zero_quantized_weights,
        zero_quantized_nontrainable_weights,
    ):
        return DeepSpeedZeRoOffload(module=module,
                                    timers=timers,
                                    ds_config=ds_config,
                                    overlap_comm=overlap_comm,
                                    prefetch_bucket_size=prefetch_bucket_size,
                                    max_reuse_distance=max_reuse_distance,
                                    max_live_parameters=max_live_parameters,
                                    param_persistence_threshold=param_persistence_threshold,
                                    model_persistence_threshold=model_persistence_threshold,
                                    dp_process_group=dp_process_group,
                                    offload_param_config=offload_param_config,
                                    mpu=mpu,
                                    zero_param_parallel_group=zero_param_parallel_group,
                                    zero_quantized_weights=zero_quantized_weights,
                                    zero_quantized_nontrainable_weights=zero_quantized_nontrainable_weights)

    def _get_trainable_parameter_groups(self):
        param_groups = []
        PARAMS_KEY = "params"
        for param_group in self.optimizer.param_groups:
            trainable_params = [p for p in param_group[PARAMS_KEY] if p.requires_grad]
            if len(trainable_params) == 0:
                continue

            trainable_param_group = {}
            for key in param_group.keys():
                if key == PARAMS_KEY:
                    trainable_param_group[PARAMS_KEY] = trainable_params
                else:
                    trainable_param_group[key] = param_group[key]
            param_groups.append(trainable_param_group)

        return param_groups

    def _set_zero_group_parallelism(self):
        groups._create_zero_param_parallel_group(self.zero_hpz_partition_size)

    def invalidate_secondary_tensor(self):
        for fpg in self.fp16_groups:
            for param in fpg:
                if param.ds_secondary_tensor is not None:
                    param.ds_secondary_tensor = None

    def _setup_for_real_optimizer(self):
        see_memory_usage("Before creating fp32 partitions", force=True)
        self._create_fp32_partitions()
        see_memory_usage("After creating fp32 partitions", force=True)
        dist.barrier()

        # To support pipelined optimizer swapping
        self._create_next_swappable_fp32_groups()

        see_memory_usage("Before initializing optimizer states", force=True)

        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=True)
        dist.barrier()

        if dist.get_rank() == 0:
            logger.info(f"optimizer state initialized")

        # IPG
        if self.contiguous_gradients:
            self.__ipg_bucket_flat_buffer: Tensor = torch.empty(self.reduce_bucket_size,
                                                                dtype=self.dtype,
                                                                device=get_accelerator().current_device_name())

        self.grad_partitions_flat_buffer = None
        self.__param_id_to_grad_partition: Dict[int, Tensor] = {}

        all_params = list(itertools.chain.from_iterable(self.fp16_groups))

        self.grad_partitions_flat_buffer: Tensor = torch.zeros(sum(p.partition_numel() for p in all_params),
                                                               dtype=self.gradient_accumulation_dtype,
                                                               device=self.device)
        if self.offload_optimizer_pin_memory:
            self.grad_partitions_flat_buffer = get_accelerator().pin_memory(self.grad_partitions_flat_buffer)

        offset = 0
        for param in all_params:
            self.__param_id_to_grad_partition[param.ds_id] = self.grad_partitions_flat_buffer.narrow(
                0, offset, param.partition_numel())
            offset += param.partition_numel()

    def _link_all_hp_params(self):
        for p in self.module.parameters():
            p._z3_optimizer = self

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    # TODO. factor out to a utility outside of stage3
    @staticmethod
    def defragment(tensors: List[Tensor]) -> Tensor:
        """move provided tensors into a contiguous flat buffer, with some additional
        measures taken to reduce memory fragmentation"""
        assert len(set(t.dtype for t in tensors)) == 1
        assert len(set(t.device for t in tensors)) == 1

        cpu_buffer = torch.empty(sum(p.numel() for p in tensors),
                                 dtype=get_only_unique_item(t.dtype for t in tensors),
                                 device="cpu")
        tensor_infos: List[Tuple[Tensor, int, int]] = []
        orig_device = get_only_unique_item(t.device for t in tensors)

        offset = 0
        for tensor in tensors:
            tensor_numel = tensor.numel()
            # move the tensor from device memory to host memory
            cpu_buffer.narrow(0, offset, tensor_numel).copy_(tensor)
            tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)

            # record some data so we can restore the device tensor later
            tensor_infos.append((tensor, offset, tensor_numel))

            offset += tensor_numel

        gc.collect()
        get_accelerator().empty_cache()

        # copy tensors (now flattened and contiguous) back to GPU
        device_buffer = cpu_buffer.to(orig_device)

        # restore device tensors
        for tensor, offset, tensor_numel in tensor_infos:
            tensor.data = device_buffer.narrow(0, offset, tensor_numel)

        return device_buffer

    def _get_param_coordinator(self, training):
        return self.parameter_offload.get_param_coordinator(training)

    def _configure_offloading(self, offload_optimizer_config, offload_param_config):
        ###################### offload optimizer setup ##################################
        if offload_optimizer_config is not None and offload_optimizer_config.device != OffloadDeviceEnum.none:
            self.offload_optimizer = True
            self.offload_optimizer_pin_memory = offload_optimizer_config.pin_memory
            self.swap_optimizer = offload_optimizer_config.device == OffloadDeviceEnum.nvme
            self.offload_optimizer_fast_init = offload_optimizer_config.fast_init

        ###################### offload param setup ##################################
        if offload_param_config is not None and offload_param_config.device != OffloadDeviceEnum.none:
            self.offload_param = True
            self.offload_param_pin_memory = offload_param_config.pin_memory
            self.params_in_nvme_and_cpu = offload_param_config.device == OffloadDeviceEnum.nvme
            self.max_params_in_cpu = offload_param_config.max_in_cpu
            print_rank_0(
                f"FP16 params swapping is {self.params_in_nvme_and_cpu}, Max params in CPU is {self.max_params_in_cpu}",
                force=False)

    def _configure_tensor_swapping(self, offload_optimizer_config, aio_config):
        nvme_swap_folder = os.path.join(offload_optimizer_config.nvme_path, 'zero_stage_3')
        os.makedirs(nvme_swap_folder, exist_ok=True)
        if dist.get_rank() == 0:
            logger.info(f'Tensor Swapping: Adding optimizer tensors')

        swapper_type = PipelinedOptimizerSwapper if offload_optimizer_config.pipeline else PartitionedOptimizerSwapper

        self.optimizer_swapper = swapper_type(swap_config=offload_optimizer_config,
                                              aio_config=aio_config,
                                              base_folder=nvme_swap_folder,
                                              optimizer=self.optimizer,
                                              largest_numel=max(self.fp16_partitioned_groups_flat_numel),
                                              device=self.device,
                                              dtype=torch.float32,
                                              timers=self.timers)

    @property
    def elements_in_ipg_bucket(self):
        return sum(p.ds_numel for p in self.params_in_ipg_bucket)

    def _move_to_flat_buffer(self, param_list, flat_buffer, avoid_copy=False):
        '''If flat buffer is None then the parameters in the param_list are
        not copied to the flat buffer. This is because they exceed the number of max_params_in_cpu
        Some of these parameters may already be in CPU in unflattened buffers
        or they maybe in GPU, or they maybe in NVME. If they are in NVME, then
        they will be marked as NOT_AVAILABLE, and will be moved to CPU when they are
        needed during training.'''
        if flat_buffer is None:
            # this dst buffer is on NVMe, so skip this
            return

        start = 0
        for param in param_list:
            src = param.ds_tensor
            dest = flat_buffer.narrow(0, start, src.ds_numel)
            start = start + src.ds_numel
            '''if the parameter was initialized in nvme then bring it to the destination buffer directly'''
            if src.status == PartitionedParamStatus.NOT_AVAILABLE:
                print_rank_0(
                    f"Swapping in {param.ds_id} with partition size {param.partition_numel()} permanently to CPU")
                param.nvme_swapper.swap_into_buffer(param, dest)
                src.data = dest.data
                src.status = PartitionedParamStatus.AVAILABLE
            else:
                assert src.status == PartitionedParamStatus.AVAILABLE, "Partitioned Param must be available here"
                if not avoid_copy:
                    dest.data.copy_(src.data)
                src.data = dest.data

            # Final location must be gpu/cpu in this case
            param.ds_tensor.final_location = 'not-nvme'

    def _create_param_groups_fp16_flat_cpu_memory(self):

        aggregate_params_count = 0

        for j, param_group in enumerate(self.trainable_param_groups):
            params_in_group = sum([p.partition_numel() for p in param_group['params']])

            flat_buffer_size = params_in_group

            if self.params_in_nvme_and_cpu and \
                aggregate_params_count + params_in_group > self.max_params_in_cpu:

                flat_buffer_size = max(0, self.max_params_in_cpu - aggregate_params_count)

            aggregate_params_count += params_in_group

            if flat_buffer_size > 0:
                print_rank_0(f"group {j} flat buffer size {flat_buffer_size}", force=False)
                self.param_groups_fp16_flat_cpu_memory.append(get_accelerator().pin_memory(
                    torch.empty(int(flat_buffer_size), dtype=self.dtype)))
            else:
                print_rank_0(f"No flat buffer size. Param group size was  {params_in_group}", force=False)

                self.param_groups_fp16_flat_cpu_memory.append(torch.empty(1, dtype=self.dtype))

    def _create_fp16_partitions_with_defragmentation(self, fp16_param_groups):
        dist.barrier()

        param_groups: List[List[Parameter]] = tuple(
            self._create_fp16_sub_groups(param_group["params"]) for param_group in fp16_param_groups)

        # bookkeeping related to param groups
        for param_group_idx, param_group in enumerate(param_groups):
            for sub_group in param_group:
                sub_group_idx = len(self.fp16_groups)

                # record sub group and partitions
                self.fp16_groups.append(sub_group)
                self.fp16_partitioned_groups.append([param.ds_tensor for param in sub_group])

                # record sub group -> group mapping
                self.sub_group_to_group_id[sub_group_idx] = param_group_idx

                # record total elements of parameter partitions in sub group
                self.fp16_partitioned_groups_flat_numel.append(sum(p.partition_numel() for p in sub_group))

                # record ds_ids of parameter partitions in sub group
                self.fp16_partitioned_groups_flat_id.append([p.ds_id for p in sub_group])

                # record padding required to align group to world size (only applies to last rank)
                rank_requires_padding = dist.get_rank(
                    self.dp_process_group) == dist.get_world_size(self.dp_process_group) - 1
                self.groups_padding.append([p.padding_size() if rank_requires_padding else 0 for p in sub_group])

        # move parameters to flattened buffer
        if not self.offload_param:  # partitioned params remain in GPU during training
            # move parameter partitions into a single contiguous flat buffer
            parameter_partitions: List[Tensor] = []
            for sub_group in self.fp16_groups:
                for param in sub_group:
                    parameter_partitions.append(param.ds_tensor)
            device_buffer = __class__.defragment(parameter_partitions)

            # setup flat buffers per subgroup, these are each just sections of the
            # contiguous flat buffer for all parameters that we created earlier
            offset = 0
            for sub_group in self.fp16_groups:
                sub_group_numel = sum(param.partition_numel() for param in sub_group)
                self.fp16_partitioned_groups_flat.append(device_buffer.narrow(0, offset, sub_group_numel))
                offset += sub_group_numel
        else:  # partitioned params offloaded to CPU when not in use
            # create a flat CPU memory allocation for each param group
            self._create_param_groups_fp16_flat_cpu_memory()
            for param_group_idx, param_group in enumerate(param_groups):
                flat_offset = 0
                for i, sub_group in enumerate(param_group):
                    total_elements = sum(p.partition_numel() for p in sub_group)
                    print_rank_0(f"Params in nvme and cpu {self.params_in_nvme_and_cpu}")
                    #Flat buffer may not be available for parameters that reside in NVME
                    if not self.params_in_nvme_and_cpu or flat_offset + total_elements <= self.param_groups_fp16_flat_cpu_memory[
                            param_group_idx].numel():
                        fp16_partitioned_group_flat = self.param_groups_fp16_flat_cpu_memory[param_group_idx].narrow(
                            0, flat_offset, total_elements)
                        print_rank_0(
                            f"Creating a flat buffer for subgroup {i} requiring {total_elements} elements, and cumulative CPU elements {flat_offset + total_elements}",
                            force=False)

                    elif self.params_in_nvme_and_cpu:
                        fp16_partitioned_group_flat = None
                        print_rank_0(f"No flat buffer for sub group {i} of {total_elements} elements", force=False)
                    else:
                        assert False, "Either params are in nvme, or they are in CPU memory. This code path should not be triggered. Please see you max_params_in_cpu and params_in_nvme configs"

                    self.fp16_partitioned_groups_flat.append(fp16_partitioned_group_flat)
                    flat_offset += total_elements

                    self._move_to_flat_buffer(sub_group,
                                              fp16_partitioned_group_flat,
                                              avoid_copy=not self.offload_param)

        # if necessary, create a pinned memory buffer to be used for swapping out
        # params to NVME after optimizer step
        should_create_fp16_flat_reuse_buffer = any(flattened_partition_group is None
                                                   for flattened_partition_group in self.fp16_partitioned_groups_flat)
        if should_create_fp16_flat_reuse_buffer:
            max_partition_numel, largest_partition_numel = 0, None
            for sub_group in self.fp16_groups:
                total_elements = sum(t.partition_numel() for t in sub_group)
                if total_elements > max_partition_numel:
                    largest_partition_numel = [t.ds_numel for t in sub_group]
                    max_partition_numel = total_elements

            assert len(largest_partition_numel) > 0, f'Unexpected that largest partition is empty'
            self.fp16_groups[0][0].nvme_swapper.reserve_partitioned_swap_space(largest_partition_numel)

    def _swap_in_sub_group_to_flat_buffer(self, flat_buffer, sub_group_id):
        offset = 0
        elements_in_sub_group = sum([t.ds_numel for t in self.fp16_partitioned_groups[sub_group_id]])
        assert (flat_buffer.numel() == elements_in_sub_group)
        for param, partitioned_param in zip(self.fp16_groups[sub_group_id],
                                            self.fp16_partitioned_groups[sub_group_id]):
            dest = flat_buffer.narrow(0, offset, partitioned_param.ds_numel)
            if partitioned_param.status == PartitionedParamStatus.NOT_AVAILABLE:
                print_rank_0(
                    f"Swapping in {param.ds_id} with elements {param.ds_numel} and partition {param.partition_numel()}"
                )
                param.nvme_swapper.swap_in([param], async_op=False)
                dest.data.copy_(partitioned_param.data)
                param.nvme_swapper.remove_partition_and_release_buffers([param])
                print_rank_0(f"Swapping in {param.ds_id} done")
            else:
                dest.data.copy_(partitioned_param.data)
            offset += partitioned_param.ds_numel

    def _create_next_swappable_fp32_groups(self):
        reverse_order_indices = [i for i in range(len(self.fp32_partitioned_groups_flat))]
        reverse_order_indices.reverse()

        next_group = None
        for i in reverse_order_indices:
            self.next_swappable_fp32_partitioned_groups.append(next_group)
            if self._swappable_optimizer_subgroup(i):
                next_group = self.fp32_partitioned_groups_flat[i]

        self.next_swappable_fp32_partitioned_groups.reverse()

    def _get_sub_group_partitions(self, sub_group_id):
        sub_group_partitions = []
        for param, partitioned_param in zip(self.fp16_groups[sub_group_id],
                                            self.fp16_partitioned_groups[sub_group_id]):
            if partitioned_param.status == PartitionedParamStatus.NOT_AVAILABLE:
                swap_path = param.nvme_swapper.get_path(param, True)
                sub_group_partitions.append((partitioned_param, param.partition_numel(), swap_path))
            else:
                sub_group_partitions.append((partitioned_param, partitioned_param.ds_numel, None))

        return sub_group_partitions

    def _create_fp32_partitions(self):
        cpu_memory_usage = 0
        cpu_memory_sub_groups = 0
        nvme_memory_usage = 0
        num_swappable_partitions = 0
        num_swap_from_nvme_partitions = 0
        num_swap_from_cpu_partitions = 0
        swap_from_nvme_memory_usage = 0
        swap_from_cpu_memory_usage = 0
        GIGA_BYTES = (1024**3)

        swappable_fp32_tensors = []
        swappable_fp16_src_tensors = []
        nvme_fp16_partitions_info = []
        nvme_fp16_num_elems = []
        nvme_fp32_dest_tensors = []
        fp32_element_size = torch.tensor([], dtype=torch.float32).element_size()

        # Assign portion of subgroup to cpu, the other to gpu.
        if self.offload_optimizer:
            self.subgroup_to_device = {}
            sub_group_size = len(self.fp16_partitioned_groups_flat)
            # print(f"Partial offload sub_group_size is {sub_group_size}, ratio is {self.partial_offload}\n")
            for i in range(sub_group_size):
                if i < int(self.partial_offload * sub_group_size):
                    self.subgroup_to_device[i] = 'cpu'
                else:
                    self.subgroup_to_device[i] = get_accelerator()._name

        for i, tensor in enumerate(self.fp16_partitioned_groups_flat):
            num_elements = self.fp16_partitioned_groups_flat_numel[i]

            # a partition of the fp32 master weights that will be updated by this process
            if self._swappable_optimizer_subgroup(i):
                self.fp32_partitioned_groups_flat.append(torch.Tensor())
                nvme_memory_usage += (fp32_element_size * num_elements)
                num_swappable_partitions += 1

                if self.params_in_nvme_and_cpu and tensor is None:
                    num_swap_from_nvme_partitions += 1
                    swap_from_nvme_memory_usage += (fp32_element_size * num_elements)
                    if self.offload_optimizer_fast_init:
                        sub_group_partitions = self._get_sub_group_partitions(i)
                        nvme_fp16_partitions_info.append(sub_group_partitions)
                        nvme_fp16_num_elems.append(num_elements)
                        nvme_fp32_dest_tensors.append(self.fp32_partitioned_groups_flat[i])
                    else:
                        unpinned_fp32_buffer = torch.empty(num_elements, device=self.device, dtype=torch.float)
                        self._swap_in_sub_group_to_flat_buffer(unpinned_fp32_buffer, i)
                        self.optimizer_swapper.initialize_parameters(parameters=[self.fp32_partitioned_groups_flat[i]],
                                                                     src_tensors=[unpinned_fp32_buffer])
                else:
                    num_swap_from_cpu_partitions += 1
                    swap_from_cpu_memory_usage += (fp32_element_size * num_elements)
                    swappable_fp32_tensors.append(self.fp32_partitioned_groups_flat[i])
                    swappable_fp16_src_tensors.append(self.fp16_partitioned_groups_flat[i])
            else:
                cpu_memory_usage += (fp32_element_size * num_elements)
                cpu_memory_sub_groups += 1

                if self.params_in_nvme_and_cpu and tensor is None:
                    unpinned_fp32_buffer = torch.empty(num_elements, device=self.device, dtype=torch.float)
                    self._swap_in_sub_group_to_flat_buffer(unpinned_fp32_buffer, i)
                    self.fp32_partitioned_groups_flat.append(unpinned_fp32_buffer)
                else:
                    if self.offload_optimizer:
                        self.fp32_partitioned_groups_flat.append(self.fp16_partitioned_groups_flat[i].to(
                            self.subgroup_to_device[i]).clone().float().detach())
                    else:
                        self.fp32_partitioned_groups_flat.append(self.fp16_partitioned_groups_flat[i].to(
                            self.device).clone().float().detach())

            self.fp32_partitioned_groups_flat[i].requires_grad = True  # keep this in case internal optimizer uses it
            ds_id_begin = str(self.fp16_partitioned_groups_flat_id[i][0])
            ds_id_end = str(self.fp16_partitioned_groups_flat_id[i][-1])
            self.fp32_partitioned_groups_flat[i].ds_id = ds_id_begin + '_' + ds_id_end

        if len(swappable_fp32_tensors) > 0:
            self.optimizer_swapper.initialize_parameters(parameters=swappable_fp32_tensors,
                                                         src_tensors=swappable_fp16_src_tensors)

        if len(nvme_fp32_dest_tensors) > 0:
            fp16_pinned_buffers = self.fp16_groups[0][0].nvme_swapper.reserve_available_buffers()
            assert len(fp16_pinned_buffers) > 0
            self.optimizer_swapper.initialize_from_swapped_fp16_params(fp16_partitions_info=nvme_fp16_partitions_info,
                                                                       fp16_num_elems=nvme_fp16_num_elems,
                                                                       fp16_pinned_buffers=fp16_pinned_buffers,
                                                                       fp32_parameters=nvme_fp32_dest_tensors)
            self.fp16_groups[0][0].nvme_swapper.release_reserved_buffers()

        nvme_gigabytes = nvme_memory_usage / GIGA_BYTES
        print_rank_0(f'Swappable FP32 Partitions: count={num_swappable_partitions} size={nvme_gigabytes:5.2f} GB',
                     force=False)
        if self.params_in_nvme_and_cpu:
            print_rank_0(
                f'Swap from NVMe Partitions: count = {num_swap_from_nvme_partitions}, size = {swap_from_nvme_memory_usage/GIGA_BYTES:5.2f}GB',
                force=False)
            print_rank_0(
                f'Swap from CPU Partitions: count = {num_swap_from_cpu_partitions}, size = {swap_from_cpu_memory_usage/GIGA_BYTES:5.2f}GB',
                force=False)

        cpu_memory_gigabytes = cpu_memory_usage / GIGA_BYTES
        print_rank_0(f'In-Memory FP32 Partitions: count={cpu_memory_sub_groups} size={cpu_memory_gigabytes:5.2f} GB',
                     force=False)

        # Clear for on-the-fly population before the optimizer step
        for param_group in self.optimizer.param_groups:
            param_group['params'] = []

    def _create_fp16_sub_groups(self, params_group):

        params_group_numel = sum([param.partition_numel() for param in params_group])
        sub_group_size = self.sub_group_size

        if sub_group_size is None or sub_group_size >= params_group_numel:
            return [params_group]

        sub_groups = []
        sub_group = []
        local_sub_group_size = 0
        for param in params_group:

            sub_group.append(param)
            local_sub_group_size += param.partition_numel()

            if local_sub_group_size >= sub_group_size or id(param) == id(params_group[-1]):

                sub_groups.append(sub_group)

                sub_group = []
                local_sub_group_size = 0

        return sub_groups

    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None

    def _optimizer_step(self, sub_group_id):
        param_group_id = self.sub_group_to_group_id[sub_group_id]
        fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
        if self.offload_optimizer:
            cur_device = self.subgroup_to_device[sub_group_id]
            if cur_device == 'cpu':
                self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
                cpu_loss = self.optimizer.step()
                self.optimizer.param_groups[param_group_id]['params'] = []
            else:
                self.backup_optimizer.param_groups[param_group_id]['params'] = [fp32_param]
                gpu_loss = self.backup_optimizer.step()
                self.backup_optimizer.param_groups[param_group_id]['params'] = []
        else:
            self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
            self.optimizer.step()
            self.optimizer.param_groups[param_group_id]['params'] = []

    def _swappable_optimizer_subgroup(self, sub_group_id):
        if not self.swap_optimizer:
            return False

        return self.optimizer_swapper.swappable_tensor(None,
                                                       numel=self.fp16_partitioned_groups_flat_numel[sub_group_id])

    def _partitioned_params_swap_out(self, i):
        offset = 0
        fp32_param = self.fp32_partitioned_groups_flat[i]
        assert fp32_param is not None, \
        f'fp32 parameters of sub_group {i} is None'

        swap_fp16_params = []
        swap_fp32_params = []
        for param, partitioned_param in zip(self.fp16_groups[i], self.fp16_partitioned_groups[i]):
            src = fp32_param.narrow(0, offset, partitioned_param.ds_numel)
            if partitioned_param.status == PartitionedParamStatus.AVAILABLE:
                partitioned_param.data.copy_(src.data)
            else:
                swap_fp32_params.append(src)
                swap_fp16_params.append(param)
            offset += partitioned_param.ds_numel

        if len(swap_fp16_params):
            swap_fp16_params[0].nvme_swapper.swap_out_partitioned_params(dst_fp16_params=swap_fp16_params,
                                                                         src_fp32_params=swap_fp32_params)

    def initialize_optimizer_states(self):
        num_subgroups = len(self.fp16_groups)

        largest_numel = max([sum([p.ds_numel for p in psg]) for psg in self.fp16_partitioned_groups])
        gradient_dtype = self.fp32_partitioned_groups_flat[0].dtype
        gradient_buffer = torch.zeros(int(largest_numel), dtype=gradient_dtype, device=self.device)

        timer_names = set()

        # State initialization for the Adagrad optimizer occurs at construction as opposed to other optimizers
        # which do lazy initialization of the state at the first call to step.
        is_adagrad = isinstance(self.optimizer, torch.optim.Adagrad)

        if self.swap_optimizer:
            self.optimizer_swapper.init_timers()

        timer_names.add(INIT_OPTIMIZER_TIMER)
        self.timers(INIT_OPTIMIZER_TIMER).start()

        for i, group in enumerate(self.fp16_groups):
            swappable_optimizer_subgroup = self._swappable_optimizer_subgroup(i)
            swappable_param_subgroup = self.fp16_partitioned_groups_flat[i] is None

            num_elements = int(self.fp16_partitioned_groups_flat_numel[i])

            see_memory_usage(
                f'[Begin] Initialize optimizer states {i} / {num_subgroups} subgroups, num_elems: {num_elements}, swappable opt/param:{swappable_optimizer_subgroup}/{swappable_param_subgroup}',
                force=False)

            if swappable_optimizer_subgroup:
                self._optimizer_states_and_gradient_swap_in(i, timer_names)

            if self.offload_optimizer and not swappable_optimizer_subgroup:
                subgroup_gradient_buffer = torch.zeros(num_elements, dtype=gradient_dtype, device=self.device)
                if self.offload_optimizer_pin_memory:
                    subgroup_gradient_buffer = get_accelerator().pin_memory(subgroup_gradient_buffer)

                self.fp32_partitioned_groups_flat[i].grad = subgroup_gradient_buffer.to(self.subgroup_to_device[i])
            else:
                self.fp32_partitioned_groups_flat[i].grad = gradient_buffer.narrow(0, 0, num_elements)

            if swappable_param_subgroup:
                self._partitioned_params_swap_out(i)

            if swappable_optimizer_subgroup:
                self._optimizer_states_and_gradient_swap_out(i, timer_names)

            see_memory_usage(
                f'[End] Initialize optimizer states {i} / {num_subgroups} subgroups, num_elems: {num_elements}, swappable opt/param:{swappable_optimizer_subgroup}/{swappable_param_subgroup}',
                force=False)

        # Initialize the optimizer states with the flattened fp32 partition.
        if is_adagrad:
            self.optimizer = torch.optim.Adagrad(self.fp32_partitioned_groups_flat, **self.optimizer.defaults)

        self.timers(INIT_OPTIMIZER_TIMER).stop()
        self.timers.log(timer_names)

        if self.swap_optimizer:
            self.optimizer_swapper.log_timers()

        if not self.offload_optimizer:
            for group in self.fp32_partitioned_groups_flat:
                group.grad = None

        # Reset steps
        return

    #########################################################################
    #########################ZeRO Partition Gradients########################
    #########################################################################

    def get_first_param_index(self, group_id, param_group, partition_id):
        for index, param in enumerate(param_group):
            param_id = self.get_param_id(param)
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                return index
        return None

    def initialize_gradient_partitioning_data_structures(self):

        total_partitions = dist.get_world_size(group=self.dp_process_group)

        for i, param_group in enumerate(self.fp16_groups):

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
                self.initialize_gradient_partition(i, param_group, partition_id)
                self.is_partition_reduced[i][partition_id] = False
                self.first_param_index_in_partition[i][partition_id] = self.get_first_param_index(
                    i, param_group, partition_id)

    @instrument_w_nvtx
    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)
        self.__reduce_and_partition_ipg_grads()
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        if not get_accelerator().resolves_data_dependency():
            self.reduce_and_partition_stream.synchronize()

        for param_id in self.params_already_reduced.keys():
            self.params_already_reduced[param_id] = False

        #in case of cpu offload, averaged gradients are already in fp32_partitioned_groups_flat.grad
        #TODO: use a similar code path for both cpu_offload and non-cpu offload
        if not self.offload_optimizer:
            for i, sub_group in enumerate(self.fp16_groups):
                #TODO: This is redundant
                self.averaged_gradients[i] = [
                    self.__param_id_to_grad_partition[param.ds_id]
                    if param.requires_grad else torch.zeros_like(param.ds_tensor) for param in sub_group
                ]
        # this method gets called after every backward. need to increment
        # here because if it gets incremented in backward() the micro step
        # id will be off by one when we do the reduce and partition at the.
        # start of this method.
        # TODO. make this less error prone
        self.micro_step_id += 1

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()

    def create_reduce_and_remove_grad_hooks(self):
        print_rank_0(f'[Begin] Create gradient reduction hooks')
        self.grad_accs = []
        self.leaf_parameters = defaultdict(list)
        for i, param_group in enumerate(self.fp16_groups):
            for param in param_group:
                if param.requires_grad:
                    #print_rank_0(f" Before all gather {param.device}, {param.shape}")
                    print_rank_0(f"Before all gather {param.device}, {param.shape}", force=False)

                    # The hook must be created in un-partitioned parameter
                    param.all_gather()

                    #print(f"After all gather {param.device}, {param.shape}")
                    def wrapper(param):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        @instrument_w_nvtx
                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param)

                        self._grad_acc_hooks.append(grad_acc.register_hook(reduce_partition_and_remove_grads))
                        self.grad_accs.append(grad_acc)

                    #print(f"param grad fn {param.expand_as(param).grad_fn}")
                    if z3_leaf_parameter(param):
                        self.leaf_parameters[param.ds_z3_leaf_module].append(param)
                    else:
                        wrapper(param)

                    # Partition the parameter after creating the hook
                    param.partition()

        # We delay reduce-scatter for all gradients in the leaf modules until the backward pass of the leaf module is done
        for leaf_module, leaf_parameters in self.leaf_parameters.items():

            def wrapper_pre_hook(params):

                def forward_pre_hook(module, input):
                    """Pre-forward hook to set backward hook on input tensors to the leaf module"""
                    module._leaf_module_inputs_remaining = 0

                    @instrument_w_nvtx
                    def reduce_leaf_module_grads(grad):
                        module._leaf_module_inputs_remaining -= 1
                        # Make sure everything is done in the leaf module
                        if module._leaf_module_inputs_remaining == 0:
                            for param in params:
                                if param.grad is None:
                                    param.grad = torch.zeros_like(param)
                                self.reduce_ready_partitions_and_remove_grads(param)

                    def set_module_bwd_hook(tensor):
                        if tensor.requires_grad:
                            module._leaf_module_inputs_remaining += 1
                            tensor.register_hook(reduce_leaf_module_grads)
                        return tensor

                    output = apply_to_tensors_only(set_module_bwd_hook, input)

                    return output

                return forward_pre_hook

            def wrapper_post_hook():

                def forward_post_hook(module, input, output):
                    """Pre-forward hook to set backward hook on input tensors to the leaf module"""
                    module._leaf_output_required_grad_num = 0

                    def increment_rg_count_bwd_hook(tensor):
                        if tensor.requires_grad:
                            module._leaf_output_required_grad_num += 1
                        return tensor

                    apply_to_tensors_only(increment_rg_count_bwd_hook, output)

                    if module._leaf_module_inputs_remaining == 0 and module._leaf_output_required_grad_num > 0:
                        raise RuntimeError(
                            "A module cannot be set as a leaf module when it does not have any input tensors that require gradients and has output tensors that require gradients. This is because the gradient reduction hook will not be called in this case."
                        )

                return forward_post_hook

            self._leaf_module_hooks.append(leaf_module.register_forward_pre_hook(wrapper_pre_hook(leaf_parameters)))
            self._leaf_module_hooks.append(leaf_module.register_forward_hook(wrapper_post_hook()))

        print_rank_0(f'[End] Create gradient reduction hooks')

    def get_param_id(self, param):
        return OptimizerSwapper.parameter_id(param)

    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = (100.0 * elem_count) // self.reduce_bucket_size
        see_memory_usage(
            f"{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}",
            force=False)

    ###############Independent Partition Gradient ########################
    def reduce_independent_p_g_buckets_and_remove_grads(self, param):
        #print_rank_0(f"Inside reduce ipg buckets. {debug_param2name_id_shape(param)}, ipg elements {self.elements_in_ipg_bucket}, reduce bucket size {self.reduce_bucket_size}", force=True)

        # Because the ipg bucket is initialized with a random place holder tensor, we must
        # explicitly check that the bucket has any real data in it (self.elements_in_ipg_bucket >
        # 0). Otherwise if the incoming param.ds_numel is large, this branch may get triggered on a
        # garbage data and `self.average_tensor()` will crash because its params_to_reduce will be
        # empty, while reduction_list will have that garbage data.
        if self.elements_in_ipg_bucket + param.ds_numel > self.reduce_bucket_size and self.elements_in_ipg_bucket > 0:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.ds_numel)

            self.__reduce_and_partition_ipg_grads()

        self.__add_grad_to_ipg_bucket(param)

    @instrument_w_nvtx
    @torch.no_grad()
    def __add_grad_to_ipg_bucket(self, param: Parameter) -> None:
        if not get_accelerator().resolves_data_dependency():
            self.reduce_and_partition_stream.wait_stream(get_accelerator().default_stream())

        if self.contiguous_gradients and self.elements_in_ipg_bucket + param.grad.numel() <= self.reduce_bucket_size:
            # move the gradient to a contiguous buffer
            with get_accelerator().stream(self.reduce_and_partition_stream):
                # move the parameter's gradient to the contiguous flat buffer
                new_grad_tensor = self.__ipg_bucket_flat_buffer.narrow(0, self.elements_in_ipg_bucket,
                                                                       param.grad.numel()).view_as(param.grad)
                new_grad_tensor.copy_(param.grad, non_blocking=True)
                if not get_accelerator().is_synchronized_device():
                    param.grad.record_stream(get_accelerator().current_stream())
                param.grad.data = new_grad_tensor

        self.params_in_ipg_bucket.append(param)

    @instrument_w_nvtx
    @torch.no_grad()
    def __reduce_and_partition_ipg_grads(self, safe_mode: bool = False) -> None:
        if not self.params_in_ipg_bucket:
            return

        for param in self.params_in_ipg_bucket:
            if param.grad.numel() != param.ds_numel:
                raise RuntimeError(f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter "
                                   f"gradients whose size is not same as the params")

        assert len(set(p.ds_id for p in self.params_in_ipg_bucket)) == len(self.params_in_ipg_bucket)

        while self.param_reduce_events and self.param_reduce_events[0].query():
            self.param_reduce_events.popleft()
        if len(self.param_reduce_events) > self.max_param_reduce_events:
            self.param_reduce_events.popleft().synchronize()

        with get_accelerator().stream(self.reduce_and_partition_stream):
            if safe_mode:
                assert_ints_same_as_other_ranks([p.ds_id for p in self.params_in_ipg_bucket])

            if self.contiguous_gradients and self.elements_in_ipg_bucket <= self.reduce_bucket_size and not self.reduce_scatter:
                grad_bucket = self.__ipg_bucket_flat_buffer.narrow(0, 0, self.elements_in_ipg_bucket)
                grad_partitions = self.__avg_scatter_contiguous_grads(grad_bucket)
            else:
                self.params_in_ipg_bucket.sort(key=lambda p: p.ds_id)
                grad_partitions = self.__avg_scatter_grads(self.params_in_ipg_bucket)

            self.partition_grads(self.params_in_ipg_bucket, grad_partitions)

            self.params_in_ipg_bucket.clear()

            if not get_accelerator().handles_memory_backpressure():
                event = get_accelerator().Event()
                event.record()
                self.param_reduce_events.append(event)

    @instrument_w_nvtx
    def __avg_scatter_contiguous_grads(self, buffer_to_reduce: Tensor) -> List[Tensor]:
        dtype = buffer_to_reduce.dtype
        if self.communication_data_type != dtype:
            buffer_to_reduce = buffer_to_reduce.to(self.communication_data_type)
        if self.postscale_gradients and self.gradient_predivide_factor != 1.0:
            buffer_to_reduce = buffer_to_reduce.div_(self.gradient_predivide_factor)

        world_sz = dist.get_world_size(self.dp_process_group)
        rank = dist.get_rank(self.dp_process_group)
        buffer_to_reduce.div_(world_sz / float(self.sequence_parallel_size))

        dist.all_reduce(buffer_to_reduce, group=self.dp_process_group)

        if self.postscale_gradients and self.gradient_predivide_factor != world_sz:
            buffer_to_reduce = buffer_to_reduce.mul(self.gradient_predivide_factor)

        if self.communication_data_type != self.dtype:
            buffer_to_reduce = buffer_to_reduce.to(self.dtype)

        grad_partitions = []
        grad_offset_in_buffer = 0
        for param in self.params_in_ipg_bucket:
            grad = param.grad
            chunk_sz = math.ceil(grad.numel() / world_sz)

            start_offset = grad_offset_in_buffer + min(rank * chunk_sz, grad.numel())
            end_offset = grad_offset_in_buffer + min(rank * chunk_sz + chunk_sz, grad.numel())

            partition = buffer_to_reduce[start_offset:end_offset]
            if param.partition_numel() != partition.numel():
                padded_partition = torch.zeros(param.partition_numel(), device=grad.device, dtype=grad.dtype)
                if partition.numel() > 0:
                    padded_partition[:partition.numel()] = partition
                grad_partitions.append(padded_partition)
            else:
                grad_partitions.append(partition)
            grad_offset_in_buffer += grad.numel()

        return grad_partitions

    @instrument_w_nvtx
    def __avg_scatter_grads(self, params_to_reduce: List[Parameter]) -> List[Tensor]:
        """average gradients and scatter partitions across ranks"""

        full_grads_for_rank = [p.grad for p in params_to_reduce]
        if self.communication_data_type != self.dtype:
            full_grads_for_rank = [g.to(self.communication_data_type) for g in full_grads_for_rank]

        if self.postscale_gradients and self.gradient_predivide_factor != 1.0:
            full_grads_for_rank = [g.div(self.gradient_predivide_factor) for g in full_grads_for_rank]

        local_world_size = get_accelerator().device_count()
        global_world_size = dist.get_world_size()
        num_nodes = global_world_size // local_world_size
        if self.all2all_process_group is not None and num_nodes > 1:
            grad_partitions_for_rank = all_to_all_quant_reduce(full_grads_for_rank, self.all2all_process_group)
        else:
            grad_partitions_for_rank = reduce_scatter_coalesced(full_grads_for_rank, self.dp_process_group)

        if self.postscale_gradients and self.gradient_predivide_factor != 1.0 and self.gradient_predivide_factor != dist.get_world_size(
                self.dp_process_group):
            grad_partitions_for_rank = [g.mul(self.gradient_predivide_factor) for g in grad_partitions_for_rank]

        if self.communication_data_type != self.dtype:
            grad_partitions_for_rank = [g.to(self.dtype) for g in grad_partitions_for_rank]

        return grad_partitions_for_rank

    def set_grad_positions(self):
        for i, group in enumerate(self.fp16_groups):
            current_offset = 0
            for param in group:
                param_id = self.get_param_id(param)
                num_elements = param.partition_numel()

                self.grad_position[param_id] = [int(i), int(current_offset), int(num_elements)]
                #print(f"param id {param_id} i:{i}, ds_tensor {num_elements} numel {param.numel()}")
                current_offset += num_elements
        see_memory_usage(f"After Set Grad positions", force=False)

    def _constant_buffered_norm2(self, input, buffer_size=250000000):
        norm = None
        for part in input.view(-1).split(buffer_size):
            if norm is None:
                norm = part.data.double().norm(2)**2.0
            else:
                norm += part.data.double().norm(2)**2.0
        return norm**0.5

    def set_norm_for_param_grad_in_gpu(self, param):
        param_id = self.get_param_id(param)
        #self.norm_for_param_grads[param_id] = param.grad.data.double().norm(2)
        #Using a more memory efficient version
        self.norm_for_param_grads[param_id] = self._constant_buffered_norm2(param.grad)

    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param, fp32_grad_tensor):
        with get_accelerator().stream(self.copy_grad_stream):
            param_id = self.get_param_id(param)
            src_tensor = param.grad.view(-1).float()
            #print(f"src_tensor {src_tensor.size()} and fp32 grad {fp32_grad_tensor.size()}")
            fp32_grad_tensor.copy_(src_tensor, non_blocking=True)
            param.grad = None

    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        total_norm = 0.0
        norm_type = 2.0
        for p in params:
            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                param_id = self.get_param_id(p)
                if param_id in self.norm_for_param_grads.keys():
                    param_norm = self.norm_for_param_grads[param_id]
                    total_norm += param_norm**2

        # Sum across all model parallel GPUs.
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])

        dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

        total_norm = total_norm_cuda[0]**(1. / norm_type)

        norm_is_inf = total_norm.isinf()
        norm_is_nan = total_norm.isnan()
        inf_or_nan = norm_is_nan.logical_or(norm_is_inf)

        err = torch.tensor(-1.0, device=self.device, dtype=torch.float)
        total_norm = inf_or_nan * err + inf_or_nan.logical_not() * total_norm

        return total_norm

    @instrument_w_nvtx
    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        offload_fp32_gradients = {}
        offload_fp32_offsets = {}
        buffers = []
        for param, grad_partition in zip(params_to_release, grad_partitions):

            contains_real_data = param.partition_numel() * dist.get_rank(self.dp_process_group) < param.ds_numel
            if not contains_real_data:
                # this grad partition is empty - don't need to do anything
                param.grad = None
                continue

            # move or accumulate gradient partition to target buffer
            grad_buffer = self.__param_id_to_grad_partition[param.ds_id].narrow(0, 0, grad_partition.numel())
            buffers.append(grad_buffer)
            if self.micro_step_id == 0:  # don't accumulate
                grad_buffer.copy_(grad_partition, non_blocking=True)
                # ensure grad buffer is a CUDA buffer to speed up the next few
                # operations and so it can be used asynchronously
                grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
            elif get_accelerator().on_accelerator(grad_buffer):
                grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(grad_buffer.shape))
            else:
                # if dst is CPU, copy first to src device, do the addition
                # there, then move back to dst. adding directly to cpu is very slow
                cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
                cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(cuda_grad_buffer.shape))
                grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
                # ensure grad buffer is a CUDA buffer to speed up the next few
                # operations and so it can be used asynchronously
                grad_buffer = cuda_grad_buffer

            # offload the gradient partition if applicable
            if self.offload_optimizer:
                i, dest_offset, _ = self.grad_position[self.get_param_id(param)]

                if self.is_gradient_accumulation_boundary:
                    self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_buffer)

                    if self._swappable_optimizer_subgroup(i):
                        if not i in offload_fp32_gradients.keys():
                            offload_fp32_gradients[i] = []
                            offload_fp32_offsets[i] = []

                        offload_fp32_gradients[i].append(grad_buffer.float())
                        offload_fp32_offsets[i].append(dest_offset)
                    else:
                        fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(
                            0, dest_offset, grad_buffer.numel())
                        fp32_grad_tensor.copy_(grad_buffer)

            # free the gradient
            if not get_accelerator().is_synchronized_device():
                param.grad.record_stream(get_accelerator().current_stream())
            param.grad = None

        if self.offload_optimizer and self.swap_optimizer:
            for i in offload_fp32_gradients.keys():
                self.optimizer_swapper.swap_out_gradients(parameter=self.fp32_partitioned_groups_flat[i],
                                                          gradient_offsets=offload_fp32_offsets[i],
                                                          gradient_tensors=offload_fp32_gradients[i])
        return buffers

    def reduce_ready_partitions_and_remove_grads(self, param):
        #print_rank_0(f"Backward {debug_param2name_id_shape(param)}", force=True)
        self.reduce_independent_p_g_buckets_and_remove_grads(param)

    def zero_reduced_gradients(self, partition_id, i):

        def are_all_related_partitions_reduced(params_id):
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            return True

        for params_id in self.is_grad_computed[i][partition_id]:
            if are_all_related_partitions_reduced(params_id):
                self.param_dict[params_id].grad = None

    def quantize_nontrainable_params(self):
        """ In ZeRO-3, when the zero_quantized_nontrainable_weights flag is set, we quantize the non-trainable weights and also store them in quantized format. However, this check for trainable/non-trainable is done when deepspeed initializes the partitioning. So, if the user changes the trainable/non-trainable status of a parameter after the partitioning is done (e.g. LoRA), the user needs to re-quantize the non-trainable weights by calling this function.
        """
        if not self.zero_quantized_nontrainable_weights:
            print_rank_0(
                f"Warning: quantize_nontrainable_params() called with zero_quantized_nontrainable_weights disabled, return without doing anything",
                force=True)
            return
        quantizer_module = CUDAQuantizer()

        def quantize_dstensor(tensor):
            assert tensor.dtype == torch.float16, f"quantize_dstensor() expects tensor.dtype == torch.float16, got {tensor.dtype}"
            partition_size = tensor.ds_numel
            ds_status = tensor.status
            final_location = tensor.final_location
            tensor, tensor.ds_quant_scale = quantizer_module.quantize(tensor)
            tensor.ds_numel = partition_size
            tensor.status = ds_status
            tensor.final_location = final_location
            tensor.requires_grad = False
            return tensor

        for param in self.module.parameters():
            if hasattr(param, "ds_tensor") and (param.ds_tensor.numel() <= 2048 or param.ds_numel <= 500000):
                # skip small parameters
                continue
            if hasattr(param,
                       "ds_tensor") and not param.requires_grad and not hasattr(param.ds_tensor, "ds_quant_scale"):
                param.ds_tensor = quantize_dstensor(param.ds_tensor)
            if hasattr(param, "ds_secondary_tensor") and not param.requires_grad and not hasattr(
                    param.ds_secondary_tensor, "ds_quant_scale") and param.ds_secondary_tensor is not None:
                param.ds_secondary_tensor = quantize_dstensor(param.ds_secondary_tensor)
        get_accelerator().synchronize()

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
                param.grad = torch.zero_like(param)

    ######################Reduction Related Methods##############################

    def allreduce_bucket(self, bucket, rank=None, log=None):
        rank = None
        tensor = self.flatten(bucket)

        tensor_to_allreduce = tensor

        if pg_correctness_test:
            communication_data_type = torch.float32
        else:
            communication_data_type = self.communication_data_type

        if communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(communication_data_type)

        tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))

        if rank is None:
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        else:
            global_rank = dist.get_global_rank(self.dp_process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)

        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                tensor.copy_(tensor_to_allreduce)

        return tensor

    # if rank is specified do a reduction instead of an allreduce
    def allreduce_and_copy(self, small_bucket, rank=None, log=None):
        with get_accelerator().stream(self.reduction_stream):
            allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(self, bucket, numel_per_bucket=500000000, rank=None, log=None):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, rank=rank, log=None)
                small_bucket = []
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, rank=rank, log=log)

    #############################################################################
    #############################################################################
    #############################################################################

    # views the tensor as multiple partitions and returns
    # those partitions
    def get_data_parallel_partitions(self, tensor):
        partitions = []

        dp = dist.get_world_size(group=self.dp_process_group)
        dp_id = dist.get_rank(group=self.dp_process_group)

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

    @instrument_w_nvtx
    def zero_grad(self, set_to_none=True):
        """
        Zero FP16 parameter grads.
        """
        self.micro_step_id = 0

        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_to_none:
                    if p.grad is not None and get_accelerator().on_accelerator(p.grad):
                        p.grad.record_stream(get_accelerator().current_stream())
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        if self.model_parallel_group is None:
            pass
        else:
            dist.all_reduce(tensor=tensor, op=op, group=self.model_parallel_group)

    @instrument_w_nvtx
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
        if norm_type == inf:
            total_norm = max(g.data.abs().max() for g in gradients)
            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.MAX)
            total_norm = total_norm_cuda[0]
        else:
            # if dist.get_rank() == 0:
            #    logger.info(f"Total Norm beginning {total_norm}")
            grad_norms = []
            for g, p in zip(gradients, params):
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    grad_norms.append(g.to(get_accelerator().device_name(), non_blocking=True).double().norm(2))

            # Sum across all model parallel GPUs.
            if len(grad_norms) == 0:
                # FIX https://github.com/microsoft/DeepSpeed/issues/3564
                total_norm_cuda = torch.tensor(0,
                                               dtype=gradients[0].dtype).to(get_accelerator().device_name()).double()
            else:
                total_norm_cuda = torch.sum(torch.pow(torch.stack(grad_norms), 2))

            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

            total_norm = total_norm_cuda**(1. / norm_type)

        norm_is_inf = total_norm.isinf()
        norm_is_nan = total_norm.isnan()
        inf_or_nan = norm_is_nan.logical_or(norm_is_inf)

        err = torch.tensor(-1.0, device=self.device, dtype=torch.float)
        total_norm = inf_or_nan * err + inf_or_nan.logical_not() * total_norm

        return total_norm

    # creates a flat fused tensor from the tensor list starting at the first_offset
    # in the first tensor of the list. If there are not enough elements in the tensor
    # list then the flat tensor will be padded with zeros
    def get_flat_partition(self, tensor_list, first_offset, partition_size, return_tensor_list=False):
        flat_tensor_list = []
        current_size = 0
        for i, tensor in enumerate(tensor_list):
            if tensor.grad is None:
                tensor.grad = torch.zeros_like(tensor)

            tensor = tensor.grad
            num_elements = tensor.numel()
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
            flat_tensor_list.append(
                torch.zeros(int(partition_size - current_size),
                            dtype=tensor_list[0].dtype,
                            device=tensor_list[0].device))

        if return_tensor_list:
            return flat_tensor_list

        return self.flatten(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None

    def reset_cpu_buffers(self):
        self.norm_for_param_grads = {}

    def _pre_step(self):
        self.micro_step_id = 0

        print_rank_0(f"Inside Step function")
        see_memory_usage(f"In step before checking overflow", force=False)

        print_rank_0("Finished Tracing at Beginning of Step")
        self._get_param_coordinator(training=True).hierarchy = 0

        print_rank_0("Finished Tracing at Beginning of Step")

    @instrument_w_nvtx
    def _get_norm_groups(self):
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            if self.offload_optimizer:
                norm_groups.append(self.complete_grad_norm_calculation_for_cpu_offload(self.fp16_groups[i]))
            else:
                norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.fp16_groups[i]))
        return norm_groups

    @instrument_w_nvtx
    def _prepare_fp32_grad_for_sub_group(self, sub_group_id):
        partition_id = dist.get_rank(group=self.dp_process_group)

        single_grad_partition = self.flatten(self.averaged_gradients[sub_group_id]).to(
            self.fp32_partitioned_groups_flat[sub_group_id].dtype)

        assert single_grad_partition.numel() == self.fp32_partitioned_groups_flat[sub_group_id].numel(), \
            "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                single_grad_partition.numel(), self.fp32_partitioned_groups_flat[sub_group_id].numel(), sub_group_id, partition_id)

        self.fp32_partitioned_groups_flat[sub_group_id].grad = single_grad_partition

        # release all the gradient since we have already created a necessary copy in dp_grad_partition
        self.zero_grad(set_to_none=True)

        if not get_accelerator().is_synchronized_device():
            for grad in filter(lambda g: get_accelerator().on_accelerator(g), self.averaged_gradients[sub_group_id]):
                grad.record_stream(get_accelerator().current_stream())

        self.averaged_gradients[sub_group_id] = None

    @instrument_w_nvtx
    def _prepare_sub_group(self, sub_group_id, timer_names):
        see_memory_usage(f'Before prepare optimizer sub group {sub_group_id}', force=False)
        if self._swappable_optimizer_subgroup(sub_group_id):
            self._optimizer_states_and_gradient_swap_in(sub_group_id, timer_names)
        elif not self.offload_optimizer:
            self._prepare_fp32_grad_for_sub_group(sub_group_id)
        see_memory_usage(f'After prepare optimizer sub group {sub_group_id}', force=False)

    def _optimizer_states_and_gradient_swap_in(self, sub_group_id, timer_names):
        param_length = self.fp16_partitioned_groups_flat_numel[sub_group_id]
        fp32_param_id = self.get_param_id(self.fp32_partitioned_groups_flat[sub_group_id])
        assert self._swappable_optimizer_subgroup(sub_group_id), \
            f'Parameter {fp32_param_id} of numel={param_length} is not swappable'

        see_memory_usage(f'pre-step Before swapping in optimizer tensors {sub_group_id}', force=False)
        timer_names.add(OPTIMIZER_SWAP_IN_STATE_TIMER)
        self.timers(OPTIMIZER_SWAP_IN_STATE_TIMER).start()

        self.optimizer_swapper.swap_in_optimizer_state(
            parameter=self.fp32_partitioned_groups_flat[sub_group_id],
            async_parameter=self.next_swappable_fp32_partitioned_groups[sub_group_id])

        self.timers(OPTIMIZER_SWAP_IN_STATE_TIMER).stop()
        see_memory_usage(f'pre-step After swapping in optimizer tensors {sub_group_id}', force=False)

    @instrument_w_nvtx
    def _release_sub_group(self, sub_group_id, timer_names):
        see_memory_usage(f'Before release optimizer sub group {sub_group_id}', force=False)
        # get rid of the fp32 gradients. Not needed anymore
        if not self.offload_optimizer:
            self.fp32_partitioned_groups_flat[sub_group_id].grad = None

        if self._swappable_optimizer_subgroup(sub_group_id):
            self._optimizer_states_and_gradient_swap_out(sub_group_id, timer_names)
        see_memory_usage(f'After release optimizer sub group {sub_group_id}', force=False)

    # create a flat tensor aligned at the alignment boundary
    @instrument_w_nvtx
    def flatten_dense_tensors_aligned(self, tensor_list, alignment):
        num_elements = 0
        for tens in tensor_list:
            num_elements = num_elements + tens.numel()

        remaining = num_elements % alignment

        if remaining:
            elements_to_add = alignment - remaining
            pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
            padded_tensor_list = tensor_list + [pad_tensor]

            num_elements = num_elements + elements_to_add
        else:
            padded_tensor_list = tensor_list

        return self.flatten(padded_tensor_list)

    def _optimizer_states_and_gradient_swap_out(self, sub_group_id, timer_names):
        param_length = self.fp16_partitioned_groups_flat_numel[sub_group_id]
        fp32_param_id = self.get_param_id(self.fp32_partitioned_groups_flat[sub_group_id])
        assert self._swappable_optimizer_subgroup(sub_group_id), \
            f'Parameter {fp32_param_id} of numel={param_length} is not swappable'

        see_memory_usage(f'post-step Before swapping out optimizer tensors {sub_group_id}', force=False)
        timer_names.add(OPTIMIZER_SWAP_OUT_STATE_TIMER)
        self.timers(OPTIMIZER_SWAP_OUT_STATE_TIMER).start()

        self.optimizer_swapper.swap_out_optimizer_state(
            parameter=self.fp32_partitioned_groups_flat[sub_group_id],
            async_swap=self.next_swappable_fp32_partitioned_groups[sub_group_id] is not None)

        self.timers(OPTIMIZER_SWAP_OUT_STATE_TIMER).stop()
        see_memory_usage(f'post-step After swapping out optimizer tensors {sub_group_id}', force=False)

        # get rid of the fp32 gradients. Not needed anymore
        self.fp32_partitioned_groups_flat[sub_group_id].grad = None

    def _unflatten_partitioned_parameters(self, sub_group_id):
        updated_params = self.unflatten(self.fp16_partitioned_groups_flat[sub_group_id],
                                        self.fp16_partitioned_groups[sub_group_id])

        for partitioned_param, q in zip(self.fp16_partitioned_groups[sub_group_id], updated_params):
            partitioned_param.data = q.data

    def _overflow_clean_up(self, prev_scale):
        see_memory_usage('After overflow before clearing gradients', force=False)
        self.zero_grad(set_to_none=True)

        if self.offload_optimizer:
            self.reset_cpu_buffers()
        else:
            self.averaged_gradients = {}

        see_memory_usage('After overflow after clearing gradients', force=False)

    @instrument_w_nvtx
    def _overflow_check_and_loss_scale_update(self):

        # First compute norm for all group so we know if there is overflow
        if self.dtype == torch.float16:
            self.check_overflow()

        #loss scaling related computation
        prev_scale = self.loss_scale
        self._update_scale(self.overflow)

        if self.overflow:
            self._overflow_clean_up(prev_scale)

        return self.overflow

    @instrument_w_nvtx
    def _post_step(self, timer_names):
        if self.offload_optimizer:
            self.reset_cpu_buffers()

        #Gathering persisting parameters
        if len(self.persistent_parameters) > 0:
            self.persistent_parameters[0].all_gather(self.persistent_parameters)

        if self.swap_optimizer:
            self.optimizer_swapper.log_timers()

        self.invalidate_secondary_tensor()

        self.timers.log(timer_names)

        see_memory_usage('After zero_optimizer step', force=False)
        print_rank_0(f"------------------Finishing Step-----------------------")

    @instrument_w_nvtx
    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
            self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                self.fp32_partitioned_groups_flat[sub_group_id].data)

            #unflatten fp16 parameter subgroup
            self._unflatten_partitioned_parameters(sub_group_id)
        else:
            self._partitioned_params_swap_out(sub_group_id)

    def override_loss_scale(self, loss_scale):
        if loss_scale != self.external_loss_scale:
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    @instrument_w_nvtx
    def step(self, closure=None):
        """
            Not supporting closure.
        """
        self._pre_step()
        self._partition_all_parameters()

        #checks for overflow, adjust the loss scale accordingly
        if self._overflow_check_and_loss_scale_update():
            if self.swap_optimizer:
                self.optimizer_swapper.log_timers()
            return

        norm_groups = self._get_norm_groups()
        scaled_global_grad_norm = get_global_norm(norm_list=norm_groups)

        # Stash unscaled gradient norm
        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale

        timer_names = set()

        timer_names.add(OPTIMIZER_STEP_TIMER)
        self.timers(OPTIMIZER_STEP_TIMER).start()

        #update parameters one sub group at a time
        for sub_group_id, group in enumerate(self.fp16_groups):

            #prepare optimizer states, gradients and fp32 parameters for update
            self._prepare_sub_group(sub_group_id, timer_names)

            #scale the fp32 gradients
            self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)

            #apply the optimizer step on the sub group and copy fp32 parameters to fp16
            self._optimizer_step(sub_group_id)

            #put fp16 parameters in appropriate location
            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)

            #release memory or swap out optimizer states of fp32 parameters
            self._release_sub_group(sub_group_id, timer_names)

        self.timers(OPTIMIZER_STEP_TIMER).stop()

        self._post_step(timer_names)

        # warn user about caching allocator flushes
        memory_stats = get_accelerator().memory_stats()
        alloc_retries = memory_stats.get("num_alloc_retries")
        if alloc_retries is None:
            alloc_retries = 0
        if alloc_retries > self.n_caching_allocator_flushes:
            if dist.get_rank() == 0:
                logger.warning(
                    "%d pytorch allocator cache flushes since last step. this happens "
                    "when there is high memory pressure and is detrimental to "
                    "performance. if this is happening frequently consider adjusting "
                    "settings to reduce memory consumption. If you are unable to "
                    "make the cache flushes go away consider adding "
                    "get_accelerator().empty_cache() calls in your training loop to ensure "
                    "that all ranks flush their caches at the same time",
                    alloc_retries - self.n_caching_allocator_flushes)
            self.n_caching_allocator_flushes = alloc_retries

    def dump_pre_step_gradients(self, debug_fp32_grads):
        # Dump gradient norms for debugging
        for i, _ in enumerate(self.fp16_groups):
            print(f'Pre-Step Dump Norms for Group {i} FP16P, FP16G, FP32G, FP32GUC')
            for fp16_param, fp32_grad in zip(self.fp16_groups[i], debug_fp32_grads[i]):
                param_id = self.get_param_id(fp16_param)
                fp16_grad_norm = self.debug_fp16_grads[i][param_id]

                fp32_grad_norm = [float(t.data.float().norm(2)) for t in fp32_grad]
                norm_list = [fp16_grad_norm, fp32_grad_norm]
                print(f'Pre-Step Norms {i} {param_id} = {norm_list}')

    def dump_post_step_gradients(self):
        # Dump gradient norms for debugging
        for i, group in enumerate(self.fp16_groups):
            print(f'Post-Step Dump Norms for Group {i} FP16P, FP16DS, FP16FLAT, FP32FLAT')
            unflat_fp16 = self.unflatten(self.fp16_groups_flat[i], self.fp16_groups[i])
            unflat_fp32 = self.unflatten(self.fp32_partitioned_groups_flat[i], self.fp16_groups[i])
            for j, p in enumerate(self.fp16_groups[i]):
                param_id = self.get_param_id(p)
                param_norm = float(p.data.float().norm(2))
                ds_norm = float(p.ds_tensor.data.float().norm(2))

                unflat_norm = [float(t.data.float().norm(2)) for t in [unflat_fp16[j], unflat_fp32[j]]]
                norm_list = [param_norm, ds_norm] + unflat_norm
                print(f'Post-Step Norms {i} {param_id} = {norm_list}')

    @instrument_w_nvtx
    def unscale_and_clip_grads(self, sub_group_id, total_norm):
        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale

        self.fp32_partitioned_groups_flat[sub_group_id].grad.mul_(1. / combined_scale)

    def _check_overflow(self, partition_gradients=True):
        self.overflow = self.has_overflow(partition_gradients)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params, is_grad_list=False):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False

    def has_overflow_partitioned_grads_serial(self):
        for i in range(len(self.fp16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    @instrument_w_nvtx
    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            with get_accelerator().stream(self.reduce_and_partition_stream):
                if hasattr(self.inf_or_nan_tracker, "logical_or_"):
                    self.inf_or_nan_tracker.logical_or_(torch.isinf(self.grad_partitions_flat_buffer).any())
                    self.inf_or_nan_tracker.logical_or_(torch.isnan(self.grad_partitions_flat_buffer).any())
                else:
                    # logical_or_ not available in older versions of pytorch
                    self.inf_or_nan_tracker += torch.isinf(self.grad_partitions_flat_buffer).any()
                    self.inf_or_nan_tracker += torch.isnan(self.grad_partitions_flat_buffer).any()
                    self.inf_or_nan_tracker = self.inf_or_nan_tracker > 0

                overflow_gpu = self.inf_or_nan_tracker.clone().to(torch.uint8)
                self.inf_or_nan_tracker.zero_()

            if not get_accelerator().resolves_data_dependency():
                get_accelerator().default_stream().wait_stream(self.reduce_and_partition_stream)
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        else:
            params = []
            for group in self.fp16_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = get_accelerator().ByteTensor([overflow])

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, j=None):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    @instrument_w_nvtx
    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        if self.swap_optimizer:
            self.optimizer_swapper.pre_backward()

        see_memory_usage(f"Before backward", force=False)

        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

        self._get_param_coordinator(training=True).reset_step()

        if self.swap_optimizer:
            self.optimizer_swapper.post_backward()

    def get_fp32_grad_partitions(self) -> Dict[int, Dict[int, Tensor]]:
        """get fp32 gradient partition dictionary
        accessed as grad_dict[parameter_group_index][parameter_index]
        """
        if not get_accelerator().resolves_data_dependency():
            self.reduce_and_partition_stream.synchronize()
        grad_dict = collections.defaultdict(dict)
        if self.offload_optimizer:
            for group in self.fp16_groups:
                for param_idx, param in enumerate(group):
                    group_idx, dest_offset, num_elements = self.grad_position[self.get_param_id(param)]
                    fp32_grad = self.fp32_partitioned_groups_flat[group_idx].grad.narrow(0, dest_offset, num_elements)
                    grad_dict[group_idx][param_idx] = fp32_grad
        else:
            for group_idx, group in self.averaged_gradients.items():
                for param_idx, gradient in enumerate(group):
                    grad_dict[group_idx][param_idx] = gradient.float()

        return grad_dict

    def _fp32_state_allgather(self, param, fp32_state_partition):
        reduce_buffer = torch.zeros(self.partition_count * fp32_state_partition.numel(),
                                    dtype=torch.float32,
                                    device=param.device)
        my_rank = dist.get_rank(group=self.dp_process_group)
        partition = reduce_buffer.narrow(0, fp32_state_partition.numel() * my_rank, fp32_state_partition.numel())
        partition.data.copy_(fp32_state_partition.data, non_blocking=False)
        dist.all_gather_into_tensor(reduce_buffer, partition, group=self.dp_process_group)
        return reduce_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape)

    def get_fp32_grad_for_param(self, param) -> Tensor:
        if not param.requires_grad:
            return None

        if not get_accelerator().resolves_data_dependency():
            self.reduce_and_partition_stream.synchronize()

        if self.offload_optimizer:
            group_idx, dest_offset, num_elements = self.grad_position[self.get_param_id(param)]
            fp32_grad = self.fp32_partitioned_groups_flat[group_idx].grad.narrow(0, dest_offset, num_elements)
        else:
            fp32_grad = self.__param_id_to_grad_partition[param.ds_id].float()

        return self._fp32_state_allgather(param, fp32_grad)

    def _get_fp32_opt_state_partition(self, param, optim_state_key=None):
        if not get_accelerator().resolves_data_dependency():
            self.reduce_and_partition_stream.synchronize()

        group_idx, dest_offset, num_elements = self.grad_position[self.get_param_id(param)]

        if self._swappable_optimizer_subgroup(group_idx):
            self._optimizer_states_and_gradient_swap_in(group_idx)

        fp32_param = self.fp32_partitioned_groups_flat[group_idx]
        if optim_state_key is None:
            fp32_opt_state = fp32_param.narrow(0, dest_offset, num_elements)
        else:
            fp32_opt_state = self.optimizer.state[fp32_param][optim_state_key].narrow(0, dest_offset, num_elements)

        return fp32_opt_state, group_idx

    def get_full_hp_param(self, param, optim_state_key=None) -> Tensor:
        if not param.requires_grad:
            return None

        fp32_opt_state, group_idx = self._get_fp32_opt_state_partition(param, optim_state_key)
        hp_param = self._fp32_state_allgather(param, fp32_opt_state)

        if self._swappable_optimizer_subgroup(group_idx):
            self._optimizer_states_and_gradient_swap_out(group_idx)

        return hp_param

    def set_full_hp_param(self, value, param, optim_state_key=None):
        if not param.requires_grad:
            return

        assert value.numel(
        ) == param.ds_numel, f" Number of elements do not match: {value.numel()} != {param.ds_numel}"

        fp32_opt_state_partition, group_idx = self._get_fp32_opt_state_partition(param, optim_state_key)
        my_rank = dist.get_rank(group=self.dp_process_group)
        value_partition = value.flatten().narrow(0,
                                                 fp32_opt_state_partition.numel() * my_rank,
                                                 fp32_opt_state_partition.numel())
        fp32_opt_state_partition.data.copy_(value_partition.data)

        if self._swappable_optimizer_subgroup(group_idx):
            self._optimizer_states_and_gradient_swap_out(group_idx)

    ### Local API START ###

    def get_local_fp32_param(self, param, optim_state_key=None) -> Tensor:
        if not param.requires_grad:
            return None
        fp32_opt_state, group_idx = self._get_fp32_opt_state_partition(param, optim_state_key)
        return fp32_opt_state

    def get_local_fp32_grad_for_param(self, param) -> Tensor:
        if not param.requires_grad:
            return None

        if not get_accelerator().resolves_data_dependency():
            self.reduce_and_partition_stream.synchronize()

        if self.offload_optimizer:
            group_idx, dest_offset, num_elements = self.grad_position[self.get_param_id(param)]
            fp32_grad = self.fp32_partitioned_groups_flat[group_idx].grad.narrow(0, dest_offset, num_elements)
        else:
            fp32_grad = self.__param_id_to_grad_partition[param.ds_id].float()
        return fp32_grad

    def set_local_hp_param(self, value, param, optim_state_key=None):
        if not param.requires_grad:
            return

        assert hasattr(param, "ds_tensor"), f" The parameter does not contain the partitioned copy of the tensor."
        assert value.numel() == param.ds_tensor.numel(
        ), f" Number of elements do not match: {value.numel()} != {param.ds_tensor.ds_numel}"

        fp32_opt_state_partition, group_idx = self._get_fp32_opt_state_partition(param, optim_state_key)
        value_partition = value.flatten()
        fp32_opt_state_partition.data.copy_(value_partition.data)

        if self._swappable_optimizer_subgroup(group_idx):
            self._optimizer_states_and_gradient_swap_out(group_idx)
        logger.info(f"[set_local_hp_param][update the params' value successfully]")

    ### Local API END ###

    @instrument_w_nvtx
    def _partition_all_parameters(self):
        self.parameter_offload.partition_all_parameters()

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
        self.trainable_param_groups = self._get_trainable_parameter_groups()

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

    def _get_lean_tensors(self, padded_flattened_tensor, group_tensors, paddings):
        # Remove paddings from flattened tensor
        individual_tensors = self.unflatten(padded_flattened_tensor, group_tensors)
        lean_lengths = [t.numel() - pad for t, pad in zip(group_tensors, paddings)]
        lean_tensors = [t[:len] for t, len in zip(individual_tensors, lean_lengths)]
        #logger.info(f'rank {dist.get_rank()}: lean_tensors = {[t.numel() for t in lean_tensors]}')
        return lean_tensors

    #TODO REVISIT this for stage 3
    def get_lean_optimizer_state(self):
        # Return optimizer states after removing paddings.
        # This method assumes that each param group contains a single flattened tensor.
        optimizer_groups_state = []

        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            lean_state = {}
            for key, value in self.optimizer.state[p].items():
                if torch.is_tensor(value):
                    padded_lens = [t.numel() for t in self.fp16_partitioned_groups[i]]
                    lean_state[key] = self._get_lean_tensors(value, self.fp16_partitioned_groups[i],
                                                             self.groups_padding[i])
                    lean_flat_len = sum([t.numel() for t in lean_state[key]])
                else:
                    lean_state[key] = value

            optimizer_groups_state.append(lean_state)

        return optimizer_groups_state

    def get_groups_without_padding(self, groups_with_padding):
        # Return group tensor after removing paddings added for alignment to DP world size.
        groups_without_padding = []
        for i, group in enumerate(groups_with_padding):
            lean_group = self._get_lean_tensors(group, self.fp16_partitioned_groups[i], self.groups_padding[i])
            groups_without_padding.append(lean_group)

        return groups_without_padding

    def _set_fp32_optimizer_param_groups(self):
        for sub_group_id, _ in enumerate(self.fp16_groups):
            param_group_id = self.sub_group_to_group_id[sub_group_id]
            self.optimizer.param_groups[param_group_id]['params'].append(
                self.fp32_partitioned_groups_flat[sub_group_id])

    def _clear_fp32_optimizer_param_groups(self):
        for param_group in self.optimizer.param_groups:
            param_group['params'] = []

    def _rigid_state_dict(self):
        state_dict = {}
        state_dict[ZERO_STAGE] = ZeroStageEnum.weights
        state_dict[LOSS_SCALER] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict[PARTITION_COUNT] = self.partition_count

        self._set_fp32_optimizer_param_groups()
        state_dict[OPTIMIZER_STATE_DICT] = self.optimizer.state_dict()
        state_dict[FP32_FLAT_GROUPS] = self.fp32_partitioned_groups_flat
        self._clear_fp32_optimizer_param_groups()

        return state_dict

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
        if self.elastic_checkpoint:
            raise NotImplementedError("ZeRO-3 does not yet support elastic checkpointing, please disable for now.")

        return self._rigid_state_dict()


# Restore base optimizer fp32 weights from checkpoint by:
# 1) Merging fp32 weights from checkpoints of all partitions
# 2) Extracting fp32 weights for current partition from merged weights
# 3) Using extracted weights to update base optimizer weights directly.

    def _restore_from_fp32_weights(self, all_state_dict):

        flat_local_partition = []
        for i in range(len(self.fp32_partitioned_groups_flat)):
            merged_partitions = [sd['fp32_groups'][i] for sd in all_state_dict]
            flat_local_partition.append(self._get_flattened_partition(merged_partitions))

        for current, saved in zip(self.fp32_partitioned_groups_flat, flat_local_partition):
            current.data.copy_(saved.data)

    # Restore base optimizer fp32 weights from ZeRO fp16 weights
    def _restore_from_bit16_weights(self):
        for fp16_partitions, fp32_partition in zip(self.fp16_partitioned_groups_flat,
                                                   self.fp32_partitioned_groups_flat):
            fp32_partition.data.copy_(fp16_partitions.data)

    # Refresh the fp32 master params from the fp16 copies.
    def refresh_fp32_params(self):
        self._restore_from_bit16_weights()

    # Extract flattened partition for current rank from all partitions
    def _get_flattened_partition(self, all_partition_states):
        partition_id = dist.get_rank(group=self.dp_process_group)
        alignment = dist.get_world_size(group=self.dp_process_group)

        param_partitions = [[] for _ in range(len(all_partition_states[0]))]
        for i, partition in enumerate(all_partition_states):
            for j, param in enumerate(partition):
                param_partitions[j].append(param)

        local_state_partitions = []
        for param_index, param_slices in enumerate(param_partitions):
            flattened_merged_tensor = self.flatten_dense_tensors_aligned(param_slices, alignment)
            new_partitions = self.get_data_parallel_partitions(flattened_merged_tensor)
            local_state_partitions.append(new_partitions[partition_id])

        if torch.is_tensor(local_state_partitions[0]):
            return self.flatten_dense_tensors_aligned(local_state_partitions, alignment)

        # Assume non-tensor states are not partitioned and equal across ranks, so return first one
        return local_state_partitions[0]

    # Restore base optimizer state from checkpoint by
    # 1) Merging optimizer state from checkpoints of all partitions
    # 2) Extracting optimizer state for current partition from the merged state
    # 3) Using the extracted value to directly update the base optimizer.
    def _restore_base_optimizer_state(self, all_state_dict):
        base_optimizer_group_states = []
        for i in range(len(self.optimizer.param_groups)):
            partition_states = {}
            all_partition_group_states = [sd['base_optimizer_state'][i] for sd in all_state_dict]
            for key in all_partition_group_states[0].keys():
                all_partition_states = [all_states[key] for all_states in all_partition_group_states]
                partition_states[key] = self._get_flattened_partition(all_partition_states)
            base_optimizer_group_states.append(partition_states)

        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            for key, saved in base_optimizer_group_states[i].items():
                if torch.is_tensor(self.optimizer.state[p][key]):
                    self.optimizer.state[p][key].data.copy_(saved.data)
                else:
                    self.optimizer.state[p][key] = saved

    def _rigid_load_state_dict(self, state_dict, load_optimizer_states=True):
        # I think it should actually be ok to reload the optimizer before the model.
        self.loss_scaler = state_dict[LOSS_SCALER]
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']

        if load_optimizer_states:
            self._set_fp32_optimizer_param_groups()
            self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_DICT])
            self._clear_fp32_optimizer_param_groups()

        if self.swap_optimizer or self.params_in_nvme_and_cpu:
            # Purge the swapped optimizer state, it was initialized to the freshly created model and not the checkpoint
            for swap_info in self.optimizer_swapper.swap_params_info.values():
                swap_info.tensors = [swap_info.tensors[0]]
                swap_info.has_state_tensors = False

        if self.swap_optimizer:
            # Touch all parameters to synchronize all buffers
            timer_names = set()
            self._partition_all_parameters()
            for sub_group_id, group in enumerate(self.fp16_groups):
                self._prepare_sub_group(sub_group_id, timer_names)
                self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
                self._release_sub_group(sub_group_id, timer_names)
            self._post_step(timer_names)

        # restore fp32 partitions
        for curr_param, saved_param in zip(self.fp32_partitioned_groups_flat, state_dict[FP32_FLAT_GROUPS]):
            curr_param.data.copy_(saved_param.data)

        # restore fp16 partitions from fp32
        for sub_group_id in range(len(self.fp32_partitioned_groups_flat)):
            fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
            if sum(fp32_param.size()) > 0:
                fp16_param = self.fp16_partitioned_groups_flat[sub_group_id]
                fp16_param.data.copy_(fp32_param.data)

        # update fp16 unflattened params
        for sub_group_id in range(len(self.fp16_partitioned_groups_flat)):
            updated_params = self.unflatten(self.fp16_partitioned_groups_flat[sub_group_id],
                                            self.fp16_partitioned_groups[sub_group_id])

            for partitioned_param, q in zip(self.fp16_partitioned_groups[sub_group_id], updated_params):
                partitioned_param.data = q.data

    # TODO: Support different/changing load/save DP degree.
    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None,
                        load_serial=None):
        r"""Loading a ZeRO checkpoint
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

        if self.elastic_checkpoint:
            raise NotImplementedError("ZeRO-3 does not yet support elastic checkpointing, please disable for now.")

        self._rigid_load_state_dict(state_dict_list[dist.get_rank(group=self.dp_process_group)],
                                    load_optimizer_states=load_optimizer_states)

        # when use loading checkpoint serial, after finish loading, we need to
        # delete the temp state_dict_list variable to save memory, then trigger
        # the next rank's loading
        if load_serial is not None:
            load_serial += 1
            rank = dist.get_rank(group=self.dp_process_group)
            local_rank = dist.get_local_rank()
            del state_dict_list[rank]
            rank_end = dist.get_world_size() - 1
            if local_rank != rank_end:
                dist.send(tensor=load_serial, dst=rank + 1)

        if len(self.persistent_parameters) > 0:
            self.persistent_parameters[0].partition(self.persistent_parameters)
            # self.persistent_parameters[0].all_gather(self.persistent_parameters) # this will be done in checkpoint_event_epilogue() so remove it to prevent double all_gather

    def reset_swap_buffers(self):
        timer_names = set()
        for sub_group_id, group in enumerate(self.fp16_groups):
            self._prepare_sub_group(sub_group_id, timer_names)
            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
            self._release_sub_group(sub_group_id, timer_names)

    def checkpoint_event_prologue(self):
        self._partition_all_parameters()

    def checkpoint_event_epilogue(self):
        if len(self.persistent_parameters) > 0:
            self.persistent_parameters[0].all_gather(self.persistent_parameters)

    def empty_partition_cache(self):
        self.parameter_offload.empty_partition_cache()


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


def estimate_zero3_model_states_mem_needs(total_params,
                                          largest_layer_params,
                                          num_gpus_per_node=1,
                                          num_nodes=1,
                                          cpu_offload=True,
                                          cpu_offload_params=True,
                                          zero_init=True,
                                          additional_buffer_factor=1.5):

    total_gpus = num_nodes * num_gpus_per_node
    gpus_factor = 1 / num_nodes
    largest_layer_memory = (4 * largest_layer_params)

    if cpu_offload:
        if cpu_offload_params:
            gpu_mem = largest_layer_memory

            if zero_init:
                cpu_mem = total_params * 18 * gpus_factor * additional_buffer_factor
            else:
                cpu_mem = total_params * max(4 * num_gpus_per_node, 18 * gpus_factor) * additional_buffer_factor

        else:
            gpu_mem = largest_layer_memory + int(2 * total_params / total_gpus)

            if zero_init:
                cpu_mem = total_params * 16 * gpus_factor * additional_buffer_factor
            else:
                cpu_mem = total_params * max(4 * num_gpus_per_node, 16 * gpus_factor) * additional_buffer_factor
    else:
        gpu_mem = largest_layer_memory + int(18 * total_params / total_gpus)
        if zero_init:
            cpu_mem = largest_layer_params * 4 * num_gpus_per_node * additional_buffer_factor
        else:
            cpu_mem = total_params * 4 * num_gpus_per_node * additional_buffer_factor

    return int(cpu_mem), int(gpu_mem), largest_layer_memory


def model_to_params(model):
    # shared params calculated only once
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

    largest_layer_params = 0
    for m in model.modules():
        # assuming no shared params within a single layer
        layer_params = sum(p.numel() for p in m.parameters(recurse=False))
        largest_layer_params = max(largest_layer_params, layer_params)

    return total_params, largest_layer_params


def estimate_zero3_model_states_mem_needs_all_live(model,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """
    Print out estimates on memory usage requirements for ZeRO 3 params, optim states and gradients
    for a given ``model`` and hardware setup.

    If you have an actual model object, use this function and everything will be derived
    automatically.

    If it's a hypothetical model, use ``estimate_zero3_model_states_mem_needs_all_cold`` where you have to pass
    the ``total_params`` and ``largest_layer_params`` explicitly.

    Args:
        - ``model``: ``nn.Module`` object
        - ``num_gpus_per_node``: how many gpus per node (defaults to 1)
        - ``num_nodes``: how many nodes (defaults to 1),
        - ``additional_buffer_factor``: estimation factor (defaults to 1.5):

    """

    total_params, largest_layer_params = model_to_params(model)

    estimate_zero3_model_states_mem_needs_all_cold(total_params=total_params,
                                                   largest_layer_params=largest_layer_params,
                                                   num_gpus_per_node=num_gpus_per_node,
                                                   num_nodes=num_nodes,
                                                   additional_buffer_factor=additional_buffer_factor)


def estimate_zero3_model_states_mem_needs_all_cold(total_params,
                                                   largest_layer_params,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """
    Print out estimates on memory usage requirements for ZeRO 3 params, optim states and gradients
    for a given ``model`` and hardware setup.

    If it's a hypothetical model, use this function where you have to pass
    the ``total_params`` and ``largest_layer_params`` explicitly.

    If you have an actual model object, use ``estimate_zero3_model_states_mem_needs_all_live`` and everything
    will be derived automatically.

    Args:
        - ``total_params``: total  model params
        - ``largest_layer_params``: largest layer's params
        - ``num_gpus_per_node``: how many gpus per node (defaults to 1)
        - ``num_nodes``: how many nodes (defaults to 1),
        - ``additional_buffer_factor``: estimation factor (defaults to 1.5):

    """

    def format_options(cpu_offload, cpu_offload_params, zero_init):
        enabled = []
        padded_cpu_str = f'{OffloadDeviceEnum.cpu:4}'
        param_device = padded_cpu_str if cpu_offload_params else "none"
        enabled.append(f"offload_param={param_device}")
        optimizer_device = padded_cpu_str if cpu_offload else "none"
        enabled.append(f"offload_optimizer={optimizer_device}")
        enabled.append(f"zero_init={1 if zero_init else 0}")
        return ", ".join(enabled)

    nodes_str = "nodes" if num_nodes > 1 else "node"
    gpus_str = "GPUs" if num_gpus_per_node > 1 else "GPU"
    print(
        "Estimated memory needed for params, optim states and gradients for a:\n"
        f"HW: Setup with {num_nodes} {nodes_str}, {num_gpus_per_node} {gpus_str} per node.\n"
        f"SW: Model with {int(total_params/1e6)}M total params, {int(largest_layer_params/1e6)}M largest layer params."
    )
    print("  per CPU  |  per GPU |   Options")
    for cpu_offload in [True, False]:
        for cpu_offload_params in [True, False]:
            if not cpu_offload and cpu_offload_params:
                continue
            for zero_init in [True, False]:
                cpu_mem, gpu_mem, largest_layer_memory = estimate_zero3_model_states_mem_needs(
                    total_params=total_params,
                    largest_layer_params=largest_layer_params,
                    num_gpus_per_node=num_gpus_per_node,
                    num_nodes=num_nodes,
                    cpu_offload=cpu_offload,
                    cpu_offload_params=cpu_offload_params,
                    zero_init=zero_init,
                    additional_buffer_factor=additional_buffer_factor)

                options_str = format_options(cpu_offload=cpu_offload,
                                             cpu_offload_params=cpu_offload_params,
                                             zero_init=zero_init)
                print(f" {cpu_mem/2**30:7.2f}GB | {gpu_mem/2**30:6.2f}GB | {options_str}")
