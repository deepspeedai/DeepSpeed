from deepspeed.utils.logging import logger
'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import os

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.distributed_c10d import _get_global_rank
import torch.distributed as dist
import math
from torch._six import inf
from torch.autograd import Variable

from deepspeed.runtime.fp16.loss_scaler import LossScaler, DynamicLossScaler
from deepspeed.runtime.utils import see_memory_usage, is_model_parallel_parameter
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus, ZeroParamType, _init_external_params, Init, is_zero_param
from deepspeed.runtime.zero.constants import ZERO_OPTIMIZATION_WEIGHTS
from deepspeed.ops.adam import DeepSpeedCPUAdam

import itertools
# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False

try:
    from apex_C import flatten
    from apex_C import unflatten
except ImportError:
    try:
        _ = warned_flatten
    except NameError:
        logger.warning(
            "apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten."
        )
        warned_flatten = True
    from torch._utils import _flatten_dense_tensors as flatten
    from torch._utils import _unflatten_dense_tensors as unflatten


def print_rank_0(message, debug=False, force=False):
    if torch.distributed.get_rank() == 0 and (debug or force):
        logger.info(message)


def input(msg):
    return


def split_half_float_double(tensors):
    dtypes = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor"
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


# create a flat tensor aligned at the alignment boundary
def flatten_dense_tensors_aligned(tensor_list, alignment):
    num_elements = 0
    for tens in tensor_list:
        num_elements = num_elements + tens.numel()

    remaining = num_elements % alignment

    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add,
                                 device=tensor_list[0].device,
                                 dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]

        num_elements = num_elements + elements_to_add
    else:
        padded_tensor_list = tensor_list

    return _flatten_dense_tensors(padded_tensor_list)


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()


def get_all_parameters(sub_module):
    return itertools.chain(sub_module.named_parameters(recurse=False),
                           sub_module.ds_external_parameters())


#apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module,
                                                    functional,
                                                    backward_function,
                                                    output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        return functional.apply(module, backward_function, outputs)
    else:
        return outputs


#for each tensor in outputs run the forward_funciton and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(module,
                                                forward_function,
                                                backward_function,
                                                outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(
                module,
                forward_function,
                backward_function,
                output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            outputs.register_hook(backward_function)
        return outputs
    else:
        return outputs


# TODO Needs to be implemented
class PrefetchCoordinator(object):
    def __init__(self):
        # step_id keeps track of the number of sub-modules invoked so far
        # the step_id is tracking forward and backward sequence of sub-modules
        self.step_id = 0

        # stores the sequence of sub modules in forward+backward pass
        self.sub_module_trace = []

        # maps sub_module id to submodule objects
        self.id_to_sub_module_map = {}

        # stores the total number of parmeters in each sub_module
        self.id_to_sub_module_size_map = {}

        self.trace_completed = False

        self.most_recent_sub_module_step = {}

        # reuse distances
        self.reuse_numel_for_step_id = {}

    def record_trace(self, sub_module):
        if not self.trace_completed:
            self.sub_module_trace.append(sub_module.id)
            self.id_to_sub_module_map[sub_module.id] = sub_module

    def print_trace(self):
        print_rank_0(
            f"The module trace is : {[self.id_to_sub_module_map[module_id].id for module_id in self.sub_module_trace]}"
        )

    def increment_step(self, sub_module):
        self.most_recent_sub_module_step[sub_module.id] = self.step_id
        self.step_id += 1

    def reset_step(self):
        self.step_id = 0

    # returns the next numel parameters that will be used next but are not available or inflight
    def get_params_to_prefetch(self, sub_module, numel=2000000):

        # numel_in_sub_module = 0
        # for name, param in sub_module.named_parameters(recurse=False):
        #     numel_in_sub_module += param.ds_numel

        # #if numel_in_sub_module < (numel // 2):
        #    return []

        # tracing failed. The sub_module passed at the step_id must match with the sub_module during tracing
        if sub_module.id != self.sub_module_trace[self.step_id]:
            print_rank_0(
                f"Tracing failed. Prefetching is disabled at sub-module: {sub_module.id}"
            )
            return []

        params_to_prefetch = []
        total_numel_to_prefetch = 0

        for i in range(self.step_id, len(self.sub_module_trace)):
            module_id = self.sub_module_trace[i]
            for _, param in get_all_parameters(self.id_to_sub_module_map[module_id]):
                if param.ds_status is ZeroParamStatus.NOT_AVAILABLE and (
                        param.ds_id not in [p.ds_id for p in params_to_prefetch]):
                    params_to_prefetch.append(param)
                    total_numel_to_prefetch += param.ds_numel
                    #print_rank_0(f"Total numel to prefetch: {total_numel_to_prefetch}. Param: {param.ds_shape} and numel {param.ds_numel}, numel limit {numel}")
                    if total_numel_to_prefetch >= numel:  # and total_numel_to_prefetch > (numel_in_sub_module // 2):
                        return params_to_prefetch

        return params_to_prefetch

    # checks if this sub_module will be used again and if so then returns the number of elements
    # in the parameters used between this sub_module and the reuse of this sub_module
    def get_reuse_distance_in_numel(self, sub_module, sub_module_step_id=None):
        #assert is_forward is not None, "is_forward must be set to True for Forward Propagation and False for backward Propagation"
        is_there_reuse = False
        reuse_distance_in_numel = 1000000000000

        # set the appropriate trace
        trace = self.sub_module_trace
        total_steps = len(trace)
        if sub_module_step_id is None:
            sub_module_step_id = self.most_recent_sub_module_step[sub_module.id]

        # tracing failed. The sub_module passed at the step_id must match with the sub_module during tracing
        if sub_module.id != trace[sub_module_step_id]:
            print_rank_0(
                f"Tracing failed. Cannot tell if the sub_module: {sub_module.id} is reused"
            )
            return reuse_distance_in_numel

        # return cached value
        if sub_module_step_id in self.reuse_numel_for_step_id:
            return self.reuse_numel_for_step_id[sub_module_step_id]

        start_step = self.step_id
        print_rank_0(f"Step id is {self.step_id} ")
        for step_id in range(start_step, total_steps):
            print_rank_0(f"Trace id {trace[step_id]} and sub_module id {sub_module.id}")
            if sub_module.id == trace[step_id]:
                end_step = step_id

                is_there_reuse = True
                reuse_distance_in_numel = self._distance_in_numel(
                    start_step,
                    end_step,
                    trace)

                break

        self.reuse_numel_for_step_id[sub_module_step_id] = reuse_distance_in_numel

        return reuse_distance_in_numel

    def _distance_in_numel(self, start_step, end_step, trace):
        distance_in_numel = 0
        for step_id in range(start_step, end_step):
            module_id = trace[step_id]
            for _, param in self.id_to_sub_module_map[module_id].named_parameters(recurse=False):
                distance_in_numel += param.ds_numel
            for _, param in self.id_to_sub_module_map[module_id].ds_external_parameters():
                distance_in_numel += param.ds_numel
        return distance_in_numel


class PartitionedParameterCoordinator(object):
    def __init__(self,
                 comm_stream=None,
                 max_reuse_distance_in_numel=500000000,
                 max_available_parameters_in_numel=700000000):

        self.in_flight_handles = []
        self.params_in_flight = []
        self.comm_stream = comm_stream if comm_stream is not None else torch.cuda.current_stream(
        )
        self.prefetch_coordinator = PrefetchCoordinator()
        self.hierarchy = 0

        self.total_available_parameter_numel = 0
        self.max_available_parameters_in_numel = max_available_parameters_in_numel

        # max distance between two use of the module beyond which module is released
        self.max_reuse_distance_in_numel = max_reuse_distance_in_numel

    def _increment_available_parameter_numel(self, increment):
        self.total_available_parameter_numel += increment

    def _decrement_available_parameter_numel(self, decrement):
        self.total_available_parameter_numel -= decrement

    '''-----------------------Tracing and Prefetching ---------------'''

    def record_trace(self, sub_module):
        self.prefetch_coordinator.record_trace(sub_module)

    def finish_tracing(self, print_trace=False):
        self.prefetch_coordinator.trace_completed = True

        if print_trace:
            self.prefetch_coordinator.print_trace()

    # Pre fetches the parameters for sub_modules that comes after
    #  the current sub_module. This call is asynchronous
    def prefetch_next_sub_modules(self, sub_module, numel=5000000):

        params_to_prefetch = []
        if not self.prefetch_coordinator.trace_completed:
            return params_to_prefetch

        # prefetch if there is no current prefetching in flight
        if not self.in_flight_handles and self.total_available_parameter_numel < self.max_available_parameters_in_numel:
            params_to_prefetch = self.prefetch_coordinator.get_params_to_prefetch(
                sub_module,
                numel=numel)

            self._all_gather(params_to_prefetch, async_op=True)
            for param in params_to_prefetch:
                param.ds_status = ZeroParamStatus.INFLIGHT

                # keeping track of number of elements consumed by available parmaeters
                self._increment_available_parameter_numel(param.ds_numel)

        self._print_prefetch_elements_info(sub_module, params_to_prefetch)
        print_rank_0(
            f"{'--' * self.hierarchy}--PreFetching parameters {[param.ds_id for param in params_to_prefetch]} and available {self.total_available_parameter_numel}, max limit {self.max_available_parameters_in_numel}",
            force=False)

    def _print_prefetch_elements_info(self, sub_module, params_to_prefetch):
        sub_module_numel = 0.0
        for name, param in sub_module.named_parameters(recurse=False):
            sub_module_numel += param.ds_numel
        numel_being_prefetched = 0
        for param in params_to_prefetch:
            numel_being_prefetched = param.ds_numel
        print_rank_0(
            f"{'--' * self.hierarchy}--PreFetching  {numel_being_prefetched} numels and number of numel in the next sub module is {sub_module_numel}",
            force=False)

    def increment_step(self, sub_module):
        self.prefetch_coordinator.increment_step(sub_module)

    def reset_step(self):
        self.prefetch_coordinator.reset_step()

    '''----------------------------------------------------------------------'''

    # Fetches the parameters in the sub_module
    # This call is blocking
    def fetch_sub_module(self, sub_module):
        partitioned_params = []
        params_in_flight = False
        #print_rank_0(f"{'--' * self.hierarchy}Fetching params in module {sub_module.__class__.__name__}")
        params_to_fetch = [
            param for _,
            param in sub_module.named_parameters(recurse=False)
        ]
        if hasattr(sub_module, 'ds_external_parameters'):
            print_rank_0(
                f"{'--' * self.hierarchy}--Fetching external parameters {sub_module.ds_external_parameters()}"
            )
            params_to_fetch += [
                param for _,
                param in sub_module.ds_external_parameters()
            ]
        # for _, param in sub_module.named_parameters(recurse=False):
        for param in params_to_fetch:
            param.ds_active_sub_modules += 1
            print_rank_0(
                f"{'--' * self.hierarchy}--Fetching parameters {param.ds_id} with active sub modules {param.ds_active_sub_modules}"
            )

            if param.ds_status == ZeroParamStatus.AVAILABLE:
                print_rank_0(
                    f"{'--' * self.hierarchy}--Parameter {param.ds_id} is already available"
                )

            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                print_rank_0(
                    f"{'--' * self.hierarchy}--Parameter {param.ds_id} is being fetched")
                partitioned_params.append(param)

                # keeping track of number of elements consumed by available parmaeters
                self._increment_available_parameter_numel(param.ds_numel)
                print_rank_0(f"Incrementing with parameter id {param.ds_id}")

            if param.ds_status == ZeroParamStatus.INFLIGHT:
                params_in_flight = True
                print_rank_0(
                    f"{'--' * self.hierarchy}--Parameters {param.ds_id} is already in flight (prefetched)"
                )
        self.hierarchy += 1

        # parameters are partitioned and need to be allgathered
        self._all_gather(partitioned_params, async_op=True)

        # parameters are inflight and communication needs to be completed
        if partitioned_params or params_in_flight:
            self._synchronize_communication()

        for _, param in sub_module.named_parameters(recurse=False):
            param.ds_status = ZeroParamStatus.AVAILABLE
            #print(f"Param id {param.ds_id}, Shape {param.shape}, device {param.device} ")
        #print_rank_0(f"After fetching (id, shape, device): {[(param.ds_id, param.shape, param.device) for param in sub_module.named_parameters(recurse=False)]}")

    def release_sub_module(self, sub_module):
        self.hierarchy -= 1
        print_rank_0(
            f"{'--' * self.hierarchy}Releasing params in module {sub_module.__class__.__name__}"
        )
        params_to_release = [
            param for _,
            param in sub_module.named_parameters(recurse=False)
        ]
        if hasattr(sub_module, 'ds_external_parameters'):
            #print_rank_0(f"Releasing external parameters {sub_module.ds_external_parameters()}")
            params_to_release += [
                param for _,
                param in sub_module.ds_external_parameters()
            ]

        # for _, param in sub_module.named_parameters(recurse=False):
        for param in params_to_release:
            param.ds_active_sub_modules -= 1
            if not param.ds_active_sub_modules and not self._keep_for_later(
                    sub_module) and not param.ds_persist:
                print_rank_0(
                    f"{'--' * self.hierarchy}--Releasing parameters {param.ds_id} with numel {param.numel()} active sub modules {param.ds_active_sub_modules} and keep for later {self._keep_for_later(sub_module)}"
                )

                # Keeping track of number of elements that are consumed by available parameters
                self._decrement_available_parameter_numel(param.ds_numel)
                see_memory_usage(
                    f"Before releasing param {param.ds_id} with numel{param.numel()}",
                    force=False)
                param.partition(hierarchy=self.hierarchy)
                see_memory_usage(
                    f"After releasing param {param.ds_id} has numel{param.numel()} ",
                    force=False)

                param.ds_status = ZeroParamStatus.NOT_AVAILABLE
            else:

                print_rank_0(
                    f"{'--' * self.hierarchy}--Did not release parameters {param.ds_id} with numel {param.numel()} with active sub modules {param.ds_active_sub_modules}, keep for later {self._keep_for_later(sub_module)} and persistence {param.ds_persist}"
                )

    def release_and_reset_parameter(self, param):
        param.ds_active_sub_modules = 0
        if param.ds_status == ZeroParamStatus.AVAILABLE:
            print_rank_0(
                f"Releasing unpartitioned {param.ds_id} active sub-modules {param.ds_active_sub_modules} size {param.ds_numel} and persisitence {param.ds_persist}"
            )
            self._decrement_available_parameter_numel(param.ds_numel)
            param.partition()

    def _keep_for_later(self, sub_module):
        if not self.prefetch_coordinator.trace_completed:
            return False
        reuse_distance_in_numel = self.prefetch_coordinator.get_reuse_distance_in_numel(
            sub_module)
        #print_rank_0(f"Reuse distance and numel for sub_module id {sub_module.id} is {reuse_distance_in_numel}")
        return reuse_distance_in_numel < self.max_reuse_distance_in_numel

    def _all_gather(self, partitioned_params, async_op=False):
        with torch.cuda.stream(self.comm_stream):
            handles = partitioned_params[0].all_gather(
                param_list=partitioned_params,
                async_op=async_op,
                hierarchy=self.hierarchy) if partitioned_params else None

        if handles is not None:
            self.in_flight_handles.extend(handles)
            self.params_in_flight.extend(partitioned_params)

    def _synchronize_communication(self, synchronize_streams=True):
        assert len(self.params_in_flight) == len(self.in_flight_handles)
        for handle, param in zip(self.in_flight_handles, self.params_in_flight):
            if handle is not None:
                with torch.cuda.stream(self.comm_stream):
                    handle.wait()
            param.ds_status = ZeroParamStatus.AVAILABLE
        self.comm_stream.synchronize()
        torch.cuda.synchronize() if synchronize_streams else None
        self.in_flight_handles = []
        self.params_in_flight = []


class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        #print(f"After Forward: {ctx.module.__class__.__name__}")
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        #print(f"Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        if output.requires_grad:
            #TODO SOME TIMES post backward does not seem to be triggered debug in detail
            #Should only cause increase in memory not correctness issue
            #if output.grad_fn.__class__.__name__ == 'ViewBackward':
            #    ctx.view=True
            #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
            #assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
            #if module.ds_grads_remaining == 0:
            #    print(f"Before Forward: {ctx.module.__class__.__name__}")
            module.ds_grads_remaining += 1
            ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        return output

    @staticmethod
    def backward(ctx, *args):
        ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        if ctx.module.ds_grads_remaining == 0:
            ctx.pre_backward_function(ctx.module)
            #print(f"After Backward: {ctx.module.__class__.__name__}")
        return (None, None) + args


INITIAL_MICRO_STEP_ID = -1


class FP16_DeepSpeedZeroOptimizer_Stage3(object):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """
    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
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
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 cpu_offload_optimizer_state=False,
                 cpu_offload_params=False,
                 cpu_offload_use_pin_memory=False,
                 sub_group_size=1000000000000,
                 mpu=None,
                 clip_grad=0.0,
                 allreduce_always_fp32=False,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 elastic_checkpoint=False):

        see_memory_usage("Stage 3 intialize beginning", force=True)

        if dist.get_rank() == 0:
            logger.info(f"Reduce bucket size {reduce_bucket_size}")
            logger.info(f"Allgather bucket size {prefetch_bucket_size}")
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master gard and unflat master weight never exist. TODO: a way to save out unflat master?
        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        if not all(is_zero_param(p) for p in module.parameters()):
            group = None
            if mpu:
                group = mpu.get_data_parallel_group()
            Init(module=module, data_parallel_group=group)

        for m in module.modules():
            _init_external_params(m)

        self.module = module
        self.elastic_checkpoint = elastic_checkpoint
        self.overlap_comm = overlap_comm

        if self.overlap_comm:
            self.gpu_sum = torch.zeros(1, dtype=torch.float).cuda()

        ######################cpu offload setup##################################
        self.cpu_offload = cpu_offload_optimizer_state
        self.cpu_offload_use_pin_memory = cpu_offload_use_pin_memory

        if cpu_offload_params:
            assert cpu_offload_optimizer_state, "parameter offload is only available with optimizer state offload"
        self.cpu_offload_params = cpu_offload_optimizer_state and cpu_offload_params

        self.deepspeed_adam_offload = (self.cpu_offload
                                       and type(init_optimizer) == DeepSpeedCPUAdam)

        self.device = torch.cuda.current_device() if not self.cpu_offload else 'cpu'
        ############################################################################

        see_memory_usage("Before Partitioned Parameter Coordinator", force=False)

        fetch_stream = torch.cuda.Stream() if self.overlap_comm else None
        self.param_coordinator = PartitionedParameterCoordinator(
            comm_stream=fetch_stream,
            max_reuse_distance_in_numel=int(max_reuse_distance),
            max_available_parameters_in_numel=int(max_live_parameters))

        see_memory_usage("After Partitioned Parameter Coordinator", force=False)

        #self.param_coordinator = PartitionedParameterCoordinator(comm_stream=torch.cuda.Stream())
        #-------------Stage 3 Setup-------------------#
        # parameters smaller than the threshold will be collectively gathered at the
        # end of the optimizer step and will be kept till the end of the backward pass
        # TODO maybe worth just replicating these parameters and doing all reduce for them
        self.persistence_threshold = int(param_persistence_threshold)

        self.persistent_parameters = self.persistent_parameters()

        self.setup_zero_stage3_hooks()

        #resetting ds_tensor just in case parameters have been changed after initialization
        #example .half() or .to()
        #self.reset_ds_tensor()
        #---------------------------------------------#

        self.timers = timers

        self.reduce_scatter = reduce_scatter

        self.dp_process_group = dp_process_group

        self.partition_count = dist.get_world_size(group=self.dp_process_group)

        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_rank = mpu.get_model_parallel_rank()

        self.overflow = False
        self.clip_grad = clip_grad
        self.allreduce_always_fp32 = allreduce_always_fp32
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = INITIAL_MICRO_STEP_ID

        if self.reduce_scatter:
            assert not self.allreduce_always_fp32, "allreduce_always_fp32 is not yet supported with ZeRO-2 with reduce scatter enabled"
            assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with ZeRO-2 with reduce scatter enabled"
            assert self.postscale_gradients, "pre-scale gradients is not yet supported with ZeRO-2 with reduce scatter enabled"

        # Holds the mode parameter
        # The param.data may not hold any meaningful data
        # when param's status is NOT_AVAILABLE or IN_FLGHT
        self.fp16_groups = []

        # Hold partitioned parameters
        self.fp16_partitioned_groups = []

        # Holds a fused and flattened copy of the parameters
        self.fp16_partitioned_groups_flat = []

        #a single 32-bit partition of the parallel partitioned parameters
        #that this process will update
        self.fp32_partitioned_groups_flat = []

        # number of elements per partition in each group
        self.partition_size = []

        self.all_reduce_print = False

        self.prefetch_elements = int(prefetch_bucket_size)

        # padding on each partition for alignment purposes
        self.groups_padding = []

        self.sub_group_size = sub_group_size

        self.sub_group_to_group_id = {}

        see_memory_usage("Before creating fp16 partitions", force=False)
        #self._create_fp16_partitions()
        self._create_fp16_partitions_with_defragmentation()
        num_fp16_subgroups = len(self.fp16_partitioned_groups_flat)
        see_memory_usage(f"After creating fp16 partitions: {num_fp16_subgroups}",
                         force=False)

        see_memory_usage("Before creating fp32 partitions", force=False)
        self._create_fp32_partitions()
        see_memory_usage("After creating fp32 partitions", force=False)

        see_memory_usage("Before initializing optimizer states", force=False)
        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=False)

        if dist.get_rank() == 0:
            logger.info(f"optimizer state initialized")

        self.reduce_bucket_size = int(reduce_bucket_size)

        self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False)

        self.reduction_stream = torch.cuda.Stream(
        ) if self.overlap_comm else torch.cuda.current_stream()
        self.callback_queued = False
        self.copy_grad_stream = torch.cuda.Stream()

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.contiguous_gradients = contiguous_gradients
        self.extra_large_param_to_reduce = None
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None

        # simplified param id
        self.param_id = {}

        count = 0
        for i, params_group in enumerate(self.fp16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                count = count + 1

        #Largest partitioned param
        largest_partitioned_param_numel = self._get_largest_partitioned_numel()

        see_memory_usage(f"Before Set Grad positions", force=False)

        self.grad_position = {}
        self.set_grad_positions()
        see_memory_usage(f"Before CPU Offload initialization", force=False)

        self.grads_in_partition = None

        if self.cpu_offload:
            self.accumulated_grads_in_cpu = {}
            self.norm_for_param_grads = {}
            self.local_overflow = False
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(
                largest_partitioned_param_numel,
                device=torch.cuda.current_device()).half()
            self.temp_grad_gpu_buffer = torch.zeros(
                largest_partitioned_param_numel,
                device=torch.cuda.current_device()).half()
        see_memory_usage(f"After CPU Offload initialization", force=False)

        # stores if a partition has been reduced in this step
        self.is_partition_reduced = {}

        # stores if a grad in a partition has been computed or not
        self.is_grad_computed = {}

        # will store the averaged gradients required by this parititon
        self.averaged_gradients = {}

        #creates backward hooks for gradient partitioning
        self.create_reduce_and_remove_grad_hooks()

        #exit(0)

        # we may have a way of fusing dynamic scale. Do not support for now
        if dynamic_loss_scale:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)

            self.dynamic_loss_scale = True

        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=static_loss_scale)
            self.cur_iter = 0

        self.debug_fp16_grads = [{} for _ in self.fp16_groups]

        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)

    def _get_largest_partitioned_numel(self):
        largest_partitioned_param_numel = 0
        for partitioned_params_group in self.fp16_partitioned_groups:
            for partitioned_param in partitioned_params_group:
                if partitioned_param.numel() > largest_partitioned_param_numel:
                    largest_partitioned_param_numel = partitioned_param.numel()

        return largest_partitioned_param_numel

    def _create_fp16_partitions(self):
        dist.barrier()
        partition_id = dist.get_rank(group=self.dp_process_group)

        # loop to deal with groups
        for j, param_group in enumerate(self.optimizer.param_groups):

            sub_groups = self._create_fp16_sub_groups(param_group['params'])
            for sub_group in sub_groups:
                i = len(self.fp16_groups)

                # push this group to list before modify
                self.fp16_groups.append(sub_group)
                self.sub_group_to_group_id[i] = j

                #These are the list of the partitoned parameters
                self.fp16_partitioned_groups.append(
                    [param.ds_tensor for param in self.fp16_groups[i]])

                print_rank_0(
                    f"fp16 group {i} partitioned_param norms : {[param.ds_tensor.norm().item() for param in self.fp16_groups[i]]}"
                )

                # Record padding required to align group to world size (only applies to last rank)
                if partition_id == dist.get_world_size(group=self.dp_process_group) - 1:
                    padding = [p.padding_size() for p in self.fp16_groups[i]]
                else:
                    padding = [0] * len(self.fp16_groups[i])
                self.groups_padding.append(padding)

                #not sure why apex was cloning the weights before flattening
                #removing cloning here
                see_memory_usage(f"Before Flattening param group {i}", force=False)

                if not self.cpu_offload_params:
                    see_memory_usage(f"Before moving param group {i} to CPU",
                                     force=False)
                    #move all the parameters to cpu to free up GPU space for creating flat buffer
                    move_to_cpu(self.fp16_partitioned_groups[i])
                    see_memory_usage(f"After moving param group {i} to CPU", force=False)

                    #create flat buffer in CPU and move to GPU
                    self.fp16_partitioned_groups_flat.append(
                        flatten_dense_tensors_aligned(
                            self.fp16_partitioned_groups[i],
                            dist.get_world_size(group=self.dp_process_group)).cuda(
                                torch.cuda.current_device()))
                    see_memory_usage(
                        f"After flattening and moving param group {i} to GPU",
                        force=False)
                else:
                    #Without the detach, seems like the flattening becomes part of the
                    #model graph causing errors downstream
                    self.fp16_partitioned_groups_flat.append(
                        flatten_dense_tensors_aligned(
                            self.fp16_partitioned_groups[i],
                            dist.get_world_size(
                                group=self.dp_process_group)).detach().pin_memory())

                see_memory_usage(f"After Flattening param group {i}", force=False)

                see_memory_usage(f"After Flattening param group {i}", force=False)

                #set model fp16 weight to slices of flattened buffer
                updated_params = _unflatten_dense_tensors(
                    self.fp16_partitioned_groups_flat[i],
                    self.fp16_partitioned_groups[i])

                for partitioned_param, q in zip(self.fp16_partitioned_groups[i], updated_params):
                    partitioned_param.data = q.data

    def _move_to_flat_buffer(self, src_list, flat_buffer):
        start = 0
        for src in src_list:
            dest = flat_buffer.narrow(0, start, src.numel())
            start = start + src.numel()
            dest.data.copy_(src.data)
            src.data = dest.data

    def _create_fp16_partitions_with_defragmentation(self):
        dist.barrier()
        partition_id = dist.get_rank(group=self.dp_process_group)

        if self.cpu_offload_params:
            self.param_groups_fp16_flat_cpu_memory = []
            for j, param_group in enumerate(self.optimizer.param_groups):
                total_params = sum([p.ds_tensor.numel() for p in param_group['params']])
                self.param_groups_fp16_flat_cpu_memory.append(
                    torch.empty(total_params,
                                dtype=torch.half,
                                pin_memory=True))

        # loop to deal with groups
        for j, param_group in enumerate(self.optimizer.param_groups):

            sub_groups = self._create_fp16_sub_groups(param_group['params'])
            flat_offset = 0
            for sub_group in sub_groups:
                i = len(self.fp16_groups)

                # push this group to list before modify
                self.fp16_groups.append(sub_group)
                self.sub_group_to_group_id[i] = j

                #These are the list of the partitoned parameters
                self.fp16_partitioned_groups.append(
                    [param.ds_tensor for param in self.fp16_groups[i]])

                print_rank_0(
                    f"fp16 group {i} partitioned_param norms : {[param.ds_tensor.norm().item() for param in self.fp16_groups[i]]}"
                )

                # Record padding required to align group to world size (only applies to last rank)
                if partition_id == dist.get_world_size(group=self.dp_process_group) - 1:
                    padding = [p.padding_size() for p in self.fp16_groups[i]]
                else:
                    padding = [0] * len(self.fp16_groups[i])
                self.groups_padding.append(padding)

                #not sure why apex was cloning the weights before flattening
                #removing cloning here
                see_memory_usage(f"Before Flattening param group {i}", force=False)

                if not self.cpu_offload_params:
                    see_memory_usage(f"Before moving param group {i} to CPU",
                                     force=False)
                    #move all the parameters to cpu to free up GPU space for creating flat buffer
                    move_to_cpu(self.fp16_partitioned_groups[i])
                    see_memory_usage(f"After moving param group {i} to CPU", force=False)

                    #create flat buffer in CPU and move to GPU
                    self.fp16_partitioned_groups_flat.append(
                        flatten_dense_tensors_aligned(self.fp16_partitioned_groups[i],
                                                      1).cuda(
                                                          torch.cuda.current_device()))
                    see_memory_usage(
                        f"After flattening and moving param group {i} to GPU",
                        force=False)
                else:
                    total_elements = sum(
                        [t.numel() for t in self.fp16_partitioned_groups[i]])
                    fp16_partitioned_group_flat = self.param_groups_fp16_flat_cpu_memory[
                        j].narrow(0,
                                  flat_offset,
                                  total_elements)
                    self.fp16_partitioned_groups_flat.append(fp16_partitioned_group_flat)
                    flat_offset += total_elements

                # move param to flat buffer for both param offload on/off
                self._move_to_flat_buffer(self.fp16_partitioned_groups[i],
                                          self.fp16_partitioned_groups_flat[i])

                see_memory_usage(f"After Flattening param group {i}", force=False)

    def _create_fp32_partitions(self):
        for i, tensor in enumerate(self.fp16_partitioned_groups_flat):
            # a partition of the fp32 master weights that will be updated by this process

            self.fp32_partitioned_groups_flat.append(
                self.fp16_partitioned_groups_flat[i].to(
                    self.device).clone().float().detach())
            element_size = self.fp32_partitioned_groups_flat[i].element_size()
            num_elements = self.fp32_partitioned_groups_flat[i].numel()

            self.fp32_partitioned_groups_flat[
                i].requires_grad = True  # keep this in case internal optimizer uses it

        # Clear for on-the-fly population before the optimizer step
        for param_group in self.optimizer.param_groups:
            param_group['params'] = []

    def _create_fp16_sub_groups(self, params_group):

        params_group_numel = sum([param.ds_tensor.numel() for param in params_group])

        sub_group_size = self.sub_group_size

        if sub_group_size is None or sub_group_size >= params_group_numel:
            return [params_group]

        sub_groups = []
        sub_group = []
        local_sub_group_size = 0
        for param in params_group:

            sub_group.append(param)
            local_sub_group_size += param.ds_tensor.numel()

            if local_sub_group_size >= sub_group_size or id(param) == id(
                    params_group[-1]):

                sub_groups.append(sub_group)

                sub_group = []
                local_sub_group_size = 0

        return sub_groups

    # def reset_ds_tensor(self):
    #     for name, param in self.module.named_parameters(recurse=True):
    #         assert hasattr(param,'ds_id'), "Parameters have not been converted to be Zero 3 compatible"
    #         assert (param.ds_status == ZeroParamStatus.NOT_AVAILABLE), "All the parameters must have been partitioned by now"
    #         param.ds_tensor.data = param.data

    def setup_zero_stage3_hooks(self):
        self.hierarchy = 0
        self._register_hooks_recursively(self.module)

        #reset step if in inference mode
        def _end_of_forward_hook(module, *args):

            if not torch._C.is_grad_enabled():
                self.param_coordinator.reset_step()

        self.module.register_forward_hook(_end_of_forward_hook)

    def persistent_parameters(self):
        persistent_params = []
        total_persistent_parameters = 0
        for _, param in self.module.named_parameters(recurse=True):
            if param.ds_numel < self.persistence_threshold:
                param.ds_persist = True
                persistent_params.append(param)
                total_persistent_parameters += param.ds_numel

        print_rank_0(
            f'ZeRO 3: Total persistent parameters: {total_persistent_parameters}',
            force=False)
        return persistent_params

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count

        #print(f"{module.__class__} : {module.id}")

        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)

        def _pre_forward_module_hook(module, *args):
            self.pre_sub_module_forward_function(module)

        def _post_forward_module_hook(module, *args):
            self.post_sub_module_forward_function(module)

        def _pre_backward_module_hook(module, inputs, output):
            def _run_before_backward_function(sub_module):
                if sub_module.applied_pre_backward is False:
                    self.pre_sub_module_backward_function(sub_module)
                    sub_module.applied_pre_backward = True

            return _apply_to_tensors_only(module,
                                          PreBackwardFunction,
                                          _run_before_backward_function,
                                          output)

        #This is an alternate to doing _post_backward_module_hook
        #it uses tensor.register_hook instead of using torch.autograd.Function
        def _alternate_post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            #print(f"Before Forward {module.__class__.__name__}")

            def _run_after_backward_hook(*unused):
                module.ds_grads_remaining = module.ds_grads_remaining - 1
                if module.ds_grads_remaining == 0:
                    #print(f"After backward {module.__class__.__name__}")
                    self.post_sub_module_backward_function(module)

            def _run_before_forward_function(input):
                if input.requires_grad:
                    module.ds_grads_remaining += 1

            return _apply_forward_and_backward_to_tensors_only(
                module,
                _run_before_forward_function,
                _run_after_backward_hook,
                inputs)

        def _post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            def _run_after_backward_function(sub_module):
                if sub_module.ds_grads_remaining == 0:
                    self.post_sub_module_backward_function(sub_module)

            return _apply_to_tensors_only(module,
                                          PostBackwardFunction,
                                          _run_after_backward_function,
                                          inputs)

        # Pre forward hook
        module.register_forward_pre_hook(_pre_forward_module_hook)
        # Post forward hook
        module.register_forward_hook(_post_forward_module_hook)

        # Pre backward hook
        module.register_forward_hook(_pre_backward_module_hook)

        # post backward hook
        module.register_forward_pre_hook(_post_backward_module_hook)

    def pre_sub_module_forward_function(self, sub_module):
        see_memory_usage(f"Before sub module function {sub_module.__class__.__name__}",
                         force=False)

        self.param_coordinator.record_trace(sub_module)

        self.param_coordinator.fetch_sub_module(sub_module)
        see_memory_usage(
            f"Before sub module function {sub_module.__class__.__name__} after fetch",
            force=False)

        self.param_coordinator.prefetch_next_sub_modules(sub_module,
                                                         numel=self.prefetch_elements)
        see_memory_usage(
            f"Before sub module function {sub_module.__class__.__name__} after prefetch",
            force=False)

        self.param_coordinator.increment_step(sub_module)

    def post_sub_module_forward_function(self, sub_module):
        see_memory_usage(
            f"After sub module function {sub_module.__class__.__name__} before release",
            force=False)
        self.param_coordinator.release_sub_module(sub_module)
        see_memory_usage(
            f"After sub module function {sub_module.__class__.__name__} after release",
            force=False)

    def pre_sub_module_backward_function(self, sub_module):
        self.param_coordinator.record_trace(sub_module)

        self.param_coordinator.fetch_sub_module(sub_module)

        self.param_coordinator.prefetch_next_sub_modules(sub_module,
                                                         numel=self.prefetch_elements)

        self.param_coordinator.increment_step(sub_module)

    def post_sub_module_backward_function(self, sub_module):
        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} before release",
            force=False)
        self.param_coordinator.release_sub_module(sub_module)
        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} after release",
            force=False)

    def _release_ipg_buffers(self):
        if self.contiguous_gradients:
            self.ipg_buffer = None
            if not self.cpu_offload:
                self.grads_in_partition = None

            self.grads_in_partition_offset = 0

    def _optimizer_step(self, sub_group_id):
        param_group_id = self.sub_group_to_group_id[sub_group_id]
        fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
        fp16_param = self.fp16_partitioned_groups_flat[sub_group_id]
        self.optimizer.param_groups[param_group_id]['params'] = [fp32_param]
        self.optimizer.step()
        self.optimizer.param_groups[param_group_id]['params'] = []
        fp16_param.data.copy_(fp32_param.data)

    def initialize_optimizer_states(self):
        num_subgroups = len(self.fp16_groups)

        largest_numel = max([t.numel() for t in self.fp16_partitioned_groups_flat])
        gradient_dtype = self.fp32_partitioned_groups_flat[0].dtype
        gradient_buffer = torch.zeros(int(largest_numel),
                                      dtype=gradient_dtype,
                                      device=self.device)

        for i, group in enumerate(self.fp16_groups):
            see_memory_usage(
                f'[Begin] Initialize optimizer states {i} / {num_subgroups} subgroups',
                force=False)

            num_elements = int(self.fp16_partitioned_groups_flat[i].numel())
            if self.cpu_offload and not self.cpu_offload_use_pin_memory:
                self.fp32_partitioned_groups_flat[i].grad = torch.zeros(
                    num_elements,
                    dtype=gradient_dtype,
                    device=self.device)
            elif self.cpu_offload_use_pin_memory:
                self.fp32_partitioned_groups_flat[i].grad = torch.zeros(
                    num_elements,
                    dtype=gradient_dtype,
                    device=self.device).pin_memory()
            else:
                self.fp32_partitioned_groups_flat[i].grad = gradient_buffer.narrow(
                    0,
                    0,
                    num_elements)

            self._optimizer_step(i)

            see_memory_usage(
                f'[End] Initialize optimizer states {i} / {num_subgroups} subgroups',
                force=False)

        if not self.cpu_offload:
            for group in self.fp32_partitioned_groups_flat:
                group.grad = None

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
                self.first_param_index_in_partition[i][
                    partition_id] = self.get_first_param_index(
                        i,
                        param_group,
                        partition_id)

    def independent_gradient_partition_epilogue(self):
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        if self.overlap_comm:
            self.reduction_stream.synchronize()

        with torch.cuda.stream(self.reduction_stream):
            self.partition_previous_reduced_grads()

        # if dist.get_rank() == 0:
        #    logger.info("Params already reduced %s", self.params_already_reduced)
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False

        #in case of cpu offload, averaged gradients are already in fp32_partitioned_groups_flat.grad
        #TODO: use a similar code path for both cpu_offload and non-cpu offload
        if not self.cpu_offload:
            for i, sub_group in enumerate(self.fp16_groups):
                self.averaged_gradients[i] = [
                    torch.zeros_like(param.ds_tensor) if param.grad is None else
                    param.grad.data.narrow(0,
                                           0,
                                           param.ds_tensor.numel())
                    for param in sub_group
                ]
                # self.averaged_gradients[i] = self.get_flat_partition(
                #     self.fp16_groups[i],
                #     0,
                #     self.fp32_partitioned_groups_flat[i].numel(),
                #     return_tensor_list=True)

        self._release_ipg_buffers()

        see_memory_usage(f"End ipg_epilogue", force=False)

    # resets all partition to no reduced
    # sets remianing grads to the total number of grads in each partition
    # set is grad computed to false for all grads in partition
    def reset_partition_gradient_structures(self):
        total_partitions = dist.get_world_size(group=self.dp_process_group)
        for i, _ in enumerate(self.fp16_groups):
            for partition_id in range(total_partitions):
                self.is_partition_reduced[i][partition_id] = False
                self.remaining_grads_in_partition[i][
                    partition_id] = self.total_grads_in_partition[i][partition_id]

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

            if (current_index >= start_index and current_index < end_index):
                set_key_value_list(self.param_to_partition_ids[i],
                                   param_id,
                                   partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][
                    param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0

            elif start_index > current_index and start_index < (current_index +
                                                                param_size):
                assert (first_offset == 0), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

                set_key_value_list(self.param_to_partition_ids[i],
                                   param_id,
                                   partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset

            current_index = current_index + param_size

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.independent_gradient_partition_epilogue()
        self.zero_grad()

    def create_reduce_and_remove_grad_hooks(self):
        print_rank_0(f'[Begin] Create gradient reduction hooks')
        self.grad_accs = []
        for i, param_group in enumerate(self.fp16_groups):
            for param in param_group:
                if param.requires_grad:
                    #print_rank_0(f" Before all gather {param.device}, {param.shape}")

                    # The hook must be created in un-partitioned parameter
                    param.all_gather()

                    #print(f"After all gather {param.device}, {param.shape}")
                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        grad_acc.register_hook(reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)

                    #print(f"param grad fn {param.expand_as(param).grad_fn}")
                    wrapper(param, i)

                    # Partition the parameter after creating the hook
                    param.partition()
        print_rank_0(f'[End] Create gradient reduction hooks')

    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    def report_ipg_memory_usage(self, tag, param_elems):
        elem_count = self.elements_in_ipg_bucket + param_elems
        percent_of_bucket_size = (100.0 * elem_count) // self.reduce_bucket_size
        see_memory_usage(
            f"{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}",
            force=False)

    ###############Idependent Partition Gradient ########################
    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        #print_rank_0(f"Inside reduce ipg buckets. Param ID {param.ds_id}, ipg elements {self.elements_in_ipg_bucket}, reduce bucket size {self.reduce_bucket_size}", force=True)
        if self.elements_in_ipg_bucket + param.ds_numel > self.reduce_bucket_size:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads",
                                         param.ds_numel)

            self.reduce_ipg_grads()

            if self.contiguous_gradients and self.overlap_comm:
                # Swap ipg_index between 0 and 1
                self.ipg_index = 1 - self.ipg_index
            self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads",
                                         param.ds_numel)

        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, \
            f"The parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported"

        # keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening
        if param.ds_numel > self.reduce_bucket_size:
            self.extra_large_param_to_reduce = param

        elif self.contiguous_gradients:
            #print_rank_0("before new grad tensor move")
            new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(
                0,
                self.elements_in_ipg_bucket,
                param.ds_numel)
            #print_rank_0("after new grad tensor move")
            new_grad_tensor.copy_(param.grad.view(-1))
            param.grad.data = new_grad_tensor.data.view_as(param.grad)

        self.elements_in_ipg_bucket += param.ds_numel
        self.grads_in_ipg_bucket.append(param.grad)
        self.params_in_ipg_bucket.append((i, param, param_id))
        self.report_ipg_memory_usage("End ipg_remove_grads", 0)

    def gradient_reduction_w_predivide(self, tensor):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)

        tensor_to_allreduce = tensor

        if self.allreduce_always_fp32:
            tensor_to_allreduce = tensor.float()

        if self.postscale_gradients:
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)

            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

            if self.gradient_predivide_factor() != dp_world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor() /
                                         dp_world_size)
        else:
            tensor_to_allreduce.div_(dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

        if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def average_tensor(self, tensors, params_to_reduce):
        with torch.cuda.stream(self.reduction_stream):
            if not self.reduce_scatter:
                for tensor in tensors:
                    self.gradient_reduction_w_predivide(tensor)
                return

            for tensor in tensors:
                tensor.div_(dist.get_world_size(group=self.dp_process_group))

            # reduction resulting with each rank only holding the gradient partition it owns
            # This could either be a reduce scatter or a reduce op depending on how
            # parameters are partitionied. The method is impelemnted by the
            # DeepSpeed param extensions to the pytroch parameter, so its up to
            # the extension to define what happens here
            params_to_reduce[0].reduce_gradients_at_owner(
                param_list=params_to_reduce,
                hierarchy=self.param_coordinator.hierarchy)

    def set_grad_positions(self):
        for i, group in enumerate(self.fp16_groups):
            current_offset = 0
            for param in group:
                param_id = self.get_param_id(param)
                num_elements = param.ds_tensor.numel()

                self.grad_position[param_id] = [
                    int(i),
                    int(current_offset),
                    int(num_elements)
                ]
                #print(f"param id {param_id} i:{i}, ds_tensor {num_elements} numel {param.numel()}")
                current_offset += num_elements

    def async_accumulate_grad_in_cpu_via_gpu(self, param, acc_grad_cpu_partition):

        # copy to a preexisiting buffer to avoid memory allocation penalty
        dest_buffer = self.temp_grad_buffer_for_gpu_offload.view(-1).narrow(
            0,
            0,
            param.ds_tensor.numel())

        if self.micro_step_id > 0:
            dest_buffer.copy_(acc_grad_cpu_partition.view(-1), non_blocking=True)
            param.grad.data.view(-1).add_(dest_buffer)

        # at the boundary we will send 32bit directly
        if not self.is_gradient_accumulation_boundary:
            acc_grad_cpu_partition.data.copy_(param.grad.data.view(-1),
                                              non_blocking=True)

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

    def update_overflow_tracker_for_param_grad(self, param):
        #Credit to our user David Minn
        if param.grad is not None:
            if self.overlap_comm:
                self.gpu_sum = self.gpu_sum + param.grad.data.float().sum()
            elif self._has_inf_or_nan(param.grad.data):
                self.local_overflow = True

    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param, fp32_grad_tensor):
        with torch.cuda.stream(self.copy_grad_stream):
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
                    total_norm += param_norm.item()**2

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])

        torch.distributed.all_reduce(total_norm_cuda,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_cuda,
                                        op=torch.distributed.ReduceOp.SUM)

        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float(
                'inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    def partition_previous_reduced_grads(self):
        if not self.previous_reduced_grads:
            return

        if self.cpu_offload:
            allocate_grads_in_partition = self.grads_in_partition is None\
            and self.gradient_accumulation_steps > 1
        else:
            allocate_grads_in_partition = self.grads_in_partition is None

        if allocate_grads_in_partition:
            self.grads_in_partition = []

            for i, group in enumerate(self.fp16_groups):
                total_size = 0
                for param_in_partition in group:
                    total_size += param_in_partition.ds_tensor.numel()

                see_memory_usage(
                    f"group {i} before creating {total_size} reduced gradients into partition",
                    force=False)
                if self.cpu_offload_use_pin_memory:
                    self.grads_in_partition.append(
                        torch.zeros(int(total_size),
                                    dtype=torch.half,
                                    device=self.device).pin_memory())
                else:
                    self.grads_in_partition.append(
                        torch.zeros(int(total_size),
                                    dtype=torch.half,
                                    device=self.device))
                see_memory_usage(
                    f"group {i} after creating {total_size} reduced gradients into partition",
                    force=False)

        for param in self.previous_reduced_grads:

            [i, dest_offset, num_elements] = self.grad_position[self.get_param_id(param)]

            # self.debug_fp16_grads[i][self.get_param_id(param)] = (
            #     float(param.data.float().norm(2)),
            #     float(param.grad.data.float().norm(2)))

            if self.cpu_offload:

                param.partition_gradients(partition_buffers=self.temp_grad_gpu_buffer)
                with torch.cuda.stream(self.copy_grad_stream):
                    self.reduction_stream.synchronize()

                if self.gradient_accumulation_steps > 1:
                    # The allreduce buffer will be rewritted. Copy the gradients in partition to a new buffer
                    fp16_grad_tensor = self.grads_in_partition[i].narrow(
                        0,
                        dest_offset,
                        num_elements)
                    self.async_accumulate_grad_in_cpu_via_gpu(param, fp16_grad_tensor)

                if self.is_gradient_accumulation_boundary:

                    self.set_norm_for_param_grad_in_gpu(param)

                    self.update_overflow_tracker_for_param_grad(param)

                    fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(
                        0,
                        dest_offset,
                        num_elements)

                    self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(
                        param,
                        fp32_grad_tensor)
            else:
                # The allreduce buffer will be rewritted. Copy the gradients in partition to a new buffer
                fp16_grad_tensor = self.grads_in_partition[i].narrow(
                    0,
                    dest_offset,
                    num_elements)
                param.partition_gradients(
                    partition_buffers=fp16_grad_tensor,
                    accumulate=True if self.micro_step_id > 0 else False)

        self.previous_reduced_grads = []

    def reduce_ipg_grads(self, extra_param=None):
        if self.overlap_comm:
            self.reduction_stream.synchronize()

        with torch.cuda.stream(self.reduction_stream):
            self.partition_previous_reduced_grads()

        params_to_reduce = [param for i, param, param_id in self.params_in_ipg_bucket]
        #print(f"Params in ipg bucket {self.params_in_ipg_bucket}")
        #print(f"Reducing {[(param.ds_id, param.grad) for param in params_to_reduce]}")
        #exit(0)
        if self.contiguous_gradients:
            reduction_list = [self.ipg_buffer[self.ipg_index]]
            if self.extra_large_param_to_reduce is not None:
                reduction_list.append(self.extra_large_param_to_reduce.grad)
                self.extra_large_param_to_reduce = None
            self.average_tensor(reduction_list, params_to_reduce)
        else:
            self.buffered_reduce_fallback(
                None,
                self.grads_in_ipg_bucket,
                elements_per_buffer=self.elements_in_ipg_bucket)

        for _, param, param_id in self.params_in_ipg_bucket:
            self.params_already_reduced[param_id] = True

        self.previous_reduced_grads = params_to_reduce

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        #####################################################################

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        #print(f"Backward {param.ds_id}")
        self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def zero_reduced_gradients(self, partition_id, i):
        def are_all_related_partitions_reduced(params_id):
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            return True

        for params_id in self.is_grad_computed[i][partition_id]:
            if are_all_related_partitions_reduced(params_id):
                self.param_dict[params_id].grad = None

    def flatten_and_print(self, message, tensors, start=0, n=5):
        flatten_tensor = _flatten_dense_tensors(tensors)

        def print_func():
            logger.info(flatten_tensor.contiguous().view(-1).narrow(0, start, n))

        self.sequential_execution(print_func, message)

    def get_grads_to_reduce(self, i, partition_id):
        def get_reducable_portion(key):
            grad = self.param_dict[key].grad
            total_elements = grad.numel()
            start = self.grad_start_offset[i][partition_id][key]
            num_elements = min(
                total_elements - start,
                self.partition_size[i] -
                self.grad_partition_insertion_offset[i][partition_id][key])
            if not pg_correctness_test:
                if num_elements == total_elements:
                    return grad
                else:
                    return grad.contiguous().view(-1).narrow(0,
                                                             int(start),
                                                             int(num_elements))
            else:
                if num_elements == total_elements:
                    return grad.clone()
                else:
                    return grad.clone().contiguous().view(-1).narrow(
                        0,
                        int(start),
                        int(num_elements))

        grads_to_reduce = []
        for key in self.is_grad_computed[i][partition_id]:
            grad = get_reducable_portion(key)
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

    def allreduce_bucket(self, bucket, allreduce_always_fp32=False, rank=None, log=None):
        rank = None
        tensor = flatten(bucket)

        tensor_to_allreduce = tensor

        if pg_correctness_test:
            allreduce_always_fp32 = True

        if allreduce_always_fp32:
            tensor_to_allreduce = tensor.float()

        tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group))

        if rank is None:
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        else:
            global_rank = _get_global_rank(self.dp_process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)

        if allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                tensor.copy_(tensor_to_allreduce)

        return tensor

    # if rank is specified do a reduction instead of an allreduce
    def allreduce_and_copy(self, small_bucket, rank=None, log=None):
        with torch.cuda.stream(self.reduction_stream):
            allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(self,
                            bucket,
                            numel_per_bucket=500000000,
                            rank=None,
                            log=None):
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

    # allows using reduction of gradients instead of using all_reduce
    def buffered_reduce_fallback(self,
                                 rank,
                                 grads,
                                 elements_per_buffer=500000000,
                                 log=None):
        split_buckets = split_half_float_double(grads)

        for i, bucket in enumerate(split_buckets):
            self.allreduce_no_retain(bucket,
                                     numel_per_bucket=elements_per_buffer,
                                     rank=rank,
                                     log=log)

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

            if (current_index >= start_index and current_index < end_index):
                params_in_partition.append(tensor)

            elif start_index > current_index and start_index < (current_index +
                                                                tensor_size):
                params_in_partition.append(tensor)

                assert (first_offset == 0), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                params_not_in_partition.append(tensor)

            current_index = current_index + tensor_size

        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        if self.model_parallel_group is None:
            torch.distributed.all_reduce(tensor=tensor, op=op)
        else:
            torch.distributed.all_reduce(tensor=tensor,
                                         op=op,
                                         group=self.model_parallel_group)

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
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm_cuda,
                                            op=torch.distributed.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
        else:
            total_norm = 0.0
            # if dist.get_rank() == 0:
            #    logger.info(f"Total Norm begining {total_norm}")
            for g, p in zip(gradients, params):
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    param_norm = g.data.double().norm(2)
                    total_norm += param_norm.item()**2
            # Sum across all model parallel GPUs.
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])

            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm_cuda,
                                            op=torch.distributed.ReduceOp.SUM)

            total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        if total_norm == float(
                'inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    # creates a flat fused tensor from the tensor list starting at the first_offset
    # in the first tensor of the list. If there are not enough elements in the tensor
    # list then the flat tensor will be padded with zeros
    def get_flat_partition(self,
                           tensor_list,
                           first_offset,
                           partition_size,
                           return_tensor_list=False):
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
                flat_tensor_list.append(tensor.contiguous().view(-1).narrow(
                    0,
                    int(tensor_offset),
                    int(num_elements)))
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

        return _flatten_dense_tensors(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            p.grad = None

    def reset_cpu_buffers(self):
        self.norm_for_param_grads = {}
        self.local_overflow = False

    def log_timers(self, timer_names):
        if self.timers is None:
            return

        self.timers.log(names=list(timer_names))

    def start_timers(self, timer_names):
        if self.timers is None:
            return

        for name in timer_names:
            self.timers(name).start()

    def stop_timers(self, timer_names):
        if self.timers is None:
            return

        for name in timer_names:
            self.timers(name).stop()

    def old_step(self, closure=None):
        """
        Not supporting closure.
        """

        self.micro_step_id = INITIAL_MICRO_STEP_ID

        # if self.cpu_offload:
        #    torch.cuda.current_stream().wait_stream(self.migration_stream)

        print_rank_0(f"Inside Step function")
        see_memory_usage(f"In step before checking overflow", force=False)

        print_rank_0("Finished Tracing at Beginning of Step")
        self.param_coordinator.hierarchy = 0
        self.param_coordinator.finish_tracing(print_trace=True)

        self.param_coordinator.reset_step()

        print_rank_0("Finished Tracing at Beginning of Step")

        # First compute norm for all group so we know if there is overflow
        self.check_overflow()

        timers = self.timers

        OPTIMIZER_STEP = 'optimizer_step'
        OPTIMIZER_FP16_UPDATE = 'optimizer_fp16_update'
        OPTIMIZER_FP32_GRADIENT = 'optimizer_fp32_gradient'
        timer_names = [OPTIMIZER_STEP, OPTIMIZER_FP16_UPDATE, OPTIMIZER_FP32_GRADIENT]

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients', force=False)
            self.zero_grad()

            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}

            see_memory_usage('After overflow after clearing gradients', force=False)

            logger.info(
                "[deepscale] OVERFLOW! Rank {} Skipping step. Attempted loss scale: {}, "
                "reducing to {}".format(dist.get_rank(),
                                        prev_scale,
                                        self.loss_scale))
            self.start_timers(timer_names)
            self.stop_timers(timer_names)
            return

        norm_groups = []
        single_partition_grad_groups = []
        skip = False
        partition_id = dist.get_rank(group=self.dp_process_group)

        debug_fp32_grads = [{} for _ in self.fp16_groups]

        self.start_timers([OPTIMIZER_FP32_GRADIENT])
        for i, group in enumerate(self.fp16_groups):

            if self.cpu_offload:
                norm_groups.append(
                    self.complete_grad_norm_calculation_for_cpu_offload(
                        self.fp16_groups[i]))

                single_grad_partition = self.fp32_partitioned_groups_flat[i].grad
            else:
                norm_groups.append(
                    self.get_grad_norm_direct(self.averaged_gradients[i],
                                              self.fp16_groups[i]))

                # free gradients for all the prameters that are not updated by this process
                # self.free_grad_in_param_list(self.params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process

                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                single_grad_partition = _flatten_dense_tensors(
                    self.averaged_gradients[i]).to(
                        self.fp32_partitioned_groups_flat[i].dtype)

                assert single_grad_partition.numel() == self.fp32_partitioned_groups_flat[i].numel(), \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.fp32_partitioned_groups_flat[i].grad = single_grad_partition

                # release all the gradient since we have already created a necessary copy in dp_grad_partition
                self.zero_grad()

                self.averaged_gradients[i] = None

            single_partition_grad_groups.append(single_grad_partition)
            debug_fp32_grads[i] = [
                (t.clone().detach(),
                 t) for t in _unflatten_dense_tensors(single_grad_partition,
                                                      group)
            ]

        self.stop_timers([OPTIMIZER_FP32_GRADIENT])

        print(f"Norm groups: {norm_groups}")

        self.unscale_and_clip_grads(single_partition_grad_groups, norm_groups)

        #self.dump_pre_step_gradients(debug_fp32_grads)

        self.start_timers([OPTIMIZER_STEP])
        self.optimizer.step()
        self.stop_timers([OPTIMIZER_STEP])

        # get rid of the fp32 gradients. Not needed anymore
        if not self.cpu_offload:
            for group in self.fp32_partitioned_groups_flat:
                group.grad = None

        self.start_timers([OPTIMIZER_FP16_UPDATE])
        for fp16_partitions, fp32_partition in zip(self.fp16_partitioned_groups_flat, self.fp32_partitioned_groups_flat):
            fp16_partitions.data.copy_(fp32_partition.data)
        self.stop_timers([OPTIMIZER_FP16_UPDATE])

        print(
            f"fp16 groups norm : {[group_flat.norm() for group_flat in self.fp16_partitioned_groups_flat]}"
        )
        if self.cpu_offload:
            self.reset_cpu_buffers()

        # TODO: we probably don't need this? just to be safe
        for i in range(len(norm_groups)):
            #for p in self.fp16_groups[i]:
            #    p.data=p.ds_tensor

            updated_params = _unflatten_dense_tensors(
                self.fp16_partitioned_groups_flat[i],
                self.fp16_partitioned_groups[i])
            for partitioned_param, q in zip(self.fp16_partitioned_groups[i], updated_params):
                # print(f"Grad fn: {p.grad_fn}")
                # p.data = torch.ones(1).half().cuda()
                partitioned_param.data = q.data

        #Gathering persisting parameters
        self.persistent_parameters[0].all_gather(self.persistent_parameters)

        #self.dump_post_step_gradients()
        self.debug_fp16_grads = [{} for _ in self.fp16_groups]

        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.log_timers(timer_names)

        see_memory_usage('After zero_optimizer step', force=False)
        print_rank_0(f"------------------Finishing Step-----------------------",
                     force=False)
        return

    def _pre_step(self):

        self.micro_step_id = INITIAL_MICRO_STEP_ID

        print_rank_0(f"Inside Step function")
        see_memory_usage(f"In step before checking overflow", force=False)

        print_rank_0("Finished Tracing at Beginning of Step")
        self.param_coordinator.hierarchy = 0
        self.param_coordinator.finish_tracing(print_trace=True)

        self.param_coordinator.reset_step()

        print_rank_0("Finished Tracing at Beginning of Step")

    def _get_norm_groups(self):
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            if self.cpu_offload:
                norm_groups.append(
                    self.complete_grad_norm_calculation_for_cpu_offload(
                        self.fp16_groups[i]))
            else:
                norm_groups.append(
                    self.get_grad_norm_direct(self.averaged_gradients[i],
                                              self.fp16_groups[i]))
        return norm_groups

    def _prepare_fp32_grad_for_sub_group(self, sub_group_id):

        partition_id = dist.get_rank(group=self.dp_process_group)

        single_grad_partition = _flatten_dense_tensors(
            self.averaged_gradients[sub_group_id]).to(
                self.fp32_partitioned_groups_flat[sub_group_id].dtype)

        assert single_grad_partition.numel() == self.fp32_partitioned_groups_flat[sub_group_id].numel(), \
            "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                single_grad_partition.numel(), self.fp32_partitioned_groups_flat[sub_group_id].numel(), sub_group_id, partition_id)

        self.fp32_partitioned_groups_flat[sub_group_id].grad = single_grad_partition

        # release all the gradient since we have already created a necessary copy in dp_grad_partition
        self.zero_grad()

        self.averaged_gradients[sub_group_id] = None

    def _prepare_sub_group(self, sub_group_id, timer_names=set()):
        see_memory_usage(f'Before prepare optimizer sub group {sub_group_id}',
                         force=False)
        if not self.cpu_offload:
            self._prepare_fp32_grad_for_sub_group(sub_group_id)
        see_memory_usage(f'After prepare optimizer sub group {sub_group_id}',
                         force=False)

    def _release_sub_group(self, sub_group_id, timer_names=set()):
        see_memory_usage(f'Before release optimizer sub group {sub_group_id}',
                         force=False)
        # get rid of the fp32 gradients. Not needed anymore
        if not self.cpu_offload:
            self.fp32_partitioned_groups_flat[sub_group_id].grad = None

        see_memory_usage(f'After release optimizer sub group {sub_group_id}',
                         force=False)

    def _unflatten_partitioned_parameters(self, sub_group_id):
        updated_params = _unflatten_dense_tensors(
            self.fp16_partitioned_groups_flat[sub_group_id],
            self.fp16_partitioned_groups[sub_group_id])

        for partitioned_param, q in zip(self.fp16_partitioned_groups[sub_group_id], updated_params):
            partitioned_param.data = q.data

    def _overflow_clean_up(self, prev_scale):
        see_memory_usage('After overflow before clearing gradients', force=False)
        self.zero_grad()

        if self.cpu_offload:
            self.reset_cpu_buffers()
        else:
            self.averaged_gradients = {}

        see_memory_usage('After overflow after clearing gradients', force=False)

        if torch.distributed.get_rank() == 0:
            logger.info(
                "[deepscale] OVERFLOW! Rank {} Skipping step. Attempted loss scale: {}, "
                "reducing to {}".format(dist.get_rank(),
                                        prev_scale,
                                        self.loss_scale))

    def _overflow_check_and_loss_scale_update(self):

        # First compute norm for all group so we know if there is overflow
        self.check_overflow()

        #loss scaling related computation
        prev_scale = self.loss_scale
        self._update_scale(self.overflow)

        if self.overflow:
            self._overflow_clean_up(prev_scale)

        return self.overflow

    def _post_step(self, timer_names=set()):
        if self.cpu_offload:
            self.reset_cpu_buffers()

        #Gathering persisting parameters
        self.persistent_parameters[0].all_gather(self.persistent_parameters)

        self.log_timers(timer_names)

        see_memory_usage('After zero_optimizer step', force=False)
        print_rank_0(f"------------------Finishing Step-----------------------")

    def step(self, closure=None):
        """
            Not supporting closure.
            """
        self._pre_step()

        #checks for overflow, adjust the loss scale accordingly
        if self._overflow_check_and_loss_scale_update():
            return

        norm_groups = self._get_norm_groups()

        timer_names = set()

        timer_names.add('optimizer_step')
        self.start_timers(['optimizer_step'])

        #update parameters one sub group at a time
        for sub_group_id, group in enumerate(self.fp16_groups):

            #prepare optimizer states, gradients and fp32 parameters for update
            self._prepare_sub_group(sub_group_id, timer_names)

            #scale the fp32 gradients
            self.unscale_and_clip_grads(sub_group_id, norm_groups)

            #apply the optimizer step on the sub group and copy fp32 parameters to fp16
            self._optimizer_step(sub_group_id)

            #release memory or swap out optimizer states of fp32 parameters
            self._release_sub_group(sub_group_id, timer_names)

            #unflatten fp16 parameter subgroup
            self._unflatten_partitioned_parameters(sub_group_id)

        self.stop_timers(['optimizer_step'])

        self._post_step(timer_names)
        return

    def dump_pre_step_gradients(self, debug_fp32_grads):
        # Dump gradient norms for debbuging
        for i, _ in enumerate(self.fp16_groups):
            print(f'Pre-Step Dump Norms for Group {i} FP16P, FP16G, FP32G, FP32GUC')
            for fp16_param, fp32_grad in zip(self.fp16_groups[i], debug_fp32_grads[i]):
                param_id = self.get_param_id(fp16_param)
                fp16_grad_norm = self.debug_fp16_grads[i][param_id]

                fp32_grad_norm = [float(t.data.float().norm(2)) for t in fp32_grad]
                norm_list = [fp16_grad_norm, fp32_grad_norm]
                print(f'Pre-Step Norms {i} {param_id} = {norm_list}')

    def dump_post_step_gradients(self):
        # Dump gradient norms for debbuging
        for i, group in enumerate(self.fp16_groups):
            print(
                f'Post-Step Dump Norms for Group {i} FP16P, FP16DS, FP16FLAT, FP32FLAT')
            unflat_fp16 = _unflatten_dense_tensors(self.fp16_groups_flat[i],
                                                   self.fp16_groups[i])
            unflat_fp32 = _unflatten_dense_tensors(self.fp32_partitioned_groups_flat[i],
                                                   self.fp16_groups[i])
            for j, p in enumerate(self.fp16_groups[i]):
                param_id = self.get_param_id(p)
                param_norm = float(p.data.float().norm(2))
                ds_norm = float(p.ds_tensor.data.float().norm(2))

                unflat_norm = [
                    float(t.data.float().norm(2))
                    for t in [unflat_fp16[j],
                              unflat_fp32[j]]
                ]
                norm_list = [param_norm, ds_norm] + unflat_norm
                print(f'Post-Step Norms {i} {param_id} = {norm_list}')

    def unscale_and_clip_grads(self, sub_group_id, norm_groups):

        grad_groups_flat = [self.fp32_partitioned_groups_flat[sub_group_id].grad]

        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm**2.0
        total_norm = math.sqrt(total_norm)

        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
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

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            if self.overlap_comm:
                self.local_overflow = self._has_inf_or_nan(self.gpu_sum)
                self.gpu_sum = torch.zeros(1, dtype=torch.float).cuda()

            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial(
            )
            #overflow = self.has_overflow_partitioned_grads_serial()
            overflow_gpu = torch.cuda.ByteTensor([overflow])
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self.dp_process_group)

        else:
            params = []
            for group in self.fp16_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = torch.cuda.ByteTensor([overflow])

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu,
                                        op=torch.distributed.ReduceOp.MAX)

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

    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.micro_step_id += 1
        print_rank_0(
            f"Total fully available parameters {self.param_coordinator.total_available_parameter_numel}"
        )
        see_memory_usage(f"Before backward", force=False)
        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(self.reduce_bucket_size,
                                dtype=torch.half,
                                device=torch.cuda.current_device())
            self.ipg_buffer.append(buf_0)

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                buf_1 = torch.empty(self.reduce_bucket_size,
                                    dtype=torch.half,
                                    device=torch.cuda.current_device())
                self.ipg_buffer.append(buf_1)
            self.ipg_index = 0

        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)
        '''Partitioning Parameters that were not partitioned
        Usually if parameters of modules whose input parameters do not require
        grad computation do not trigger post call and will therefore will remain unpartitioned '''
        self._partition_all_parameters()

    def _partition_all_parameters(self):
        for name, param in self.module.named_parameters(recurse=True):
            self.param_coordinator.release_and_reset_parameter(param)

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
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def _get_lean_tensors(self, padded_flattened_tensor, group_tensors, paddings):
        # Remove paddings from flattened tensor
        individual_tensors = _unflatten_dense_tensors(padded_flattened_tensor,
                                                      group_tensors)
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
                    lean_state[key] = self._get_lean_tensors(
                        value,
                        self.fp16_partitioned_groups[i],
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
            lean_group = self._get_lean_tensors(group,
                                                self.fp16_partitioned_groups[i],
                                                self.groups_padding[i])
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
        state_dict['zero_stage'] = ZERO_OPTIMIZATION_WEIGHTS
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['partition_count'] = self.partition_count

        self._set_fp32_optimizer_param_groups()
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_flat_groups'] = self.fp32_partitioned_groups_flat
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
            raise NotImplementedError(
                "ZeRO-3 does not yet support elastic checkpointing, please disable for now."
            )

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
    def _restore_from_fp16_weights(self):
        for fp16_partitions, fp32_partition in zip(self.fp16_partitioned_groups_flat, self.fp32_partitioned_groups_flat):
            fp32_partition.data.copy_(fp16_partitions.data)

    # Refresh the fp32 master params from the fp16 copies.
    def refresh_fp32_params(self):
        self._restore_from_fp16_weights()

    # Extract flattened partion for current rank from all partitions
    def _get_flattened_partition(self, all_partition_states):
        partition_id = dist.get_rank(group=self.dp_process_group)
        alignment = dist.get_world_size(group=self.dp_process_group)

        param_partitions = [[] for _ in range(len(all_partition_states[0]))]
        for i, partition in enumerate(all_partition_states):
            for j, param in enumerate(partition):
                param_partitions[j].append(param)

        local_state_partitions = []
        for param_index, param_slices in enumerate(param_partitions):
            flattened_merged_tensor = flatten_dense_tensors_aligned(
                param_slices,
                alignment)
            new_partitions = self.get_data_parallel_partitions(flattened_merged_tensor)
            local_state_partitions.append(new_partitions[partition_id])

        if torch.is_tensor(local_state_partitions[0]):
            return flatten_dense_tensors_aligned(local_state_partitions, alignment)

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
            all_partition_group_states = [
                sd['base_optimizer_state'][i] for sd in all_state_dict
            ]
            for key in all_partition_group_states[0].keys():
                all_partition_states = [
                    all_states[key] for all_states in all_partition_group_states
                ]
                partition_states[key] = self._get_flattened_partition(
                    all_partition_states)
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
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']

        if load_optimizer_states:
            self._set_fp32_optimizer_param_groups()
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self._clear_fp32_optimizer_param_groups()

        # restore fp32 partitions
        for curr_param, saved_param in zip(self.fp32_partitioned_groups_flat, state_dict['fp32_flat_groups']):
            curr_param.data.copy_(saved_param.data)

        # restore fp16 partitions from fp32
        for sub_group_id in range(len(self.fp32_partitioned_groups_flat)):
            fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
            fp16_param = self.fp16_partitioned_groups_flat[sub_group_id]
            fp16_param.data.copy_(fp32_param.data)

        # update fp16 unflattened params
        for sub_group_id in range(len(self.fp16_partitioned_groups_flat)):
            updated_params = _unflatten_dense_tensors(
                self.fp16_partitioned_groups_flat[sub_group_id],
                self.fp16_partitioned_groups[sub_group_id])

            for partitioned_param, q in zip(self.fp16_partitioned_groups[sub_group_id], updated_params):
                partitioned_param.data = q.data

    # TODO: Support different/changing load/save DP degree.
    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False):
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
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """

        if self.elastic_checkpoint:
            raise NotImplementedError(
                "ZeRO-3 does not yet support elastic checkpointing, please disable for now."
            )
        else:
            self._rigid_load_state_dict(
                state_dict_list[dist.get_rank(group=self.dp_process_group)],
                load_optimizer_states=load_optimizer_states)

        self.persistent_parameters[0].partition(self.persistent_parameters)
        self.persistent_parameters[0].all_gather(self.persistent_parameters)

    def save_checkpoint_prologue(self):
        self._partition_all_parameters()

    def save_checkpoint_epilogue(self):
        self.persistent_parameters[0].all_gather(self.persistent_parameters)


def _handle_overflow(cpu_sum, x, i):
    import math
    rank = torch.distributed.get_rank()
    if rank == 0:
        t_i = -1
        for v_i, v in enumerate(x.data.contiguous().view(-1)):
            if not math.isfinite(float(v)):
                t_i = v_i
                break
        logger.info(
            f"rank {rank} detected overflow {cpu_sum} in tensor {i}:{t_i} shape {x.shape}"
        )
