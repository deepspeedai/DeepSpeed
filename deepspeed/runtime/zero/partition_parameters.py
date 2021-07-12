"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

import os
import time
import types
from enum import Enum
import functools
import itertools

import torch
from torch.distributed.distributed_c10d import _get_global_rank

from .linear import LinearModuleForZeroStage3, LinearFunctionForZeroStage3
from .offload_constants import *

from ..utils import see_memory_usage
from deepspeed.utils import log_dist, init_distributed
from deepspeed.utils.debug import debug_param2name_id_shape, debug_param2name_id_shape_device, debug_module2name, debug_param2name, debug_param2name_id_shape_status, printflock, log_rank_file

from ..swap_tensor.partitioned_param_swapper import AsyncPartitionedParameterSwapper, PartitionedParamStatus
from ..config import DeepSpeedConfig

param_count = 0
partitioned_param_data_shape = [1]


def print_rank_0(message, debug=False, force=False):
    rank = torch.distributed.get_rank()
    if rank == 0 and (debug or force):
        print(message)
    # other variations
    # - print for all ranks w/o interleaving
    # printflock(f"[{rank}] {message}")
    # - print to log file per rank
    # log_rank_file(rank, message)


def is_zero_param(parameter):
    if not torch.is_tensor(parameter):
        return False
    return hasattr(parameter, 'ds_id')


def _init_external_params(module):
    if not hasattr(module, '_external_params'):
        module._external_params = {}

        def external_parameters(self):
            return self._external_params.items()

        def all_parameters(self):
            return itertools.chain(self.named_parameters(self,
                                                         recurse=False),
                                   external_parameters(self))

        module.ds_external_parameters = types.MethodType(external_parameters, module)
        module.all_parameters = types.MethodType(all_parameters, module)


def register_external_parameter(module, parameter):
    """Instruct DeepSpeed to coordinate ``parameter``'s collection and partitioning in
    the forward and backward passes of ``module``.

    This is used when a parameter is accessed outside of its owning module's
    ``forward()``. DeepSpeed must know to collect it from its partitioned
    state and when to release the memory.

    .. note::
        This is only applicable to training with ZeRO stage 3.

    Args:
        module (``torch.nn.Module``): The module that requires ``parameter`` in its forward pass.
        parameter (``torch.nn.Parameter``): The parameter to register.

    Raises:
        RuntimeError: If ``parameter`` is not of type ``torch.nn.Parameter``.


    Examples
    ========

    #. Register a weight that is used in another module's forward pass (line 6).
       Parameter ``layer1.weight`` is used by ``layer2`` (line 11).

        .. code-block:: python
            :linenos:
            :emphasize-lines: 6,11

            class ModuleZ3(torch.nn.Module):
                def __init__(self, *args):
                    super().__init__(self, *args)
                    self.layer1 = SomeLayer()
                    self.layer2 = OtherLayer()
                    deepspeed.zero.register_external_parameter(self, self.layer1.weight)

                def forward(self, input):
                    x = self.layer1(input)
                    # self.layer1.weight is required by self.layer2.forward
                    y = self.layer2(x, self.layer1.weight)
                    return y
    """
    if not isinstance(parameter, torch.nn.Parameter):
        raise RuntimeError('Parameter is not a torch.nn.Parameter')

    if not hasattr(module, '_external_params'):
        _init_external_params(module)

    key = id(parameter)
    module._external_params[key] = parameter


def unregister_external_parameter(module, parameter):
    """Reverses the effects of :meth:`register_external_parameter`.

    Args:
        module (``torch.nn.Module``): The module to affect.
        parameter (``torch.nn.Parameter``): The parameter to unregister.

    Raises:
        RuntimeError: If ``parameter`` is not of type ``torch.nn.Parameter``.
        RuntimeError: If ``parameter`` is not a registered external parameter of ``module``.
    """
    if not isinstance(parameter, torch.nn.Parameter):
        raise RuntimeError('Parameter is not a torch.nn.Parameter')

    if not hasattr(module,
                   '_external_params') or id(parameter) not in module._external_params:
        raise RuntimeError('Parameter is not a registered external parameter of module.')

    key = id(parameter)
    del module._external_params[key]


class ZeroParamType(Enum):

    # same as regular pytorch parameters
    NORMAL = 1

    # parameters are partitioned across data parallel process
    PARTITIONED = 2

    # the parameter is held with a unique process rank
    # and is not available on all other process
    REMOTE = 3


class ZeroParamStatus(Enum):
    # parameters are fully present and ready for use on all processes
    AVAILABLE = 1

    # parameters are either partitioned or remote in some or all process
    NOT_AVAILABLE = 2

    # parameters are being gathered.
    INFLIGHT = 3


_orig_torch_empty = torch.empty


def empty_cuda_tensor_half(*size, **kwargs):
    if not 'device' in kwargs.keys():
        kwargs['device'] = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
    tensor = _orig_torch_empty(*size, **kwargs)
    if tensor.is_floating_point():
        return tensor.half()
    else:
        return tensor


def new_cuda_tensor_half(cls, *args):
    device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
    tensor = torch.ones((1, 1), device=device).new_empty(*args).half()
    if tensor.is_floating_point():
        return tensor.half()
    else:
        return tensor


def empty_cuda_tensor(*size, **kwargs):
    if not 'device' in kwargs.keys():
        kwargs['device'] = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
    tensor = _orig_torch_empty(*size, **kwargs)
    return tensor


def new_cuda_tensor(cls, *args):
    device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))
    tensor = torch.ones((1, 1), device=device).new_empty(*args)
    return tensor


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


reuse_buffers = False
temp_contiguous_tensor = None
empty_buffers = {}


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self, enabled=True, mem_efficient_linear=True, config=None, dtype=None):
        self.mem_efficient_linear = mem_efficient_linear
        self.enabled = enabled
        self._set_dtype(config, dtype)
        assert self.dtype in [torch.half, torch.float], f"Invalid data type {self.dtype}, allowed values are [torch.half, torch.float]"

    def __enter__(self):
        if not self.enabled:
            return

        def partition_after(f):
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):

                # important logic: We want to run post_init only after child's __init__ is
                # completed, and do nothing after __init__ of any of its parents and grandparents in
                # the inheritance ancestry. This way the partitioning will need to happen only once
                # when the whole object is ready to be partitioned and not before. This is because
                # often the child module will need to tweak the weights - for example running a
                # custom weights init function. So if a parent created the weights param, the child
                # won't need to gather it in order to tweak it

                print_rank_0(f'Before initializing {module.__class__.__name__}',
                             force=False)

                is_child_module = False
                if not hasattr(module, "_ds_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    is_child_module = True
                    setattr(module, "_ds_child_entered", True)

                f(module, *args, **kwargs)

                if is_child_module:
                    # child's __init__ is done, now we can run a single post_init on the child object
                    delattr(module, "_ds_child_entered")

                    print_rank_0(f'Running post_init for {module.__class__.__name__}',
                                 force=False)
                    self._post_init_method(module)

                print_rank_0(
                    f'After initializing followed by post init for {module.__class__.__name__}',
                    force=False)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = partition_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module recursively
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            # print(f"subclass={subclass.__module__}.{subclass.__qualname__}")
            _enable_class(subclass)

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.Tensor.__old_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        if self.dtype == torch.half:
            torch.Tensor.__new__ = new_cuda_tensor_half
            torch.empty = empty_cuda_tensor_half
        else:
            torch.Tensor.__new__ = new_cuda_tensor
            torch.empty = empty_cuda_tensor

        if self.mem_efficient_linear:
            print_rank_0(
                "nn.functional.linear has been overridden with a more memory efficient version. This will persist unless manually reset.",
                force=False)
            self.linear_bk = torch.nn.functional.linear
            torch.nn.functional.linear = LinearFunctionForZeroStage3.apply

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass

        torch.Tensor.__new__ = torch.Tensor.__old_new__
        torch.empty = _orig_torch_empty

        #un doing it here will undo it during training
        #if self.mem_efficient_linear:
        #    torch.nn.functional.linear = self.linear_bk
        #        if self.mem_efficient_linear:
        #            torch.nn.functional.linear = self.linear_bk

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _set_dtype(self, ds_config, dtype):
        if ds_config is not None and dtype is None:
            _ds_config = DeepSpeedConfig(ds_config)
            self.dtype = torch.half if _ds_config.fp16_enabled else torch.float
        elif dtype is None:
            self.dtype = torch.half
        else:
            self.dtype = dtype


# Replaces all parameters in module with Scattered Parameters
class Init(InsertPostInitMethodToModuleSubClasses):
    param_id = 0

    def __init__(self,
                 module=None,
                 data_parallel_group=None,
                 mem_efficient_linear=True,
                 remote_device=None,
                 pin_memory=False,
                 config=None,
                 enabled=True,
                 dtype=None):
        """A context to enable massive model construction for training with
        ZeRO-3. Models are automatically partitioned (or, sharded) across the
        system and converted to half precision.

        Args:
            module (``torch.nn.Module``, optional): If provided, partition the model as
                if it was constructed in the context.
            data_parallel_group (``torch.distributed`` process group, optional):
                The group of processes to partition among. Defaults to all processes.
            mem_efficient_linear (bool, optional): Replace
                torch.nn.functional.linear with an implementation that allows
                DeepSpeed to partition parameters. Defaults to ``True``.
            remote_device (string, optional): The initial device to store model
                weights e.g., ``cpu``, ``nvme``. Passing ``"cpu"`` will create the model in CPU
                memory. The model may still be moved to GPU based on the
                offload settings for training. Defaults to the local GPU.
            pin_memory (bool, optional): Potentially increase performance by
                using pinned memory for model weights. ``remote_device`` must be
                ``"cpu"``. Defaults to ``False``.
            config (``json file`` or dict, optional): If provided, provides configuration
                for swapping fp16 params to NVMe.
            enabled (bool, optional): If ``False``, this context has no
                effect. Defaults to ``True``.
            dtype (``dtype``, optional): Can be used to change the data type of the parameters.
                Supported options are ``torch.half`` and ``torch.float``. Defaults to ``None``

        This context accelerates model initialization and enables models that
        are too large to allocate in their entirety in CPU memory. It has the
        following effects:

        #. allocates tensors to either GPU or CPU memory or NVMe
        #. converts floating point tensors to half precision
        #. immediately partitions tensors among the group of data-parallel devices
        #. (*optional*) replaces ``torch.nn.functional.linear`` with a more
           memory-efficient implementation

        These modifications allow for models that exceed the size of local CPU/GPU
        memory/NVMe, but fit within the total NVMe capacity (*i.e.*, aggregate CPU
        or GPU memory or NVMe) across all nodes. Consider initializing a model with one
        trillion parameters, whose weights occupy two terabytes (TB) in half
        precision. The initial CPU allocation in full precision requires 4TB of
        memory *per process*, and so a system with 8 GPUs per node would need 32TB of
        CPU memory due to data-parallel redundancies. Instead, by immediately
        partitioning tensors we remove the redundancies. The result is that
        regardless of the number of GPUs, we still only require the original 4TB. This
        allows for a linear increase in model size with the aggregate system memory.
        For example, if a node has 1TB of memory and 8 GPUs, we could fit a trillion
        parameter model with 4 nodes and 32 GPUs.

        Important: If the fp16 weights of the model can't fit onto a single GPU memory
        this feature must be used.

        .. note::
            Initializes ``torch.distributed`` if it has not already been done so.
            See :meth:`deepseed.init_distributed` for more information.

        .. note::
            Can also be used as a decorator:

            .. code-block:: python

                @deepspeed.zero.Init()
                def get_model():
                    return MyLargeModel()

        .. note::
            Only applicable to training with ZeRO-3.

        Examples
        --------

        #. Allocate a model and partition it among all processes:

            .. code-block:: python

                with deepspeed.zero.Init():
                    model = MyLargeModel()


        #. Allocate a model in pinned CPU memory and partition it among a subgroup of processes:

            .. code-block:: python

                with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                                         remote_device="cpu",
                                         pin_memory=True):
                    model = MyLargeModel()


        #. Partition an already-allocated model in CPU memory:

            .. code-block:: python

                model = deepspeed.zero.Init(module=model)
        """

        super().__init__(enabled=enabled,
                         mem_efficient_linear=mem_efficient_linear,
                         config=config,
                         dtype=dtype)
        if not torch.distributed.is_initialized():
            init_distributed()
            assert torch.distributed.is_initialized(), "Parameters cannot be scattered without initializing torch.distributed"
        if data_parallel_group is None:
            self.ds_process_group = torch.distributed.group.WORLD
        else:
            self.ds_process_group = data_parallel_group

        self.rank = torch.distributed.get_rank(group=self.ds_process_group)
        self.world_size = torch.distributed.get_world_size(group=self.ds_process_group)

        #Local device is the device where the parameters are consumed
        #It is the device where parameters are fully instantiated using allgather
        self.local_device = torch.device('cuda:{}'.format(os.environ["LOCAL_RANK"]))

        self._validate_remote_device(remote_device, config)

        #Remote device is the device where parameter partiitons are stored
        #It can be same as local_device or it could be CPU or NVMe.
        self.remote_device = self.local_device if remote_device is None else remote_device
        self.pin_memory = pin_memory if (
            self.remote_device == OFFLOAD_CPU_DEVICE) else False

        # Enable fp16 param swapping to NVMe
        if self.remote_device == OFFLOAD_NVME_DEVICE:
            _ds_config = DeepSpeedConfig(config)
            self.param_swapper = AsyncPartitionedParameterSwapper(_ds_config)
        else:
            self.param_swapper = None

        # If we are provided an already-allocated module to prepare.
        if module is not None:
            assert isinstance(module, torch.nn.Module)
            for param in module.parameters(recurse=True):
                if is_zero_param(param):
                    continue
                self._convert_to_deepspeed_param(param)
                param.partition()

    def _validate_remote_device(self, remote_device, ds_config):
        if ds_config is not None:
            _ds_config = DeepSpeedConfig(ds_config)
            if remote_device in [None, OFFLOAD_CPU_DEVICE]:
                if _ds_config.zero_config.offload_param is not None:
                    offload_param_device = _ds_config.zero_config.offload_param[
                        OFFLOAD_PARAM_DEVICE]
                    assert offload_param_device != OFFLOAD_NVME_DEVICE, \
                    f"{OFFLOAD_PARAM_DEVICE} in DeepSpeed Config cannot be {offload_param_device} if remote device is {remote_device}."

            if remote_device == OFFLOAD_NVME_DEVICE:
                assert _ds_config.zero_config.offload_param is not None, \
                f'{OFFLOAD_PARAM} must be defined in DeepSpeed Config if remote device is {OFFLOAD_NVME_DEVICE}.'

                assert _ds_config.zero_config.offload_param[OFFLOAD_PARAM_NVME_PATH] is not None, \
                f'{OFFLOAD_PARAM_NVME_PATH} in DeepSpeed Config cannot be None if remote device is {OFFLOAD_NVME_DEVICE}'

    def _post_init_method(self, module):
        #see_memory_usage(f"Before converting parmas in {module.__class__.__name__}", force=False)
        print_rank_0(f'Converting Params in {module.__class__.__name__}', force=False)
        see_memory_usage(
            f"Before converting and partitioning parmas in {module.__class__.__name__}",
            force=False)

        global param_count
        for param in module.parameters(recurse=False):
            param_count += param.numel()
            if not is_zero_param(param):
                self._convert_to_deepspeed_param(param)
                print_rank_0(
                    f"Partitioning param {debug_param2name_id_shape(param)} module={debug_module2name(module)}"
                )
                param.partition()
        see_memory_usage(
            f"Param count {param_count}. After converting and partitioning parmas in {module.__class__.__name__}",
            force=False)

    def _convert_to_deepspeed_param(self, param):

        # Partitioned, Normal, Remote
        param.ds_param_type = ZeroParamType.PARTITIONED

        # Replicated vs Partitioned vs Inflight
        param.ds_status = ZeroParamStatus.AVAILABLE

        # Stores the shape of the original tensor
        param.ds_shape = param.shape

        # Stores the number of elements in the original parameter without padding
        param.ds_numel = param.numel()

        # Stores the partitioned copy of the tensor
        param.ds_tensor = None

        # Keeps track of how many active sub-modules need this param at any given point in time
        param.ds_active_sub_modules = 0

        # If this flag is true, then the parameters are replicated throughput training
        # And only partitioned before the step
        param.ds_persist = False

        # The group that the parameter is scattered across.
        param.ds_process_group = self.ds_process_group

        # This is set to the Async Param swapper if remote device is nvme
        # else this is set to None
        param.nvme_swapper = self.param_swapper

        # DeepSped Param ID
        param.ds_id = Init.param_id
        Init.param_id += 1

        def all_gather(param_list=None, async_op=False, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            return self._all_gather(param_list, async_op=async_op, hierarchy=hierarchy)

        def partition(param_list=None, hierarchy=0, has_been_updated=False):
            cls = param
            print_rank_0(
                f"{'--'*hierarchy}----Partitioning param {debug_param2name_id_shape_device(cls)}"
            )
            if param_list is None:
                param_list = [cls]
            self._partition(param_list, has_been_updated=has_been_updated)

        def reduce_gradients_at_owner(param_list=None, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            print_rank_0(
                f"{'--'*hierarchy}----Reducing Gradients for param with ids {[param.ds_id for param in param_list]} to owner"
            )
            self._reduce_scatter_gradients(param_list)

        def partition_gradients(param_list=None,
                                partition_buffers=None,
                                hierarchy=0,
                                accumulate=False):
            cls = param
            print_rank_0(
                f"{'--'*hierarchy}----Partitioning param gradient with id {debug_param2name_id_shape_device(cls)}"
            )
            if param_list is None:
                param_list = [cls]
                if isinstance(partition_buffers, torch.Tensor):
                    partition_buffers = [partition_buffers]

            self._partition_gradients(param_list,
                                      partition_buffers=partition_buffers,
                                      accumulate=accumulate)

        def aligned_size():
            return self._aligned_size(param)

        def padding_size():
            return self._padding_size(param)

        def partitioned_size():
            return self._partitioned_size(param)

        # Collectives for gathering and partitioning parameters
        param.all_gather = all_gather
        param.partition = partition

        # Collective for averaging gradients
        param.reduce_gradients_at_owner = reduce_gradients_at_owner
        param.partition_gradients = partition_gradients

        # Partitioning size utilities
        param.aligned_size = aligned_size
        param.padding_size = padding_size
        param.partitioned_size = partitioned_size

    def _aligned_size(self, param):
        return param.ds_numel + self._padding_size(param)

    def _padding_size(self, param):
        remainder = param.ds_numel % self.world_size
        return (self.world_size - remainder) if remainder else 0

    def _partitioned_size(self, param):
        return param.ds_tensor.ds_numel

    def _ensure_availability_of_partitioned_params(self, params):
        swap_in_list = []
        swap_in_flight = []
        for param in params:
            if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                assert param.ds_tensor.final_location == OFFLOAD_NVME_DEVICE and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                swap_in_list.append(param)
            if param.ds_tensor.status == PartitionedParamStatus.INFLIGHT:
                assert param.ds_tensor.final_location == OFFLOAD_NVME_DEVICE and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                swap_in_flight.append(param)
        if len(swap_in_list) > 0:
            swap_in_list[0].nvme_swapper.swap_in(swap_in_list, async_op=False)
        elif len(swap_in_flight) > 0:
            swap_in_flight[0].nvme_swapper.synchronize_reads()

    def _all_gather(self, param_list, async_op=False, hierarchy=None):

        #fetches from nvme if the partition is not available and in nvme
        self._ensure_availability_of_partitioned_params(param_list)

        handles = []
        all_gather_list = []
        for param in param_list:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if async_op:
                    handle = self._allgather_param(param,
                                                   async_op=async_op,
                                                   hierarchy=hierarchy)
                    param.ds_status = ZeroParamStatus.INFLIGHT  # if async_op else ZeroParamStatus.AVAILABLE
                    handles.append(handle)
                else:
                    all_gather_list.append(param)

        if not async_op:
            ret_value = self._allgather_params(all_gather_list, hierarchy=hierarchy)
            for param in all_gather_list:
                param.ds_status = ZeroParamStatus.AVAILABLE
            return ret_value

        return handles

    def _partition(self, param_list, force=False, has_been_updated=False):
        for param in param_list:
            #print_rank_0(f"Before Partitioning Param {param.ds_id}")
            #self._param_status(param)
            self._partition_param(param, has_been_updated=has_been_updated)
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE
            #if param.ds_tensor is not None:
            #    assert id(param.data) == id(param.ds_tensor.data), \
            #    "After the parameters are initially partitioned, make sure we are not recreating the partition."
            #print_rank_0(f"After Partitioning Param {param.ds_id}")
            # self._param_status(param)

    def _partition_param(self, param, buffer=None, has_been_updated=False):
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot parititon a param in flight"

        global reuse_buffers
        #print_rank_0(f"Param id {param.ds_id} status is {param.ds_status}")
        if param.ds_status is ZeroParamStatus.AVAILABLE:
            print_rank_0(
                f"Partitioning param id {param.ds_id} reuse buffers {reuse_buffers}",
                force=False)
            # if reuse_buffers and False:
            #     numel = buffer.numel()
            #     buffer = param.data.view(-1)
            #     print_rank_0(
            #         "Returning buffer for param {param.ds_id} with numel {param.ds_numel} to empty buffers",
            #         force=False)
            #     if numel in empty_buffers:
            #         empty_buffers[numel].append(buffer)

            #if torch.distributed.get_rank():
            #    print(f"Releasing {param.data.numel()}")
            if param.ds_tensor is not None and not has_been_updated:

                #param.data = param.ds_tensor.data

                see_memory_usage(
                    f'Before partitioning param {param.ds_id} {param.shape}',
                    force=False)
                #param.data does not store anything meaningful in partitioned state
                param.data = torch.ones(1, dtype=self.dtype).to(param.device)
                see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}',
                                 force=False)

                if param.ds_tensor.final_location == OFFLOAD_NVME_DEVICE:
                    print_rank_0(
                        f"Param {param.ds_id} partition released since it exists in nvme",
                        force=False)
                    param.nvme_swapper.remove_partition_and_release_buffers([param])

                return

            tensor_size = self._aligned_size(param)
            partition_size = tensor_size // self.world_size

            if param.ds_tensor is None:
                final_location = None
                if self.remote_device == OFFLOAD_NVME_DEVICE and self.param_swapper.swappable_tensor(
                        numel=partition_size):
                    final_location = OFFLOAD_NVME_DEVICE
                    buffer = self.param_swapper.get_buffer(param, partition_size)
                    partitioned_tensor = torch.zeros(1,
                                                     dtype=param.dtype,
                                                     device=buffer.device)
                    partitioned_tensor.data = buffer.data
                    print_rank_0(
                        f"ID {param.ds_id} Initializing partition for the first time for nvme offload."
                    )

                else:
                    partitioned_tensor = torch.zeros(
                        partition_size,
                        dtype=param.dtype,
                        device=OFFLOAD_CPU_DEVICE
                        if self.remote_device == OFFLOAD_NVME_DEVICE else
                        self.remote_device)
                    if self.pin_memory:
                        partitioned_tensor = partitioned_tensor.pin_memory()

                partitioned_tensor.requires_grad = False
                param.ds_tensor = partitioned_tensor
                param.ds_tensor.ds_numel = partition_size
                param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_tensor.final_location = final_location

            start = partition_size * self.rank
            end = start + partition_size

            one_dim_param = param.contiguous().view(-1)

            if start < param.ds_numel and end <= param.ds_numel:
                src_tensor = one_dim_param.narrow(0, start, partition_size)

                param.ds_tensor.copy_(src_tensor)
                #partitioned_tensor = src_tensor.clone().detach().to(self.remote_device)

            else:
                # partitioned_tensor = torch.zeros(partition_size,
                #                                  dtype=param.dtype,
                #                                  device=self.remote_device )

                if start < param.ds_numel:
                    elements_to_copy = param.ds_numel - start
                    param.ds_tensor.narrow(0,
                                           0,
                                           elements_to_copy).copy_(
                                               one_dim_param.narrow(
                                                   0,
                                                   start,
                                                   elements_to_copy))

            #print(f"Remote device {self.remote_device}")

            #param.ds_tensor = partitioned_tensor

            #param.data = param.ds_tensor.data

            #param.data does not store anything meaningful in partitioned state

            see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}',
                             force=False)
            param.data = torch.ones(1, dtype=self.dtype).to(param.device)
            see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}',
                             force=False)

            if param.ds_tensor.final_location == OFFLOAD_NVME_DEVICE:
                self.param_swapper.swap_out_and_release([param])
                print_rank_0(
                    f"ID {param.ds_id} Offloaded to nvme offload and buffers released.")
                see_memory_usage(
                    f"ID {param.ds_id} Offloaded to nvme offload and buffers released.",
                    force=False)

            print_rank_0(
                f"ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}"
            )

    def _param_status(self, param):
        if param.ds_tensor is not None:
            print_rank_0(
                f"Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned numel {param.ds_tensor.numel()}, data numel {param.data.numel()}"
            )
        else:
            print_rank_0(
                f"Param id {param.ds_id}, param status: {param.ds_status}, param numel {param.ds_numel}, partitioned ds_tensor {param.ds_tensor}, data numel {param.data.numel()}"
            )

    def _allgather_param(self, param, async_op=False, hierarchy=0):

        partition_size = param.ds_tensor.ds_numel

        tensor_size = partition_size * self.world_size
        aligned_param_size = self._aligned_size(param)
        assert tensor_size == aligned_param_size, f'param id {param.ds_id} aligned size {aligned_param_size} does not match tensor size {tensor_size}'

        print_rank_0(
            f"{'--'* hierarchy}---- Before allocating allgather param {debug_param2name_id_shape_status(param)} partition size={partition_size}"
        )

        see_memory_usage(
            f'Before allocate allgather param {debug_param2name_id_shape_status(param)} partition_size={partition_size} ',
            force=False)
        flat_tensor = torch.zeros(aligned_param_size,
                                  dtype=param.dtype,
                                  device=param.device).view(-1)
        see_memory_usage(
            f'After allocate allgather param {debug_param2name_id_shape_status(param)} {aligned_param_size} {partition_size} ',
            force=False)

        torch.cuda.synchronize()

        print_rank_0(
            f"{'--'* hierarchy}----allgather param with {debug_param2name_id_shape_status(param)} partition size={partition_size}"
        )
        #        if not flat_tensor.numel() > 100000:
        #            replicated_tensor = flat_tensor.narrow(0,
        #                                                   0,
        #                                                   param.ds_numel).view(param.ds_shape)
        #            param.data = replicated_tensor.data
        #            return None
        partitions = []
        for i in range(self.world_size):
            partitions.append(flat_tensor.narrow(0, partition_size * i, partition_size))

            if i == torch.distributed.get_rank(group=self.ds_process_group):
                partitions[i].data.copy_(param.ds_tensor.data, non_blocking=True)

        handle = torch.distributed.all_gather(partitions,
                                              partitions[self.rank],
                                              group=self.ds_process_group,
                                              async_op=async_op)

        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
        param.data = replicated_tensor.data
        return handle

    def _allgather_params(self, param_list, hierarchy=0):
        if len(param_list) == 0:
            return

        partition_size = sum([param.ds_tensor.ds_numel for param in param_list])

        tensor_size = partition_size * self.world_size
        flat_tensor = torch.empty(tensor_size,
                                  dtype=param_list[0].dtype,
                                  device=self.local_device)
        flat_tensor.requres_grad = False
        partitions = []
        for i in range(self.world_size):
            start = partition_size * i

            partitions.append(flat_tensor.narrow(0, start, partition_size))

            if i == self.rank:
                offset = 0
                for param in param_list:
                    param_numel = param.ds_tensor.ds_numel

                    partitions[i].narrow(0,
                                         offset,
                                         param_numel).copy_(param.ds_tensor.data)

                    offset += param_numel

        torch.distributed.all_gather(partitions,
                                     partitions[self.rank],
                                     group=self.ds_process_group,
                                     async_op=False)
        param_offset = 0

        for param in param_list:
            param_partition_size = param.ds_tensor.ds_numel
            param_size = param.ds_numel
            replicated_tensor = torch.empty(param.ds_shape,
                                            dtype=param.dtype,
                                            device=self.local_device)

            for i in range(self.world_size):

                start = i * partition_size

                param_start = i * param_partition_size

                if param_start < param_size:
                    numel_to_copy = min(param_size - param_start, param_partition_size)

                    part_to_copy = partitions[i].narrow(0, param_offset, numel_to_copy)

                    replicated_tensor.view(-1).narrow(0,
                                                      param_start,
                                                      numel_to_copy).copy_(part_to_copy)
            #param_offset += param.data.numel()
            param_offset += param.ds_tensor.ds_numel

            param.data = replicated_tensor.data

        return None

    def _reduce_scatter_gradients(self, param_list):
        #print_rank_0([param.grad for param in param_list])
        #assert any([param.grad is None for param in param_list]), "None gradients cannot be reduce scattered"

        handles_and_reduced_partitions = []
        for param in param_list:
            assert param.grad.numel(
            ) == param.ds_numel, f"{param.grad.numel()} != {param.ds_numel} Cannot reduce scatter gradients whose size is not same as the params"

            handles_and_reduced_partitions.append(self._reduce_scatter_gradient(param))

        for param, (handle, reduced_partition) in zip(param_list, handles_and_reduced_partitions):
            if handle is not None:
                handle.wait()

            # some ranks may have partitions that are padded to go beyond the grad size.
            # For these ranks the output of reduce scatter is a separate buffer and needs
            # to be copied in
            partition_size = param.ds_tensor.ds_numel
            start = self.rank * partition_size
            end = start + partition_size
            #print_rank_0("REduce scatter was executed for praam {param.ds_id}")
            if start < param.ds_numel and end > param.ds_numel:
                elements = param.ds_numel - start
                param.grad.view(-1).narrow(0,
                                           start,
                                           elements).copy_(
                                               reduced_partition.narrow(0,
                                                                        0,
                                                                        elements))

    def _reduce_scatter_gradient(self, param):

        partition_size = param.ds_tensor.ds_numel
        #output = torch.empty(partition_size, dtype=param.dtype, device=param.device)

        total_size = partition_size * self.world_size
        input_list = []

        for i in range(self.world_size):

            start = i * partition_size
            end = start + partition_size

            #print("before reduce scatter gradients")
            if start < param.ds_numel and end <= param.ds_numel:
                input = param.grad.view(-1).narrow(0, start, partition_size)
            else:
                input = torch.zeros(partition_size,
                                    dtype=param.dtype,
                                    device=param.device)

                if start < param.ds_numel:
                    elements = param.ds_numel - start
                    input.narrow(0,
                                 0,
                                 elements).copy_(
                                     param.grad.view(-1).narrow(0,
                                                                start,
                                                                elements))
            #print("after reduce scatter gradients")
            input_list.append(input)

        rank = torch.distributed.get_rank(group=self.ds_process_group)
        handle = torch.distributed.reduce_scatter(input_list[rank],
                                                  input_list,
                                                  group=self.ds_process_group,
                                                  async_op=True)

        return handle, input_list[rank]

    def _partition_gradients(self, param_list, partition_buffers=None, accumulate=False):
        if partition_buffers is None:
            partition_buffers = [None] * len(param_list)

        for param, partition_buffer in zip(param_list, partition_buffers):
            self._partition_gradient(param,
                                     partition_buffer=partition_buffer,
                                     accumulate=accumulate)

    def _partition_gradient(self, param, partition_buffer=None, accumulate=False):
        #import pdb;pdb.set_trace()
        # param.grad=None
        # param.grad.test()
        print_rank_0(
            f"Partitioning param {param.ds_id} gradient of size {param.grad.numel()} type {param.grad.dtype} part_size {param.ds_tensor.ds_numel}"
        )
        see_memory_usage("Before partitioning gradients", force=False)
        partition_size = param.ds_tensor.ds_numel

        if partition_buffer is None:
            assert not accumulate, "No buffer to accumulate to"
            partition_buffer = torch.zeros(partition_size,
                                           dtype=param.dtype,
                                           device=param.device)
        else:
            assert partition_buffer.numel() >= partition_size, f"The partition buffer size {partition_buffer.numel()} should match the size of param.ds_tensor {partition_size}"

        rank = torch.distributed.get_rank(group=self.ds_process_group)
        start = partition_size * rank
        end = start + partition_size

        dest_tensor_full_buffer = partition_buffer.view(-1).narrow(0, 0, partition_size)

        #print("before partition gradients")
        if start < param.ds_numel:
            elements = min(param.ds_numel - start, partition_size)

            dest_tensor = dest_tensor_full_buffer.narrow(0, 0, elements)
            src_tensor = param.grad.view(-1).narrow(0, start, elements)

            # just copy the grad partition to the buffer
            if not accumulate:
                dest_tensor.copy_(src_tensor)

            # if source and destinatoin are on same device,
            # add to the provided buffer
            elif src_tensor.device == dest_tensor.device:
                dest_tensor.add_(src_tensor)

            # if source and destination are on different device, copy first to src
            # then add and move back to the destination. This seems to run faster
            # when src is gpu and dest is cpu
            # adding directly to cpu is very slow
            else:
                acc_tensor = torch.empty(src_tensor.numel(),
                                         dtype=param.dtype,
                                         device=param.device)

                acc_tensor.copy_(dest_tensor)
                acc_tensor.add_(src_tensor)
                dest_tensor.copy_(acc_tensor)

            # partition_buffer.view(-1).narrow(
            #     0,
            #     0,
            #     elements).copy_(param.grad.view(-1).narrow(0,
            #                                             start,
            #                                             elements))

        #print("after partition gradients")
        param.grad.data = dest_tensor_full_buffer.data
        see_memory_usage("After partitioning gradients", force=False)


class GatheredParameters:
    def __init__(self, params, modifier_rank=None, fwd_module=None, enabled=True):
        """A context that collects parameters that were partitioned via a
        :class:`deepspeed.zero.Init` context. The parameters are partitioned
        again upon exit.

        Args:
            params (``torch.nn.Parameter``): A single parameter or a list of parameters to collect.
                It's assumed that all parameters are zero params.
            modifier_rank (int, optional): If specified, this rank's parameter will be
                broadcasted on exit from the context. This argument is required if ``params`` are
                modified, so that all processes have a consistent view of the data. Defaults
                to ``None``.
            fwd_module (``torch.nn.Module``, optional): If specified, ``params`` will be
                registered as external parameters of ``fwd_module``. See :meth:`deepspeed.zero.register_external_parameter`.
            enabled (bool, optional): If ``False``, this context is a no-op. Defaults to ``True``.

        Important: Make sure to use ``modifier_rank`` that is not ``None`` (e.g. ``modifier_rank=0``)
        if you need the GPU memory allocated by gather to be released upon exit from the context manager.

        Examples
        ========

        #. Allocate a partitioned module, initialize its weight on rank 0, and update all
           processes.

            .. code-block:: python

                with deepspeed.zero.Init():
                    linear = torch.nn.Linear(1000,1000)

                with deepspeed.zero.GatheredParameters(linear.weight,
                                                       modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        linear.weight.zero_()

                with deepspeed.zero.GatheredParameters(linear.weight,
                                                       modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        linear.weight.zero_()

        #. Collect a partitioned weight to pass to another module during
           training. The parameter will be registered as an external parameter
           and made available during the backward pass.

            .. code-block:: python
                :emphasize-lines: 6

                def forward(self, input):
                    x = self.layer1(input)

                    # self.layer1.weight is required by self.layer2.forward
                    with deepspeed.zero.GatheredParameters(self.layer1.weight,
                                                           fwd_module=self):
                        y = self.layer2(x, self.layer1.weight)
                    return y


        #. Pretrained model loading

            .. code-block:: python

                with deepspeed.zero.Init():
                    model = MyModel()

                state_dict = torch.load(model_path, map_location="cpu")

                def load(module: nn.Module, prefix=""):
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(state_dict, prefix)

                    for name, child in module._modules.items():
                        if child is not None:
                            load(child, prefix + name + ".")

                load(model, prefix="")

        If this approach is not used, then the full model will first get copied to each GPU. For models
        bigger than the memory of a single gpu this method is required.
        """

        self.enabled = enabled
        if not enabled:
            return

        if not isinstance(params, list):
            params = [params]

        # enable if at least one is zero-param, otherwise a noop
        if not any(is_zero_param(p) for p in params):
            self.enabled = False
            return

        self.params = [p for p in params if hasattr(p, "ds_id")]
        self.src_rank = None
        if modifier_rank is not None:
            if self.params[0].ds_process_group == torch.distributed.group.WORLD:
                self.src_rank = modifier_rank
            else:
                # A group was specified; convert DP rank to global rank
                self.src_rank = _get_global_rank(self.params[0].ds_process_group,
                                                 modifier_rank)
        self.fwd_module = fwd_module
        if self.fwd_module is not None:
            # is a no-op if already registered
            for p in self.params:
                register_external_parameter(self.fwd_module, p)

    def __enter__(self):
        if not self.enabled:
            return
        self.params[0].all_gather(param_list=self.params)

    def __exit__(self, *exc):
        if not self.enabled:
            return
        if self.src_rank is None:
            return

        handles = [
            torch.distributed.broadcast(p,
                                        self.src_rank,
                                        group=p.ds_process_group,
                                        async_op=True) for p in self.params
        ]
        for h in handles:
            h.wait()
        self.params[0].partition(param_list=self.params, has_been_updated=True)
