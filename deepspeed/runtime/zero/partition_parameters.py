# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import os
import types
from typing import Callable, Iterable
from enum import Enum
import functools
import itertools
from typing import List
from collections import defaultdict
import logging
import torch
from torch import Tensor
from deepspeed import comm as dist
from torch.nn import Module
from torch.nn import Parameter

from .linear import zero3_linear_wrap

from deepspeed.utils import groups
import deepspeed
from ..utils import see_memory_usage
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from deepspeed.runtime.zero.utils import assert_ints_same_as_other_ranks, is_zero_param
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.config_utils import get_config_default
from deepspeed.utils import instrument_w_nvtx, logger
from deepspeed.comm.comm import init_distributed
from deepspeed.utils.debug import (debug_param2name_id_shape, debug_param2name_id_shape_device, debug_module2name,
                                   debug_param2name_id, debug_param2name_id_shape_status)
from deepspeed.accelerator import get_accelerator
from ..swap_tensor.partitioned_param_swapper import AsyncPartitionedParameterSwapper, PartitionedParamStatus
from deepspeed.inference.quantization.utils import _quantize_param, WEIGHT_QUANTIZATION_LAYERS, wrap_quantized_functional, wrap_load_from_state_dict

partitioned_param_data_shape = [0]
zero_init_context = 0
top_level_context = None


class NoGatherHandle:

    def __init__(self, param: Parameter) -> None:
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"expected param {param.ds_summary()} to be available")

        if hasattr(param.ds_tensor, "ds_quant_scale"):
            param.data = Init.quantizer_module.dequantize(param.ds_tensor.data, param.ds_tensor.ds_quant_scale).to(
                device=get_accelerator().current_device_name(), non_blocking=True).view(param.ds_shape)
        else:
            param.data = param.ds_tensor.data.to(device=get_accelerator().current_device_name(),
                                                 non_blocking=True).view(param.ds_shape)
        self.__param = param

    def wait(self) -> None:
        if not get_accelerator().is_synchronized_device():
            get_accelerator().current_stream().synchronize()
        self.__param.ds_status = ZeroParamStatus.AVAILABLE


class NoGatherCoalescedHandle:

    def __init__(self, params: List[Parameter]) -> None:
        self.__params = params
        self.__complete = False

        for param in self.__params:
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(f"expected param {param.ds_summary()} to not be available")
            if hasattr(param.ds_tensor, "ds_quant_scale"):
                param.data = Init.quantizer_module.dequantize(param.ds_tensor.data, param.ds_tensor.ds_quant_scale).to(
                    device=get_accelerator().current_device_name(), non_blocking=True).view(param.ds_shape)
            else:
                param.data = param.ds_tensor.data.to(device=get_accelerator().current_device_name(),
                                                     non_blocking=True).view(param.ds_shape)

    @instrument_w_nvtx
    def wait(self) -> None:
        if self.__complete:
            return

        if not get_accelerator().is_synchronized_device():
            get_accelerator().current_stream().synchronize()
        for param in self.__params:
            assert param.ds_status == ZeroParamStatus.INFLIGHT, f"expected param {param.ds_summary()} to be inflight"
            param.ds_status = ZeroParamStatus.AVAILABLE

        self.__complete = True


def _dist_allgather_fn(input_tensor: Tensor, output_tensor: Tensor, group=None):
    return instrument_w_nvtx(dist.allgather_fn)(output_tensor, input_tensor, group=group, async_op=True)


def print_rank_0(message, debug=False, force=False):
    rank = dist.get_rank()
    if rank == 0 and (debug or force):
        print(message)
    # other variations
    # - print for all ranks w/o interleaving
    # printflock(f"[{rank}] {message}")
    # - print to log file per rank
    # log_rank_file(rank, message)


def debug_rank0(msg: str) -> None:
    if dist.get_rank() == 0:
        logger.debug(msg)


def _init_external_params(module):
    if not hasattr(module, '_external_params'):
        module._external_params = {}

        def external_parameters(self):
            return self._external_params.items()

        def all_parameters(self):
            return itertools.chain(self.named_parameters(self, recurse=False), external_parameters(self))

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

    if not hasattr(module, '_external_params') or id(parameter) not in module._external_params:
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


_orig_torch_tensor = torch.tensor
_orig_torch_empty = torch.empty
_orig_torch_zeros = torch.zeros
_orig_torch_ones = torch.ones
_orig_torch_full = torch.full
_orig_torch_arange = torch.arange
_orig_torch_eye = torch.eye
_orig_torch_randn = torch.randn


def zero_wrapper_for_fp_tensor_constructor(fn: Callable, target_fp_dtype: torch.dtype) -> Callable:

    def wrapped_fn(*args, **kwargs) -> Tensor:
        if kwargs.get("device", None) is None:
            kwargs['device'] = torch.device(get_accelerator().device_name(os.environ["LOCAL_RANK"]))
        tensor: Tensor = fn(*args, **kwargs)
        if tensor.is_floating_point():
            tensor.data = tensor.data.to(target_fp_dtype)

        return tensor

    return wrapped_fn


def get_new_tensor_fn_for_dtype(dtype: torch.dtype) -> Callable:

    def new_tensor(cls, *args, **kwargs) -> Tensor:
        device = torch.device(get_accelerator().device_name(os.environ["LOCAL_RANK"]))
        if not args:
            args = (0, )
        tensor = _orig_torch_empty(0, device=device).new_empty(*args, **kwargs)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)

        return tensor

    return new_tensor


# https://stackoverflow.com/a/63851681/9201239
def get_all_subclasses(cls):
    subclass_list = []

    def recurse(cl):
        for subclass in cl.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(cls)

    return set(subclass_list)


@instrument_w_nvtx
def free_param(param: Parameter) -> None:
    """Free underlying storage of a parameter."""
    assert not param.ds_active_sub_modules, param.ds_summary()
    if get_accelerator().on_accelerator(param.data):
        # need to make sure that we don't free the parameter while it is still
        # being used for computation
        if not get_accelerator().is_synchronized_device():
            param.data.record_stream(get_accelerator().current_stream())
    # param.data doesn't store anything meaningful in partitioned state
    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    param.ds_status = ZeroParamStatus.NOT_AVAILABLE


reuse_buffers = False
temp_contiguous_tensor = None
empty_buffers = {}


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    num_module_parameters = 0
    num_module_elements = 0

    def __init__(self, enabled=True, mem_efficient_linear=True, ds_config=None, dtype=None):
        self.mem_efficient_linear = mem_efficient_linear
        self.enabled = enabled
        self._set_dtype(ds_config, dtype)
        assert self.dtype in [
            torch.half, torch.bfloat16, torch.float
        ], f"Invalid data type {self.dtype}, allowed values are [torch.half, torch.bfloat16, torch.float]"
        self.wrapped_cls = set()
        self.skip_init_depth = 0

        self.quantized_initialization = None
        if ds_config is not None and ds_config.weight_quantization_config and ds_config.weight_quantization_config.quantized_initialization:
            self.quantized_initialization = ds_config.weight_quantization_config.quantized_initialization

    def __enter__(self):
        if not self.enabled:
            return

        global zero_init_context
        if zero_init_context == 0:
            self.patch_init_and_builtins()
            global top_level_context
            top_level_context = self

        zero_init_context += 1

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        global zero_init_context
        zero_init_context -= 1

        # Exiting the top level context
        if zero_init_context == 0:
            self.unpatch_init_and_builtins()
            global top_level_context
            top_level_context = None

            if dist.get_rank() == 0:
                billion_elems = InsertPostInitMethodToModuleSubClasses.num_module_elements / 1e9
                num_params = InsertPostInitMethodToModuleSubClasses.num_module_parameters
                logger.info(
                    f"finished initializing model - num_params = {num_params}, num_elems = {billion_elems:.2f}B")

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _set_dtype(self, ds_config, dtype):
        if ds_config is not None and dtype is None:
            if ds_config.bfloat16_enabled and ds_config.fp16_enabled:
                raise RuntimeError("bfloat16 and fp16 cannot be enabled at once")

            if ds_config.bfloat16_enabled:
                self.dtype = torch.bfloat16
            elif ds_config.fp16_enabled:
                self.dtype = torch.half
            else:
                self.dtype = torch.float
        else:
            self.dtype = dtype or torch.float16 if get_accelerator().is_fp16_supported(
            ) else torch.bfloat16 if get_accelerator().is_bf16_supported else torch.float32

    def patch_init_and_builtins(self):

        def apply_with_gather(orig_module_apply_fn: Callable) -> Callable:
            """many models make use of child modules like Linear or Embedding which
            perform their own weight initialization in their __init__ methods,
            but will then have more weight initialization in a parent module's __init__
            method that modifies weights of child modules, which is typically done
            using the Module.apply method.

            since the Init context manager partitions child modules immediately after
            they are initialized, without modifying apply we would entirely skip
            any initialization done by parent modules.

            to get around this issue, we wrap the function passed to Module.apply
            so that the applied function is applied to child modules correctly.
            """

            def get_wrapped_fn_to_apply(fn_to_apply: Callable) -> Callable:
                if hasattr(fn_to_apply, "wrapped"):
                    return fn_to_apply

                @functools.wraps(fn_to_apply)
                def wrapped_fn_to_apply(module_to_apply_fn_to: Module) -> None:
                    """gathers parameters before calling apply function. afterwards
                    parameters are broadcasted to ensure consistency across all ranks
                    then re-partitioned.

                    takes the following steps:
                    1. allgathers parameters for the current module being worked on
                    2. calls the original function
                    3. broadcasts root rank's parameters to the other ranks
                    4. re-partitions the parameters
                    """

                    # TODO Delay error checking for dangling partitioned parameters to post module init
                    # raise RuntimeError(f"not all parameters for {module_to_apply_fn_to.__class__.__name__}, "
                    #                    f"were zero params, is it possible that the parameters were "
                    #                    f"overwritten after they were initialized? "
                    #                    f"params: {[p for p in module_to_apply_fn_to.parameters(recurse=False)]} ")

                    params_to_apply_fn_to: Iterable[Parameter] = list(
                        sorted([p for p in module_to_apply_fn_to.parameters(recurse=False) if is_zero_param(p)],
                               key=lambda p: p.ds_id))

                    for param in params_to_apply_fn_to:
                        param.all_gather()

                    fn_to_apply(module_to_apply_fn_to)

                    for param in params_to_apply_fn_to:
                        dist.broadcast(param.data, 0, group=param.ds_process_group)

                    for param in params_to_apply_fn_to:
                        param.partition(has_been_updated=True)

                wrapped_fn_to_apply.wrapped = True

                return wrapped_fn_to_apply

            @functools.wraps(orig_module_apply_fn)
            def wrapped_apply(module: Module, fn_to_apply: Callable) -> None:
                orig_module_apply_fn(module, get_wrapped_fn_to_apply(fn_to_apply))

            return wrapped_apply

        def hook_for_skip_init(module):
            # this function is intended for handling the logic of torch.nn.utils.skip_init
            # skip_init:module_cls(*args, **kwargs).to_empty(device=final_device), where kwargs['device']='meta'
            # the function call occurs between module_cls(*args, **kwargs) and to_empty(device=final_device).
            def partition_after_empty_init(f):

                @functools.wraps(f)
                def wrapper(module, *args, **kwargs):
                    _module = f(module, *args, **kwargs)
                    # here is the post-hook for module.apply(empty_like...)
                    # after module.apply(empty_like...), the module has completed its empty init on real device
                    # since skip_init won't involve any computations or weight adjustments, we can directly utilize post_init
                    self._post_init_method(_module)
                    return _module

                return wrapper

            def post_wrapper_to_empty(f):
                # append some wrapper restoration after to_empty() call
                @functools.wraps(f)
                def wrapper(*args, **kwargs):
                    res = f(*args, **kwargs)
                    # restore _apply hook
                    for subclass in get_all_subclasses(torch.nn.modules.module.Module):
                        _disable_class_apply(subclass)
                    # self restore
                    module.to_empty = f
                    return res

                return wrapper

            def _enable_class_apply(cls):
                cls._old_apply_of_skip_init_hook = cls._apply
                cls._apply = partition_after_empty_init(cls._apply)

            def _disable_class_apply(cls):
                cls._apply = cls._old_apply_of_skip_init_hook

            # add hooks for to_empty: apply_(empty_like)
            for subclass in get_all_subclasses(torch.nn.modules.module.Module):
                _enable_class_apply(subclass)

            # add a restore hook when exiting skip_init
            module.to_empty = post_wrapper_to_empty(module.to_empty)

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

                print_rank_0(f'Before initializing {module.__class__.__name__}', force=False)

                is_child_module = False
                if not hasattr(module, "_ds_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    is_child_module = True
                    setattr(module, "_ds_child_entered", True)

                init_on_meta = 'device' in kwargs and kwargs['device'] == 'meta'
                if init_on_meta:
                    self.skip_init_depth += 1

                f(module, *args, **kwargs)
                if init_on_meta and self.skip_init_depth == 1:
                    # check and handle the logic of empty_init
                    hook_for_skip_init(module)
                if is_child_module:
                    # child's __init__ is done, now we can run a single post_init on the child object
                    delattr(module, "_ds_child_entered")

                    print_rank_0(f'Running post_init for {module.__class__.__name__}', force=False)
                    if self.skip_init_depth == 0:
                        self._post_init_method(module)

                print_rank_0(f'After initializing followed by post init for {module.__class__.__name__}', force=False)
                if init_on_meta:
                    self.skip_init_depth -= 1

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls._old_init = cls.__init__
            cls.__init__ = partition_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module recursively
        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            _enable_class(subclass)

        # holding onto some methods so we can put them back the way they were in __exit__
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.nn.modules.module.Module._old_apply = torch.nn.modules.module.Module.apply
        torch.Tensor.__old_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        if Init.override_module_apply:
            torch.nn.modules.module.Module.apply = apply_with_gather(torch.nn.modules.module.Module._old_apply)

        self._add_tensor_creation_wrappers()

        if self.mem_efficient_linear:
            print_rank_0(
                "nn.functional.linear has been overridden with a more memory efficient version. This will persist unless manually reset.",
                force=False)
            self.linear_bk = torch.nn.functional.linear
            torch.nn.functional.linear = zero3_linear_wrap

            if self.quantized_initialization:
                print_rank_0("nn.functional.linear has been overridden with quantized linear version.", force=False)
                torch.nn.functional.linear = wrap_quantized_functional(torch.nn.functional.linear)
                torch.nn.functional.embedding = wrap_quantized_functional(torch.nn.functional.embedding)
                for cls in WEIGHT_QUANTIZATION_LAYERS:
                    cls._load_from_state_dict = wrap_load_from_state_dict(cls._load_from_state_dict)

                logger.info("Enable Zero3 engine with INT4 quantization.")

        self.patched = True

    def unpatch_init_and_builtins(self):
        if self.patched:

            def _disable_class(cls):
                cls.__init__ = cls._old_init

            for subclass in get_all_subclasses(torch.nn.modules.module.Module):
                _disable_class(subclass)

            # putting methods back the way we found them
            torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass
            if Init.override_module_apply:
                torch.nn.modules.module.Module.apply = torch.nn.modules.module.Module._old_apply

            self._remove_tensor_creation_wrappers()

            self.patched = False

    def _add_tensor_creation_wrappers(self):
        torch.Tensor.__new__ = get_new_tensor_fn_for_dtype(self.dtype)
        torch.tensor = zero_wrapper_for_fp_tensor_constructor(_orig_torch_tensor, self.dtype)
        torch.empty = zero_wrapper_for_fp_tensor_constructor(_orig_torch_empty, self.dtype)
        torch.zeros = zero_wrapper_for_fp_tensor_constructor(_orig_torch_zeros, self.dtype)
        torch.ones = zero_wrapper_for_fp_tensor_constructor(_orig_torch_ones, self.dtype)
        torch.full = zero_wrapper_for_fp_tensor_constructor(_orig_torch_full, self.dtype)
        torch.arange = zero_wrapper_for_fp_tensor_constructor(_orig_torch_arange, self.dtype)
        torch.eye = zero_wrapper_for_fp_tensor_constructor(_orig_torch_eye, self.dtype)
        torch.randn = zero_wrapper_for_fp_tensor_constructor(_orig_torch_randn, self.dtype)

    def _remove_tensor_creation_wrappers(self):
        torch.Tensor.__new__ = torch.Tensor.__old_new__
        torch.tensor = _orig_torch_tensor
        torch.empty = _orig_torch_empty
        torch.zeros = _orig_torch_zeros
        torch.ones = _orig_torch_ones
        torch.full = _orig_torch_full
        torch.arange = _orig_torch_arange
        torch.eye = _orig_torch_eye
        torch.randn = _orig_torch_randn


def shutdown_init_context():
    """
    This function is used to initialize deepspeed engine inside the context of Init.
    We need to remove the wrappers but keep the context.
    """
    if top_level_context:
        top_level_context.unpatch_init_and_builtins()


def restore_init_context():
    """
    This function is used to restore the wrappers after deepspeed engine is initialized.
    """
    if top_level_context:
        top_level_context.patch_init_and_builtins()


class AllGatherHandle:

    def __init__(self, handle, param: Parameter, quantization=None) -> None:
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"expected param {param.ds_summary()} to be available")

        self.__handle = handle
        self.__param = param
        self.__quantization = quantization

    def wait(self) -> None:
        instrument_w_nvtx(self.__handle.wait)()
        if self.__quantization:
            instrument_w_nvtx(self.__quantization.quant_handle.wait)()
            self.__param.data = self.__quantization.backend.dequantize(
                self.__quantization.quantized_param, self.__quantization.scale_buffer).to(self.__param.device)
        self.__param.ds_status = ZeroParamStatus.AVAILABLE


class AllGatherCoalescedHandle:

    def __init__(
        self,
        allgather_handle,
        params: List[Parameter],
        partitions: List[Tensor],
        world_size: int,
        use_secondary_tensor=False,
        quantization=None,
    ) -> None:
        self.allgather_handle = allgather_handle
        self.params = params
        self.partitions = partitions
        self.world_size = world_size
        self.use_secondary_tensor = use_secondary_tensor
        self.complete = False
        self.quantization = quantization

        for param in self.params:
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(f"expected param {param.ds_summary()} to not be available")

    @instrument_w_nvtx
    def wait(self) -> None:
        if self.complete:
            return

        instrument_w_nvtx(self.allgather_handle.wait)()

        if self.quantization:
            instrument_w_nvtx(self.quantization.quant_handle.wait)()
            flat_tensor = self.quantization.backend.dequantize(
                self.quantization.quantized_param, self.quantization.scale_buffer).to(self.params[0].device)

            self.partitions: List[Parameter] = []
            for i in range(self.world_size):
                self.partitions.append(
                    flat_tensor.narrow(0, self.quantization.partition_sz * i, self.quantization.partition_sz))

        # split the single tensor out into individual tensors
        param_offset = 0
        for param in self.params:
            assert param.ds_status == ZeroParamStatus.INFLIGHT, f"expected param {param.ds_summary()} to be inflight"
            partitions: List[Tensor] = []
            ds_tensor_numel = param.ds_tensor.ds_numel
            if self.use_secondary_tensor:
                ds_tensor_numel *= param.ds_secondary_tensor_num_of_groups
            for rank in range(self.world_size):
                param_start = rank * ds_tensor_numel
                if param_start < param.ds_numel:
                    part_to_copy = self.partitions[rank].narrow(0, param_offset,
                                                                min(param.ds_numel - param_start, ds_tensor_numel))
                    partitions.append(part_to_copy)
            param.data = instrument_w_nvtx(torch.cat)(partitions).view(param.ds_shape)
            param.ds_status = ZeroParamStatus.AVAILABLE

            for part_to_copy in partitions:
                if not get_accelerator().is_synchronized_device():
                    part_to_copy.record_stream(get_accelerator().current_stream())

            param_offset += ds_tensor_numel

        self.complete = True


class MultipleAllGatherHandles:

    def __init__(self, handles: List[AllGatherCoalescedHandle]):
        self.handles = handles

    def wait(self) -> None:
        for handle in self.handles:
            handle.wait()


class QuantizationInfo:
    # a placeholder object to store all quant related vars used in handles
    def __init__(self) -> None:
        self.quantized_param = None
        self.backend = None
        self.quant_handle = None
        self.scale_buffer = None


class CUDAQuantizer:
    async_flag = True
    target_group_size = 8000  # the optimal size is 4k, so we set the target to be below 8k
    group_size_cache = dict()
    quantizer_cuda_module = None

    def __init__(self) -> None:
        if CUDAQuantizer.quantizer_cuda_module is None:
            CUDAQuantizer.quantizer_cuda_module = deepspeed.ops.op_builder.QuantizerBuilder().load()

    def quantize(self, param, groups=None):
        if groups is None:
            try:
                groups = self.group_size_cache[param.numel()]
            except KeyError:
                groups = math.ceil(param.numel() / self.target_group_size)
                while groups < param.numel():
                    if param.numel() % (8 * groups) == 0:
                        break
                    groups += 1
                while True:
                    if param.numel() % (8 * groups * 2) == 0 and param.numel(
                    ) / groups > self.target_group_size:  #hard limit of 16k group_size
                        groups *= 2
                    else:
                        break
                assert (
                    param.numel() % (8 * groups) == 0
                ), f"Qantized weight requires the number of weights be a multiple of 8. Yet {param.numel()} cannot be divided by 8*{groups}"
                assert (param.numel() / groups < 16000), f"{param.numel()} / {groups} is larger than 16k"
                assert param.numel(
                ) > groups, f"Adaptive grouping algorithm cannot find a group size for input tensor of size {param.numel()}"
                self.group_size_cache[param.numel()] = groups
        return self.quantizer_cuda_module.quantize(param.to(get_accelerator().device_name()), groups, 8,
                                                   self.quantizer_cuda_module.Symmetric)

    def dequantize(self, quantized_param, scale):
        return self.quantizer_cuda_module.dequantize(quantized_param, scale, scale.numel(), 8,
                                                     self.quantizer_cuda_module.Symmetric)


def _no_gather_coalesced(params: Iterable[Parameter]) -> AllGatherCoalescedHandle:
    for param in params:
        if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
            raise RuntimeError(f"expect param.ds_status == ZeroParamStatus.NOT_AVAILABLE, got{param.ds_summary()}")
        param.ds_status = ZeroParamStatus.INFLIGHT

    params = sorted(params, key=lambda p: p.ds_id)
    if len(params) == 1:
        param, = params
        return NoGatherHandle(param)
    return NoGatherCoalescedHandle(params)


# Replaces all parameters in module with Scattered Parameters
class Init(InsertPostInitMethodToModuleSubClasses):
    param_id = 0
    param_persistence_threshold = get_config_default(DeepSpeedZeroConfig, "param_persistence_threshold")
    model_persistence_threshold = get_config_default(DeepSpeedZeroConfig, "model_persistence_threshold")
    num_persisted_parameters = 0
    num_persisted_elements = 0
    apply_param_persistence = False
    override_module_apply = get_config_default(DeepSpeedZeroConfig, "override_module_apply")

    def __init__(
        self,
        module=None,
        data_parallel_group=None,
        mem_efficient_linear=True,
        remote_device=None,
        pin_memory=False,
        config_dict_or_path=None,
        config=None,
        enabled=True,
        dtype=None,
        mpu=None,
        zero_param_parallel_group=None,
        zero_quantized_weights=False,
        zero_quantized_nontrainable_weights=False,
        sequence_data_parallel_group=None,
        param_swapper=None,
    ):
        """A context to enable massive model construction for training with
        ZeRO-3. Models are automatically partitioned (or, sharded) across the
        system and converted to half precision.

        Args:
            module (``torch.nn.Module``, optional): If provided, partition the model as
                if it was constructed in the context.
            data_parallel_group (``deepspeed.comm`` process group, optional):
                The group of processes to partition among. Defaults to all processes.
            mem_efficient_linear (bool, optional): Replace
                torch.nn.functional.linear with an implementation that allows
                DeepSpeed to partition parameters. Defaults to ``True``.
            remote_device (string, optional): The initial device to store model
                weights e.g., ``cpu``, ``nvme``. Passing ``"cpu"`` will create the model in CPU
                memory. The model may still be moved to GPU based on the
                offload settings for training. Defaults to param offload device if a config is
                defined, otherwise GPU.
            pin_memory (bool, optional): Potentially increase performance by
                using pinned memory for model weights. ``remote_device`` must be
                ``"cpu"``. Defaults to pin_memory value in config, otherwise ``False``.
            config_dict_or_path (dict or ``json file``, optional): If provided, provides configuration
                for swapping fp16 params to NVMe.
            config (dict or ``json file``, optional): Deprecated, use config_dict_or_path instead.
            enabled (bool, optional): If ``False``, this context has no
                effect. Defaults to ``True``.
            dtype (``dtype``, optional): Can be used to change the data type of the parameters.
                Supported options are ``torch.half`` and ``torch.float``. Defaults to ``None``
            mpu (``object``, optional): A model parallelism unit object that implements get_{model,data}_parallel_{rank,group,world_size}.
            zero_param_parallel_group(``object``, optional): Parallel (comm) group for dual partitioning of ZeRO params.
            zero_quantized_weights (bool, optional): If ``True``, turn on quantized weights in all gather weights. Default is ``False``
            zero_quantized_nontrainable_weights (bool, optional): If ``True``, nontrainable weights will be stored in quantized format. Default is ``False``
            param_swapper (``deepspeed.runtime.swap_tensor.partitioned_param_swapper.AsyncPartitionedParameterSwapper``, optional): [Experimental] Use existing parameter swapper. Defaults to ``None``.
                This argument will be removed in the near future.

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
            Initializes ``deepspeed.comm`` if it has not already been done so.
            See :meth:`deepspeed.init_distributed` for more information.

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
        if config is not None:
            config_dict_or_path = config
            logger.warning(
                f'zero.Init: the `config` argument is deprecated. Please use `config_dict_or_path` instead.')
        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(config_dict_or_path,
                                                              mpu) if config_dict_or_path is not None else None
        if _ds_config is not None:
            if _ds_config.zero_config.memory_efficient_linear and _ds_config.compile_config.enabled:
                # memory_efficient_linear displays numerous errors when torch.compile is enabled.
                # Refer to https://github.com/pytorch/pytorch/issues/119059 for details.
                # Further investigation into performance is necessary, even after resolving this issue because
                # the `memory_efficient_linear` module may lead to more graph breaks compared to the original implementation.
                logger.warning(f'memory_efficient_linear is disabled when torch.compile is enabled.')
                mem_efficient_linear = False
            else:
                mem_efficient_linear = _ds_config.zero_config.memory_efficient_linear

        super().__init__(enabled=enabled, mem_efficient_linear=mem_efficient_linear, ds_config=_ds_config, dtype=dtype)
        if not dist.is_initialized():
            init_distributed()
            assert dist.is_initialized(), "Parameters cannot be scattered without initializing deepspeed.comm"

        if data_parallel_group is None and sequence_data_parallel_group is None:
            self.ds_process_group = dist.get_world_group()
        elif sequence_data_parallel_group is not None:
            self.ds_process_group = sequence_data_parallel_group
        elif data_parallel_group is not None:
            self.ds_process_group = data_parallel_group
        else:  # both given
            raise ValueError(
                "Both 'data_parallel_group' and 'sequence_data_parallel_group' were specified. Please provide only one of these arguments."
            )

        self.rank = dist.get_rank(group=self.ds_process_group)
        self.dp_world_size = dist.get_world_size(group=self.ds_process_group)

        self.zero_param_process_group = zero_param_parallel_group
        if _ds_config is not None and _ds_config.zero_config.zero_hpz_partition_size > 1 and self.zero_param_process_group is None:
            groups._create_zero_param_parallel_group(_ds_config.zero_config.zero_hpz_partition_size)
            self.zero_param_process_group = groups._get_zero_param_intra_parallel_group()

        self.num_ranks_in_param_group = self.dp_world_size
        self.rank_in_group = self.rank
        self.num_param_groups = 1

        if self.zero_param_process_group is not None:
            self.num_ranks_in_param_group = groups._get_zero_param_intra_parallel_group_world_size()
            self.num_param_groups = int(self.dp_world_size / self.num_ranks_in_param_group)
            self.rank_in_group = groups._get_zero_param_intra_parallel_rank_in_mygroup()
            print_rank_0(f"hpZeRO group size: {self.num_ranks_in_param_group}", force=True)

            logger.debug(
                "hpZeRO partition parameter my rank in world {} my rank in group {} ranks in my param partition group: {} "
                .format(self.rank, self.rank_in_group, groups._get_zero_param_intra_parallel_group_ranks()))

        # Local device is the device where the parameters are consumed, must be default device.
        # It is the device where parameters are fully instantiated using allgather
        self.local_device = torch.device(get_accelerator().device_name(os.environ["LOCAL_RANK"]))
        get_accelerator().set_device(self.local_device)

        self.quantized_weights = zero_quantized_weights
        if _ds_config is not None and _ds_config.zero_config.zero_quantized_weights and not self.quantized_weights:
            self.quantized_weights = _ds_config.zero_config.zero_quantized_weights
        self.quantized_nontrainable_weights = zero_quantized_nontrainable_weights
        if _ds_config is not None and _ds_config.zero_config.zero_quantized_nontrainable_weights and not self.quantized_nontrainable_weights:
            self.quantized_nontrainable_weights = _ds_config.zero_config.zero_quantized_nontrainable_weights

        self.module = module
        if (self.quantized_weights or self.quantized_nontrainable_weights):
            self.quantizer_module = CUDAQuantizer()
            print_rank_0(f'Using quantizer for weights: {self.quantizer_module.__class__.__name__}', force=True)

        if _ds_config is not None:
            Init.override_module_apply = _ds_config.zero_config.override_module_apply

            if _ds_config.zero_config.offload_param is not None:
                remote_device = _ds_config.zero_config.offload_param.device
                pin_memory = _ds_config.zero_config.offload_param.pin_memory

        self._validate_remote_device(remote_device, _ds_config)

        # Remote device is the device where parameter partitions are stored
        # It can be same as local_device or it could be CPU or NVMe.
        self.remote_device = self.local_device if remote_device in [None, OffloadDeviceEnum.none] else remote_device
        self.pin_memory = pin_memory if (self.remote_device in [OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme
                                                                ]) else False

        # Enable fp16 param swapping to NVMe
        if self.remote_device == OffloadDeviceEnum.nvme:
            self.param_swapper = param_swapper or AsyncPartitionedParameterSwapper(_ds_config, self.dtype)
        else:
            self.param_swapper = None

        # If we are provided an already-allocated module to prepare.
        if module is not None:
            assert isinstance(module, torch.nn.Module)
            self._convert_to_zero_parameters(module.parameters(recurse=True))

        self.use_all_gather_into_tensor = dist.has_all_gather_into_tensor()
        if not self.use_all_gather_into_tensor:
            logger.info(f"all_gather_into_tensor API is not available in torch {torch.__version__}")

    def _update_persist_config(self, ds_config):
        Init.apply_param_persistence = True
        Init.param_persistence_threshold = ds_config.zero_config.param_persistence_threshold
        Init.model_persistence_threshold = ds_config.zero_config.model_persistence_threshold // self.num_partitions

    def _zero_init_param(self, param):
        self._convert_to_deepspeed_param(param)
        if dist.get_world_group() == self.get_dp_process_group():
            dist.broadcast(param.data, 0, self.get_dp_process_group())
        else:
            dist.broadcast(param.data, dist.get_global_rank(self.get_dp_process_group(), 0),
                           self.get_dp_process_group())
        param.partition()

    def _convert_to_zero_parameters(self, param_list):
        for param in param_list:
            if is_zero_param(param):
                continue

            param.data = param.data.to(self.local_device)
            self._zero_init_param(param)

    def _validate_remote_device(self, remote_device, ds_config):
        if ds_config is not None:
            if remote_device in [None, OffloadDeviceEnum.cpu]:
                if ds_config.zero_config.offload_param is not None:
                    offload_param_device = ds_config.zero_config.offload_param.device
                    assert offload_param_device != OffloadDeviceEnum.nvme, \
                        f"'device' in DeepSpeed Config cannot be {offload_param_device} if remote device is {remote_device}."

            if remote_device == OffloadDeviceEnum.nvme:
                assert ds_config.zero_config.offload_param is not None, \
                f'"offload_param" must be defined in DeepSpeed Config if remote device is {OffloadDeviceEnum.nvme}.'

                assert ds_config.zero_config.offload_param.nvme_path is not None, \
                f'"nvme_path" in DeepSpeed Config cannot be None if remote device is {OffloadDeviceEnum.nvme}'

    def _post_init_method(self, module):
        #see_memory_usage(f"Before converting params in {module.__class__.__name__}", force=False)
        print_rank_0(f'Converting Params in {module.__class__.__name__}', force=False)
        see_memory_usage(f"Before converting and partitioning params in {module.__class__.__name__}", force=False)

        for name, param in module.named_parameters(recurse=False):
            print_rank_0(f'Analyzing param {name} in {module.__class__.__name__}', force=False)
            InsertPostInitMethodToModuleSubClasses.num_module_parameters += 1
            InsertPostInitMethodToModuleSubClasses.num_module_elements += param.numel()
            if not is_zero_param(param):
                if not get_accelerator().on_accelerator(param):
                    param.data = param.data.to(self.local_device)

                if name == 'weight' and self.quantized_initialization and type(module) in WEIGHT_QUANTIZATION_LAYERS:
                    _quantize_param(param, self.quantized_initialization)

                self._zero_init_param(param)
                print_rank_0(
                    f"Partitioning param {debug_param2name_id_shape(param)} module={debug_module2name(module)}")

        see_memory_usage(
            f"Param count {InsertPostInitMethodToModuleSubClasses.num_module_elements}. After converting and partitioning params in {module.__class__.__name__}",
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
        param.ds_active_sub_modules = set()

        # If this flag is true, then the parameters are replicated throughput training
        # And only partitioned before the step
        if Init.apply_param_persistence and param.ds_numel <= Init.param_persistence_threshold and Init.num_persisted_elements + param.ds_numel <= Init.model_persistence_threshold:
            param.ds_persist = True
            Init.num_persisted_parameters += 1
            Init.num_persisted_elements += param.ds_numel
        else:
            param.ds_persist = False

        param.is_external_param = False

        # The group that the parameter is scattered across.
        param.ds_process_group = self.ds_process_group

        # Stores the secondary partitioned copy of the tensor
        param.ds_secondary_tensor = None

        #Process group for secondary partition all (group) gather
        param.ds_zero_param_process_group = self.zero_param_process_group
        param.ds_secondary_tensor_group_size = self.num_ranks_in_param_group
        param.ds_secondary_tensor_num_of_groups = self.num_param_groups

        # This is set to the Async Param swapper if remote device is nvme
        # else this is set to None
        param.nvme_swapper = self.param_swapper

        # DeepSpeed Param ID
        param.ds_id = Init.param_id
        Init.param_id += 1

        def all_gather(param_list=None, async_op=False, hierarchy=0):
            cls = param
            if param_list is None:
                param_list = [cls]
            return self._all_gather(param_list, async_op=async_op, hierarchy=hierarchy)

        def _all_gather_dtype(dtype, params, world_size, rank_in_group, ds_process_group):
            partition_sz = sum(p.ds_tensor.ds_numel for p in params)

            use_secondary_tensor = params[0].ds_secondary_tensor is not None

            if use_secondary_tensor:
                partition_sz = sum(p.ds_tensor.ds_numel * p.ds_secondary_tensor_num_of_groups for p in params)

            flat_tensor = torch.empty(partition_sz * world_size,
                                      dtype=dtype,
                                      device=get_accelerator().current_device_name(),
                                      requires_grad=False)

            partitions: List[Parameter] = []
            for i in range(world_size):
                partitions.append(flat_tensor.narrow(0, partition_sz * i, partition_sz))

            if use_secondary_tensor:
                instrument_w_nvtx(
                    torch.cat)([p.ds_secondary_tensor.to(get_accelerator().current_device_name()) for p in params],
                               out=partitions[rank_in_group])
            else:
                instrument_w_nvtx(torch.cat)([p.ds_tensor.to(get_accelerator().current_device_name()) for p in params],
                                             out=partitions[rank_in_group])
            handle = _dist_allgather_fn(partitions[rank_in_group], flat_tensor, ds_process_group)
            #Fix get_partition_dp_group(params[0]))

            return AllGatherCoalescedHandle(
                allgather_handle=handle,
                params=params,
                partitions=partitions,
                world_size=world_size,
                use_secondary_tensor=use_secondary_tensor,
            )

        @instrument_w_nvtx
        def all_gather_coalesced(params: Iterable[Parameter],
                                 safe_mode: bool = False,
                                 quantize: bool = False) -> AllGatherCoalescedHandle:

            # fetches from nvme if the partition is not available and in nvme
            self._ensure_availability_of_partitioned_params(params)

            if self.num_partitions == 1:
                return _no_gather_coalesced(params)

            for param in params:
                if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                    raise RuntimeError(param.ds_summary())
                param.ds_status = ZeroParamStatus.INFLIGHT

            #use appropriate all gather process group
            ds_process_group = self.ds_process_group
            rank_in_group = self.rank
            world_size = self.dp_world_size
            use_secondary_tensor = params[0].ds_secondary_tensor is not None
            if self.zero_param_process_group and use_secondary_tensor:
                ds_process_group = self.zero_param_process_group  #intragroup
                rank_in_group = self.rank_in_group
                world_size = self.num_ranks_in_param_group

            #pprint(dir(ds_process_group))
            # ensure that each rank has params in same order. the allgather
            # is done by flattening the parameter list into a single tensor that
            # can be allgathered in a single call - this means that if each rank
            # gives a list of the same parameters in a different order we will
            # silently get incorrect parameter values, and have very difficult
            # to debug correctness issues.
            params = sorted(params, key=lambda p: p.ds_id)

            if logger.isEnabledFor(logging.DEBUG):
                debug_rank0(f"-allgather_coalesced: {[p.ds_id for p in params]}")

            if safe_mode:
                # ensure that same list (with same ordering) of parameters are
                # being allgathered across all ranks, otherwise could mix
                # data between tensors.
                assert_ints_same_as_other_ranks([p.ds_id for p in params])
                # ensure that tensors from each rank agree on the same ds_numel
                # otherwise could mix data between tensors.
                assert_ints_same_as_other_ranks([p.ds_tensor.ds_numel for p in params])

            if len(params) == 1:
                # have an opportunity to avoid some intermediate memory allocations
                param = params[0]
                buffer_size = math.ceil(param.ds_numel / world_size) * world_size
                if use_secondary_tensor:
                    buffer_size = param.ds_secondary_tensor.shape[0] * world_size  #make sure out is appropriately sized

                param_ds_tensor = param.ds_secondary_tensor if use_secondary_tensor else param.ds_tensor
                param_buffer = torch.empty(
                    buffer_size,
                    dtype=param_ds_tensor.dtype if not quantize else torch.int8,
                    device=get_accelerator().current_device_name(),
                    requires_grad=False,
                )
                if not quantize:
                    handles = _dist_allgather_fn(
                        param_ds_tensor.to(get_accelerator().current_device_name()),
                        param_buffer,
                        ds_process_group,
                    )
                    param.data = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape).to(param.device)
                    return AllGatherHandle(handles, param)
                else:
                    if hasattr(param_ds_tensor, "ds_quant_scale"):
                        scales = param_ds_tensor.ds_quant_scale
                        quantized_param = param_ds_tensor.data
                    else:
                        quantized_param, scales = self.quantizer_module.quantize(param_ds_tensor)
                    handle = _dist_allgather_fn(quantized_param.to(get_accelerator().current_device_name()),
                                                param_buffer, ds_process_group)

                    quant_scale_buffer = torch.empty(
                        scales.numel() * world_size,
                        dtype=scales.dtype,
                        device=get_accelerator().current_device_name(),
                        requires_grad=False,
                    )
                    quant_handle = _dist_allgather_fn(scales.to(get_accelerator().current_device_name()),
                                                      quant_scale_buffer, ds_process_group)
                    quant_info = QuantizationInfo()
                    quant_info.quantized_param = param_buffer.narrow(0, 0, param.ds_numel).view(param.ds_shape).to(
                        param.device)
                    quant_info.backend = self.quantizer_module
                    quant_info.quant_handle = quant_handle
                    quant_info.scale_buffer = quant_scale_buffer
                    return AllGatherHandle(handle, param, quantization=quant_info)

            else:
                if not quantize:
                    dtype_params = defaultdict(list)
                    for p in params:
                        dtype_params[p.ds_tensor.dtype].append(p)
                    handles = []
                    for dtype, params in dtype_params.items():
                        handles.append(_all_gather_dtype(dtype, params, world_size, rank_in_group, ds_process_group))

                    return MultipleAllGatherHandles(handles)

                else:
                    partition_sz = sum(p.ds_tensor.ds_numel for p in params)

                    if use_secondary_tensor:
                        partition_sz = sum(p.ds_tensor.ds_numel * p.ds_secondary_tensor_num_of_groups for p in params)

                    flat_tensor = torch.empty(partition_sz * world_size,
                                              dtype=torch.int8,
                                              device=get_accelerator().current_device_name(),
                                              requires_grad=False)

                    if use_secondary_tensor:
                        if hasattr(params[0].ds_secondary_tensor, "ds_quant_scale"):
                            quantized_param = instrument_w_nvtx(torch.cat)([
                                p.ds_secondary_tensor.data.to(get_accelerator().current_device_name()) for p in params
                            ])
                            scales = instrument_w_nvtx(torch.cat)([
                                p.ds_secondary_tensor.ds_quant_scale.to(get_accelerator().current_device_name())
                                for p in params
                            ])
                        else:
                            quantized_param, scales = self.quantizer_module.quantize(
                                instrument_w_nvtx(torch.cat)([
                                    p.ds_secondary_tensor.to(get_accelerator().current_device_name()) for p in params
                                ]))
                    else:
                        if hasattr(params[0].ds_tensor, "ds_quant_scale"):
                            quantized_param = instrument_w_nvtx(torch.cat)(
                                [p.ds_tensor.data.to(get_accelerator().current_device_name()) for p in params])
                            scales = instrument_w_nvtx(torch.cat)([
                                p.ds_tensor.ds_quant_scale.to(get_accelerator().current_device_name()) for p in params
                            ])
                        else:
                            quantized_param, scales = self.quantizer_module.quantize(
                                instrument_w_nvtx(torch.cat)(
                                    [p.ds_tensor.to(get_accelerator().current_device_name()) for p in params]))
                    quant_scale_buffer = torch.empty(
                        scales.numel() * world_size,
                        dtype=torch.float32,
                        device=get_accelerator().current_device_name(),
                        requires_grad=False,
                    )
                    handle = _dist_allgather_fn(quantized_param, flat_tensor, ds_process_group)
                    quant_handle = _dist_allgather_fn(scales, quant_scale_buffer, ds_process_group)
                    quant_info = QuantizationInfo()
                    quant_info.quantized_param = flat_tensor
                    quant_info.backend = self.quantizer_module
                    quant_info.quant_handle = quant_handle
                    quant_info.scale_buffer = quant_scale_buffer
                    quant_info.partition_sz = partition_sz
                    quant_info.world_size = world_size
                    return AllGatherCoalescedHandle(
                        allgather_handle=handle,
                        params=params,
                        partitions=None,
                        world_size=world_size,
                        use_secondary_tensor=use_secondary_tensor,
                        quantization=quant_info,
                    )

        def partition(param_list=None, hierarchy=0, has_been_updated=False):
            cls = param
            print_rank_0(f"{'--'*hierarchy}----Partitioning param {debug_param2name_id_shape_device(cls)}",
                         force=False)
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

        def partition_gradients(param_list=None, partition_buffers=None, hierarchy=0, accumulate=False):
            cls = param
            print_rank_0(
                f"{'--'*hierarchy}----Partitioning param gradient with id {debug_param2name_id_shape_device(cls)}")
            if param_list is None:
                param_list = [cls]
                if isinstance(partition_buffers, torch.Tensor):
                    partition_buffers = [partition_buffers]

            self._partition_gradients(param_list, partition_buffers=partition_buffers, accumulate=accumulate)

        def aligned_size():
            return self._aligned_size(param)

        def padding_size():
            return self._padding_size(param)

        def partition_numel():
            return self._partition_numel(param)

        def item_override():
            param.all_gather()
            return param._orig_item()

        def ds_summary(slf: torch.Tensor, use_debug_name: bool = False) -> dict:
            return {
                "id": debug_param2name_id(slf) if use_debug_name else slf.ds_id,
                "status": slf.ds_status.name,
                "numel": slf.numel(),
                "ds_numel": slf.ds_numel,
                "shape": tuple(slf.shape),
                "ds_shape": tuple(slf.ds_shape),
                "requires_grad": slf.requires_grad,
                "grad_shape": tuple(slf.grad.shape) if slf.grad is not None else None,
                "persist": slf.ds_persist,
                "active_sub_modules": slf.ds_active_sub_modules,
                "ds_tensor.shape": slf.ds_tensor.shape if slf.ds_tensor is not None else None
            }

        def convert_to_zero_parameters(param_list):
            self._convert_to_zero_parameters(param_list)

        def allgather_before(func: Callable) -> Callable:

            def wrapped(*args, **kwargs):
                param.all_gather()
                return func(*args, **kwargs)

            return wrapped

        # Collectives for gathering and partitioning parameters
        param.all_gather = all_gather
        param.all_gather_coalesced = all_gather_coalesced
        param.partition = partition

        # Collective for averaging gradients
        param.reduce_gradients_at_owner = reduce_gradients_at_owner
        param.partition_gradients = partition_gradients

        # Partitioning size utilities
        param.aligned_size = aligned_size
        param.padding_size = padding_size
        param.partition_numel = partition_numel
        param.ds_summary = types.MethodType(ds_summary, param)

        param.item = allgather_before(param.item)

        param.convert_to_zero_parameters = convert_to_zero_parameters

    def _aligned_size(self, param):
        return param.ds_numel + self._padding_size(param)

    def _padding_size(self, param):
        remainder = param.ds_numel % self.num_partitions
        return (self.num_partitions - remainder) if remainder else 0

    def _partition_numel(self, param):
        return param.ds_tensor.ds_numel

    def _ensure_availability_of_partitioned_params(self, params):
        swap_in_list = []
        swap_in_flight = []
        for param in params:
            if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                assert param.ds_tensor.final_location == OffloadDeviceEnum.nvme and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                swap_in_list.append(param)
            if param.ds_tensor.status == PartitionedParamStatus.INFLIGHT:
                assert param.ds_tensor.final_location == OffloadDeviceEnum.nvme and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                swap_in_flight.append(param)
        if len(swap_in_list) > 0:
            swap_in_list[0].nvme_swapper.swap_in(swap_in_list, async_op=False)
        elif len(swap_in_flight) > 0:
            swap_in_flight[0].nvme_swapper.synchronize_reads()

    @instrument_w_nvtx
    def _all_gather(self, param_list, async_op=False, hierarchy=None):

        # fetches from nvme if the partition is not available and in nvme
        self._ensure_availability_of_partitioned_params(param_list)

        handles = []
        all_gather_list = []
        for param in param_list:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if async_op:
                    handle = self._allgather_param(param, async_op=async_op, hierarchy=hierarchy)
                    param.ds_status = ZeroParamStatus.INFLIGHT  # if async_op else ZeroParamStatus.AVAILABLE
                    handles.append(handle)
                else:
                    all_gather_list.append(param)
        # note: param_list may contain params that are already in flight / aviailable. So we need to use all_gather_list
        if not async_op:
            if len(all_gather_list) == 1:
                ret_value = self._allgather_params(all_gather_list, hierarchy=hierarchy)
            else:
                all_gather_quantize_list = []
                all_gather_nonquantize_list = []
                for param in all_gather_list:
                    if hasattr(param.ds_tensor,
                               "ds_quant_scale") or (hasattr(param, "ds_secondary_tensor")
                                                     and hasattr(param.ds_secondary_tensor, "ds_quant_scale")):
                        all_gather_quantize_list.append(param)
                    else:
                        all_gather_nonquantize_list.append(param)
                # _allgather_params_coalesced always return None
                self._allgather_params_coalesced(all_gather_nonquantize_list, hierarchy, quantize=False)
                self._allgather_params_coalesced(all_gather_quantize_list, hierarchy, quantize=True)
            for param in all_gather_list:
                param.ds_status = ZeroParamStatus.AVAILABLE
            return None

        return handles

    def _partition(self, param_list, force=False, has_been_updated=False):
        for param in param_list:
            print_rank_0(f"Before Partitioning Param {param.ds_id}", force=False)
            if self.zero_param_process_group is not None:
                self._partition_param_sec(param)
            self._partition_param(param, has_been_updated=has_been_updated)

            param.ds_status = ZeroParamStatus.NOT_AVAILABLE
            # if param.ds_tensor is not None:
            #    assert id(param.data) == id(param.ds_tensor.data), \
            #    "After the parameters are initially partitioned, make sure we are not recreating the partition."
            #print_rank_0(f"After Partitioning Param {param.ds_id} {param.ds_tensor.size()} {param.ds_tensor}",force=False)
    @instrument_w_nvtx
    def _partition_param(self, param, buffer=None, has_been_updated=False):
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot partition a param in flight"
        global reuse_buffers
        print_rank_0(f"Param id {param.ds_id} status is {param.ds_status}", force=False)
        if param.ds_status is ZeroParamStatus.AVAILABLE:
            print_rank_0(f"Partitioning param id {param.ds_id} reuse buffers {reuse_buffers}", force=False)
            # if reuse_buffers and False:
            #     numel = buffer.numel()
            #     buffer = param.data.view(-1)
            #     print_rank_0(
            #         "Returning buffer for param {param.ds_id} with numel {param.ds_numel} to empty buffers",
            #         force=False)
            #     if numel in empty_buffers:
            #         empty_buffers[numel].append(buffer)

            # if deepspeed.comm.get_rank():
            #    print(f"Releasing {param.data.numel()}")

            if param.ds_tensor is not None and not has_been_updated:  ##param already partitioned

                #print_rank_0(f"Param  {param.ds_id} pri {param.ds_tensor.size()}  loc? {param.ds_tensor.final_location}", force=True)
                #param.data = param.ds_tensor.data

                see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=False)
                # param.data does not store anything meaningful in partitioned state
                free_param(param)
                see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=False)

                if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                    print_rank_0(f"Param {param.ds_id} partition released since it exists in nvme", force=False)
                    param.nvme_swapper.remove_partition_and_release_buffers([param])
                    print_rank_0(
                        f"after swap Param {param.ds_id} {param.ds_tensor.shape} partition released since it exists in nvme",
                        force=False)

                return

            tensor_size = self._aligned_size(param)
            partition_size = tensor_size // self.num_partitions
            if param.ds_tensor is None:
                final_location = None
                if self.remote_device == OffloadDeviceEnum.nvme and self.param_swapper.swappable_tensor(
                        numel=partition_size):
                    final_location = OffloadDeviceEnum.nvme
                    buffer = self.param_swapper.get_buffer(param, partition_size)
                    partitioned_tensor = torch.empty(0, dtype=param.dtype, device=buffer.device)
                    partitioned_tensor.data = buffer.data
                    print_rank_0(f"ID {param.ds_id} Initializing partition for the first time for nvme offload.")

                else:
                    if param.ds_persist:
                        device = self.local_device
                    elif self.remote_device == OffloadDeviceEnum.nvme:
                        device = OffloadDeviceEnum.cpu
                    else:
                        device = self.remote_device

                    partitioned_tensor = torch.empty(partition_size, dtype=param.dtype, device=device)
                    # quantize the tensor if it's not trainable
                    if not param.requires_grad and self.quantized_nontrainable_weights:
                        partitioned_tensor, partitioned_tensor.ds_quant_scale = self.quantizer_module.quantize(
                            partitioned_tensor)

                    if device == OffloadDeviceEnum.cpu and self.pin_memory:
                        partitioned_tensor = get_accelerator().pin_memory(partitioned_tensor)

                partitioned_tensor.requires_grad = False
                param.ds_tensor = partitioned_tensor
                param.ds_tensor.ds_numel = partition_size
                param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_tensor.final_location = final_location

            start = partition_size * self.get_partition_rank()
            end = start + partition_size

            one_dim_param = param.contiguous().view(-1)

            if start < param.ds_numel and end <= param.ds_numel:
                src_tensor = one_dim_param.narrow(0, start, partition_size)

                with torch.no_grad():
                    # make sure param.ds_tensor requires_grad always be false,
                    # otherwise, torch tracer will complain.
                    param.ds_tensor.copy_(src_tensor)

                #partitioned_tensor = src_tensor.clone().detach().to(self.remote_device)

            else:
                # partitioned_tensor = torch.zeros(partition_size,
                #                                  dtype=param.dtype,
                #                                  device=self.remote_device )

                if start < param.ds_numel:
                    elems_to_copy = param.ds_numel - start
                    with torch.no_grad():
                        # make sure param.ds_tensor requires_grad always be false,
                        # otherwise, torch tracer will complain.
                        param.ds_tensor.narrow(0, 0,
                                               elems_to_copy).copy_(one_dim_param.narrow(0, start, elems_to_copy))

            #print(f"Remote device {self.remote_device}")

            #param.ds_tensor = partitioned_tensor

            #param.data = param.ds_tensor.data

            # param.data does not store anything meaningful in partitioned state

            see_memory_usage(f'Before partitioning param {param.ds_id} {param.shape}', force=False)
            free_param(param)
            see_memory_usage(f'After partitioning param {param.ds_id} {param.shape}', force=False)

            if param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                self.param_swapper.swap_out_and_release([param])
                print_rank_0(f"ID {param.ds_id} Offloaded to nvme offload and buffers released.")
                see_memory_usage(f"ID {param.ds_id} Offloaded to nvme offload and buffers released.", force=False)

            print_rank_0(f"ID {param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}")

    @instrument_w_nvtx
    def _partition_param_sec(self, param, buffer=None, has_been_updated=False):
        assert param.ds_status is not ZeroParamStatus.INFLIGHT, f" {param} Cannot partition a param in flight"
        global reuse_buffers
        ##support for NVME secondary param offload
        #print_rank_0(f"SEC Param id {param.ds_id} status is {param.ds_status}", force=True)
        if param.ds_status is ZeroParamStatus.AVAILABLE:
            #check padding
            tensor_size = self._aligned_size(param)
            partition_size = tensor_size // self.dp_world_size

            secondary_partition_size = int(tensor_size // self.num_ranks_in_param_group)
            if param.ds_secondary_tensor is None:
                final_location = None
                secondary_partitioned_tensor = torch.empty(secondary_partition_size,
                                                           dtype=param.dtype,
                                                           device=self.remote_device)

                if self.pin_memory:
                    secondary_partitioned_tensor = secondary_partitioned_tensor.pin_memory()
                # quantize the tensor if it's not trainable
                if not param.requires_grad and self.quantized_nontrainable_weights:
                    secondary_partitioned_tensor, secondary_partitioned_tensor.ds_quant_scale = self.quantizer_module.quantize(
                        secondary_partitioned_tensor)
                secondary_partitioned_tensor.requires_grad = False
                param.ds_secondary_tensor = secondary_partitioned_tensor
                param.ds_secondary_tensor.ds_numel = secondary_partition_size
                param.ds_secondary_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_secondary_tensor.final_location = final_location

            #use rank in group for secondary tensor
            secondary_start = secondary_partition_size * self.rank_in_group

            secondary_end = secondary_start + secondary_partition_size

            one_dim_param = param.contiguous().view(-1)

            # ds_numel is unpadded, so the last chunk of the secondary tensor might not be secondary_partition_size
            sec_numel = param.ds_numel - secondary_start if secondary_end > param.ds_numel else secondary_partition_size

            # copy from full tensor to secondary tensor
            param.ds_secondary_tensor.narrow(0, 0,
                                             sec_numel).copy_(one_dim_param.narrow(0, secondary_start, sec_numel))

            # TODO: This is a temporary fix to avoid the issue that 2nd tensor all-gather happens before 2nd tensor partition is done
            get_accelerator().current_stream().synchronize()

            print_rank_0(f"{param.ds_id} partitioned type {param.dtype} dev {param.device} shape {param.shape}",
                         force=False)

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

        tensor_size = partition_size * self.num_partitions
        aligned_param_size = self._aligned_size(param)
        assert tensor_size == aligned_param_size, f'param id {param.ds_id} aligned size {aligned_param_size} does not match tensor size {tensor_size}'

        print_rank_0(
            f"{'--'* hierarchy}---- Before allocating allgather param {debug_param2name_id_shape_status(param)} partition size={partition_size}"
        )

        see_memory_usage(
            f'Before allocate allgather param {debug_param2name_id_shape_status(param)} partition_size={partition_size} ',
            force=False)
        flat_tensor = torch.zeros(aligned_param_size, dtype=param.dtype, device=param.device).view(-1)
        see_memory_usage(
            f'After allocate allgather param {debug_param2name_id_shape_status(param)} {aligned_param_size} {partition_size} ',
            force=False)

        get_accelerator().synchronize()

        print_rank_0(
            f"{'--'* hierarchy}----allgather param with {debug_param2name_id_shape_status(param)} partition size={partition_size}"
        )
        #        if not flat_tensor.numel() > 100000:
        #            replicated_tensor = flat_tensor.narrow(0,
        #                                                   0,
        #                                                   param.ds_numel).view(param.ds_shape)
        #            param.data = replicated_tensor.data
        #            return None
        if self.use_all_gather_into_tensor:
            handle = dist.all_gather_into_tensor(flat_tensor,
                                                 param.ds_tensor.to(get_accelerator().device_name()),
                                                 group=self.get_partition_dp_group(param),
                                                 async_op=async_op)
        else:
            partitions = []
            for i in range(self.num_partitions):
                partitions.append(flat_tensor.narrow(0, partition_size * i, partition_size))

                if i == dist.get_rank(group=self.get_partition_dp_group(param)):
                    partitions[i].data.copy_(param.ds_tensor.data, non_blocking=True)

            handle = dist.all_gather(partitions,
                                     partitions[self.get_partition_rank()],
                                     group=self.get_partition_dp_group(param),
                                     async_op=async_op)

        replicated_tensor = flat_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape)
        param.data = replicated_tensor.data
        return handle

    def _allgather_params_coalesced(self, param_list, hierarchy=0, quantize=False):
        """ blocking call
        avoid explicit memory copy in _allgather_params
        """
        if len(param_list) == 0:
            return

        if self.num_partitions == 1:
            handle = _no_gather_coalesced(param_list)
            handle.wait()
            return None

        # collect local tensors and partition sizes
        partition_sizes = []
        local_tensors = []
        if quantize:
            quantize_scale_sizes = []
            quantize_scale_tensors = []
        for param in param_list:
            partition_sizes.append(param.ds_tensor.ds_numel)
            local_tensors.append(param.ds_tensor.to(get_accelerator().device_name()))
            if quantize:
                quantize_scale_sizes.append(param.ds_tensor.ds_quant_scale.numel())
                quantize_scale_tensors.append(param.ds_tensor.ds_quant_scale.to(get_accelerator().device_name()))
        # allocate memory for allgather params
        allgather_params = []
        if quantize:
            allgather_quantize_scale = []
        for psize in partition_sizes:
            tensor_size = psize * self.num_partitions
            flat_tensor = torch.empty(tensor_size, dtype=param_list[0].ds_tensor.dtype,
                                      device=self.local_device).view(-1)
            flat_tensor.requires_grad = False
            allgather_params.append(flat_tensor)
        if quantize:
            for psize in quantize_scale_sizes:
                tensor_size = psize * self.num_partitions
                flat_tensor = torch.empty(tensor_size,
                                          dtype=param_list[0].ds_tensor.ds_quant_scale.dtype,
                                          device=self.local_device).view(-1)
                flat_tensor.requires_grad = False
                allgather_quantize_scale.append(flat_tensor)

        # launch
        launch_handles = []
        launch_quantize_handles = []
        for param_idx, param in enumerate(param_list):
            input_tensor = local_tensors[param_idx].view(-1)

            if self.use_all_gather_into_tensor:
                # try the _all_gather_base from Pytorch master
                h = dist.all_gather_into_tensor(allgather_params[param_idx],
                                                input_tensor,
                                                group=self.get_partition_dp_group(param),
                                                async_op=True)
                if quantize:
                    quantize_handle = dist.all_gather_into_tensor(allgather_quantize_scale[param_idx],
                                                                  quantize_scale_tensors[param_idx],
                                                                  group=self.get_partition_dp_group(param),
                                                                  async_op=True)
                    launch_quantize_handles.append(quantize_handle)
            else:
                output_list = []
                for i in range(self.num_partitions):
                    psize = partition_sizes[param_idx]
                    partition = allgather_params[param_idx].narrow(0, i * psize, psize)
                    output_list.append(partition)
                    if not get_accelerator().on_accelerator(partition):
                        logger.warning(
                            f'param {param_idx}, partition {i} is not on CUDA, partition shape {partition.size()}')

                # back to old all_gather function
                h = dist.all_gather(output_list, input_tensor, group=self.get_partition_dp_group(param), async_op=True)
                if quantize:
                    output_scale_list = []
                    for i in range(self.num_partitions):
                        psize = quantize_scale_sizes[param_idx]
                        partition = allgather_quantize_scale[param_idx].narrow(0, i * psize, psize)
                        output_scale_list.append(partition)
                    quant_handle = dist.all_gather(output_scale_list,
                                                   quantize_scale_tensors[param_idx],
                                                   group=self.get_partition_dp_group(param),
                                                   async_op=True)
                    launch_quantize_handles.append(quant_handle)
            launch_handles.append(h)

        # Wait ensures the operation is enqueued, but not necessarily complete.
        launch_handles[-1].wait()
        if quantize:
            for quant_handle in launch_quantize_handles:
                quant_handle.wait()

        # assign to param.data (not copy)
        for i, param in enumerate(param_list):
            gathered_tensor = allgather_params[i]
            if quantize:
                gathered_tensor = self.quantizer_module.dequantize(gathered_tensor, allgather_quantize_scale[i])
            param.data = gathered_tensor.narrow(0, 0, param.ds_numel).view(param.ds_shape).data

        # guarantee the communication to be completed
        get_accelerator().synchronize()

        return None

    def _allgather_params(self, param_list, hierarchy=0):
        if len(param_list) == 0:
            return

        partition_size = sum([param.ds_tensor.ds_numel for param in param_list])

        tensor_size = partition_size * self.num_partitions
        flat_tensor = torch.empty(tensor_size, dtype=param_list[0].ds_tensor.dtype, device=self.local_device)
        flat_tensor.requires_grad = False
        partitions = []
        for i in range(self.num_partitions):
            start = partition_size * i

            partitions.append(flat_tensor.narrow(0, start, partition_size))

            if i == self.get_partition_rank():
                offset = 0
                for param in param_list:
                    param_numel = param.ds_tensor.ds_numel

                    partitions[i].narrow(0, offset, param_numel).copy_(param.ds_tensor.data)

                    offset += param_numel

        if hasattr(param_list[0], 'ds_quant_scale'):
            scale_size = sum([param.ds_tensor.ds_quant_scale.numel() for param in param_list])
            scale_tensor_size = scale_size * self.world_size
            flat_scale_tensor = torch.empty(scale_tensor_size,
                                            dtype=param_list[0].ds_tensor.ds_quant_scale.dtype,
                                            device=self.local_device)
            flat_scale_tensor.requires_grad = False
            scale_partitions = []
            for i in range(self.world_size):
                start = scale_tensor_size * i
                scale_partitions.append(flat_scale_tensor.narrow(0, start, scale_tensor_size))
                if i == self.rank:
                    offset = 0
                    for param in param_list:
                        param_scale_numel = param.ds_tensor.ds_quant_scale.ds_numel

                        scale_partitions[i].narrow(0, offset,
                                                   param_scale_numel).copy_(param.ds_tensor.ds_quant_scale.data)

                        offset += param_scale_numel

        dist.all_gather_into_tensor(flat_tensor,
                                    partitions[self.get_partition_rank()],
                                    group=self.get_partition_dp_group(param),
                                    async_op=False)
        if hasattr(param_list[0], 'ds_quant_scale'):
            dist.all_gather(flat_scale_tensor,
                            param_list[0].ds_quant_scale,
                            group=self.get_partition_dp_group(param),
                            async_op=False)
        param_offset = 0

        for param in param_list:
            param_partition_size = param.ds_tensor.ds_numel
            param_size = param.ds_numel
            replicated_tensor = torch.empty(param.ds_shape, dtype=param.ds_tensor.dtype, device=self.local_device)

            for i in range(self.num_partitions):

                start = i * partition_size

                param_start = i * param_partition_size

                if param_start < param_size:
                    numel_to_copy = min(param_size - param_start, param_partition_size)

                    part_to_copy = partitions[i].narrow(0, param_offset, numel_to_copy)

                    replicated_tensor.view(-1).narrow(0, param_start, numel_to_copy).copy_(part_to_copy)
            #param_offset += param.data.numel()
            param_offset += param.ds_tensor.ds_numel
            if hasattr(param_list[0], 'ds_quant_scale'):
                replicated_tensor = self.quantizer_module.dequantize(replicated_tensor, flat_scale_tensor)
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
            start = self.get_partition_rank() * partition_size
            end = start + partition_size
            #print_rank_0("REduce scatter was executed for param {param.ds_id}")
            if start < param.ds_numel < end:
                elements = param.ds_numel - start
                param.grad.view(-1).narrow(0, start, elements).copy_(reduced_partition.narrow(0, 0, elements))

    def _reduce_scatter_gradient(self, param):

        partition_size = param.ds_tensor.ds_numel
        #output = torch.empty(partition_size, dtype=param.dtype, device=param.device)

        total_size = partition_size * self.num_partitions
        input_list = []

        for i in range(self.num_partitions):

            start = i * partition_size
            end = start + partition_size

            #print("before reduce scatter gradients")
            if start < param.ds_numel and end <= param.ds_numel:
                input = param.grad.view(-1).narrow(0, start, partition_size)
            else:
                input = torch.zeros(partition_size, dtype=param.dtype, device=param.device)

                if start < param.ds_numel:
                    elements = param.ds_numel - start
                    input.narrow(0, 0, elements).copy_(param.grad.view(-1).narrow(0, start, elements))
            #print("after reduce scatter gradients")
            input_list.append(input)

        rank = dist.get_rank(group=self.get_partition_dp_group(param))
        handle = dist.reduce_scatter(input_list[rank],
                                     input_list,
                                     group=self.get_partition_dp_group(param),
                                     async_op=True)

        return handle, input_list[rank]

    def _partition_gradients(self, param_list, partition_buffers=None, accumulate=False):
        if partition_buffers is None:
            partition_buffers = [None] * len(param_list)

        for param, partition_buffer in zip(param_list, partition_buffers):
            self._partition_gradient(param, partition_buffer=partition_buffer, accumulate=accumulate)

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
            partition_buffer = torch.zeros(partition_size, dtype=param.dtype, device=param.device)
        else:
            assert partition_buffer.numel(
            ) >= partition_size, f"The partition buffer size {partition_buffer.numel()} should match the size of param.ds_tensor {partition_size}"

        rank = dist.get_rank(group=self.get_partition_dp_group(param))
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

            # if source and destination are on same device,
            # add to the provided buffer
            elif src_tensor.device == dest_tensor.device:
                dest_tensor.add_(src_tensor)

            # if source and destination are on different device, copy first to src
            # then add and move back to the destination. This seems to run faster
            # when src is gpu and dest is cpu
            # adding directly to cpu is very slow
            else:
                acc_tensor = torch.empty(src_tensor.numel(), dtype=param.dtype, device=param.device)

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

    def get_partition_dp_group(self, param):
        return param.ds_process_group

    def get_partition_rank(self):
        """subclass can overload to specify different relative rank in
        parameter partition group"""
        return self.rank

    @property
    def num_partitions(self):
        return self.dp_world_size

    def get_dp_process_group(self):
        """ Return the communication group with all data-parallel ranks """
        return self.ds_process_group


class GatheredParameters:

    def __init__(self, params, modifier_rank=None, fwd_module=None, enabled=True):
        """A context that collects parameters that were partitioned via a
        :class:`deepspeed.zero.Init` context. The parameters are partitioned
        again upon exit.

        Args:
            params (``torch.nn.Parameter``): A single parameter, or an iterable of parameters (list, tuple, generator) of parameters to collect.
                It's assumed that all parameters are zero params.
            modifier_rank (int, optional): If specified, this rank's parameter will be
                broadcasted on exit from the context. This argument is required if ``params`` are
                modified, so that all processes have a consistent view of the data. Defaults
                to ``None``.
            fwd_module (``torch.nn.Module``, optional): If specified, ``params`` will be
                registered as external parameters of ``fwd_module``. See :meth:`deepspeed.zero.register_external_parameter`.
            enabled (bool, optional): If ``False``, this context is a no-op. Defaults to ``True``.

        Important: Make sure to use ``modifier_rank`` that is not ``None`` (e.g., ``modifier_rank=0``)
        if you need the GPU memory allocated by gather to be released upon exit from the context manager.

        Important: if ``params`` isn't an iterable of parameters or a single parameter it'll be silently ignored!

        Examples
        ========

        #. Allocate a partitioned module, initialize its weight on rank 0, and update all
           processes.

            .. code-block:: python

                with deepspeed.zero.Init():
                    linear = torch.nn.Linear(1000,1000)

                with deepspeed.zero.GatheredParameters(linear.weight,
                                                       modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        linear.weight.zero_()

                with deepspeed.zero.GatheredParameters(linear.weight,
                                                       modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
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
                        if deepspeed.comm.get_rank() == 0:
                            module._load_from_state_dict(state_dict, prefix)

                    for name, child in module._modules.items():
                        if child is not None:
                            load(child, prefix + name + ".")

                load(model, prefix="")

        If this approach is not used, then the full model will first be copied to each GPU. For models
        bigger than the memory of a single GPU, this method is required.
        """

        self.enabled = enabled
        if not enabled:
            return

        if isinstance(params, Iterable) and not isinstance(params, torch.Tensor):
            # deal with generators like model.parameters()
            # must convert to list to be able to iterate more than once if we get a generator
            params = list(params)
        else:
            # single param
            params = [params]
        # enable if at least one is zero-param, otherwise a noop
        if not any(is_zero_param(p) for p in params):
            self.enabled = False
            return

        self.params = [p for p in params if hasattr(p, "ds_id")]
        self.params = sorted(
            set(self.params), key=lambda x: x.ds_id
        )  # remove the duplicates to prevent racing condition, we must also make sure the order is the same on all ranks otherwise we'll get deadlocks
        self.src_rank = None
        if modifier_rank is not None:
            if self.params[0].ds_process_group == dist.get_world_group():
                self.src_rank = modifier_rank
            else:
                # A group was specified; convert DP rank to global rank
                self.src_rank = dist.get_global_rank(self.params[0].ds_process_group, modifier_rank)
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
            self.params[0].partition(param_list=self.params, has_been_updated=False)
            return

        handles = [dist.broadcast(p.data, self.src_rank, group=p.ds_process_group, async_op=True) for p in self.params]
        for h in handles:
            h.wait()
        self.params[0].partition(param_list=self.params, has_been_updated=True)
