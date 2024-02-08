# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator
import intel_extension_for_pytorch as ipex  # noqa: F401 # type: ignore
import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore


class XPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'xpu'
        self._communication_backend_name = 'ccl'
        self.aligned_tensors = []

    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index == None:
            return 'xpu'
        return 'xpu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.xpu.device(device_index)

    def set_device(self, device_index):
        torch.xpu.set_device(device_index)

    def current_device(self):
        return torch.xpu.current_device()

    def current_device_name(self):
        return 'xpu:{}'.format(torch.xpu.current_device())

    def device_count(self):
        return torch.xpu.device_count()

    def synchronize(self, device_index=None):
        return torch.xpu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.xpu.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index == None:
            return torch.xpu.set_rng_state(new_state)
        return torch.xpu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index == None:
            return torch.xpu.get_rng_state()
        return torch.xpu.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.xpu.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.xpu.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.xpu.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.xpu.Stream

    def stream(self, stream):
        return torch.xpu.stream(stream)

    def current_stream(self, device_index=None):
        return torch.xpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        # torch.xpu does not support the sync behavior of default stream as cuda
        # use current_stream as workaround
        # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
        return torch.xpu.current_stream(device_index)

    @property
    def Event(self):
        return torch.xpu.Event

    # Memory management
    def empty_cache(self):
        return torch.xpu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.xpu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.xpu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.xpu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.xpu.memory_reserved(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.xpu.max_memory_reserved(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.xpu.reset_max_memory_reserved(device_index)

    def memory_stats(self, device_index=None):
        return torch.xpu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        return torch.xpu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        return torch.xpu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        return torch.xpu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.xpu.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Misc
    def amp(self):
        return torch.xpu.amp

    def is_available(self):
        return torch.xpu.is_available()

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return torch.xpu.lazy_init._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Graph operations
    def create_graph(self):
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    def supported_dtypes(self):
        return [torch.float, torch.half, torch.bfloat16]

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.xpu.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.xpu.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.xpu.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.xpu.FloatTensor

    @property
    def HalfTensor(self):
        return torch.xpu.HalfTensor

    @property
    def IntTensor(self):
        return torch.xpu.IntTensor

    @property
    def LongTensor(self):
        return torch.xpu.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        if align_bytes == 1:
            return tensor.pin_memory(device=self.current_device_name())
        elif align_bytes == 0:
            from intel_extension_for_deepspeed.op_builder.async_io import AsyncIOBuilder
            self.aio_handle = AsyncIOBuilder().load().aio_handle(128 * 1024, 8, False, False, False)
            aligned_t = self.aio_handle.new_cpu_locked_tensor(tensor.numel(), tensor)
            aligned_t = aligned_t[:tensor.numel()].copy_(tensor)
            self.aligned_tensors.append([aligned_t.data_ptr(), aligned_t[-1].data_ptr()])
            return aligned_t

    def is_pinned(self, tensor):
        if tensor.is_pinned(device=self.current_device_name()):
            return True
        else:
            for begin, end in self.aligned_tensors:
                if begin <= tensor.data_ptr() and tensor.data_ptr() <= end:
                    return True
        return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.xpu"
        except ImportError:
            return "deepspeed.ops.op_builder.xpu"

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('xpu:'):
            return True
        else:
            return False

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class != None:
            return builder_class()
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            from op_builder.xpu import CPUAdagradBuilder, CPUAdamBuilder, FusedAdamBuilder, AsyncIOBuilder
        except ImportError:
            from deepspeed.ops.op_builder.xpu import CPUAdagradBuilder, CPUAdamBuilder, FusedAdamBuilder, AsyncIOBuilder

        if class_name == "AsyncIOBuilder":
            return AsyncIOBuilder
        elif class_name == "CPUAdagradBuilder":
            return CPUAdagradBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        else:
            return None

    def build_extension(self):
        try:
            from intel_extension_for_pytorch.xpu.cpp_extension import DpcppBuildExtension
        except ImportError:
            from intel_extension_for_pytorch.xpu.utils import DpcppBuildExtension
        return DpcppBuildExtension

    def export_envs(self):
        return []
