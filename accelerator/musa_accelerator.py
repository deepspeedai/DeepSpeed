# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# MUSA accelerator backend for torch_musa (ported from MLU pattern).
import functools
import importlib
import inspect

from .abstract_accelerator import DeepSpeedAccelerator

try:
    import torch
except ImportError:
    # During setup stage torch may not be installed.
    torch = None


class MUSA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'musa'
        self._communication_backend_name = 'mccl'
        self._compile_backend = "inductor"
        self.class_dict = None

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
        if device_index is None:
            return 'musa'
        return 'musa:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.musa.device(device_index)

    def set_device(self, device_index):
        torch.musa.set_device(device_index)

    def current_device(self):
        return torch.musa.current_device()

    def current_device_name(self):
        return 'musa:{}'.format(torch.musa.current_device())

    def device_count(self):
        return torch.musa.device_count()

    def synchronize(self, device_index=None):
        return torch.musa.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.musa.set_rng_state(new_state)
        return torch.musa.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.musa.get_rng_state()
        return torch.musa.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.musa.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.musa.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.musa.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.musa.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.musa.Stream

    def stream(self, stream):
        return torch.musa.stream(stream)

    def current_stream(self, device_index=None):
        return torch.musa.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.musa.default_stream(device_index)

    @property
    def Event(self):
        return torch.musa.Event

    # Memory management
    def empty_cache(self):
        return torch.musa.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.musa.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.musa.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.musa.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        # torch_musa exposes memory_reserved; keep memory_cached alias for DeepSpeed callers.
        if hasattr(torch.musa, 'memory_cached'):
            return torch.musa.memory_cached(device_index)
        return torch.musa.memory_reserved(device_index)

    def max_memory_cached(self, device_index=None):
        if hasattr(torch.musa, 'max_memory_cached'):
            return torch.musa.max_memory_cached(device_index)
        if hasattr(torch.musa, 'max_memory_reserved'):
            return torch.musa.max_memory_reserved(device_index)
        return self.max_memory_allocated(device_index)

    def reset_max_memory_cached(self, device_index=None):
        if hasattr(torch.musa, 'reset_max_memory_cached'):
            return torch.musa.reset_max_memory_cached(device_index)
        if hasattr(torch.musa, 'reset_peak_memory_stats'):
            return torch.musa.reset_peak_memory_stats(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.musa, 'memory_stats'):
            return torch.musa.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.musa, 'reset_peak_memory_stats'):
            return torch.musa.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.musa, 'memory_reserved'):
            return torch.musa.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.musa, 'max_memory_reserved'):
            return torch.musa.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.musa.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if hasattr(torch.musa, 'is_bf16_supported'):
            return torch.musa.is_bf16_supported()
        return True

    def is_fp16_supported(self):
        return True

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def is_available(self):
        return torch.musa.is_available()

    def range_push(self, msg, domain=None, category=None):
        return

    def range_pop(self, domain=None):
        return

    def lazy_call(self, callback):
        return torch.musa._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Graph operations (not generally available on torch_musa yet)
    def create_graph(self):
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device='musa')

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device='musa')

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device='musa')

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device='musa')

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device='musa')

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device='musa')

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device='musa')

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        return device_str.startswith('musa:')

    def op_builder_dir(self):
        try:
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.musa"
        except ImportError:
            return "deepspeed.ops.op_builder.musa"

    def _lazy_init_class_dict(self):
        if self.class_dict:
            return
        op_builder_module = importlib.import_module(self.op_builder_dir())
        self.class_dict = {}
        for class_name, class_obj in inspect.getmembers(op_builder_module, inspect.isclass):
            self.class_dict[class_name] = class_obj

    def create_op_builder(self, class_name):
        builder_class = self.get_op_builder(class_name)
        return builder_class()

    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        return self.class_dict['NotImplementedBuilder']

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return ['MUSA_HOME', 'MCCL', 'LD_LIBRARY_PATH', 'PATH']

    def visible_devices_envs(self):
        return ['MUSA_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        return self._compile_backend

    def set_compile_backend(self, backend):
        supported_backends = torch._dynamo.list_backends(exclude_tags=())
        if backend in supported_backends:
            self._compile_backend = backend
        else:
            raise ValueError(
                f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends}")
