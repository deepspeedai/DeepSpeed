# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import os
import sys
import pkgutil
import importlib

import torch
from .abstract_accelerator import DeepSpeedAccelerator


class SUPA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'supa'
        # Use BCCL on Linux, fall back to gloo on Windows (no BCCL support yet)
        self._communication_backend_name = 'bccl' if sys.platform != 'win32' else 'gloo'
        self._compile_backend = "inductor"

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
            return 'supa'
        return 'supa:{}'.format(device_index)

    def communication_backend_version(self):
        # BCCL does not expose a version via torch.supa
        return (0, 0, 0)

    def device(self, device_index=None):
        return torch.device('supa', device_index)

    def set_device(self, device_index):
        torch.supa.set_device(device_index)

    def current_device(self):
        return torch.supa.current_device()

    def current_device_name(self):
        return 'supa:{}'.format(torch.supa.current_device())

    def device_count(self):
        return torch.supa.device_count()

    def synchronize(self, device_index=None):
        return torch.supa.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.supa.set_rng_state(new_state)
        return torch.supa.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.supa.get_rng_state()
        return torch.supa.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.supa.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.supa.manual_seed_all(seed)

    def initial_seed(self):
        return torch.supa.initial_seed()

    def default_generator(self, device_index):
        return torch.supa.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.supa.Stream

    def stream(self, stream):
        return torch.supa.stream(stream)

    def current_stream(self, device_index=None):
        return torch.supa.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.supa.default_stream(device_index)

    @property
    def Event(self):
        return torch.supa.Event

    # Memory management
    def empty_cache(self):
        return torch.supa.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.supa.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.supa.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.supa.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.supa.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.supa.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.supa.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.supa, 'memory_stats'):
            return torch.supa.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.supa, 'reset_peak_memory_stats'):
            return torch.supa.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.supa, 'memory_reserved'):
            return torch.supa.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.supa, 'max_memory_reserved'):
            return torch.supa.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.supa.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    def supported_dtypes(self):
        return [torch.float, torch.half, torch.bfloat16]

    # Misc
    def is_available(self):
        return torch.supa.is_available()

    def range_push(self, msg, domain=None, category=None):
        return None

    def range_pop(self, domain=None):
        return None

    def lazy_call(self, callback):
        return torch.supa._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return True

    # Graph operations
    def create_graph(self):
        return torch.supa.SUPAGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.supa.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return torch.supa.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.supa.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.supa.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.supa.FloatTensor

    @property
    def HalfTensor(self):
        return torch.supa.HalfTensor

    @property
    def IntTensor(self):
        return torch.supa.IntTensor

    @property
    def LongTensor(self):
        return torch.supa.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        return device_str.startswith('supa:')

    def op_builder_dir(self):
        try:
            # Local install: op_builder is a top-level package
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.supa"
        except ImportError:
            return "deepspeed.ops.op_builder.supa"

    # dict that holds class name <--> class type mapping i.e.
    # 'FusedAdamBuilder': <class 'op_builder.supa.fused_adam.FusedAdamBuilder'>
    # populated lazily on first call to create_op_builder / get_op_builder
    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict is not None:
            return
        self.class_dict = {}
        op_builder_dir = self.op_builder_dir()
        op_builder_module = importlib.import_module(op_builder_dir)
        op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
        for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
            if module_name in ('all_ops', 'builder') or os.path.isdir(
                    os.path.join(op_builder_absolute_path, module_name)):
                continue
            module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
            for member_name in module.__dir__():
                if (member_name.endswith('Builder')
                        and member_name not in ('OpBuilder', 'CUDAOpBuilder', 'TorchCPUOpBuilder', 'SUPAOpBuilder')):
                    if member_name not in self.class_dict:
                        self.class_dict[member_name] = getattr(module, member_name)

    def create_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        return None

    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return ['BCCL', 'BIREN', 'SUPA', 'LD_LIBRARY', 'PATH']

    def visible_devices_envs(self):
        return ['SUPA_VISIBLE_DEVICES']

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
            raise ValueError(f"{backend} not supported by {self.device_name()}. "
                             f"Supported backends: {supported_backends}")
