# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import importlib
import inspect
import functools

from .abstract_accelerator import DeepSpeedAccelerator
import torch
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.


class MLU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'mlu'
        self._communication_backend_name = 'cncl'
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
        if device_index == None:
            return 'mlu'
        return 'mlu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.mlu.device(device_index)

    def set_device(self, device_index):
        torch.mlu.set_device(device_index)

    def current_device(self):
        return torch.mlu.current_device()

    def current_device_name(self):
        return 'mlu:{}'.format(torch.mlu.current_device())

    def device_count(self):
        return torch.mlu.device_count()

    def synchronize(self, device_index=None):
        return torch.mlu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.mlu.set_rng_state(new_state)

        return torch.mlu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.mlu.get_rng_state()

        return torch.mlu.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.mlu.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.mlu.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.mlu.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.mlu.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.mlu.Stream

    def stream(self, stream):
        return torch.mlu.stream(stream)

    def current_stream(self, device_index=None):
        return torch.mlu.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.mlu.default_stream(device_index)

    @property
    def Event(self):
        return torch.mlu.Event

    # Memory management
    def empty_cache(self):
        return torch.mlu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.mlu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.mlu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.mlu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.mlu.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.mlu.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.mlu.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.mlu, 'memory_stats'):
            return torch.mlu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.mlu, 'reset_peak_memory_stats'):
            return torch.mlu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.mlu, 'memory_reserved'):
            return torch.mlu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.mlu, 'max_memory_reserved'):
            return torch.mlu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.mlu.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        return torch.mlu.is_bf16_supported()

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
    def amp(self):
        if hasattr(torch.mlu, 'amp'):
            return torch.mlu.amp
        return None

    def is_available(self):
        return torch.mlu.is_available()

    def range_push(self, msg):
        if hasattr(torch.mlu.cnpx, 'range_push'):
            return torch.mlu.cnpx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.mlu.cnpx, 'range_pop'):
            return torch.mlu.cnpx.range_pop()

    def lazy_call(self, callback):
        return torch.mlu._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return True

    # Graph operations
    def create_graph(self):
        torch.mlu.MLUGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.mlu.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device='mlu')

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device='mlu')

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device='mlu')

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device='mlu')

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device='mlu')

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device='mlu')

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device='mlu')

    def pin_memory(self, tensor):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('mlu:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.mlu"
        except ImportError:
            return "deepspeed.ops.op_builder.mlu"

    def _lazy_init_class_dict(self):
        if self.class_dict:
            return

        op_builder_module = importlib.import_module(self.op_builder_dir())

        # get op builder class from op_builder/mlu/__init__.py
        self.class_dict = {}
        for class_name, class_obj in inspect.getmembers(op_builder_module, inspect.isclass):
            self.class_dict[class_name] = class_obj

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        builder_class = self.get_op_builder(class_name)
        return builder_class()

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return self.class_dict['NotImplementedBuilder']

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return ['NEUWARE_HOME', 'CNCL', 'LD_LIBRARY', 'PATH']

    def visible_devices_envs(self):
        return ['MLU_VISIBLE_DEVICES']

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
                f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends }")
