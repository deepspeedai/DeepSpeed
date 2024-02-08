# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pkgutil
import importlib
import torch

from .abstract_accelerator import DeepSpeedAccelerator


class HPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'hpu'
        self._communication_backend_name = 'hccl'
        try:
            import habana_frameworks.torch.hpu as hpu
            hpu.setDeterministic(True)
            self.hpu = hpu
        except ImportError as e:
            raise ValueError(
                f"HPU_Accelerator requires habana_frameworks.torch.hpu, which is not installed on this system.")

        self.fp16_supported = None

    # Device APIs
    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return False

    def resolves_data_dependency(self):
        return True

    def handles_memory_backpressure(self):
        return True

    def device_name(self, device_index=None):
        if device_index is None:
            return 'hpu'
        return 'hpu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device(self.device_name(device_index))

    def set_device(self, device_index):
        self.hpu.set_device(device_index)

    def current_device(self):
        return (self.hpu.current_device())

    def current_device_name(self):
        return 'hpu:{}'.format(self.current_device())

    def device_count(self):
        return self.hpu.device_count()

    def synchronize(self, device_index=None):
        return self.hpu.synchronize()

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        self.hpu.random.set_rng_state(new_state)

    def get_rng_state(self, device_index=None):
        return self.hpu.random.get_rng_state()

    def manual_seed(self, seed):
        self.hpu.random.manual_seed(seed)

    def manual_seed_all(self, seed):
        self.hpu.random.manual_seed_all(seed)

    def initial_seed(self, seed):
        self.hpu.random.initial_seed(seed)

    def default_generator(self, device_index):
        return self.hpu.random.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return self.hpu.Stream

    def stream(self, stream):
        return self.hpu.stream(stream)

    def current_stream(self, device_index=None):
        return self.hpu.current_stream()

    def default_stream(self, device_index=None):
        return self.hpu.default_stream()

    @property
    def Event(self):
        import habana_frameworks.torch.core as htcore
        return htcore.hpu.Event

    # Memory management
    def empty_cache(self):
        return

    def memory_allocated(self, device_index=None):
        return self.hpu.memory_allocated()

    def max_memory_allocated(self, device_index=None):
        return self.hpu.max_memory_allocated()

    def reset_max_memory_allocated(self, device_index=None):
        return self.hpu.reset_max_memory_allocated()

    def memory_cached(self, device_index=None):
        return self.hpu.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return self.hpu.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return None

    def memory_stats(self, device_index=None):
        return self.hpu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        self.hpu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        return self.hpu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        return self.hpu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return self.memory_stats(device_index)['Limit']

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        if self.fp16_supported is None:
            import habana_frameworks.torch.utils.experimental as htexp
            self.fp16_supported = htexp._is_fp16_supported()
        return self.fp16_supported

    def supported_dtypes(self):
        supported_dtypes = [torch.float, torch.bfloat16]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        return supported_dtypes

    # Misc
    def amp(self):
        return None

    def is_available(self):
        return self.hpu.is_available()

    def range_push(self, msg):
        return

    def range_pop(self):
        return

    def lazy_call(self, callback):
        callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Graph operations
    def create_graph(self):
        return self.hpu.HPUGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return self.hpu.graph(graph, stream=stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return self.hpu.BFloat16Tensor

    @property
    def ByteTensor(self):
        return self.hpu.ByteTensor

    @property
    def DoubleTensor(self):
        return self.hpu.DoubleTensor

    @property
    def FloatTensor(self):
        return self.hpu.FloatTensor

    @property
    def HalfTensor(self):
        return self.hpu.HalfTensor

    @property
    def IntTensor(self):
        return self.hpu.IntTensor

    @property
    def LongTensor(self):
        return self.hpu.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory(self.device())

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('hpu:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.hpu"
        except ImportError:
            return "deepspeed.ops.op_builder.hpu"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
    # this dict will be filled at init stage
    class_dict = None

    def _lazy_init_class_dict(self):
        if self.class_dict is not None:
            return
        else:
            self.class_dict = {}
            # begin initialize for create_op_builder()
            # put all valid class name <--> class type mapping into class_dict
            op_builder_dir = self.op_builder_dir()
            op_builder_module = importlib.import_module(op_builder_dir)
            op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
            for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
                # avoid self references,
                # skip sub_directories which contains ops for other backend(cpu, npu, etc.).
                if module_name != 'all_ops' and module_name != 'builder' and not os.path.isdir(
                        os.path.join(op_builder_absolute_path, module_name)):
                    module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                    for member_name in module.__dir__():
                        if member_name.endswith(
                                'Builder'
                        ) and member_name != "OpBuilder" and member_name != "CPUOpBuilder" and member_name != "TorchCPUOpBuilder":  # avoid abstract classes
                            if not member_name in self.class_dict:
                                self.class_dict[member_name] = getattr(module, member_name)
            # end initialize for create_op_builder()

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        else:
            return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        self._lazy_init_class_dict()
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return self.class_dict['NotImplementedBuilder'] if 'NotImplementedBuilder' in self.class_dict else None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return []
