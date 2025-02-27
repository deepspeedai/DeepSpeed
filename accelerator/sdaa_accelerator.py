# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

# DeepSpeed Team

import importlib
import inspect
import functools

from .abstract_accelerator import DeepSpeedAccelerator
# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.sdaa
except ImportError:
    pass


class SDAA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'sdaa'
        self._communication_backend_name = 'tccl'
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
            return 'sdaa'
        return 'sdaa:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.sdaa.device(device_index)

    def set_device(self, device_index):
        torch.sdaa.set_device(device_index)

    def current_device(self):
        return torch.sdaa.current_device()

    def current_device_name(self):
        return 'sdaa:{}'.format(torch.sdaa.current_device())

    def device_count(self):
        return torch.sdaa.device_count()

    def synchronize(self, device_index=None):
        return torch.sdaa.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.sdaa.set_rng_state(new_state)

        return torch.sdaa.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.sdaa.get_rng_state()

        return torch.sdaa.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.sdaa.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.sdaa.manual_seed_all(seed)

    def initial_seed(self):
        return torch.sdaa.initial_seed()

    def default_generator(self, device_index):
        return torch.sdaa.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        return torch.sdaa.Stream

    def stream(self, stream):
        return torch.sdaa.stream(stream)

    def current_stream(self, device_index=None):
        return torch.sdaa.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.sdaa.default_stream(device_index)

    @property
    def Event(self):
        return torch.sdaa.Event

    # Memory management
    def empty_cache(self):
        return torch.sdaa.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.sdaa.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.sdaa.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.sdaa.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.sdaa.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.sdaa.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.sdaa.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.sdaa, 'memory_stats'):
            return torch.sdaa.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.sdaa, 'reset_peak_memory_stats'):
            return torch.sdaa.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.sdaa, 'memory_reserved'):
            return torch.sdaa.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.sdaa, 'max_memory_reserved'):
            return torch.sdaa.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.sdaa.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        return torch.sdaa.is_bf16_supported()

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
        if hasattr(torch.sdaa, 'amp'):
            return torch.sdaa.amp
        return None

    def is_available(self):
        return torch.sdaa.is_available()

    def range_push(self, msg):
        return

    def range_pop(self):
        return

    def lazy_call(self, callback):
        return torch.sdaa._lazy_call(callback)

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

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device='sdaa')

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device='sdaa')

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.double, device='sdaa')

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device='sdaa')

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device='sdaa')

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device='sdaa')

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.long, device='sdaa')

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('sdaa:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.sdaa"
        except ImportError:
            return "deepspeed.ops.op_builder.sdaa"

    def _lazy_init_class_dict(self):
        if self.class_dict:
            return

        op_builder_module = importlib.import_module(self.op_builder_dir())

        # get op builder class from op_builder/sdaa/__init__.py
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
        return ['NCCL', 'LD_LIBRARY', 'PATH']

    def visible_devices_envs(self):
        return ['SDAA_VISIBLE_DEVICES']

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
