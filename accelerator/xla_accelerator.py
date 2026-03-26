# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import functools

import torch

from .abstract_accelerator import DeepSpeedAccelerator

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError as e:
    torch_xla = None
    xm = None


class XLA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'xla'
        self._communication_backend_name = 'xla'
        self._compile_backend = None
        if xm is None:
            raise ValueError("XLA_Accelerator requires torch_xla, which is not installed on this system.")

    def _require_xm(self):
        if xm is None:
            raise RuntimeError("torch_xla is required to use the XLA_Accelerator")
        return xm

    def _tensor_factory(self, dtype):
        return functools.partial(torch.tensor, dtype=dtype, device=self.current_device_name())

    def _addressable_devices(self):
        if torch_xla is not None and hasattr(torch_xla, 'devices'):
            return list(torch_xla.devices())

        xm_module = self._require_xm()
        return [torch.device(device) for device in xm_module.get_xla_supported_devices(devkind='TPU')]

    def _normalize_device_index(self, device_index=None):
        devices = self._addressable_devices()
        if not devices:
            raise RuntimeError("No addressable XLA devices are available in the current process.")
        if device_index is None:
            return 0
        if isinstance(device_index, torch.device):
            device_index = device_index.index if device_index.index is not None else 0
        elif isinstance(device_index, str):
            device_index = int(device_index) if device_index.isdigit() else int(device_index.split(':')[-1])
        return min(device_index, len(devices) - 1)

    def is_synchronized_device(self):
        return True

    def use_host_timers(self):
        return True

    def resolves_data_dependency(self):
        return True

    def handles_memory_backpressure(self):
        return True

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'xla'
        return str(self._addressable_devices()[self._normalize_device_index(device_index)])

    def device(self, device_index=None):
        xm_module = self._require_xm()
        if device_index is None:
            return xm_module.xla_device(devkind='TPU')
        return xm_module.xla_device(n=self._normalize_device_index(device_index), devkind='TPU')

    def set_device(self, device_index):
        # XLA uses the default device selected for the current process.
        self.device(device_index)
        os.environ['LOCAL_RANK'] = str(device_index)
        os.environ.setdefault('PJRT_LOCAL_PROCESS_RANK', str(device_index))

    def current_device(self):
        current_device = self.device()
        device_index = getattr(current_device, 'index', None)
        if device_index is not None:
            return device_index
        return self._normalize_device_index()

    def current_device_name(self):
        return str(self.device())

    def device_count(self):
        xm_module = self._require_xm()
        return len(xm_module.get_xla_supported_devices(devkind='TPU'))

    def synchronize(self, device_index=None):
        xm_module = self._require_xm()
        xm_module.mark_step()
        return xm_module.wait_device_ops()

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        return torch.set_rng_state(new_state)

    def get_rng_state(self, device_index=None):
        return torch.get_rng_state()

    def manual_seed(self, seed):
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.manual_seed(seed)

    def initial_seed(self):
        return torch.initial_seed()

    def default_generator(self, device_index):
        return torch.default_generator

    # Streams/Events
    @property
    def Stream(self):
        return None

    def stream(self, stream):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):
        return

    def memory_allocated(self, device_index=None):
        return 0

    def max_memory_allocated(self, device_index=None):
        return 0

    def reset_max_memory_allocated(self, device_index=None):
        return

    def memory_cached(self, device_index=None):
        return 0

    def max_memory_cached(self, device_index=None):
        return 0

    def reset_max_memory_cached(self, device_index=None):
        return

    def memory_stats(self, device_index=None):
        return {}

    def reset_peak_memory_stats(self, device_index=None):
        return

    def memory_reserved(self, device_index=None):
        return 0

    def max_memory_reserved(self, device_index=None):
        return 0

    def total_memory(self, device_index=None):
        return 0

    def available_memory(self, device_index=None):
        return 0

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return False

    def supported_dtypes(self):
        return [torch.float32, torch.bfloat16]

    # Misc
    def is_available(self):
        return self.device_count() > 0

    def range_push(self, msg):
        return

    def range_pop(self):
        return

    def lazy_call(self, callback):
        return callback()

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
        return self._tensor_factory(torch.bfloat16)

    @property
    def ByteTensor(self):
        return self._tensor_factory(torch.uint8)

    @property
    def DoubleTensor(self):
        return self._tensor_factory(torch.float64)

    @property
    def FloatTensor(self):
        return self._tensor_factory(torch.float32)

    @property
    def HalfTensor(self):
        return self._tensor_factory(torch.float16)

    @property
    def IntTensor(self):
        return self._tensor_factory(torch.int32)

    @property
    def LongTensor(self):
        return self._tensor_factory(torch.int64)

    def pin_memory(self, tensor, align_bytes=1):
        return tensor

    def is_pinned(self, tensor):
        return False

    def on_accelerator(self, tensor):
        return getattr(tensor.device, 'type', None) == 'xla'

    def op_builder_dir(self):
        return "deepspeed.ops.op_builder.cpu"

    def create_op_builder(self, op_name):
        return None

    def get_op_builder(self, class_name):
        return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return ['PJRT_DEVICE', 'TPU_VISIBLE_CHIPS']

    def visible_devices_envs(self):
        return ['TPU_VISIBLE_CHIPS']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))
        current_env.setdefault('PJRT_DEVICE', 'TPU')

    def get_compile_backend(self):
        return self._compile_backend

    def set_compile_backend(self, backend):
        if backend is not None:
            raise ValueError(f"{backend} not supported by {self.device_name()}. Supported Backends are [None]")
