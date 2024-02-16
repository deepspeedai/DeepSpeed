# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .abstract_accelerator import DeepSpeedAccelerator

# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch.mps
except ImportError:
    pass


class MPS_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = "mps"
        self._communication_backend_name = None

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
            return "mps"
        return "mps:{}".format(device_index)

    def device(self, device_index):
        return torch.device("mps", index=0)

    def set_device(self, device_index):
        return

    def current_device(self):
        return torch.device("mps", index=0)

    def current_device_name(self):
        return "mps:0"

    def device_count(self):
        return 1

    def synchronize(self, device_index=None):
        return torch.mps.synchronize()

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        return torch.mps.set_rng_state(new_state)

    def get_rng_state(self, device_index=None):
        return torch.mps.get_rng_state()

    def manual_seed(self, seed):
        return torch.mps.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.mps.manual_seed(seed)

    def seed(self):
        return torch.mps.seed()

    def initial_seed(self, seed):
        return

    def default_generator(self, device_index):
        return

    # Streams/Events
    @property
    def Stream(self):
        return None

    def stream(self, stream):
        return None

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):
        return torch.mps.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.mps.current_allocated_memory()

    def max_memory_allocated(self, device_index=None):
        return torch.mps.driver_allocated_memory()

    def set_per_process_memory_fraction(self, fraction):
        return torch.mps.set_per_process_memory_fraction(fraction)

    def reset_max_memory_allocated(self, device_index=None):
        return

    def memory_cached(self, device_index=None):
        return

    def max_memory_cached(self, device_index=None):
        return

    def reset_max_memory_cached(self, device_index=None):
        return

    def memory_stats(self, device_index=None):
        return

    def reset_peak_memory_stats(self, device_index=None):
        return

    def memory_reserved(self, device_index=None):
        return

    def max_memory_reserved(self, device_index=None):
        return

    def total_memory(self, device_index=None):
        return

    def available_memory(self, device_index=None):
        return

    # Data types
    def is_bf16_supported(self):
        return False

    def is_fp16_supported(self):
        return False

    def supported_dtypes(self):
        return [torch.float]

    # Misc
    def amp(self):
        return

    def is_available(self):
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def range_push(self, msg):
        return

    def range_pop(self):
        return

    def lazy_call(self, callback):
        return

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
        return

    @property
    def ByteTensor(self):
        return

    @property
    def DoubleTensor(self):
        return

    @property
    def FloatTensor(self):
        return

    @property
    def HalfTensor(self):
        return

    @property
    def IntTensor(self):
        return

    @property
    def LongTensor(self):
        return

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith("mps"):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore

            return "op_builder"
        except ImportError:
            return "deepspeed.ops.op_builder"

    # create an instance of op builder, specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class is not None:
            return builder_class()
        return None

    # return an op builder class, specified by class_name
    def get_op_builder(self, class_name):
        from deepspeed.ops.op_builder.cpu import NotImplementedBuilder

        return NotImplementedBuilder

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension

        return BuildExtension

    def export_envs(self):
        return []
