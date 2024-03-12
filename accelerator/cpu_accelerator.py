# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from .abstract_accelerator import DeepSpeedAccelerator

try:
    import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
    oneccl_imported_p = True
except ImportError as e:
    oneccl_imported_p = False

import os


# accelerator for Intel CPU
class CPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cpu'
        if oneccl_imported_p:
            self._communication_backend_name = 'ccl'
        else:
            # fallback to gloo if oneccl_binding_for_pytorch is not installed
            self._communication_backend_name = 'gloo'
        try:
            import psutil
            mem = psutil.Process().memory_info().rss
            self.max_mem = mem
        except ImportError as e:
            self.max_mem = 0

    def is_synchronized_device(self):
        return True

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        return 'cpu'

    def device(self, device_index=None):
        return None

    def set_device(self, device_index):
        return

    def current_device(self):
        return os.environ.get('LOCAL_RANK', 0)

    def current_device_name(self):
        return 'cpu'

    def device_count(self):
        device_count = int(os.environ.get('LOCAL_SIZE', 0))
        if device_count > 0:
            return device_count
        else:
            from deepspeed.utils.numa import get_numa_cores
            # Count NUMA node for number of cpu accelerators. On machine with HBM
            # In flat mode, HBM is in separate NUMA node with no cores on this node.
            # Ignore these NUMA nodes with no cores.
            numa_core_lists = get_numa_cores()
            numa_count = 0
            prev_core_list = []
            for core_list in numa_core_lists:
                if len(core_list) > 0 and core_list != prev_core_list:
                    numa_count += 1
                    prev_core_list = core_list
            return numa_count

    def synchronize(self, device_index=None):
        return

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.set_rng_state(new_state)
        return torch.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        return torch.get_rng_state()

    def manual_seed(self, seed):
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.manual_seed(seed)

    def initial_seed(self, seed):
        return torch.initial_seed(seed)

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

    def get_rss(self):
        import psutil
        mem = psutil.Process().memory_info().rss
        if mem > self.max_mem:
            self.max_mem = mem
        return mem

    def reset_rss(self):
        import psutil
        mem = psutil.Process().memory_info().rss
        self.max_mem = mem
        return mem

    def memory_allocated(self, device_index=None):
        return self.get_rss()

    def max_memory_allocated(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_allocated(self, device_index=None):
        self.reset_rss()
        return

    def memory_cached(self, device_index=None):
        return self.get_rss()

    def max_memory_cached(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_cached(self, device_index=None):
        self.reset_rss()
        return

    def memory_stats(self, device_index=None):
        mem = self.get_rss()
        mem_stat = {}
        mem_stat['allocated_bytes.all.current'] = mem
        mem_stat['allocated_bytes.all.peak'] = self.max_mem
        return mem_stat

    def reset_peak_memory_stats(self, device_index=None):
        self.reset_rss()
        return

    def memory_reserved(self, device_index=None):
        return self.get_rss()

    def max_memory_reserved(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def total_memory(self, device_index=None):
        import psutil
        return psutil.virtual_memory().total

    def available_memory(self, device_index=None):
        import psutil
        return psutil.virtual_memory().available

    # Misc
    def amp(self):
        return torch.cpu.amp

    def is_available(self):
        return True

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return False

    def supported_dtypes(self):
        return [torch.float, torch.bfloat16]

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
        return torch.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.FloatTensor

    @property
    def HalfTensor(self):
        return torch.HalfTensor

    @property
    def IntTensor(self):
        return torch.IntTensor

    @property
    def LongTensor(self):
        return torch.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        return tensor

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.cpu"
        except ImportError:
            return "deepspeed.ops.op_builder.cpu"

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cpu'):
            return True
        else:
            return False

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, op_name):
        builder_class = self.get_op_builder(op_name)
        if builder_class is not None:
            return builder_class()
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            from op_builder.cpu import CCLCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder
        except ImportError:
            from deepspeed.ops.op_builder.cpu import CCLCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder

        if class_name == "CCLCommBuilder":
            return CCLCommBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        else:
            # return a NotImplementedBuilder to avoid get NoneType[Name] in unit tests
            return NotImplementedBuilder

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return []
