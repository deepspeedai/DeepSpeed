# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import abc
from abc import ABC


class DeepSpeedAccelerator(ABC):
    supports_nvtx_domain = False

    def __init__(self):
        self._name = None
        self._communication_backend_name = None
        self._compile_backend = None

    @abc.abstractmethod
    def is_synchronized_device(self):
        ...

    @abc.abstractmethod
    def use_host_timers(self):
        ...

    @abc.abstractmethod
    def resolves_data_dependency(self):
        ...

    @abc.abstractmethod
    def handles_memory_backpressure(self):
        ...

    # Device APIs
    @abc.abstractmethod
    def device_name(self, device_index):
        ...

    @abc.abstractmethod
    def device(self, device_index):
        ...

    @abc.abstractmethod
    def set_device(self, device_index):
        ...

    @abc.abstractmethod
    def current_device(self):
        ...

    @abc.abstractmethod
    def current_device_name(self):
        ...

    @abc.abstractmethod
    def device_count(self):
        ...

    @abc.abstractmethod
    def synchronize(self, device_index=None):
        ...

    # RNG APIs
    @abc.abstractmethod
    def random(self):
        ...

    @abc.abstractmethod
    def set_rng_state(self, new_state, device_index=None):
        ...

    @abc.abstractmethod
    def get_rng_state(self, device_index=None):
        ...

    @abc.abstractmethod
    def manual_seed(self, seed):
        ...

    @abc.abstractmethod
    def manual_seed_all(self, seed):
        ...

    @abc.abstractmethod
    def initial_seed(self):
        ...

    @abc.abstractmethod
    def default_generator(self, device_index):
        ...

    # Streams/Events
    @property
    @abc.abstractmethod
    def Stream(self):
        ...

    @abc.abstractmethod
    def stream(self, stream):
        ...

    @abc.abstractmethod
    def current_stream(self, device_index=None):
        ...

    @abc.abstractmethod
    def default_stream(self, device_index=None):
        ...

    @property
    @abc.abstractmethod
    def Event(self):
        ...

    # Memory management
    @abc.abstractmethod
    def empty_cache(self):
        ...

    @abc.abstractmethod
    def memory_allocated(self, device_index=None):
        ...

    @abc.abstractmethod
    def max_memory_allocated(self, device_index=None):
        ...

    @abc.abstractmethod
    def reset_max_memory_allocated(self, device_index=None):
        ...

    @abc.abstractmethod
    def memory_cached(self, device_index=None):
        ...

    @abc.abstractmethod
    def max_memory_cached(self, device_index=None):
        ...

    @abc.abstractmethod
    def reset_max_memory_cached(self, device_index=None):
        ...

    @abc.abstractmethod
    def memory_stats(self, device_index=None):
        ...

    @abc.abstractmethod
    def reset_peak_memory_stats(self, device_index=None):
        ...

    @abc.abstractmethod
    def memory_reserved(self, device_index=None):
        ...

    @abc.abstractmethod
    def max_memory_reserved(self, device_index=None):
        ...

    @abc.abstractmethod
    def total_memory(self, device_index=None):
        ...

    @abc.abstractmethod
    def available_memory(self, device_index=None):
        ...

    # Data types
    @abc.abstractmethod
    def is_bf16_supported(self):
        ...

    @abc.abstractmethod
    def is_fp16_supported(self):
        ...

    # Not abstract: nearly every accelerator supports fp64, so only those that do not need to override.
    def is_fp64_supported(self):
        return True

    @abc.abstractmethod
    def supported_dtypes(self):
        ...

    # Misc
    @abc.abstractmethod
    def is_available(self):
        ...

    @abc.abstractmethod
    def range_push(self, msg, domain=None, category=None):
        ...

    @abc.abstractmethod
    def range_pop(self, domain=None):
        ...

    @abc.abstractmethod
    def lazy_call(self, callback):
        ...

    @abc.abstractmethod
    def communication_backend_name(self):
        ...

    @abc.abstractmethod
    def is_triton_supported(self):
        ...

    # Whether the fused MoE expert path should prefer a Triton grouped-GEMM
    # kernel over ``torch._grouped_mm`` on this accelerator. Backends override
    # this to return True when the Triton path is faster than the native op
    # (e.g. CUDA sm8x, where ``torch._grouped_mm`` falls back to a slow
    # per-group loop). Defaults to False so backends opt in explicitly.
    def prefer_triton_grouped_mm(self):
        return False

    # Graph operations
    @abc.abstractmethod
    def create_graph(self):
        ...

    @abc.abstractmethod
    def capture_to_graph(self, graph, pool=None, stream=None):
        ...

    @abc.abstractmethod
    def replay_graph(self, graph):
        ...

    # Tensor operations
    @property
    @abc.abstractmethod
    def BFloat16Tensor(self):
        ...

    @property
    @abc.abstractmethod
    def ByteTensor(self):
        ...

    @property
    @abc.abstractmethod
    def DoubleTensor(self):
        ...

    @property
    @abc.abstractmethod
    def FloatTensor(self):
        ...

    @property
    @abc.abstractmethod
    def HalfTensor(self):
        ...

    @property
    @abc.abstractmethod
    def IntTensor(self):
        ...

    @property
    @abc.abstractmethod
    def LongTensor(self):
        ...

    # Memory pinning. The public methods below dispatch between the native
    # backend (``deepspeed.utils.pin_memory``, selected via ``DS_PIN_MEMORY_BACKEND``)
    # and the device-specific torch primitives ``_torch_pin_memory``/``_torch_is_pinned``,
    # which subclasses override as needed. The native utility is backend-only and
    # never calls back here.
    def _torch_pin_memory(self, tensor):
        return tensor.pin_memory()

    def _torch_empty_pinned(self, tensor, shape):
        return tensor.new_empty(shape, pin_memory=True)

    def _torch_is_pinned(self, tensor):
        return tensor.is_pinned()

    def register_host_memory(self, address, num_bytes):
        """Register page-locked host memory with the active device runtime."""
        return False

    def unregister_host_memory(self, address):
        """Unregister host memory previously registered with the device runtime."""
        return None

    @staticmethod
    def _shape_numel(shape):
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel

    def _active_native_pins(self, nbytes):
        from deepspeed.utils.pin_memory_tracker import track_pinned_memory
        track_pinned_memory(nbytes)
        from deepspeed.utils.pin_memory import get_active_native_pinned_memory
        return get_active_native_pinned_memory()

    def pin_memory(self, tensor, make_copy=True, match_shape=True, alloc_shape=None):
        # ``alloc_shape`` is the size/dtype-only path used by pin_empty: ``tensor``
        # is just a dtype template and must not dictate the allocation numel.
        if alloc_shape is None:
            nbytes = tensor.nbytes
            out_shape = tensor.shape if match_shape else (tensor.numel(), )
        else:
            numel = self._shape_numel(alloc_shape)
            nbytes = numel * tensor.element_size()
            out_shape = tuple(alloc_shape) if match_shape else (numel, )
        pins = self._active_native_pins(nbytes)
        if pins is not None:
            if alloc_shape is not None:
                return pins.pin_empty(tensor, out_shape)
            return pins.pin(tensor, make_copy=make_copy, match_shape=match_shape)
        if make_copy and alloc_shape is None:
            return self._torch_pin_memory(tensor)
        # ``tensor`` is only a shape/dtype template here, so page-lock a fresh
        # buffer instead of faulting it in and copying it into a second one.
        return self._torch_empty_pinned(tensor, out_shape)

    def pin_empty(self, *size, dtype=None, device='cpu'):
        # Scratch destinations: shape/dtype only, no pageable template of ``size``.
        import torch
        if len(size) == 1 and not isinstance(size[0], int):
            shape = tuple(size[0])
        else:
            shape = tuple(size)
        template = torch.empty(0, dtype=dtype, device=device)
        return self.pin_memory(template, make_copy=False, alloc_shape=shape)

    def pin_empty_like(self, tensor, device='cpu', dtype=None):
        import torch
        template = torch.empty(0, dtype=tensor.dtype if dtype is None else dtype, device=device)
        return self.pin_memory(template, make_copy=False, alloc_shape=tuple(tensor.shape))

    def is_pinned(self, tensor):
        from deepspeed.utils.pin_memory import get_active_native_pinned_memory
        pins = get_active_native_pinned_memory()
        if pins is not None and pins.is_pinned(tensor):
            return True
        return self._torch_is_pinned(tensor)

    def unpin_memory(self, tensor):
        from deepspeed.utils.pin_memory import get_active_native_pinned_memory
        pins = get_active_native_pinned_memory()
        if pins is not None:
            return pins.unpin(tensor)
        return None

    @abc.abstractmethod
    def on_accelerator(self, tensor):
        ...

    @abc.abstractmethod
    def op_builder_dir(self):
        ...

    # create an instance of op builder, specified by class_name
    @abc.abstractmethod
    def create_op_builder(self, class_name):
        ...

    # return an op builder class, specified by class_name
    @abc.abstractmethod
    def get_op_builder(self, class_name):
        ...

    @abc.abstractmethod
    def build_extension(self):
        ...

    @abc.abstractmethod
    def export_envs(self):
        ...

    @abc.abstractmethod
    def visible_devices_envs(self):
        ...

    @abc.abstractmethod
    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        ...

    @abc.abstractmethod
    def get_compile_backend(self):
        ...

    @abc.abstractmethod
    def set_compile_backend(self, backend):
        ...
