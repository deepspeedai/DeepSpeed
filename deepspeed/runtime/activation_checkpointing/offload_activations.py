# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Offload activation checkpoint inputs (hidden_states) to CPU for non-reentrant
checkpointing (use_reentrant=False, the HF Transformers default)

A saved_tensors_hooks context manager offloads only marked checkpoint inputs to
a pinned CPU buffer pool on a side stream, overlapping copies with compute
Independent of deepspeed.checkpointing.configure() state; eager-mode
counterpart of the DeepCompile offload_activation pass
transformers is imported lazily so importing this module works without it

Adapted from axolotl checkpoint_activation_offload.py
https://github.com/axolotl-ai-cloud/axolotl/pull/3776
"""

from __future__ import annotations

import contextlib
import importlib.util
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.autograd.graph import saved_tensors_hooks

from deepspeed.accelerator import get_accelerator


@dataclass
class CheckpointActivationOffloadStats:
    saved_tensors_seen: int = 0
    marked_tensors: int = 0
    offloaded_tensors: int = 0
    restored_tensors: int = 0
    skipped_marked_tensors: int = 0
    offloaded_bytes: int = 0
    restored_bytes: int = 0


@dataclass(frozen=True)
class _OffloadedTensorRef:
    tensor_id: int


_BufferKey = Tuple[Tuple[int, ...], Tuple[int, ...], torch.dtype, torch.layout, bool]
_MANAGER_STACK: list["CheckpointHiddenStatesOffload"] = []
_ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL = None


def _current_manager() -> "CheckpointHiddenStatesOffload | None":
    return _MANAGER_STACK[-1] if _MANAGER_STACK else None


def patch_gradient_checkpointing_layer_marker() -> None:
    global _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL

    if _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL is not None:
        return

    try:
        from transformers import GradientCheckpointingLayer
    except ImportError as e:
        raise RuntimeError("patch_gradient_checkpointing_layer_marker requires the `transformers` "
                           "package (>= 4.52, which introduced GradientCheckpointingLayer). "
                           "Install it with `pip install transformers`.") from e

    _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL = GradientCheckpointingLayer.__call__

    def _checkpoint_offload_call(self, *args, **kwargs):
        manager = _current_manager()
        if (manager is not None and self.training and torch.is_grad_enabled()
                and getattr(self, "gradient_checkpointing", False)):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if torch.is_tensor(hidden_states):
                manager.mark(hidden_states)
        return _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL(self, *args, **kwargs)

    GradientCheckpointingLayer.__call__ = _checkpoint_offload_call


class CheckpointHiddenStatesOffload(saved_tensors_hooks):
    """Offload only marked checkpoint inputs.

    The marker is installed on ``GradientCheckpointingLayer.__call__`` before
    non-reentrant checkpointing saves positional layer inputs. All other saved
    tensors pass through unchanged, including tensors from the final norm/head
    and any non-checkpointed modules.
    """

    def __init__(
        self,
        use_pin_memory: bool = True,
        use_streams: bool = True,
        min_offload_size: int = 1024,
        max_fwd_stash_size: int = 2,
        max_cpu_buffer_pool_size: int = 64,
    ) -> None:
        self.use_pin_memory = use_pin_memory
        self.use_streams = use_streams
        self.min_tensor_size_bytes = min_offload_size
        self.max_fwd_stash_size = max_fwd_stash_size
        self.max_cpu_buffer_pool_size = max_cpu_buffer_pool_size
        self.stats = CheckpointActivationOffloadStats()
        self._allowed: dict[int, int] = {}
        self._next_id = 0
        self._tracker: dict[int, tuple[torch.Tensor, torch.device, torch.Size, tuple[int, ...], _BufferKey]] = {}
        self._fwd_stash: dict[int, tuple[torch.Tensor, torch.Event]] = {}
        self._cpu_buffer_pool: dict[_BufferKey, list[torch.Tensor]] = {}
        self._cpu_buffer_pool_size = 0
        self._pending_cpu_buffers: list[tuple[_BufferKey, torch.Tensor, torch.Event]] = []

        # cpu-like accelerators have no streams; _pack_tensor never offloads there
        stream_cls = get_accelerator().Stream
        self.s1 = stream_cls() if (use_streams and stream_cls is not None) else None

        super().__init__(self._pack_tensor, self._unpack_tensor)

    @property
    def compute_stream(self):
        # resolved per use: the compute stream may differ from the construction-time one
        return get_accelerator().current_stream()

    def _stream_context(self, stream):
        if stream is None:
            return contextlib.nullcontext()
        return get_accelerator().stream(stream)

    def mark(self, tensor: torch.Tensor) -> None:
        self._allowed[id(tensor)] = self._allowed.get(id(tensor), 0) + 1
        self.stats.marked_tensors += 1

    def _consume_mark(self, tensor: torch.Tensor) -> bool:
        tensor_id = id(tensor)
        count = self._allowed.get(tensor_id, 0)
        if count <= 0:
            return False
        if count == 1:
            del self._allowed[tensor_id]
        else:
            self._allowed[tensor_id] = count - 1
        return True

    @staticmethod
    def _num_bytes(tensor: torch.Tensor) -> int:
        return tensor.element_size() * tensor.nelement()

    def _next_tensor_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _reap_forward_stash(self, new_tensor_id: int) -> None:
        for tensor_id in list(self._fwd_stash):
            if tensor_id > new_tensor_id - self.max_fwd_stash_size:
                continue
            _, event = self._fwd_stash.pop(tensor_id)
            self.compute_stream.wait_event(event)

    def _buffer_key(self, tensor: torch.Tensor) -> _BufferKey:
        return (
            tuple(tensor.size()),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.layout,
            self.use_pin_memory,
        )

    def _pool_cpu_buffer(self, key: _BufferKey, tensor: torch.Tensor) -> None:
        if self._cpu_buffer_pool_size >= self.max_cpu_buffer_pool_size:
            return
        self._cpu_buffer_pool.setdefault(key, []).append(tensor)
        self._cpu_buffer_pool_size += 1

    def _reap_cpu_buffer_pool(self, force: bool = False) -> None:
        pending = self._pending_cpu_buffers
        self._pending_cpu_buffers = []
        for key, tensor, event in pending:
            if force:
                event.synchronize()
                self._pool_cpu_buffer(key, tensor)
            elif event.query():
                self._pool_cpu_buffer(key, tensor)
            else:
                self._pending_cpu_buffers.append((key, tensor, event))

    def _empty_cpu_like(self, tensor: torch.Tensor) -> tuple[torch.Tensor, _BufferKey]:
        self._reap_cpu_buffer_pool()
        key = self._buffer_key(tensor)
        pool = self._cpu_buffer_pool.get(key)
        if pool:
            self._cpu_buffer_pool_size -= 1
            return pool.pop(), key
        # empty_strided(pin_memory=True) preserves strides; get_accelerator().pin_memory() copies and may not
        return torch.empty_strided(
            tuple(tensor.size()),
            tuple(tensor.stride()),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device="cpu",
            pin_memory=self.use_pin_memory,
        ), key

    def _pack_tensor(self, tensor: torch.Tensor):
        self.stats.saved_tensors_seen += 1
        if not self._consume_mark(tensor):
            return tensor

        num_bytes = self._num_bytes(tensor)
        if (get_accelerator().is_synchronized_device() or not get_accelerator().on_accelerator(tensor)
                or num_bytes < self.min_tensor_size_bytes or isinstance(tensor, torch.nn.Parameter)
                or (hasattr(torch.nn, "Buffer") and isinstance(tensor, torch.nn.Buffer))):
            self.stats.skipped_marked_tensors += 1
            return tensor

        tensor_id = self._next_tensor_id()
        if self.use_streams:
            self._reap_forward_stash(tensor_id)
            self.s1.wait_stream(self.compute_stream)
            stream = self.s1
        else:
            stream = self.compute_stream

        with self._stream_context(stream):
            cpu_tensor, buffer_key = self._empty_cpu_like(tensor)
            cpu_tensor.copy_(tensor.detach(), non_blocking=self.use_streams)

        self._tracker[tensor_id] = (
            cpu_tensor,
            tensor.device,
            tensor.size(),
            tensor.stride(),
            buffer_key,
        )
        if self.use_streams:
            self._fwd_stash[tensor_id] = (tensor, self.s1.record_event())

        self.stats.offloaded_tensors += 1
        self.stats.offloaded_bytes += num_bytes
        return _OffloadedTensorRef(tensor_id)

    def _restore_tensor(self, tensor_id: int) -> torch.Tensor:
        if tensor_id in self._fwd_stash:
            tensor, event = self._fwd_stash.pop(tensor_id)
            self.compute_stream.wait_event(event)
            cpu_tensor, *_unused, buffer_key = self._tracker.pop(tensor_id)
            self._pool_cpu_buffer(buffer_key, cpu_tensor)
            self.stats.restored_tensors += 1
            self.stats.restored_bytes += self._num_bytes(tensor)
            return tensor

        cpu_tensor, device, shape, stride, buffer_key = self._tracker.pop(tensor_id)
        stream = self.s1 if self.use_streams else self.compute_stream
        with self._stream_context(stream):
            # restore into fresh offset-0 storage; reapplying the source storage_offset
            # would index past the pool buffer's storage for sliced views
            gpu_tensor = torch.empty_strided(shape, stride, dtype=cpu_tensor.dtype, device=device)
            gpu_tensor.copy_(cpu_tensor, non_blocking=self.use_streams)
        if self.use_streams:
            compute = self.compute_stream
            event = self.s1.record_event()
            compute.wait_event(event)
            gpu_tensor.record_stream(compute)
            self._pending_cpu_buffers.append((buffer_key, cpu_tensor, event))
        else:
            self._pool_cpu_buffer(buffer_key, cpu_tensor)

        self.stats.restored_tensors += 1
        self.stats.restored_bytes += self._num_bytes(gpu_tensor)
        return gpu_tensor

    def _unpack_tensor(self, maybe_ref):
        if isinstance(maybe_ref, _OffloadedTensorRef):
            return self._restore_tensor(maybe_ref.tensor_id)
        return maybe_ref

    def reset(self) -> None:
        self.stats = CheckpointActivationOffloadStats()
        self._allowed.clear()
        self._tracker.clear()
        self._fwd_stash.clear()
        self._pending_cpu_buffers.clear()
        self._next_id = 0

    def _sync_and_clear(self) -> None:
        if self.use_streams:
            compute = self.compute_stream
            if compute is not None:
                compute.synchronize()
            if self.s1 is not None:
                self.s1.synchronize()
            self._reap_cpu_buffer_pool(force=True)
        for tracked in self._tracker.values():
            self._pool_cpu_buffer(tracked[-1], tracked[0])
        self._allowed.clear()
        self._tracker.clear()
        self._fwd_stash.clear()
        self._pending_cpu_buffers.clear()

    def __enter__(self):
        # auto-install the marker if transformers is present; mark() still works without it
        if _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL is None and importlib.util.find_spec("transformers") is not None:
            patch_gradient_checkpointing_layer_marker()
        self.reset()
        _MANAGER_STACK.append(self)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        try:
            self._sync_and_clear()
        finally:
            if _MANAGER_STACK and _MANAGER_STACK[-1] is self:
                _MANAGER_STACK.pop()
            elif self in _MANAGER_STACK:
                _MANAGER_STACK.remove(self)
        return super().__exit__(*args, **kwargs)


def get_checkpoint_hidden_states_offloading_ctx_manager(
    use_pin_memory: bool = True,
    use_streams: bool = True,
    min_offload_size: int = 1024,
    max_fwd_stash_size: int = 2,
    max_cpu_buffer_pool_size: int = 64,
) -> CheckpointHiddenStatesOffload:
    return CheckpointHiddenStatesOffload(
        use_pin_memory=use_pin_memory,
        use_streams=use_streams,
        min_offload_size=min_offload_size,
        max_fwd_stash_size=max_fwd_stash_size,
        max_cpu_buffer_pool_size=max_cpu_buffer_pool_size,
    )
