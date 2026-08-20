# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Offload activation checkpoint inputs to CPU on a pinned side-stream buffer pool.

Two consumers share one ``_ActivationOffloadEngine``:

* ``CheckpointHiddenStatesOffload``: a ``saved_tensors_hooks`` context manager for
  HF non-reentrant checkpointing that offloads only marked inputs.
* DeepSpeed native ``cpu_checkpointing`` (checkpointing.py), which drives the
  engine directly instead of blocking ``.to('cpu')`` / ``.to(cuda)``.

transformers is imported lazily. Adapted from axolotl checkpoint_activation_offload.py
https://github.com/axolotl-ai-cloud/axolotl/pull/3776
"""

from __future__ import annotations

import contextlib
import importlib.util
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

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
    kept_last_tensors: int = 0
    offloaded_bytes: int = 0
    restored_bytes: int = 0


@dataclass(frozen=True)
class _OffloadedTensorRef:
    tensor_id: int


_BufferKey = Tuple[Tuple[int, ...], torch.dtype, torch.layout, bool]
_TLS = threading.local()
_PATCH_LOCK = threading.Lock()
_ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL = None
_MARKER_WRAPPER = None
_MARKER_INSTALLED_BY_CONTEXT = False
_PATCH_USERS = 0


class _ActivationOffloadEngine:
    """Async CPU offload of activation tensors on a pinned side stream.

    Owns the buffer pool, stream, stash, keep-last, and copy logic. Callers drive
    it with ``offload_input`` / ``restore_input``.
    """

    def __init__(
        self,
        use_pin_memory: bool = True,
        use_streams: bool = True,
        min_offload_bytes: int = 1024,
        max_fwd_stash_count: int = 2,
        max_cpu_buffer_pool_count: int = 64,
        keep_last_count: int = 1,
    ) -> None:
        if keep_last_count < 0:
            raise ValueError(f"keep_last_count must be >= 0, got {keep_last_count}")
        if max_fwd_stash_count < 0:
            raise ValueError(f"max_fwd_stash_count must be >= 0, got {max_fwd_stash_count}")
        if max_cpu_buffer_pool_count < 0:
            raise ValueError(f"max_cpu_buffer_pool_count must be >= 0, got {max_cpu_buffer_pool_count}")
        if min_offload_bytes < 0:
            raise ValueError(f"min_offload_bytes must be >= 0, got {min_offload_bytes}")

        self.use_pin_memory = use_pin_memory
        self.min_offload_bytes = min_offload_bytes
        self.max_fwd_stash_count = max_fwd_stash_count
        self.max_cpu_buffer_pool_count = max_cpu_buffer_pool_count
        # Keep the last inputs on GPU; the first backward needs them before a
        # D2H would finish.
        self.keep_last_count = keep_last_count
        self.stats = CheckpointActivationOffloadStats()
        self._next_id = 0
        self._tracker: dict[int, tuple[torch.Tensor, torch.device, torch.Size, tuple[int, ...], _BufferKey]] = {}
        self._fwd_stash: dict[int, tuple[torch.Tensor, torch.Event]] = {}
        self._keep_last: dict[int, torch.Tensor] = {}
        self._restored: dict[int, torch.Tensor] = {}
        self._cpu_buffer_pool: dict[_BufferKey, list[torch.Tensor]] = {}
        self._cpu_buffer_pool_count = 0
        self._pending_cpu_buffers: list[tuple[_BufferKey, torch.Tensor, torch.Event]] = []

        # cpu-like accelerators have no streams; offload_input never offloads there
        stream_cls = get_accelerator().Stream
        self.s1 = stream_cls() if (use_streams and stream_cls is not None) else None
        self.use_streams = bool(use_streams and self.s1 is not None)

    @property
    def compute_stream(self):
        # resolve per use; the compute stream may change after construction
        return get_accelerator().current_stream()

    def _stream_context(self, stream):
        if stream is None:
            return contextlib.nullcontext()
        return get_accelerator().stream(stream)

    @staticmethod
    def _num_bytes(tensor: torch.Tensor) -> int:
        return tensor.element_size() * tensor.nelement()

    def _next_tensor_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _should_skip(self, tensor: torch.Tensor) -> bool:
        # Overlapping (stride-0) views are skipped because restore rebuilds the
        # stride and copy_ cannot write into aliased storage.
        num_bytes = self._num_bytes(tensor)
        return (get_accelerator().is_synchronized_device() or not get_accelerator().on_accelerator(tensor)
                or num_bytes < self.min_offload_bytes or isinstance(tensor, torch.nn.Parameter)
                or (hasattr(torch.nn, "Buffer") and isinstance(tensor, torch.nn.Buffer))
                or (tensor.numel() > 1 and 0 in tensor.stride()))

    def _reap_forward_stash(self, new_tensor_id: int) -> None:
        # Drop GPU refs past the stash window; record_stream (offload) holds the
        # allocation until s1 drains, so compute never stalls.
        for tensor_id in list(self._fwd_stash):
            if tensor_id > new_tensor_id - self.max_fwd_stash_count:
                continue
            self._fwd_stash.pop(tensor_id)

    def _buffer_key(self, tensor: torch.Tensor) -> _BufferKey:
        # Pooled host buffers are dense, so sources that differ only in stride can
        # share one buffer.
        return (
            tuple(tensor.size()),
            tensor.dtype,
            tensor.layout,
            self.use_pin_memory,
        )

    def _pool_cpu_buffer(self, key: _BufferKey, tensor: torch.Tensor) -> None:
        if self._cpu_buffer_pool_count >= self.max_cpu_buffer_pool_count:
            return
        self._cpu_buffer_pool.setdefault(key, []).append(tensor)
        self._cpu_buffer_pool_count += 1

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
            self._cpu_buffer_pool_count -= 1
            return pool.pop(), key
        # Dense host buffer: restore rebuilds strides device-side, so matching
        # shape/dtype is enough and we pin via the accelerator API (honors
        # DS_PIN_MEMORY_BACKEND and its accounting).
        host = torch.empty(tuple(tensor.size()), dtype=tensor.dtype, layout=tensor.layout, device="cpu")
        if self.use_pin_memory:
            host = get_accelerator().pin_memory(host, make_copy=False)
        return host, key

    def _start_offload(self, tensor_id: int, tensor: torch.Tensor, reap_id: int | None = None) -> None:
        if self.use_streams:
            self._reap_forward_stash(tensor_id if reap_id is None else reap_id)
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
            # record_stream holds the allocation until s1 drains, so a zero stash
            # can drop the GPU ref now.
            tensor.record_stream(self.s1)
            if self.max_fwd_stash_count > 0:
                self._fwd_stash[tensor_id] = (tensor, self.s1.record_event())

        self.stats.offloaded_tensors += 1
        self.stats.offloaded_bytes += self._num_bytes(cpu_tensor)

    def _flush_keep_last(self, newest_id: int) -> None:
        while len(self._keep_last) > self.keep_last_count:
            oldest_id, oldest = next(iter(self._keep_last.items()))
            del self._keep_last[oldest_id]
            self._start_offload(oldest_id, oldest, reap_id=newest_id)

    def offload_input(self, tensor: torch.Tensor) -> Optional[int]:
        """Register a tensor for offload; returns its id, or None if skipped.

        The newest ``keep_last_count`` tensors stay on GPU; older ones start an
        async D2H. The caller must ``restore_input`` before reusing the storage.
        """
        if self._should_skip(tensor):
            self.stats.skipped_marked_tensors += 1
            return None
        tensor_id = self._next_tensor_id()
        self._keep_last[tensor_id] = tensor
        self._flush_keep_last(tensor_id)
        return tensor_id

    def restore_input(self, tensor_id: int) -> torch.Tensor:
        """Restore an offloaded tensor for a single consumption (no caching)."""
        return self._restore_tensor(tensor_id, cache=False)

    def _restore_tensor(self, tensor_id: int, cache: bool = True) -> torch.Tensor:
        # cache=True allows repeated unpack of one id (HF retain_graph), cleared on
        # context exit; cache=False fully consumes.
        if cache:
            cached = self._restored.get(tensor_id)
            if cached is not None:
                return cached

        kept = self._keep_last.pop(tensor_id, None)
        if kept is not None:
            self.stats.kept_last_tensors += 1
            if cache:
                self._restored[tensor_id] = kept
            return kept

        if tensor_id in self._fwd_stash:
            tensor, event = self._fwd_stash.pop(tensor_id)
            self.compute_stream.wait_event(event)
            cpu_tensor, *_unused, buffer_key = self._tracker.pop(tensor_id)
            self._pool_cpu_buffer(buffer_key, cpu_tensor)
            self.stats.restored_tensors += 1
            self.stats.restored_bytes += self._num_bytes(tensor)
            if cache:
                self._restored[tensor_id] = tensor
            return tensor

        tracked = self._tracker.pop(tensor_id, None)
        if tracked is None:
            raise RuntimeError(f"offloaded activation {tensor_id} is no longer tracked. backward() must run "
                               "before the offload engine is reset.")
        cpu_tensor, device, shape, stride, buffer_key = tracked
        stream = self.s1 if self.use_streams else self.compute_stream
        with self._stream_context(stream):
            # fresh offset-0 storage; a source offset would index past the pool buffer
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
        if cache:
            self._restored[tensor_id] = gpu_tensor
        return gpu_tensor

    def _sync_copy_streams(self) -> None:
        if not self.use_streams:
            return
        if self.s1 is not None:
            self.s1.synchronize()
        self._reap_cpu_buffer_pool(force=True)

    def reset(self) -> None:
        # Wait for in-flight copies before dropping CPU/GPU refs (public API).
        self._sync_copy_streams()
        self.stats = CheckpointActivationOffloadStats()
        self._tracker.clear()
        self._fwd_stash.clear()
        self._keep_last.clear()
        self._restored.clear()
        self._pending_cpu_buffers.clear()
        self._next_id = 0

    def _sync_and_clear(self) -> None:
        self._sync_copy_streams()
        for tracked in self._tracker.values():
            self._pool_cpu_buffer(tracked[-1], tracked[0])
        self._tracker.clear()
        self._fwd_stash.clear()
        self._keep_last.clear()
        self._restored.clear()
        self._pending_cpu_buffers.clear()


def _manager_stack() -> list["CheckpointHiddenStatesOffload"]:
    stack = getattr(_TLS, "stack", None)
    if stack is None:
        stack = []
        _TLS.stack = stack
    return stack


def _current_manager() -> "CheckpointHiddenStatesOffload | None":
    stack = _manager_stack()
    return stack[-1] if stack else None


def patch_gradient_checkpointing_layer_marker() -> None:
    global _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL, _MARKER_WRAPPER

    if _MARKER_WRAPPER is not None:
        return

    try:
        from transformers import GradientCheckpointingLayer
    except ImportError as e:
        raise RuntimeError("patch_gradient_checkpointing_layer_marker requires the `transformers` "
                           "package (>= 4.52, which introduced GradientCheckpointingLayer). "
                           "Install it with `pip install transformers`.") from e

    # Class-owned __call__, or None if HF is using nn.Module.__call__ via MRO.
    orig_owned = GradientCheckpointingLayer.__dict__.get("__call__")
    orig_call = GradientCheckpointingLayer.__call__
    _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL = orig_owned

    def _checkpoint_offload_call(self, *args, **kwargs):
        manager = _current_manager()
        marked = None
        if (manager is not None and self.training and torch.is_grad_enabled()
                and getattr(self, "gradient_checkpointing", False)):
            # Only args[0] (hidden_states) is checkpointed; a keyword hidden_states
            # rides in partial(**kwargs) and is never packed.
            hidden_states = args[0] if args else None
            if torch.is_tensor(hidden_states):
                manager.mark(hidden_states)
                marked = hidden_states
        try:
            return orig_call(self, *args, **kwargs)
        finally:
            # id() is recycled; drop a mark the pack hook did not consume.
            if marked is not None:
                manager._consume_mark(marked)

    GradientCheckpointingLayer.__call__ = _checkpoint_offload_call
    _MARKER_WRAPPER = _checkpoint_offload_call


def _unpatch_gradient_checkpointing_layer_marker() -> None:
    # Restore __call__ once no manager is active so the model stays clean for
    # HybridEngine reuse.
    global _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL, _MARKER_WRAPPER

    wrapper = _MARKER_WRAPPER
    orig_owned = _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL
    _MARKER_WRAPPER = None
    _ORIG_GRADIENT_CHECKPOINTING_LAYER_CALL = None
    if wrapper is None:
        return

    from transformers import GradientCheckpointingLayer
    if GradientCheckpointingLayer.__call__ is not wrapper:
        return
    if orig_owned is None:
        delattr(GradientCheckpointingLayer, "__call__")
    else:
        GradientCheckpointingLayer.__call__ = orig_owned


class CheckpointHiddenStatesOffload(_ActivationOffloadEngine, saved_tensors_hooks):
    """Offload only marked checkpoint inputs.

    The marker is installed on ``GradientCheckpointingLayer.__call__`` while this
    context is active and restored on exit. All other saved tensors pass through
    unchanged, including tensors from the final norm/head and any
    non-checkpointed modules.
    """

    def __init__(
        self,
        use_pin_memory: bool = True,
        use_streams: bool = True,
        min_offload_bytes: int = 1024,
        max_fwd_stash_count: int = 2,
        max_cpu_buffer_pool_count: int = 64,
        keep_last_count: int = 1,
    ) -> None:
        _ActivationOffloadEngine.__init__(
            self,
            use_pin_memory=use_pin_memory,
            use_streams=use_streams,
            min_offload_bytes=min_offload_bytes,
            max_fwd_stash_count=max_fwd_stash_count,
            max_cpu_buffer_pool_count=max_cpu_buffer_pool_count,
            keep_last_count=keep_last_count,
        )
        self._allowed: dict[int, int] = {}
        saved_tensors_hooks.__init__(self, self._pack_tensor, self._unpack_tensor)

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

    def _pack_tensor(self, tensor: torch.Tensor):
        self.stats.saved_tensors_seen += 1
        if not self._consume_mark(tensor):
            return tensor
        tensor_id = self.offload_input(tensor)
        if tensor_id is None:
            return tensor
        return _OffloadedTensorRef(tensor_id)

    def _unpack_tensor(self, maybe_ref):
        if isinstance(maybe_ref, _OffloadedTensorRef):
            return self._restore_tensor(maybe_ref.tensor_id, cache=True)
        return maybe_ref

    def reset(self) -> None:
        super().reset()
        self._allowed.clear()

    def _sync_and_clear(self) -> None:
        super()._sync_and_clear()
        self._allowed.clear()

    def _pop_manager_stack(self) -> None:
        global _MARKER_INSTALLED_BY_CONTEXT, _PATCH_USERS
        stack = _manager_stack()
        if stack and stack[-1] is self:
            stack.pop()
        elif self in stack:
            stack.remove(self)
        with _PATCH_LOCK:
            if _PATCH_USERS > 0:
                _PATCH_USERS -= 1
            if _PATCH_USERS == 0 and _MARKER_INSTALLED_BY_CONTEXT:
                _unpatch_gradient_checkpointing_layer_marker()
                _MARKER_INSTALLED_BY_CONTEXT = False

    def __enter__(self):
        global _MARKER_INSTALLED_BY_CONTEXT, _PATCH_USERS
        if self in _manager_stack():
            raise RuntimeError("CheckpointHiddenStatesOffload is not re-entrant; use a separate "
                               "instance for a nested context")

        with _PATCH_LOCK:
            if (_MARKER_WRAPPER is None and importlib.util.find_spec("transformers") is not None):
                patch_gradient_checkpointing_layer_marker()
                _MARKER_INSTALLED_BY_CONTEXT = True
            if _MARKER_WRAPPER is not None:
                _PATCH_USERS += 1

        self.reset()
        _manager_stack().append(self)
        try:
            return super().__enter__()
        except Exception:
            self._pop_manager_stack()
            raise

    def __exit__(self, *args, **kwargs):
        sync_err = None
        try:
            self._sync_and_clear()
        except Exception as err:
            sync_err = err
        try:
            self._pop_manager_stack()
        finally:
            result = super().__exit__(*args, **kwargs)
        if sync_err is not None:
            raise sync_err
        return result


def get_checkpoint_hidden_states_offloading_ctx_manager(
    use_pin_memory: bool = True,
    use_streams: bool = True,
    min_offload_bytes: int = 1024,
    max_fwd_stash_count: int = 2,
    max_cpu_buffer_pool_count: int = 64,
    keep_last_count: int = 1,
) -> CheckpointHiddenStatesOffload:
    return CheckpointHiddenStatesOffload(
        use_pin_memory=use_pin_memory,
        use_streams=use_streams,
        min_offload_bytes=min_offload_bytes,
        max_fwd_stash_count=max_fwd_stash_count,
        max_cpu_buffer_pool_count=max_cpu_buffer_pool_count,
        keep_last_count=keep_last_count,
    )
