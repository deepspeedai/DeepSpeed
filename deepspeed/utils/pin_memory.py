# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import weakref

# ``torch._subclasses.fake_tensor`` is a private API that may be absent on some
# torch versions; guard the import so this module stays importable (including
# during setup when torch may not be installed).
try:
    from torch._subclasses.fake_tensor import is_fake as _is_fake_tensor
except ImportError:
    _is_fake_tensor = None


class NativePinnedMemory(object):
    # Host-memory pinning backed by the DeepNVMe page-locked (``mlock``) allocator.
    # This is device-independent: it always pins CPU memory, so it lives here
    # rather than in the accelerator, which only owns the device-specific torch
    # pinning path.
    def __init__(self):
        # base address -> end address; keyed by base so unpin is an O(1) delete.
        self._ranges = {}
        # base address -> weakref.finalize handle for the returned tensor, so an
        # explicit unpin() can cancel the GC-triggered free.
        self._finalizers = {}
        # Fail early: native pinning is useless without the pin_memory handle, so
        # surface the build/load failure here instead of silently degrading.
        try:
            from deepspeed.ops.op_builder import PinMemoryBuilder
            self._handle = PinMemoryBuilder().load().pin_handle()
        except Exception as e:
            raise RuntimeError(
                "DS_PIN_MEMORY_BACKEND=native requires the pin_memory op, which failed to build/load.") from e

    def pin(self, tensor, make_copy=True, match_shape=True):
        numel = tensor.numel()
        # ``base`` is the allocation root and the view root for everything derived
        # from it. Every slice/view of the returned tensor keeps ``base`` alive via
        # ``._base``, so the allocation is freed only after the returned tensor and
        # all of its aliases are gone (a live view can never outlive the free).
        base = self._handle.new_cpu_locked_tensor(numel, tensor)
        begin = base.data_ptr()
        locked = base[:numel]
        if make_copy:
            locked.copy_(tensor.reshape(-1))
        if match_shape:
            locked = locked.view(tensor.shape)
        self._ranges[begin] = begin + numel * tensor.element_size()
        locked.ds_pinned = True
        # Remember the owning allocation address so an explicit unpin() frees the
        # original region even if the tensor's ``.data`` is later redirected (e.g.
        # ZeRO offload/reload rebinds ``.data`` to a different buffer).
        locked.ds_pin_base = begin
        # Match torch.pin_memory lifetime semantics: free the page-locked allocation
        # once its root is garbage-collected, so call sites that never call unpin()
        # do not accumulate mlocked host memory. The finalizer is tied to ``base``
        # (not the returned view) and frees by address; ``base`` must not be passed
        # as a finalize argument or it would be kept alive forever.
        self._finalizers[begin] = weakref.finalize(base, self._release, self._handle, begin, self._ranges,
                                                   self._finalizers)
        return locked

    def is_pinned(self, tensor):
        if getattr(tensor, "ds_pinned", False):
            return True
        if not self._has_real_storage(tensor):
            return False
        ptr = tensor.data_ptr()
        return any(begin <= ptr < end for begin, end in self._ranges.items())

    def unpin(self, tensor):
        # After freeing, using ``tensor`` is a use-after-free and must be avoided.
        # Prefer the address recorded at pin time so a redirected ``.data`` still
        # releases the correct region; fall back to the current pointer otherwise.
        begin = getattr(tensor, "ds_pin_base", None)
        if begin is None:
            begin = tensor.data_ptr()
        finalizer = self._finalizers.pop(begin, None)
        if finalizer is not None:
            # Explicit unpin owns the free; cancel the GC finalizer to avoid a
            # redundant free.
            finalizer.detach()
        freed = self._handle.free_cpu_locked_tensor_by_ptr(begin)
        self._ranges.pop(begin, None)
        if hasattr(tensor, "ds_pinned"):
            tensor.ds_pinned = False
        return freed

    @staticmethod
    def _release(handle, begin, ranges, finalizers):
        ranges.pop(begin, None)
        finalizers.pop(begin, None)
        try:
            handle.free_cpu_locked_tensor_by_ptr(begin)
        except Exception:
            # Best-effort cleanup; the handle or torch may already be torn down
            # during interpreter shutdown.
            pass

    @staticmethod
    def _has_real_storage(tensor):
        # Fake/meta tensors have no storage; skip the range check to avoid a
        # meaningless (and warning-prone) data_ptr() call.
        device = getattr(tensor, "device", None)
        if device is not None and device.type == "meta":
            return False
        if _is_fake_tensor is not None and _is_fake_tensor(tensor):
            return False
        return True


# Process-wide shared manager. Pinned-range tracking must be consistent across
# every component that pins or queries pinned status, so the accelerator (the
# sole dispatcher) always routes through this single instance.
_shared_native_pins = None


def get_native_pinned_memory():
    global _shared_native_pins
    if _shared_native_pins is None:
        _shared_native_pins = NativePinnedMemory()
    return _shared_native_pins


def get_active_native_pinned_memory():
    # Returns the shared manager when ``DS_PIN_MEMORY_BACKEND=native``; otherwise
    # None so the accelerator uses its (torch) pinning path. When native is
    # selected but the pin_memory op cannot be built, constructing the manager
    # raises rather than silently falling back.
    if os.environ.get("DS_PIN_MEMORY_BACKEND", "torch").strip().lower() != "native":
        return None
    return get_native_pinned_memory()
