# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.utils.pin_memory import NativePinnedMemory


@pytest.fixture
def native_pins():
    try:
        return NativePinnedMemory()
    except Exception:
        pytest.skip("pin_memory op could not be built; native pinning unavailable")


def test_pin_copies_and_matches_shape(native_pins):
    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    pinned = native_pins.pin(tensor)
    assert tuple(pinned.shape) == (4, 8)
    assert torch.equal(pinned, tensor)
    assert getattr(pinned, "ds_pinned", False) is True
    assert native_pins.is_pinned(pinned)


def test_is_pinned_propagates_to_views(native_pins):
    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    pinned = native_pins.pin(tensor)
    # Views/slices lose the .ds_pinned attribute but must still be recognized
    # via the tracked pointer range.
    view = pinned.reshape(-1).narrow(0, 8, 8)
    assert getattr(view, "ds_pinned", False) is False
    assert native_pins.is_pinned(view)
    assert native_pins.is_pinned(pinned[1])


def test_pin_flags(native_pins):
    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    alloc = native_pins.pin(tensor, make_copy=False)
    assert tuple(alloc.shape) == (4, 8)
    assert native_pins.is_pinned(alloc)
    flat = native_pins.pin(tensor, match_shape=False)
    assert tuple(flat.shape) == (tensor.numel(), )


def test_unpin_frees_range(native_pins):
    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    pinned = native_pins.pin(tensor)
    view = pinned.reshape(-1).narrow(0, 8, 8)
    assert native_pins.unpin(pinned) is True
    assert not native_pins.is_pinned(pinned)
    assert not native_pins.is_pinned(view)


def test_pin_freed_on_gc(native_pins):
    import gc

    tensor = torch.arange(32, dtype=torch.float32)
    pinned = native_pins.pin(tensor)
    begin = pinned.data_ptr()
    assert begin in native_pins._ranges
    # Dropping the returned tensor must release the mlocked allocation, matching
    # torch.pin_memory lifetime semantics (no explicit unpin() required).
    del pinned
    gc.collect()
    assert begin not in native_pins._ranges


def test_allocation_survives_until_all_views_dropped(native_pins):
    import gc

    tensor = torch.arange(32, dtype=torch.float32)
    pinned = native_pins.pin(tensor)
    begin = pinned.data_ptr()
    view = pinned.narrow(0, 8, 8)
    # Dropping the returned tensor while a derived view is still live must NOT free
    # the shared allocation, or the view would alias freed (use-after-free) memory.
    del pinned
    gc.collect()
    assert begin in native_pins._ranges
    assert native_pins.is_pinned(view)
    assert float(view[0]) == 8.0
    # The allocation is released only once the last alias is gone.
    del view
    gc.collect()
    assert begin not in native_pins._ranges


def test_unpin_frees_original_after_data_redirect(native_pins):
    tensor = torch.arange(32, dtype=torch.float32)
    pinned = native_pins.pin(tensor)
    begin = pinned.data_ptr()
    # Keep the original allocation alive so the GC finalizer cannot race the
    # explicit unpin below; this mirrors live narrows into an offload buffer.
    keep_alive = pinned.narrow(0, 0, 1)
    # Simulate ZeRO offload/reload rebinding .data to a different buffer.
    pinned.data = torch.zeros(32, dtype=torch.float32)
    assert pinned.data_ptr() != begin
    # unpin() must free the original region recorded at pin time, not the buffer
    # the tensor currently points at.
    assert native_pins.unpin(pinned) is True
    assert begin not in native_pins._ranges
    del keep_alive


def test_is_pinned_handles_storageless_tensors(native_pins):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as fake_mode:
        fake_tensor = fake_mode.from_tensor(torch.zeros(2, 8))
        assert native_pins.is_pinned(fake_tensor) is False

    meta_tensor = torch.zeros(2, 8, device="meta")
    assert native_pins.is_pinned(meta_tensor) is False
