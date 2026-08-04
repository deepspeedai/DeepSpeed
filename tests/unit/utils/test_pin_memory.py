# Copyright (c) Microsoft Corporation.
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
        pytest.skip("async_io op could not be built; native pinning unavailable")


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


def test_is_pinned_handles_storageless_tensors(native_pins):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as fake_mode:
        fake_tensor = fake_mode.from_tensor(torch.zeros(2, 8))
        assert native_pins.is_pinned(fake_tensor) is False

    meta_tensor = torch.zeros(2, 8, device="meta")
    assert native_pins.is_pinned(meta_tensor) is False
