# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import os
import sys
import importlib
import re

import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator

DS_ACCEL_PATH = "deepspeed.accelerator"
IGNORE_FILES = ["abstract_accelerator.py", "real_accelerator.py"]


@pytest.fixture
def accel_class_name(module_name):
    class_list = []
    mocked_modules = []

    # Get the accelerator class name for a given module
    while True:
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as e:
            # If the environment is missing a module, mock it so we can still
            # test importing the accelerator class
            missing_module = re.search(r"\'(.*)\'", e.msg).group().strip("'")
            sys.modules[missing_module] = lambda x: None
            mocked_modules.append(missing_module)
    for name in dir(module):
        if name.endswith("_Accelerator"):
            class_list.append(name)

    assert len(class_list) == 1, f"Multiple accelerator classes found in {module_name}"

    yield class_list[0]

    # Clean up mocked modules so as to not impact other tests
    for module in mocked_modules:
        del sys.modules[module]


@pytest.mark.parametrize(
    "module_name",
    [
        DS_ACCEL_PATH + "." + f.rstrip(".py") for f in os.listdir(deepspeed.accelerator.__path__[0])
        if f.endswith("_accelerator.py") and f not in IGNORE_FILES
    ],
)
def test_abstract_methods_defined(module_name, accel_class_name):
    module = importlib.import_module(module_name)
    accel_class = getattr(module, accel_class_name)
    # Stub __init__ so the class can be instantiated without a real device, but
    # restore it afterwards to avoid polluting the shared class object for other
    # tests that construct a fully-initialized accelerator.
    original_init = accel_class.__init__
    accel_class.__init__ = lambda self: None
    try:
        _ = accel_class()
    finally:
        accel_class.__init__ = original_init


def _require_native(monkeypatch):
    # Backend selection is env-driven and evaluated per call, so set native for
    # the whole test and skip if the pin_memory op cannot be built (constructing
    # the native manager raises in that case).
    from deepspeed.utils.pin_memory import get_active_native_pinned_memory
    monkeypatch.setenv("DS_PIN_MEMORY_BACKEND", "native")
    try:
        if get_active_native_pinned_memory() is None:
            pytest.skip("native backend not selected")
    except Exception:
        pytest.skip("pin_memory op could not be built; native pinning unavailable")


def test_pin_memory_torch_backend(monkeypatch):
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    accel = get_accelerator()

    tensor = torch.randn(4, 8)
    pinned = accel.pin_memory(tensor)
    # is_pinned must agree with torch for the torch backend.
    assert accel.is_pinned(pinned) == pinned.is_pinned()
    # Unpinning a torch-pinned tensor is a no-op (freed on garbage collection).
    assert accel.unpin_memory(pinned) is None


def test_pin_memory_torch_backend_make_copy_false_allocates_pinned(monkeypatch):
    """make_copy=False must page-lock a new buffer instead of pinning a copy."""
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    calls = []

    class _StubAccelerator:
        # Exercise the shared dispatch without needing a real pinning device.
        pin_memory = DeepSpeedAccelerator.pin_memory

        def _torch_pin_memory(self, tensor):
            calls.append(("pin_copy", tuple(tensor.shape)))
            return tensor

        def _torch_empty_pinned(self, tensor, shape):
            calls.append(("empty_pinned", tuple(shape)))
            return tensor.new_empty(shape)

    accel = _StubAccelerator()
    tensor = torch.randn(4, 8)

    accel.pin_memory(tensor)
    assert tuple(accel.pin_memory(tensor, make_copy=False).shape) == (4, 8)
    assert tuple(accel.pin_memory(tensor, make_copy=False, match_shape=False).shape) == (32, )
    assert calls == [("pin_copy", (4, 8)), ("empty_pinned", (4, 8)), ("empty_pinned", (32, ))]


def test_pin_memory_torch_backend_no_copy_is_pinned(monkeypatch):
    """The buffer returned for make_copy=False is really page-locked."""
    monkeypatch.delenv("DS_PIN_MEMORY_BACKEND", raising=False)
    accel = get_accelerator()
    if accel.device_name() == "cpu":
        pytest.skip("torch cannot pin CPU tensors on the CPU accelerator")

    tensor = torch.randn(4, 8)
    buffer = accel.pin_memory(tensor, make_copy=False)
    assert buffer.is_pinned()
    assert tuple(buffer.shape) == (4, 8)
    assert buffer.data_ptr() != tensor.data_ptr()


def test_pin_memory_native_backend(monkeypatch):
    _require_native(monkeypatch)
    accel = get_accelerator()

    tensor = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    # Default: copy data and match the input shape, tag with .ds_pinned.
    pinned = accel.pin_memory(tensor)
    assert tuple(pinned.shape) == (4, 8)
    assert torch.equal(pinned, tensor)
    assert getattr(pinned, "ds_pinned", False) is True
    assert accel.is_pinned(pinned)

    # Pinned status propagates to slices/views via pointer-range tracking even
    # though the .ds_pinned attribute does not survive tensor ops.
    view = pinned.reshape(-1).narrow(0, 8, 8)
    assert getattr(view, "ds_pinned", False) is False
    assert accel.is_pinned(view)
    assert accel.is_pinned(pinned[1])

    # make_copy=False -> shaped allocation; match_shape=False -> flat buffer.
    alloc = accel.pin_memory(tensor, make_copy=False)
    assert tuple(alloc.shape) == (4, 8)
    assert accel.is_pinned(alloc)
    flat = accel.pin_memory(tensor, match_shape=False)
    assert tuple(flat.shape) == (tensor.numel(), )

    # Unpinning frees the native allocation and flips is_pinned for the buffer
    # and its derived views.
    assert accel.unpin_memory(pinned) is True
    assert not accel.is_pinned(pinned)
    assert not accel.is_pinned(view)


def test_pin_memory_native_raises_when_pin_memory_unavailable(monkeypatch):
    import deepspeed.ops.op_builder as op_builder
    import deepspeed.utils.pin_memory as pin_memory_util

    class _FailingBuilder:

        def load(self):
            raise RuntimeError("simulated pin_memory build failure")

    # Reset the shared manager so it is rebuilt with the failing builder below;
    # monkeypatch restores the original instance after the test.
    monkeypatch.setenv("DS_PIN_MEMORY_BACKEND", "native")
    monkeypatch.setattr(pin_memory_util, "_shared_native_pins", None)
    monkeypatch.setattr(op_builder, "PinMemoryBuilder", _FailingBuilder)

    # Native is selected but the pin_memory op cannot be built, so we fail early
    # (no silent torch fallback).
    with pytest.raises(RuntimeError, match="pin_memory"):
        pin_memory_util.get_active_native_pinned_memory()


def test_is_pinned_handles_storageless_tensors(monkeypatch):
    _require_native(monkeypatch)
    accel = get_accelerator()

    from torch._subclasses.fake_tensor import FakeTensorMode

    # FakeTensor/meta tensors have no real storage; is_pinned must not access
    # data_ptr() on them.
    with FakeTensorMode() as fake_mode:
        fake_tensor = fake_mode.from_tensor(torch.zeros(2, 8))
        assert accel.is_pinned(fake_tensor) is False

    meta_tensor = torch.zeros(2, 8, device="meta")
    assert accel.is_pinned(meta_tensor) is False
