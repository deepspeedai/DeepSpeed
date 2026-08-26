# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator
from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
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


class _RegisteringAccelerator:

    def __init__(self):
        self.registered = []
        self.unregistered = []

    def register_host_memory(self, address, num_bytes):
        self.registered.append((address, num_bytes))
        return True

    def unregister_host_memory(self, address):
        self.unregistered.append(address)


def test_native_device_registration_and_unpin(monkeypatch, native_pins):
    accelerator = _RegisteringAccelerator()
    monkeypatch.setattr("deepspeed.accelerator.get_accelerator", lambda: accelerator)
    monkeypatch.setenv("DS_PIN_MEMORY_REGISTER_DEVICE", "1")

    pinned = native_pins.pin(torch.empty(32), make_copy=False)
    begin = pinned.data_ptr()
    assert accelerator.registered == [(begin, pinned.nbytes)]
    assert begin in native_pins._device_registered

    assert native_pins.unpin(pinned) is True
    assert accelerator.unregistered == [begin]
    assert begin not in native_pins._device_registered
    # A second call must not unregister or free the allocation twice.
    assert native_pins.unpin(pinned) is False
    assert accelerator.unregistered == [begin]


def test_native_device_registration_disabled(monkeypatch, native_pins):
    accelerator = _RegisteringAccelerator()
    monkeypatch.setattr("deepspeed.accelerator.get_accelerator", lambda: accelerator)
    monkeypatch.setenv("DS_PIN_MEMORY_REGISTER_DEVICE", "0")

    pinned = native_pins.pin(torch.empty(32), make_copy=False)
    assert native_pins.is_pinned(pinned)
    assert accelerator.registered == []
    assert native_pins.unpin(pinned) is True
    assert accelerator.unregistered == []


def test_cpu_device_registration_is_noop():
    accelerator = CPU_Accelerator()
    assert accelerator.register_host_memory(1234, 4096) is False
    assert accelerator.unregister_host_memory(1234) is None


def test_cuda_device_registration_calls_cudart(monkeypatch):

    class _Cudart:

        def __init__(self):
            self.registered = []
            self.unregistered = []

        def cudaHostRegister(self, address, num_bytes, flags):
            self.registered.append((address, num_bytes, flags))
            return 0

        def cudaHostUnregister(self, address):
            self.unregistered.append(address)
            return 0

    cudart = _Cudart()
    errors = []
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)  #ignore-cuda
    monkeypatch.setattr(torch.cuda, "check_error", errors.append)  #ignore-cuda
    accelerator = CUDA_Accelerator.__new__(CUDA_Accelerator)

    assert accelerator.register_host_memory(1234, 4096) is True
    accelerator.unregister_host_memory(1234)
    assert cudart.registered == [(1234, 4096, 0)]
    assert cudart.unregistered == [1234]
    assert errors == [0, 0]


def test_cpu_native_pin_with_register_env_on(monkeypatch, native_pins):
    """CPU accelerator has no register hook; native pin still works with default-on."""
    monkeypatch.setenv("DS_PIN_MEMORY_REGISTER_DEVICE", "1")
    monkeypatch.setattr("deepspeed.accelerator.get_accelerator", lambda: CPU_Accelerator())
    pinned = native_pins.pin(torch.empty(32), make_copy=False)
    assert native_pins.is_pinned(pinned)
    assert native_pins.unpin(pinned) is True


def test_device_registration_failure_keeps_mlock(monkeypatch, native_pins):

    class _FailingAccelerator:

        def register_host_memory(self, address, num_bytes):
            raise RuntimeError("simulated cudaHostRegister failure")

        def unregister_host_memory(self, address):
            raise AssertionError("unregister must not run when register failed")

    monkeypatch.setattr("deepspeed.accelerator.get_accelerator", lambda: _FailingAccelerator())
    monkeypatch.setenv("DS_PIN_MEMORY_REGISTER_DEVICE", "1")
    pinned = native_pins.pin(torch.empty(32), make_copy=False)
    assert native_pins.is_pinned(pinned)
    assert pinned.data_ptr() not in native_pins._device_registered
    assert native_pins.unpin(pinned) is True


def test_device_registration_gc_unregisters(monkeypatch, native_pins):
    import gc

    accelerator = _RegisteringAccelerator()
    monkeypatch.setattr("deepspeed.accelerator.get_accelerator", lambda: accelerator)
    monkeypatch.setenv("DS_PIN_MEMORY_REGISTER_DEVICE", "1")
    pinned = native_pins.pin(torch.empty(32), make_copy=False)
    begin = pinned.data_ptr()
    del pinned
    gc.collect()
    assert begin not in native_pins._ranges
    assert accelerator.unregistered == [begin]


def test_invalid_register_device_env(monkeypatch, native_pins):
    monkeypatch.setenv("DS_PIN_MEMORY_REGISTER_DEVICE", "maybe")
    with pytest.raises(ValueError, match="DS_PIN_MEMORY_REGISTER_DEVICE"):
        native_pins.pin(torch.empty(8), make_copy=False)


def test_unpin_keeps_allocation_when_unregister_fails(monkeypatch, native_pins):

    class _UnregisterFailAccelerator(_RegisteringAccelerator):

        def __init__(self):
            super().__init__()
            self.fail_unregister = True

        def unregister_host_memory(self, address):
            if self.fail_unregister:
                raise RuntimeError("simulated cudaHostUnregister failure")
            super().unregister_host_memory(address)

    accelerator = _UnregisterFailAccelerator()
    monkeypatch.setattr("deepspeed.accelerator.get_accelerator", lambda: accelerator)
    monkeypatch.setenv("DS_PIN_MEMORY_REGISTER_DEVICE", "1")
    pinned = native_pins.pin(torch.empty(32), make_copy=False)
    begin = pinned.data_ptr()

    with pytest.raises(RuntimeError, match="cudaHostUnregister"):
        native_pins.unpin(pinned)

    assert native_pins.is_pinned(pinned)
    assert begin in native_pins._device_registered
    assert begin in native_pins._ranges
    assert accelerator.unregistered == []

    accelerator.fail_unregister = False
    assert native_pins.unpin(pinned) is True
    assert accelerator.unregistered == [begin]
    assert begin not in native_pins._device_registered
