# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Native-backend coverage for Phase 2 pin routing through offload helpers."""

import gc

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum
from deepspeed.runtime.zero.offload_states import offload_optimizer_states, reload_optimizer_states
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils.pin_memory import get_active_native_pinned_memory


def _require_native(monkeypatch):
    monkeypatch.setenv("DS_PIN_MEMORY_BACKEND", "native")
    try:
        if get_active_native_pinned_memory() is None:
            pytest.skip("native backend not selected")
    except Exception:
        pytest.skip("pin_memory op could not be built; native pinning unavailable")


def _require_gpu_native(monkeypatch):
    _require_native(monkeypatch)
    accel = get_accelerator()
    # Async host-to-device copies only exist off the CPU accelerator.
    if accel.device_name() == "cpu":
        pytest.skip("requires non-CPU accelerator")
    return accel


def test_offload_helper_pin_pattern_native(monkeypatch):
    """Empty scratch pin used by offload_optimizer_states must be is_pinned under native."""
    _require_native(monkeypatch)
    accel = get_accelerator()
    # Mirrors offload_optimizer_states: allocate empty host buffer, then copy.
    src = torch.randn(32)
    pinned_buffer = accel.pin_memory(torch.empty_like(src, device="cpu"), make_copy=False)
    pinned_buffer.copy_(src)
    assert accel.is_pinned(pinned_buffer) is True
    assert accel.unpin_memory(pinned_buffer) is True


def test_offload_optimizer_states_native_is_pinned(monkeypatch):
    """offload_optimizer_states must route through accelerator pin_memory."""
    _require_native(monkeypatch)
    accel = get_accelerator()
    # The pin path only runs when the source tensor is not already on CPU.
    if accel.device_name() == "cpu":
        pytest.skip("requires non-CPU accelerator for GPU->pinned CPU offload path")

    device = accel.current_device_name()

    class _Opt:
        pass

    opt = _Opt()
    opt.state = {0: {"exp_avg": torch.randn(32, device=device)}}

    offload_optimizer_states(opt, device="cpu", pin_memory=True, non_blocking=False)
    buf = opt.state[0]["exp_avg"]
    assert buf.device.type == "cpu"
    assert accel.is_pinned(buf) is True
    assert accel.unpin_memory(buf) is True


def test_reload_optimizer_states_returns_host_sources(monkeypatch):
    """Callers need the replaced host tensors to outlive a non-blocking copy."""
    accel = _require_gpu_native(monkeypatch)
    device = accel.current_device_name()

    class _Opt:
        pass

    opt = _Opt()
    opt.state = {0: {"exp_avg": torch.randn(32, device=device)}}
    offload_optimizer_states(opt, device="cpu", pin_memory=True, non_blocking=False)
    pinned = opt.state[0]["exp_avg"]

    host_buffers = reload_optimizer_states(opt, device, non_blocking=True)
    accel.synchronize()

    assert len(host_buffers) == 1
    assert host_buffers[0] is pinned
    assert opt.state[0]["exp_avg"].device.type != "cpu"
    assert accel.unpin_memory(pinned) is True


def test_reload_states_holds_pin_buffers_until_sync(monkeypatch):
    """ZeRO-1/2 reload must not free native pinned sources while copies are in flight."""
    accel = _require_gpu_native(monkeypatch)
    manager = get_active_native_pinned_memory()
    device = accel.current_device_name()

    hp_param = torch.randn(1024, device=device)
    pinned = accel.pin_memory(torch.empty_like(hp_param, device="cpu"), make_copy=False)
    pinned.copy_(hp_param)
    # Mirrors offload_states(): the hp param now points at the pinned host buffer.
    hp_param.data = pinned
    begin = pinned.data_ptr()

    class _FakeZeroOptimizer:
        # Run the real method against the minimal state its hp_params branch touches.
        reload_states = DeepSpeedZeroOptimizer.reload_states

        def _link_all_hp_params(self):
            pass

    optimizer = _FakeZeroOptimizer()
    optimizer.offloaded_states = {OffloadStateTypeEnum.hp_params}
    optimizer.single_partition_of_fp32_groups = [hp_param]
    optimizer.hp_params_pin_buffers = [pinned]
    # reload_states now owns the only reference to the pinned buffer.
    del pinned

    registered_while_syncing = []
    real_synchronize = accel.synchronize

    def _recording_synchronize():
        registered_while_syncing.append(begin in manager._ranges)
        real_synchronize()

    monkeypatch.setattr(accel, "synchronize", _recording_synchronize)
    optimizer.reload_states(non_blocking=True)

    assert registered_while_syncing == [True]
    assert hp_param.device.type != "cpu"
    gc.collect()
    # Released once the copies are known to have completed.
    assert begin not in manager._ranges


def test_superoffload_grad_buffer_unpinned_when_disabled():
    """offload_optimizer.pin_memory=False must allocate a regular CPU buffer."""
    from deepspeed.runtime.superoffload.superoffload_utils import _allocate_worker_grad_buffer
    buffer = _allocate_worker_grad_buffer(32, pin_memory=False)
    assert get_accelerator().is_pinned(buffer) is False


def test_superoffload_grad_buffer_pinned_when_enabled(monkeypatch):
    """offload_optimizer.pin_memory=True keeps the current pinned allocation."""
    _require_native(monkeypatch)
    from deepspeed.runtime.superoffload.superoffload_utils import _allocate_worker_grad_buffer
    accel = get_accelerator()
    buffer = _allocate_worker_grad_buffer(32, pin_memory=True)
    assert accel.is_pinned(buffer) is True
    assert accel.unpin_memory(buffer) is True
