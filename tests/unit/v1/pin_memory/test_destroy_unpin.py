# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""UTs for accelerator.unpin_memory() at ZeRO optimizer destroy().

Native-pinned host buffers owned by the optimizer must be released
deterministically in destroy(). The torch backend treats unpin_memory as a
no-op, so destroy() must still succeed under the default backend.
"""

import os

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.pin_memory import get_active_native_pinned_memory

from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import SimpleModel, random_dataloader


def _require_native():
    os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
    try:
        native = get_active_native_pinned_memory()
    except Exception:
        pytest.skip("pin_memory op could not be built; native pinning unavailable")
    if native is None:
        pytest.skip("native backend not selected")
    return native


def _restore_pin_backend(prev):
    if prev is None:
        os.environ.pop("DS_PIN_MEMORY_BACKEND", None)
    else:
        os.environ["DS_PIN_MEMORY_BACKEND"] = prev


def _config(stage, pin_memory=True, offload_param=False, offload_optimizer=True):
    zero = {"stage": stage}
    if offload_optimizer:
        zero["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": pin_memory,
        }
    if offload_param:
        zero["offload_param"] = {
            "device": "cpu",
            "pin_memory": pin_memory,
        }

    config = {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            },
        },
        "zero_optimization": zero,
        "zero_force_ds_cpu_optimizer": False,
    }

    dtype = preferred_dtype()
    if dtype == torch.bfloat16:
        config["bf16"] = {"enabled": True}
    elif dtype == torch.float16:
        config["fp16"] = {"enabled": True}
    return config, dtype


def _run_one_step(engine, hidden_dim, dtype):
    data_loader = random_dataloader(model=engine,
                                    total_samples=2,
                                    hidden_dim=hidden_dim,
                                    device=engine.device,
                                    dtype=dtype)
    batch = next(iter(data_loader))
    loss = engine(batch[0], batch[1])
    engine.backward(loss)
    engine.step()


class TestDestroyUnpinsNativeBuffers(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("stage", [2, 3])
    def test_native_destroy_frees_optimizer_pins(self, stage):
        # Optimizer-owned CPU-offload pins must be released in destroy() under the
        # native backend (param-offload pins are intentionally left to the finalizer).
        prev = os.environ.get("DS_PIN_MEMORY_BACKEND")
        try:
            native = _require_native()
            config, dtype = _config(stage, pin_memory=True, offload_param=False)
            hidden_dim = 16
            model = SimpleModel(hidden_dim, nlayers=2)
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            _run_one_step(engine, hidden_dim, dtype)

            ranges_before = len(native._ranges)
            assert ranges_before > 0, "expected native-pinned optimizer offload buffers"
            engine.destroy()
            assert len(native._ranges) == 0, (f"optimizer-owned native pins not released on destroy: "
                                              f"before={ranges_before} after={len(native._ranges)}")
        finally:
            _restore_pin_backend(prev)

    def test_native_stage3_param_offload_frees_optimizer_pins(self):
        # With param offload enabled, residual param-partition pins may remain
        # after destroy (reclaimed by the weakref finalizer). Optimizer-owned
        # pins must still decrease.
        prev = os.environ.get("DS_PIN_MEMORY_BACKEND")
        try:
            native = _require_native()
            config, dtype = _config(stage=3, pin_memory=True, offload_param=True)
            hidden_dim = 16
            model = SimpleModel(hidden_dim, nlayers=2)
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            _run_one_step(engine, hidden_dim, dtype)

            ranges_before = len(native._ranges)
            assert ranges_before > 0
            engine.destroy()
            ranges_after = len(native._ranges)
            assert ranges_after < ranges_before, (f"expected optimizer-owned pins to be freed on destroy: "
                                                  f"before={ranges_before} after={ranges_after}")
        finally:
            _restore_pin_backend(prev)

    @pytest.mark.parametrize("stage", [2, 3])
    def test_native_destroy_frees_offload_states_pins(self, stage):
        # offload_states(pin_memory=True) pins host buffers even with no ZeRO
        # CPU offload configured. destroy() must release them when the caller
        # never reloaded the states.
        prev = os.environ.get("DS_PIN_MEMORY_BACKEND")
        try:
            native = _require_native()
            config, dtype = _config(stage, pin_memory=True, offload_optimizer=False)
            hidden_dim = 16
            model = SimpleModel(hidden_dim, nlayers=2)
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            _run_one_step(engine, hidden_dim, dtype)

            ranges_before = len(native._ranges)
            engine.offload_states(pin_memory=True)
            ranges_offloaded = len(native._ranges)
            assert ranges_offloaded > ranges_before, "expected offload_states to pin host buffers"

            engine.destroy()
            assert len(native._ranges) <= ranges_before, (f"offload_states pins not released on destroy: "
                                                          f"offloaded={ranges_offloaded} after={len(native._ranges)}")
        finally:
            _restore_pin_backend(prev)

    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_native_async_offload_destroy_synchronizes_before_unpin(self, stage, monkeypatch):
        prev = os.environ.get("DS_PIN_MEMORY_BACKEND")
        try:
            native = _require_native()
            config, dtype = _config(stage, pin_memory=True, offload_optimizer=False)
            hidden_dim = 16
            model = SimpleModel(hidden_dim, nlayers=2)
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            _run_one_step(engine, hidden_dim, dtype)

            accelerator = get_accelerator()
            original_synchronize = accelerator.synchronize
            original_unpin = accelerator.unpin_memory
            events = []
            safety_sync_done = False

            def record_synchronize():
                events.append("synchronize")
                return original_synchronize()

            def safely_record_unpin(tensor):
                nonlocal safety_sync_done
                if "synchronize" not in events and not safety_sync_done:
                    original_synchronize()
                    safety_sync_done = True
                events.append("unpin")
                return original_unpin(tensor)

            monkeypatch.setattr(accelerator, "synchronize", record_synchronize)
            monkeypatch.setattr(accelerator, "unpin_memory", safely_record_unpin)

            ranges_before = len(native._ranges)
            engine.offload_states(pin_memory=True, non_blocking=True)
            assert len(native._ranges) > ranges_before, "expected offload_states to pin host buffers"
            engine.destroy()

            assert events and events[0] == "synchronize", f"destroy reached native unpin before completion: {events}"
            assert "unpin" in events

            events.clear()
            engine.destroy()
            assert events and events[0] == "synchronize", f"repeated destroy unpinned before sync: {events}"
            assert "unpin" in events
        finally:
            _restore_pin_backend(prev)

    @pytest.mark.parametrize("stage", [2, 3])
    def test_torch_destroy_succeeds(self, stage):
        # Default torch backend: unpin_memory is a no-op; destroy must still work.
        prev = os.environ.get("DS_PIN_MEMORY_BACKEND")
        try:
            os.environ.pop("DS_PIN_MEMORY_BACKEND", None)
            config, dtype = _config(stage, pin_memory=True, offload_param=False)
            hidden_dim = 16
            model = SimpleModel(hidden_dim, nlayers=2)
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            _run_one_step(engine, hidden_dim, dtype)
            engine.destroy()
            # Sanity: torch path never activates the native manager.
            assert get_active_native_pinned_memory() is None
        finally:
            _restore_pin_backend(prev)
