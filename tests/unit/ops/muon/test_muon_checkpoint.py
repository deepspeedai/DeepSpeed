# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Muon has to survive a checkpoint round trip, in the dtype the run is using.

ZeRO 1/2 keep Muon's momentum in a hand-flattened tensor under
``optimizer.state[flatten_copy]["momentum_buffer"]``, allocated in the gradient
accumulation dtype. A checkpoint stores optimizer state in fp32, so a restored buffer
comes back in a different dtype than the gradients it is combined with, and
``muon_update``'s ``momentum.lerp_(grad)`` requires the two to match.

The existing Muon suite covers neither half of this: it runs in fp16, where the two
dtypes coincide, and it never saves or loads.
"""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def _config(zero_stage, dtype):
    return {
        "train_micro_batch_size_per_gpu":
        2,
        "gradient_accumulation_steps":
        1,
        "steps_per_print":
        10**9,
        dtype: ({
            "enabled": True,
            # A loss scale that overflows spends the run backing off instead of stepping,
            # and this test is about what a checkpoint carries, not about the scaler.
            "initial_scale_power": 4,
        } if dtype == "fp16" else {
            "enabled": True
        }),
        "zero_optimization": {
            "stage": zero_stage,
            # Muon rejects reduce-scatter, as the rest of the suite does too.
            "reduce_scatter": False,
        },
        "optimizer": {
            "type": "Muon",
            "params": {
                "lr": 1e-3,
                "momentum": 0.95
            }
        },
    }


def _momentum_norm(engine):
    optimizer = getattr(engine.optimizer, "optimizer", engine.optimizer)
    total = 0.0
    for state in optimizer.state.values():
        buffer = state.get("momentum_buffer") if isinstance(state, dict) else None
        if buffer is not None:
            total += float(buffer.float().norm()**2)
    return total**0.5


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
class TestMuonCheckpointRoundTrip(DistributedTest):
    world_size = 2

    def test_resumes_with_its_momentum(self, tmpdir, zero_stage, dtype):
        """A resumed run must continue the one it resumed, not restart its momentum.

        Before the dtype was reconciled this raised on stages 1 and 2 under bf16:

            RuntimeError: expected dtype torch.float32 for `end`,
            but got dtype torch.bfloat16
        """
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.half
        if torch_dtype not in get_accelerator().supported_dtypes():
            pytest.skip(f"{dtype} not supported by {get_accelerator().device_name()}")

        hidden_dim, steps = 32, 4
        ckpt_dir = str(tmpdir)

        def build():
            torch.manual_seed(1234)
            model = SimpleModel(hidden_dim, nlayers=3)
            engine, _, _, _ = deepspeed.initialize(model=model,
                                                   model_parameters=model.parameters(),
                                                   config=_config(zero_stage, dtype))
            return engine

        uninterrupted = build()
        batch = torch.randn(2, hidden_dim, device=uninterrupted.device, dtype=torch_dtype)
        labels = torch.randn(2, hidden_dim, device=uninterrupted.device, dtype=torch_dtype)

        for _ in range(steps):
            uninterrupted.backward(uninterrupted(batch, labels))
            uninterrupted.step()
        momentum_at_save = _momentum_norm(uninterrupted)
        uninterrupted.save_checkpoint(ckpt_dir, tag="mid")
        for _ in range(steps):
            uninterrupted.backward(uninterrupted(batch, labels))
            uninterrupted.step()
        straight_through = [float(p.float().norm()) for p in uninterrupted.module.parameters()]

        resumed = build()
        resumed.load_checkpoint(ckpt_dir, tag="mid")
        assert _momentum_norm(resumed) == pytest.approx(momentum_at_save, rel=1e-3), \
            "the checkpoint carries the momentum; a resume that drops it is a different run"

        for _ in range(steps):
            resumed.backward(resumed(batch, labels))
            resumed.step()
        after_resume = [float(p.float().norm()) for p in resumed.module.parameters()]

        for straight, restored in zip(straight_through, after_resume):
            assert straight == pytest.approx(restored, abs=1e-4)
