# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A step the loss scaler discards must leave Muon's momentum as it was.

Muon folds the gradient into its momentum while the partition is filled, which happens
before the overflow check decides whether to keep the step. With nesterov the blend is
also written back into the gradient in place. One overflow would therefore leave the
momentum non-finite for the rest of the run and make every later step overflow too,
until the scaler reaches its minimum and raises.
"""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.muon.original_muon import muon_update
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def test_an_overflowed_gradient_does_not_enter_the_momentum():
    """The unit of the behaviour, without a training loop around it."""
    device = get_accelerator().device_name()
    grad = torch.randn(16, 16, device=device)
    momentum = torch.randn(16, 16, device=device)
    before = momentum.clone()

    overflowed = grad.clone()
    overflowed[0, 0] = float("inf")
    update = muon_update(overflowed, momentum)

    assert torch.equal(momentum, before), "an overflowed step must not move the momentum"
    assert not torch.isfinite(update).all(), \
        "the update has to stay non-finite, or the overflow check will not skip the step"


def test_a_finite_gradient_still_moves_the_momentum():
    """The guard must not disable the optimizer."""
    device = get_accelerator().device_name()
    grad = torch.randn(16, 16, device=device)
    momentum = torch.zeros(16, 16, device=device)

    update = muon_update(grad.clone(), momentum)

    assert momentum.abs().sum() > 0
    assert torch.isfinite(update).all()


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestMuonSurvivesLossScaleBackoff(DistributedTest):
    world_size = 2

    def test_training_recovers_from_the_initial_overflow(self, zero_stage):
        """fp16 starts at a loss scale that overflows; backing off is the normal path.

        On the parent commit this never recovers: the momentum is NaN from the first step,
        the poisoned gradient keeps the overflow check firing, and DeepSpeed raises
        "Current loss scale already at minimum - cannot decrease scale anymore".
        """
        if torch.half not in get_accelerator().supported_dtypes():
            pytest.skip("fp16 not supported")

        hidden_dim, batch_size = 128, 8
        torch.manual_seed(0)
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=5)
        config = {
            "train_batch_size": batch_size,
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 0.05
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_scatter": False
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

        # Captured after initialize: before it the parameters are still fp32, and comparing
        # across the fp16 cast makes any assertion about change trivially true.
        before = [p.clone().cpu() for p in model.parameters()]
        for _ in range(30):
            x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
            y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
            engine.backward(engine(x, y))
            engine.step()
        after = [p.clone().cpu() for p in model.parameters()]

        changed = sum(1 for b, a in zip(before, after) if not torch.equal(b, a))
        assert changed == len(before), f"only {changed}/{len(before)} parameters moved in 30 steps"

        optimizer = getattr(engine.optimizer, "optimizer", engine.optimizer)
        for state in optimizer.state.values():
            buffer = state.get("momentum_buffer") if isinstance(state, dict) else None
            if buffer is not None:
                assert torch.isfinite(buffer.float()).all(), "the momentum did not survive the backoff"
