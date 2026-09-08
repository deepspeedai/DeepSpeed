# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A tensor whose own gradient overflowed must not absorb it into its momentum.

Muon folds the gradient into its momentum while the partition is filled, which happens
before the overflow check decides whether to keep the step. With nesterov the blend is
also written back into the gradient in place. One overflow would therefore leave the
momentum non-finite for the rest of the run and make every later step overflow too,
until the scaler reaches its minimum and raises.

The guard is per tensor, and the loss scaler's decision is global: `has_overflow` reduces
`_has_inf_or_nan` over every partitioned gradient and `step()` discards the whole step on
that one flag. So a tensor whose own gradient was finite still advances its momentum on a
step discarded because some *other* tensor overflowed, and that update is thrown away.
`test_a_finite_tensor_still_absorbs_a_step_discarded_for_another` pins that, measured
rather than assumed; making it exact needs the momentum write deferred until the step is
known to survive, which costs a second buffer the size of the momentum.
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

    assert torch.equal(momentum, before), "a tensor's own overflow must not move its momentum"
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


class TestMuonMixedOverflow(DistributedTest):
    world_size = 1

    def test_a_finite_tensor_still_absorbs_a_step_discarded_for_another(self):
        """The scope of the guard, recorded rather than implied.

        Two 2-D parameters in one group; only `boom` is fed an input that overflows in
        fp16. The step is discarded for the whole model, so neither parameter moves --
        but `calm`'s gradient was finite, so its momentum advances anyway, for an update
        that is thrown away. Making that exact needs the momentum write deferred until
        the step is known to survive; this pins today's behaviour so the gap is visible.
        """
        if torch.half not in get_accelerator().supported_dtypes():
            pytest.skip("fp16 not supported")

        hidden_dim = 32
        numel = hidden_dim * hidden_dim

        class TwoMatrices(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.calm = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.boom = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

            def forward(self, calm_x, boom_x):
                return self.calm(calm_x).sum() + self.boom(boom_x).sum()

        torch.manual_seed(0)
        model = TwoMatrices()
        config = {
            "train_batch_size": 1,
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.0
                }
            },
            # low enough that a normal step does not overflow, so the only overflow is
            # the one the test injects
            "fp16": {
                "enabled": True,
                "initial_scale_power": 4
            },
            "zero_optimization": {
                "stage": 1
            },
            "zero_allow_untested_optimizer": True,
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=list(model.parameters()), config=config)

        def momentum_halves():
            """(calm, boom) slices of the group's flat momentum buffer, in parameter order."""
            inner = getattr(engine.optimizer, "optimizer", engine.optimizer)
            for state in inner.state.values():
                buffer = state.get("momentum_buffer") if isinstance(state, dict) else None
                if buffer is not None and buffer.numel() >= 2 * numel:
                    flat = buffer.detach().float()
                    return flat[:numel].norm().item(), flat[numel:2 * numel].norm().item()
            return None, None

        device = engine.device
        calm_x = torch.randn(1, hidden_dim, device=device, dtype=torch.half)
        finite_x = torch.randn(1, hidden_dim, device=device, dtype=torch.half)
        overflowing_x = torch.full((1, hidden_dim), 6e4, device=device, dtype=torch.half)

        # Step 0 establishes a momentum for both; step 1 overflows only through `boom`.
        for step in range(2):
            # muon_update runs while the partition is filled, i.e. inside backward, so the
            # momentum has to be read on either side of that rather than around step().
            calm_before, boom_before = momentum_halves()
            params_before = [p.detach().float().norm().item() for p in model.parameters()]
            engine.backward(engine(calm_x, overflowing_x if step == 1 else finite_x))
            calm_after, boom_after = momentum_halves()
            engine.step()
            params_after = [p.detach().float().norm().item() for p in model.parameters()]

            if step == 1:
                assert engine.optimizer.overflow, "step 1 was meant to overflow"
                assert params_before == params_after, "an overflowed step must not move parameters"
                assert boom_before == boom_after, \
                    "the tensor whose own gradient overflowed must not absorb it"
                assert calm_before != calm_after, (
                    "a tensor whose gradient was finite does advance its momentum on a step "
                    "discarded for another tensor -- if this starts failing, the guard has "
                    "become global and the module docstring should say so")
