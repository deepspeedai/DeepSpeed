# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""ZeRO-3 runs Muon once per optimizer step, not once per micro-batch (#8443).

Muon used to run inside the ZeRO-3 gradient reduce, which happens every micro-batch, so with
`gradient_accumulation_steps: n` the momentum advanced n times per step and Newton-Schulz saw
partial gradients. ZeRO-1/2 were already correct.
"""

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
import deepspeed.runtime.zero.muon.original_muon as original_muon
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest

HIDDEN = 64
SAMPLES_PER_STEP = 8


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(HIDDEN, HIDDEN, bias=False), torch.nn.Linear(HIDDEN, HIDDEN,
                                                                                            bias=False))


def _train(zero_stage, gas, steps):
    """Train on the same SAMPLES_PER_STEP samples per step, split into `gas` micro-batches."""
    micro_batch = SAMPLES_PER_STEP // gas
    config = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": gas,
        "optimizer": {
            "type": "muon",
            "params": {
                "lr": 0.02,
                "momentum": 0.95,
                "weight_decay": 0.0
            }
        },
        "gradient_clipping": 0.0,
        "zero_optimization": {
            "stage": zero_stage,
            # ZeRO-3 rejects Muon with reduce scatter, so keep both stages on the same reduction.
            "reduce_scatter": False
        },
    }
    model = _model()
    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    generator = torch.Generator().manual_seed(dist.get_rank() + 1)
    for _ in range(steps):
        x = torch.randn(SAMPLES_PER_STEP, HIDDEN, generator=generator).to(engine.device)
        y = torch.randn(SAMPLES_PER_STEP, HIDDEN, generator=generator).to(engine.device)
        for m in range(gas):
            batch = slice(m * micro_batch, (m + 1) * micro_batch)
            loss = torch.nn.functional.mse_loss(engine(x[batch]), y[batch])
            engine.backward(loss)
            engine.step()
    if zero_stage == 3:
        with deepspeed.zero.GatheredParameters(list(engine.module.parameters())):
            return [p.detach().float().cpu().clone() for p in engine.module.parameters()]
    return [p.detach().float().cpu().clone() for p in engine.module.parameters()]


class TestZero3MuonOncePerStep(DistributedTest):
    world_size = 2

    def test_newton_schulz_runs_once_per_matrix_per_step(self, monkeypatch):
        calls = []
        for name in ("zeropower_via_gram_newtonschulz", "zeropower_via_newtonschulz5"):
            original = getattr(original_muon, name)

            def counted(*args, _original=original, **kwargs):
                calls.append(1)
                return _original(*args, **kwargs)

            monkeypatch.setattr(original_muon, name, counted)

        steps = 2
        _train(zero_stage=3, gas=4, steps=steps)

        total = torch.tensor([len(calls)], device=get_accelerator().current_device_name())
        dist.all_reduce(total)
        # Two matrices, orthogonalized once per step across the ranks: 4, not 4 x gas = 16.
        assert total.item() == 2 * steps

    @pytest.mark.parametrize("zero_stage", [2, 3])
    def test_gradient_accumulation_matches_one_large_micro_batch(self, zero_stage):
        """Same samples per step either way, so only half-precision Newton-Schulz noise may differ."""
        one = _train(zero_stage, gas=1, steps=3)
        four = _train(zero_stage, gas=4, steps=3)

        relative = torch.cat([(a - b).flatten()
                              for a, b in zip(one, four)]).norm() / torch.cat([a.flatten() for a in one]).norm()
        # Measured on 2 GPUs: 3.6e-4 at both stages; stage 3 was 1.3e-1 before this change.
        assert relative.item() < 5e-3
