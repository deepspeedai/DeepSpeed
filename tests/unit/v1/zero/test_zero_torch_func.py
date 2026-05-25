# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression tests for torch.func transforms invoked directly on the engine.

Covers grad / grad_and_value / jacrev / vmap(grad) for ZeRO-0/1/2. Plain
``vmap`` skips the backward graph and already worked.
"""

import copy

import pytest
import torch
import torch.nn as nn

import deepspeed
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest


def _config(stage, gas=1):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": gas,
        "steps_per_print": 2147483647,
        "fp16": {
            "enabled": False
        },
        "bf16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": stage,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            },
        },
    }


class _Tiny(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).sum()


def _build_engine(stage, gas=1):
    model = _Tiny()
    baseline = copy.deepcopy(model).to(get_accelerator().device_name())
    engine, _, _, _ = deepspeed.initialize(model=model,
                                           config=_config(stage, gas),
                                           model_parameters=model.parameters())
    dtype = next(engine.module.parameters()).dtype
    x = torch.randn(8, device=engine.device, dtype=dtype)
    return engine, baseline, x


@pytest.mark.parametrize("stage", [0, 1, 2])
class TestEngineTorchFunc(DistributedTest):
    """``torch.func.grad`` and friends must work when invoked directly on the engine."""

    world_size = 1

    def test_grad_through_engine(self, stage):
        engine, baseline, x = _build_engine(stage)
        g_engine = torch.func.grad(lambda xi: engine(xi))(x)
        g_baseline = torch.func.grad(lambda xi: baseline(xi))(x)
        assert torch.allclose(g_engine, g_baseline, atol=1e-5)

    def test_grad_and_value_through_engine(self, stage):
        engine, baseline, x = _build_engine(stage)
        g_engine, v_engine = torch.func.grad_and_value(lambda xi: engine(xi))(x)
        g_baseline, v_baseline = torch.func.grad_and_value(lambda xi: baseline(xi))(x)
        assert torch.allclose(v_engine, v_baseline, atol=1e-5)
        assert torch.allclose(g_engine, g_baseline, atol=1e-5)

    def test_jacrev_through_engine(self, stage):
        engine, baseline, x = _build_engine(stage)
        j_engine = torch.func.jacrev(lambda xi: engine(xi))(x)
        j_baseline = torch.func.jacrev(lambda xi: baseline(xi))(x)
        assert torch.allclose(j_engine, j_baseline, atol=1e-5)

    def test_vmap_grad_through_engine(self, stage):
        # vmap(grad) still calls into autograd per slice, so it hits the same
        # engine backward hooks the fix short-circuits.
        engine, baseline, x = _build_engine(stage)
        x_batch = torch.stack([x, x + 0.1, x - 0.1])
        g_engine = torch.func.vmap(torch.func.grad(lambda xi: engine(xi)))(x_batch)
        g_baseline = torch.func.vmap(torch.func.grad(lambda xi: baseline(xi)))(x_batch)
        assert torch.allclose(g_engine, g_baseline, atol=1e-5)

    def test_grad_not_scaled_by_gas(self, stage):
        # Per-tensor hook divides by GAS by default; the guard must suppress that under torch.func.
        engine, baseline, x = _build_engine(stage, gas=4)
        g_engine = torch.func.grad(lambda xi: engine(xi))(x)
        g_baseline = torch.func.grad(lambda xi: baseline(xi))(x)
        assert torch.allclose(g_engine, g_baseline, atol=1e-5)

    def test_engine_backward_still_works(self, stage):
        # Regression guard: the functorch shortcut must not break the normal
        # engine.backward() path.
        engine, _, x = _build_engine(stage)
        for _ in range(2):
            loss = engine(x.unsqueeze(0))
            engine.backward(loss)
            engine.step()
        assert torch.isfinite(loss)


class TestZero0DirectBackwardStillRaises(DistributedTest):
    """ZeRO-0's direct ``loss.backward()`` safety net must still fire for non-functorch callers."""

    world_size = 1

    def test_direct_backward_raises_without_functorch(self):
        engine, _, x = _build_engine(stage=0)
        loss = engine(x.unsqueeze(0))
        with pytest.raises(RuntimeError, match="Direct calls to tensor.backward"):
            loss.backward()
