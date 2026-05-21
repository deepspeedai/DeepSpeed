# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""``torch.func`` transforms must pass cleanly through the engine for ZeRO-0/1/2.

Calling ``torch.func.grad`` / ``grad_and_value`` / ``jacrev`` on a model wrapped
by ``deepspeed.initialize`` invokes the autograd engine via
``torch.autograd.grad``, which fires the engine's output-tensor hooks. Without
the functorch-aware guard, the prologue/epilogue mutate ZeRO state that the
transformed graph never populates (no per-param post-accumulate-grad hooks,
since parameters are not leaves under the transform) and the epilogue then
indexes empty bucket bookkeeping, surfacing either a ``RuntimeError`` (ZeRO-0)
or ``IndexError`` (ZeRO-1/2).

``vmap`` alone runs only the forward graph so it does not exercise the same
hooks; it is not covered here.
"""

import copy

import pytest
import torch
import torch.nn as nn

import deepspeed
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest


def _config(stage):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
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


def _build_engine(stage):
    model = _Tiny()
    baseline = copy.deepcopy(model).to(get_accelerator().device_name())
    engine, _, _, _ = deepspeed.initialize(model=model, config=_config(stage), model_parameters=model.parameters())
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
