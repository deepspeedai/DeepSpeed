# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A zero-sized trainable parameter must survive a full ZeRO step on every stage.

Stages 1 and 2 were fixed in #8280 and #8298 (issues #8279, #8297). Stage 3 took the same
shape of failure from a different place: `fetch_sub_module` gates the all-gather on
`fetch_numel > 0`, and a submodule holding only a zero-sized parameter contributes nothing to
that sum, so the gather never runs, the parameter stays `NOT_AVAILABLE`, and the wait loop
immediately below asserts that it is `AVAILABLE`.
"""

import pytest
import torch

from unit.common import DistributedTest

import deepspeed


class EmptyTailModel(torch.nn.Module):
    """The shape from issue #8279: a trainable parameter with no elements, used in the loss."""

    def __init__(self, hidden=8):
        super().__init__()
        self.dense = torch.nn.Linear(hidden, hidden, bias=False)
        self.empty = torch.nn.Linear(hidden, 0, bias=False)

    def forward(self, x):
        hidden = self.dense(x)
        # `empty(hidden)` is (batch, 0); summing it keeps the parameter in the autograd graph.
        return hidden.sum() + self.empty(hidden).sum()


def _run_one_step(stage, hidden=8):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "zero_optimization": {
            "stage": stage
        },
        "fp16": {
            "enabled": False
        },
    }
    model = EmptyTailModel(hidden)
    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

    loss = engine(torch.randn(1, hidden, device=engine.device, dtype=next(engine.parameters()).dtype))
    engine.backward(loss)
    engine.step()

    return engine


class TestZeroSizedParameterSingleRank(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_step_completes(self, stage):
        engine = _run_one_step(stage)
        assert engine.global_steps == 1


class TestZeroSizedParameterPartitioned(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_step_completes(self, stage):
        # With more than one rank the stage-3 path goes through the real all-gather rather than
        # the single-rank shortcut, which is where the gate lives.
        engine = _run_one_step(stage)
        assert engine.global_steps == 1
