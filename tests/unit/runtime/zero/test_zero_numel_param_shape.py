# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A zero-element parameter must keep its shape across the ZeRO-1/2 flat buffer.

`_update_model_bit16_weights` repoints every parameter at its slice of the flattened
group. torch's `unflatten_dense_tensors` special-cases a zero-element tensor and returns
a freshly allocated 1-D `zeros({0})` rather than a view of the requested shape, so a
`(0, 8)` parameter came back as `(0,)` and the module's own forward then dispatched
`F.linear` to `addmv`:

    RuntimeError: size mismatch, got input (1), mat (1x8), vec (0)

The parameter is rebuilt on every `step()` as well as at init, so the shape did not
survive one iteration either.
"""

import pytest
import torch

from unit.common import DistributedTest

import deepspeed

HIDDEN = 8


class EmptyTailModel(torch.nn.Module):
    """A trainable parameter with no elements, kept in the autograd graph by the loss."""

    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.dense = torch.nn.Linear(hidden, hidden, bias=False)
        self.empty = torch.nn.Linear(hidden, 0, bias=False)

    def forward(self, x):
        hidden = self.dense(x)
        # `empty(hidden)` is (batch, 0); summing it keeps the parameter in the graph.
        return hidden.sum() + self.empty(hidden).sum()


def _engine(stage):
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
        "bf16": {
            "enabled": True
        },
    }
    model = EmptyTailModel()
    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _step(engine):
    x = torch.randn(1, HIDDEN, device=engine.device, dtype=torch.bfloat16)
    loss = engine(x)
    engine.backward(loss)
    engine.step()


@pytest.mark.parametrize("stage", [1, 2])
class TestZeroNumelParameterShape(DistributedTest):
    world_size = 1

    def test_shape_survives_initialize(self, stage):
        engine = _engine(stage)

        assert engine.module.empty.weight.shape == torch.Size([0, HIDDEN])
        # The sized parameter shares the flat buffer, which is what makes the
        # zero-element one the special case rather than the rule.
        assert engine.module.dense.weight.shape == torch.Size([HIDDEN, HIDDEN])

    def test_shape_survives_a_step(self, stage):
        engine = _engine(stage)

        _step(engine)

        assert engine.global_steps == 1
        assert engine.module.empty.weight.shape == torch.Size([0, HIDDEN])

    def test_a_second_step_still_runs(self, stage):
        # step() rebuilds the parameters from the flat buffer, so a shape lost there
        # only shows up on the forward of the iteration after it.
        engine = _engine(stage)

        _step(engine)
        _step(engine)

        assert engine.global_steps == 2
