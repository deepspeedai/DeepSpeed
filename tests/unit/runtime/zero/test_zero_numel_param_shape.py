# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A zero-element parameter must keep its shape when it is bound to a flat buffer.

Every optimizer wrapper that flattens the parameters repoints each one at its slice of
the flat buffer. torch's `unflatten_dense_tensors` special-cases a zero-element tensor
and returns a freshly allocated 1-D `zeros({0})` rather than a view of the requested
shape, so a `(0, 8)` parameter came back as `(0,)` and the module's own forward then
dispatched `F.linear` to `addmv`:

    RuntimeError: size mismatch, got input (1), mat (1x8), vec (0)

The parameters are rebuilt on every `step()` as well as at init, so the shape did not
survive one iteration either.
"""

import pytest
import torch

from unit.common import DistributedTest

import deepspeed

HIDDEN = 8

# One case per wrapper that binds parameters to a flat buffer.
CONFIGS = {
    "fp16_stage0": ({
        "fp16": {
            "enabled": True,
            "loss_scale": 1.0
        },
        "zero_optimization": {
            "stage": 0
        }
    }, torch.float16),
    "bf16_stage0": ({
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 0
        }
    }, torch.bfloat16),
    "bf16_stage1_fp32_accum": ({
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1
        },
        "data_types": {
            "grad_accum_dtype": "fp32"
        }
    }, torch.bfloat16),
    "zero1": ({
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1
        }
    }, torch.bfloat16),
    "zero2": ({
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        }
    }, torch.bfloat16),
}


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


def _engine(case):
    extra, _ = CONFIGS[case]
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        **extra,
    }
    model = EmptyTailModel()
    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine


def _step(engine, case):
    _, dtype = CONFIGS[case]
    loss = engine(torch.randn(1, HIDDEN, device=engine.device, dtype=dtype))
    engine.backward(loss)
    engine.step()


@pytest.mark.parametrize("case", list(CONFIGS))
class TestZeroNumelParameterShape(DistributedTest):
    world_size = 1

    def test_shape_survives_initialize(self, case):
        engine = _engine(case)

        assert engine.module.empty.weight.shape == torch.Size([0, HIDDEN])
        # The sized parameter shares the flat buffer, which is what makes the
        # zero-element one the special case rather than the rule.
        assert engine.module.dense.weight.shape == torch.Size([HIDDEN, HIDDEN])

    def test_shape_survives_a_step(self, case):
        engine = _engine(case)

        _step(engine, case)

        assert engine.global_steps == 1
        assert engine.module.empty.weight.shape == torch.Size([0, HIDDEN])

    def test_a_second_step_still_runs(self, case):
        # step() rebuilds the parameters from the flat buffer, so a shape lost there
        # only shows up on the forward of the iteration after it.
        engine = _engine(case)

        _step(engine, case)
        _step(engine, case)

        assert engine.global_steps == 2
