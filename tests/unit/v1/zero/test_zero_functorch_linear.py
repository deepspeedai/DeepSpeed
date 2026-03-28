# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression: ZeRO-3 linear autograd.Function must work with torch.func transforms.

ZeRO Stage 3 uses ``LinearFunctionForZeroStage3`` (via ``zero3_linear_wrap``) as
the memory-efficient linear path. After ``deepspeed.initialize``, global
``torch.nn.functional.linear`` is often the built-in again, so tests call
``zero3_linear_wrap`` directly—the same ``autograd.Function`` as when the patch
is active. Legacy ``forward(ctx, ...)`` + ``ctx.save_for_backward`` in forward
raises on strict functorch builds::

    RuntimeError: In order to use an autograd.Function with functorch
    transforms ... it must override the setup_context staticmethod.
"""

import pytest
import torch
import torch.nn as nn

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.linear import zero3_linear_wrap

from unit.common import DistributedTest


def _zero3_functorch_config():
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 2147483647,
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 0,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            },
        },
    }
    acc = get_accelerator()
    if acc.is_bf16_supported():
        config["bf16"] = {"enabled": True}
    elif acc.is_fp16_supported():
        config["fp16"] = {"enabled": True, "initial_scale_power": 8}
    return config


class TestZeroFunctorchLinearRegression(DistributedTest):
    """``torch.func.grad_and_value`` over ``zero3_linear_wrap`` / LinearFunctionForZeroStage3."""

    world_size = 1

    def test_grad_and_value_over_patched_functional_linear(self):
        if not hasattr(torch, "func"):
            pytest.skip("torch.func not available")
        if not hasattr(torch.autograd.Function, "setup_context"):
            pytest.skip("Requires PyTorch 2.0+ autograd.Function.setup_context")

        model = nn.Linear(8, 8, bias=True)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=_zero3_functorch_config(),
            model_parameters=model.parameters(),
        )

        device = engine.device
        dtype = engine.module.weight.dtype
        weight = torch.randn(8, 8, device=device, dtype=dtype, requires_grad=True)
        inp = torch.randn(2, 8, device=device, dtype=dtype, requires_grad=True)

        with torch.enable_grad():
            probe = zero3_linear_wrap(inp, weight, None)
        assert "LinearFunctionForZeroStage3" in type(probe.grad_fn).__name__

        def loss_fn(w, x):
            return zero3_linear_wrap(x, w, None).sum()

        grads, value = torch.func.grad_and_value(loss_fn, argnums=(0, 1))(weight, inp)
        assert torch.isfinite(value)
        assert grads[0] is not None and torch.isfinite(grads[0]).all()
        assert grads[1] is not None and torch.isfinite(grads[1]).all()
