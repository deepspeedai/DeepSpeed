# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression: ZeRO-3 linear autograd.Function must work with torch.func transforms.

ZeRO Stage 3 uses ``LinearFunctionForZeroStage3`` (via ``zero3_linear_wrap``) as
the memory-efficient linear path. After ``deepspeed.initialize``, global
``torch.nn.functional.linear`` is often the built-in again, so tests call
``zero3_linear_wrap`` directly-the same ``autograd.Function`` as when the patch
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


class TestZeroLinearAutocast(DistributedTest):
    """Verify autocast state is correctly propagated through forward and backward."""

    world_size = 1

    def _run_forward_backward(self, device, use_autocast, dtype=None):
        """Run zero3_linear_wrap forward+backward, optionally inside autocast."""
        weight = torch.randn(4, 4, device=device, dtype=torch.float32, requires_grad=True)
        inp = torch.randn(2, 4, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(4, device=device, dtype=torch.float32, requires_grad=True)

        if use_autocast:
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                out = zero3_linear_wrap(inp, weight, bias)
        else:
            out = zero3_linear_wrap(inp, weight, bias)

        loss = out.sum()
        loss.backward()
        return out, weight.grad, inp.grad, bias.grad

    def test_backward_without_autocast(self):
        """Backward without autocast should produce float32 gradients."""
        model = nn.Linear(4, 4)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=_zero3_functorch_config(),
            model_parameters=model.parameters(),
        )
        device = engine.device

        out, w_grad, i_grad, b_grad = self._run_forward_backward(device, use_autocast=False)
        assert out.dtype == torch.float32
        assert w_grad.dtype == torch.float32
        assert i_grad.dtype == torch.float32
        assert b_grad.dtype == torch.float32

    def test_backward_with_autocast(self):
        """Backward with autocast should produce float32 gradients (autocast only affects forward)."""
        acc = get_accelerator()
        if acc.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif acc.is_fp16_supported():
            amp_dtype = torch.float16
        else:
            pytest.skip("No half-precision support")

        model = nn.Linear(4, 4)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=_zero3_functorch_config(),
            model_parameters=model.parameters(),
        )
        device = engine.device

        out, w_grad, i_grad, b_grad = self._run_forward_backward(device, use_autocast=True, dtype=amp_dtype)
        # Forward output should be in reduced precision
        assert out.dtype == amp_dtype
        # Gradients accumulate in float32 (master weights)
        assert w_grad.dtype == torch.float32
        assert i_grad.dtype == torch.float32
        assert b_grad.dtype == torch.float32

    def test_no_autocast_leak_into_backward(self):
        """When forward runs without autocast, an outer autocast during backward must not affect gradient dtype."""
        model = nn.Linear(4, 4)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=_zero3_functorch_config(),
            model_parameters=model.parameters(),
        )
        device = engine.device

        acc = get_accelerator()
        if acc.is_bf16_supported():
            amp_dtype = torch.bfloat16
        elif acc.is_fp16_supported():
            amp_dtype = torch.float16
        else:
            pytest.skip("No half-precision support")

        weight = torch.randn(4, 4, device=device, dtype=torch.float32, requires_grad=True)
        inp = torch.randn(2, 4, device=device, dtype=torch.float32, requires_grad=True)

        # Forward WITHOUT autocast
        out = zero3_linear_wrap(inp, weight, None)
        assert out.dtype == torch.float32

        # Backward WITH an outer autocast region -- should NOT affect gradient computation
        # because setup_context captured _fwd_used_autocast=False
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            out.sum().backward()

        assert weight.grad.dtype == torch.float32
        assert inp.grad.dtype == torch.float32

    def test_setup_context_stores_autocast_attrs(self):
        """setup_context must store _fwd_used_autocast and _dtype on ctx."""
        model = nn.Linear(4, 4)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config=_zero3_functorch_config(),
            model_parameters=model.parameters(),
        )
        device = engine.device

        weight = torch.randn(4, 4, device=device, dtype=torch.float32, requires_grad=True)
        inp = torch.randn(2, 4, device=device, dtype=torch.float32, requires_grad=True)

        # Without autocast: setup_context must record that forward did not use autocast
        out = zero3_linear_wrap(inp, weight, None)
        grad_fn = out.grad_fn
        assert hasattr(grad_fn, "_fwd_used_autocast")
        assert grad_fn._fwd_used_autocast is False
        assert hasattr(grad_fn, "_dtype")
        out.sum().backward()
        assert torch.isfinite(weight.grad).all()
