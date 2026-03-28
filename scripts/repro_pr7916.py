#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# Repro: functorch over ZeRO-3 memory-efficient linear (LinearFunctionForZeroStage3).
#
# Legacy autograd.Function.forward(ctx, ...) + ctx.save_for_backward in that class
# triggers (PyTorch builds that enforce functorch custom-Function rules, e.g. 2.8+):
#
#   RuntimeError: In order to use an autograd.Function with functorch transforms
#   (vmap, grad, jvp, jacrev, ...), it must override the setup_context staticmethod.
#
# Why we call zero3_linear_wrap() instead of torch.nn.functional.linear:
# After deepspeed.initialize(), the global ZeRO Init context has usually ended, so
# torch.nn.functional.linear is often restored to PyTorch's built-in. That means
# F.linear in a post-init script does NOT hit LinearFunctionForZeroStage3. The
# Stage-3 patch uses zero3_linear_wrap (see partition_parameters.py); it is the
# same autograd.Function — calling it here reliably reproduces the bug on unfixed
# trees and validates the fix on fixed trees.
#
# Regression coverage: tests/unit/v1/zero/test_zero_functorch_linear.py
#
# Run from the DeepSpeed repo root (single GPU), after scripts/setup_pr7916.sh (or manually):
#   torchrun --standalone --nproc_per_node=1 scripts/repro_pr7916.py
#
# To test an unfixed DeepSpeed tree without importing another checkout by mistake,
# copy this file outside the repo (e.g. /tmp) and set PYTHONPATH to that tree:
#   cp scripts/repro_zero3_functorch_linear.py /tmp/ && cd /tmp && \
#   PYTHONPATH=/path/to/deepspeed-checkout torchrun --standalone --nproc_per_node=1 repro_zero3_functorch_linear.py
#
# Requires: PyTorch with torch.func and strict custom-Function checks (e.g. 2.8+),
# DeepSpeed ZeRO-3, CUDA (typical setup).

import torch
import torch.nn as nn

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.linear import zero3_linear_wrap


def _assert_hits_zero3_linear(weight, inp):
    """Sanity check: we are exercising LinearFunctionForZeroStage3, not built-in linear."""
    with torch.enable_grad():
        y = zero3_linear_wrap(inp, weight, None)
    name = type(y.grad_fn).__name__
    assert "LinearFunctionForZeroStage3" in name, (
        f"Expected LinearFunctionForZeroStage3 in grad_fn, got {name!r}. "
        "Repro would not test the intended autograd.Function.")


def main():
    if not hasattr(torch, "func"):
        raise SystemExit("This repro requires torch.func (PyTorch 2.0+).")
    if not hasattr(torch.autograd.Function, "setup_context"):
        raise SystemExit("This repro requires autograd.Function.setup_context (PyTorch 2.0+).")

    deepspeed.init_distributed()
    acc = get_accelerator()
    device = acc.device_name() + ":" + str(acc.current_device())

    model = nn.Linear(8, 8, bias=True).to(device)

    config = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 2147483647,
        "zero_optimization": {
            "stage": 3,
            "stage3_param_persistence_threshold": 0,
        },
        "optimizer": {"type": "Adam", "params": {"lr": 1e-3}},
    }
    if acc.is_bf16_supported():
        config["bf16"] = {"enabled": True}
    elif acc.is_fp16_supported():
        config["fp16"] = {"enabled": True, "initial_scale_power": 8}

    _, _, _, _ = deepspeed.initialize(
        model=model,
        config=config,
        model_parameters=model.parameters(),
    )

    weight = torch.randn(8, 8, device=device, dtype=model.weight.dtype, requires_grad=True)
    inp = torch.randn(2, 8, device=device, dtype=model.weight.dtype, requires_grad=True)

    if deepspeed.comm.get_rank() == 0:
        _assert_hits_zero3_linear(weight, inp)

    def loss_fn(w, x):
        # Same op as ZeRO-3's F.linear replacement when the patch is active.
        return zero3_linear_wrap(x, w, None).sum()

    torch.func.grad_and_value(loss_fn, argnums=(0, 1))(weight, inp)
    if deepspeed.comm.get_rank() == 0:
        print("repro: grad_and_value over zero3_linear_wrap (LinearFunctionForZeroStage3) OK.")


if __name__ == "__main__":
    main()
