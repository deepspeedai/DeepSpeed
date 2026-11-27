# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

import deepspeed
import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.lamb import FusedLamb
from deepspeed.ops.op_builder import FusedLambBuilder


def _skip_if_incompatible():
    if not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME]:
        pytest.skip("FusedLamb is not compatible")


def _reference_lamb_step(param,
                         grad,
                         exp_avg,
                         exp_avg_sq,
                         step,
                         lr,
                         beta1,
                         beta2,
                         eps,
                         weight_decay,
                         eps_inside_sqrt,
                         bias_correction,
                         min_coeff,
                         max_coeff,
                         grad_scale=1.0):
    """FP64 reference matching csrc/lamb/fused_lamb_cuda_kernel.cu."""
    p = param.double()
    g = grad.double() / grad_scale
    m = beta1 * exp_avg.double() + (1.0 - beta1) * g
    v = beta2 * exp_avg_sq.double() + (1.0 - beta2) * g * g
    denom = torch.sqrt(v + eps) if eps_inside_sqrt else torch.sqrt(v) + eps
    update = (m / denom) + (weight_decay * p)

    reg_w = torch.sqrt(torch.sum(p * p))
    reg_u = torch.sqrt(torch.sum(update * update))
    lamb_coeff = 1.0
    if reg_w.item() != 0.0 and reg_u.item() != 0.0:
        lamb_coeff = float((reg_w / reg_u).item())
        lamb_coeff = max(min_coeff, min(max_coeff, lamb_coeff))

    if bias_correction:
        step_size = lr * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
    else:
        step_size = lr

    p_new = p - step_size * lamb_coeff * update
    return p_new, m, v, lamb_coeff


@pytest.mark.parametrize('numel', [512, 700, 1024, 2048], ids=["1block", "partial_block", "2blocks", "4blocks"])
@pytest.mark.parametrize('weight_decay', [1.0, 0.5], ids=["wd1.0", "wd0.5"])
@pytest.mark.parametrize('bias_correction', [False, True], ids=["no_bc", "bc"])
def test_fused_lamb_large_norm_no_overflow(numel, weight_decay, bias_correction):
    """Regression test for Wave-64 shuffle reduction inflation in reduce_block_in_shared_memory.

    With ||w||_2^2 = 1e38 (< FLT_MAX ~ 3.4e38) and zero gradients:
      - weight_decay = 1.0 keeps ||w||_2^2 = ||u||_2^2 = 1e38 (trust ratio = 1.0);
        unpatched Wave-64 4x inflation overflows both to Inf and yields NaN.
      - weight_decay = 0.5 keeps ||w||_2^2 = 1e38 and ||u||_2^2 = 0.25e38 (trust ratio = 2.0);
        unpatched Wave-64 4x inflation overflows only ||w||_2^2 and wrongly clamps to max_coeff.
    """
    _skip_if_incompatible()

    device = get_accelerator().device_name()
    lr, betas, eps = 1e-3, (0.9, 0.999), 1e-6
    target_sq_norm = 1e38
    init_val = math.sqrt(target_sq_norm / numel)
    expected_coeff = 1.0 / weight_decay

    param = torch.nn.Parameter(torch.full((numel, ), init_val, device=device, dtype=torch.float32))
    param.grad = torch.zeros_like(param)
    init_fp64 = param.detach().double().clone()

    optimizer = FusedLamb([param],
                          lr=lr,
                          betas=betas,
                          eps=eps,
                          weight_decay=weight_decay,
                          bias_correction=bias_correction)
    optimizer.step()

    coeffs = optimizer.get_lamb_coeffs()
    assert len(coeffs) == 1
    assert math.isfinite(coeffs[0]), f"Expected finite trust coefficient, got {coeffs[0]}"
    assert coeffs[0] == pytest.approx(expected_coeff, rel=1e-5)

    step_size = lr * math.sqrt(1.0 - betas[1]) / (1.0 - betas[0]) if bias_correction else lr
    expected = init_fp64 * (1.0 - step_size * expected_coeff * weight_decay)
    assert torch.isfinite(param).all(), "Parameter update produced non-finite values"
    torch.testing.assert_close(param.detach().double(), expected, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize('numel', [512, 700, 2048], ids=["1block", "partial_block", "4blocks"])
@pytest.mark.parametrize('eps_inside_sqrt', [False, True], ids=["eps_out", "eps_in"])
@pytest.mark.parametrize('bias_correction', [False, True], ids=["no_bc", "bc"])
def test_fused_lamb_matches_reference(numel, eps_inside_sqrt, bias_correction):
    _skip_if_incompatible()

    device = get_accelerator().device_name()
    torch.manual_seed(0)
    lr, betas, eps, weight_decay = 1e-2, (0.9, 0.999), 1e-6, 0.1
    min_coeff, max_coeff = 0.01, 10.0

    param = torch.nn.Parameter(torch.randn(numel, device=device, dtype=torch.float32))
    ref_param = param.detach().clone()
    ref_m = torch.zeros_like(ref_param)
    ref_v = torch.zeros_like(ref_param)

    optimizer = FusedLamb([param],
                          lr=lr,
                          betas=betas,
                          eps=eps,
                          eps_inside_sqrt=eps_inside_sqrt,
                          weight_decay=weight_decay,
                          bias_correction=bias_correction,
                          min_coeff=min_coeff,
                          max_coeff=max_coeff)

    for step in range(1, 4):
        grad = torch.randn_like(param)
        param.grad = grad
        ref_param, ref_m, ref_v, ref_coeff = _reference_lamb_step(ref_param, grad, ref_m, ref_v, step, lr, betas[0],
                                                                  betas[1], eps, weight_decay, eps_inside_sqrt,
                                                                  bias_correction, min_coeff, max_coeff)
        optimizer.step()
        assert optimizer.get_lamb_coeffs()[0] == pytest.approx(ref_coeff, rel=1e-4, abs=1e-5)

    torch.testing.assert_close(param.detach().double(), ref_param, rtol=1e-4, atol=1e-5)


def test_fused_lamb_fp16_grad_large_norm_no_overflow():
    """Exercise the FP16-gradient / FP32-master-weight kernel dispatch path."""
    _skip_if_incompatible()
    if torch.half not in get_accelerator().supported_dtypes():
        pytest.skip(f"fp16 not supported on {get_accelerator().device_name()}")

    device = get_accelerator().device_name()
    numel = 1024
    lr, weight_decay = 1e-3, 1.0
    init_val = math.sqrt(1e38 / numel)

    param = torch.nn.Parameter(torch.full((numel, ), init_val, device=device, dtype=torch.float32))
    grad_fp16 = torch.zeros(numel, device=device, dtype=torch.float16)
    init_fp64 = param.detach().double().clone()

    optimizer = FusedLamb([param], lr=lr, weight_decay=weight_decay, bias_correction=False)
    optimizer.step(grads=[grad_fp16])

    coeff = optimizer.get_lamb_coeffs()[0]
    assert math.isfinite(coeff)
    assert coeff == pytest.approx(1.0, rel=1e-5)
    expected = init_fp64 * (1.0 - lr * weight_decay)
    assert torch.isfinite(param).all()
    torch.testing.assert_close(param.detach().double(), expected, rtol=1e-5, atol=0.0)


def test_fused_lamb_fp16_grad_with_output_params():
    """Verify FP16 gradient + FP16 output_params copy against the FP64 reference."""
    _skip_if_incompatible()
    if torch.half not in get_accelerator().supported_dtypes():
        pytest.skip(f"fp16 not supported on {get_accelerator().device_name()}")

    device = get_accelerator().device_name()
    torch.manual_seed(0)
    numel = 1024
    lr, betas, eps, weight_decay = 1e-2, (0.9, 0.999), 1e-6, 0.1
    min_coeff, max_coeff = 0.01, 10.0

    param = torch.nn.Parameter(torch.randn(numel, device=device, dtype=torch.float32))
    out_param = torch.empty(numel, device=device, dtype=torch.float16)
    grad_fp16 = torch.randn(numel, device=device, dtype=torch.float16)

    ref_param, _, _, ref_coeff = _reference_lamb_step(param.detach().clone(),
                                                      grad_fp16.float(), torch.zeros_like(param),
                                                      torch.zeros_like(param), 1, lr, betas[0], betas[1], eps,
                                                      weight_decay, False, True, min_coeff, max_coeff)

    optimizer = FusedLamb([param],
                          lr=lr,
                          betas=betas,
                          eps=eps,
                          weight_decay=weight_decay,
                          bias_correction=True,
                          min_coeff=min_coeff,
                          max_coeff=max_coeff)
    optimizer.step(grads=[grad_fp16], output_params=[out_param])

    assert optimizer.get_lamb_coeffs()[0] == pytest.approx(ref_coeff, rel=1e-4, abs=1e-5)
    torch.testing.assert_close(param.detach().double(), ref_param, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_param, param.detach().half(), rtol=0.0, atol=0.0)
