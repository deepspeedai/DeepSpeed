# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the fused Triton SwiGLU (``deepspeed.ops.triton_ops.swiglu_triton``).

Correctness of ``swiglu(gate, up) == silu(gate) * up`` is checked for the forward
output and both input gradients against an eager PyTorch reference, across dtypes
and even / uneven / empty shapes.
"""

import pytest
import torch
import torch.nn.functional as F

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import is_triton_available
from deepspeed.ops.triton_ops.swiglu_triton import swiglu, swiglu_weighted

if not is_triton_available():
    pytest.skip("Triton is not available", allow_module_level=True)

if not (get_accelerator().is_available() and get_accelerator().device_name() == "cuda"):
    pytest.skip("Fused Triton SwiGLU requires a CUDA device", allow_module_level=True)


def _tol(dtype):
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    if dtype == torch.float16:
        return dict(atol=2e-3, rtol=2e-3)
    return dict(atol=1e-2, rtol=1e-2)  # bfloat16


def _ref_swiglu(gate, up):
    return F.silu(gate) * up


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 16), (1, 1), (128, 512), (7, 13), (4, 2048), (2, 3000)])
def test_forward_matches_reference(dtype, shape):
    dev = get_accelerator().current_device_name()
    gate = torch.randn(shape, device=dev, dtype=dtype)
    up = torch.randn(shape, device=dev, dtype=dtype)

    out = swiglu(gate, up)
    ref = _ref_swiglu(gate, up)

    assert out.shape == ref.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 16), (128, 512), (7, 13), (4, 2048), (2, 3000)])
def test_backward_matches_reference(dtype, shape):
    dev = get_accelerator().current_device_name()
    gate = torch.randn(shape, device=dev, dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device=dev, dtype=dtype, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)

    grad_out = torch.randn(shape, device=dev, dtype=dtype)

    swiglu(gate, up).backward(grad_out)
    _ref_swiglu(gate_ref, up_ref).backward(grad_out)

    torch.testing.assert_close(gate.grad, gate_ref.grad, **_tol(dtype))
    torch.testing.assert_close(up.grad, up_ref.grad, **_tol(dtype))


def _ref_swiglu_weighted(gate, up, weights):
    return F.silu(gate.float()) * up.float() * weights[:, None]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 16), (1, 1), (128, 768), (7, 13), (4, 2048), (2, 3000)])
def test_weighted_forward_is_the_fp32_product_rounded_once(dtype, shape):
    dev = get_accelerator().current_device_name()
    gate = torch.randn(shape, device=dev, dtype=dtype)
    up = torch.randn(shape, device=dev, dtype=dtype)
    weights = torch.rand(shape[0], device=dev)

    out = swiglu_weighted(gate, up, weights)

    assert out.shape == gate.shape and out.dtype == dtype
    torch.testing.assert_close(out, _ref_swiglu_weighted(gate, up, weights).to(dtype), **_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 16), (128, 768), (7, 13), (4, 2048), (2, 3000)])
def test_weighted_backward_matches_reference(dtype, shape):
    dev = get_accelerator().current_device_name()
    gate = torch.randn(shape, device=dev, dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device=dev, dtype=dtype, requires_grad=True)
    weights = torch.rand(shape[0], device=dev, requires_grad=True)
    gate_ref = gate.detach().float().requires_grad_(True)
    up_ref = up.detach().float().requires_grad_(True)
    weights_ref = weights.detach().clone().requires_grad_(True)
    grad_out = torch.randn(shape, device=dev, dtype=dtype)

    swiglu_weighted(gate, up, weights).backward(grad_out)
    _ref_swiglu_weighted(gate_ref, up_ref, weights_ref).backward(grad_out.float())

    torch.testing.assert_close(gate.grad, gate_ref.grad.to(dtype), **_tol(dtype))
    torch.testing.assert_close(up.grad, up_ref.grad.to(dtype), **_tol(dtype))
    # The weight gradient is an FP32 sum over the row; only summation order separates it from the reference.
    assert weights.grad.dtype == torch.float32
    scale = weights_ref.grad.abs().max().item()
    torch.testing.assert_close(weights.grad, weights_ref.grad, rtol=1e-4, atol=1e-5 * max(scale, 1.0))


def test_weighted_empty_input():
    dev = get_accelerator().current_device_name()
    gate = torch.empty(0, 16, device=dev, dtype=torch.bfloat16, requires_grad=True)
    up = torch.empty(0, 16, device=dev, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.empty(0, device=dev, requires_grad=True)
    out = swiglu_weighted(gate, up, weights)
    out.sum().backward()
    assert out.shape == (0, 16) and weights.grad.shape == (0, )


def test_weighted_rejects_weights_that_are_not_one_fp32_value_per_row():
    dev = get_accelerator().current_device_name()
    gate = torch.randn(4, 8, device=dev, dtype=torch.bfloat16)
    up = torch.randn(4, 8, device=dev, dtype=torch.bfloat16)
    for weights in (torch.rand(4, device=dev).half(), torch.rand(4, 1, device=dev), torch.rand(3, device=dev)):
        with pytest.raises(ValueError, match="FP32 weights"):
            swiglu_weighted(gate, up, weights)
    with pytest.raises(ValueError, match="one shape"):
        swiglu_weighted(gate, up[:, :4], torch.rand(4, device=dev))


def test_empty_input():
    dev = get_accelerator().current_device_name()
    gate = torch.empty(0, 16, device=dev, dtype=torch.float32, requires_grad=True)
    up = torch.empty(0, 16, device=dev, dtype=torch.float32, requires_grad=True)

    out = swiglu(gate, up)
    assert out.shape == (0, 16)
    out.sum().backward()
    assert gate.grad.shape == gate.shape
    assert up.grad.shape == up.shape


def test_non_contiguous_input():
    dev = get_accelerator().current_device_name()
    # Transposed views are non-contiguous; the kernel must still match the reference.
    gate = torch.randn(64, 32, device=dev, dtype=torch.float32).t()
    up = torch.randn(64, 32, device=dev, dtype=torch.float32).t()

    torch.testing.assert_close(swiglu(gate, up), _ref_swiglu(gate, up), **_tol(torch.float32))


def test_shape_mismatch_raises():
    dev = get_accelerator().current_device_name()
    gate = torch.randn(8, 16, device=dev)
    up = torch.randn(8, 32, device=dev)
    with pytest.raises(ValueError):
        swiglu(gate, up)


def test_dtype_mismatch_raises():
    dev = get_accelerator().current_device_name()
    gate = torch.randn(8, 16, device=dev, dtype=torch.float16)
    up = torch.randn(8, 16, device=dev, dtype=torch.float32)
    with pytest.raises(ValueError):
        swiglu(gate, up)
