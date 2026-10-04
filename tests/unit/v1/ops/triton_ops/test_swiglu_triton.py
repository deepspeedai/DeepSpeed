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
from deepspeed.ops.triton_ops.swiglu_triton import swiglu, swiglu_backward, swiglu_packed, swiglu_packed_backward

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 16), (1, 1), (7, 13), (4, 768), (3, 5, 384), (0, 16)])
def test_packed_matches_the_separate_kernel_bitwise(dtype, shape):
    # The packed kernel reads gate and up in place from one [..., 2 * I] tensor with the same FP32 math, so
    # it must reproduce the separate kernel exactly, forward and backward.
    dev = get_accelerator().current_device_name()
    gate = torch.randn(shape, device=dev, dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device=dev, dtype=dtype, requires_grad=True)
    gate_up = torch.cat([gate, up], dim=-1).detach().requires_grad_(True)
    grad_out = torch.randn(shape, device=dev, dtype=dtype)

    out = swiglu_packed(gate_up)
    expected = swiglu(gate, up)
    assert out.shape == expected.shape and out.dtype == dtype
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(out, _ref_swiglu(gate, up), **_tol(dtype))

    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gate_up.grad, torch.cat([gate.grad, up.grad], dim=-1), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_standalone_backward_matches_autograd_bitwise(dtype):
    # Callers that run the forward inside their own autograd function differentiate it with these.
    dev = get_accelerator().current_device_name()
    gate = torch.randn(7, 384, device=dev, dtype=dtype, requires_grad=True)
    up = torch.randn(7, 384, device=dev, dtype=dtype, requires_grad=True)
    gate_up = torch.cat([gate, up], dim=-1).detach().requires_grad_(True)
    grad_out = torch.randn(7, 384, device=dev, dtype=dtype)
    swiglu(gate, up).backward(grad_out)
    swiglu_packed(gate_up).backward(grad_out)

    grad_gate, grad_up = swiglu_backward(grad_out, gate.detach(), up.detach())
    torch.testing.assert_close(grad_gate, gate.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_up, up.grad, rtol=0, atol=0)
    torch.testing.assert_close(swiglu_packed_backward(grad_out, gate_up.detach()), gate_up.grad, rtol=0, atol=0)


def test_packed_non_contiguous_input():
    dev = get_accelerator().current_device_name()
    gate_up = torch.randn(32, 64, device=dev, dtype=torch.float32).t()
    gate, up = gate_up.chunk(2, dim=-1)

    torch.testing.assert_close(swiglu_packed(gate_up), _ref_swiglu(gate, up), **_tol(torch.float32))


def test_packed_odd_width_raises():
    with pytest.raises(ValueError, match="even last dimension"):
        swiglu_packed(torch.randn(4, 7, device=get_accelerator().current_device_name()))


def test_dtype_mismatch_raises():
    dev = get_accelerator().current_device_name()
    gate = torch.randn(8, 16, device=dev, dtype=torch.float16)
    up = torch.randn(8, 16, device=dev, dtype=torch.float32)
    with pytest.raises(ValueError):
        swiglu(gate, up)
