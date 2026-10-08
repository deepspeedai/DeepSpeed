# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the fused Triton SwiGLU kernels (``deepspeed.ops.triton_ops.swiglu_triton``).

Correctness of ``swiglu(gate, up) == silu(gate) * up`` is checked for the forward
output and both input gradients against an eager PyTorch reference, across dtypes
and even / uneven / empty shapes.

``swiglu_clamped`` covers the two clamped forms, both with ``gate`` clamped from above at
``limit`` and ``up`` clamped to ``[-limit, limit]``:
``oai=True``  -> ``(up + 1) * g * sigmoid(alpha * g)``   (GPT-OSS, MiniMax-M3)
``oai=False`` -> ``silu(g) * up``                        (DeepSeek-V4)
Its forward output and both input gradients are checked against the expression evaluated
in float32 on the same inputs and rounded once to the test dtype, with inputs scaled so
that a good share of them lie past the clamp limits. Comparing against the expression
evaluated IN the low-precision dtype would not work: it rounds after each of its seven
operations and lands two to three ulps away from the kernel, which computes in float32
and rounds once. A second assertion checks that the kernel is never farther from the
float32 result than that low-precision evaluation is.
"""

import pytest
import torch
import torch.nn.functional as F

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import is_triton_available
from deepspeed.ops.triton_ops.swiglu_triton import swiglu, swiglu_clamped

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


def test_dtype_mismatch_raises():
    dev = get_accelerator().current_device_name()
    gate = torch.randn(8, 16, device=dev, dtype=torch.float16)
    up = torch.randn(8, 16, device=dev, dtype=torch.float32)
    with pytest.raises(ValueError):
        swiglu(gate, up)


# ---------------------------------------------------------------------------
# Clamped forms
# ---------------------------------------------------------------------------


def _clamped_tol(dtype):
    # about one ulp of the dtype at the output magnitude, plus a small absolute floor
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    if dtype == torch.float16:
        return dict(atol=1e-4, rtol=1.5e-3)
    return dict(atol=1e-3, rtol=1.2e-2)  # bfloat16


def _clamped_eager(gate, up, alpha, limit, oai):
    """The two forms as the model code writes them (GptOssExperts / DeepseekV4Experts._apply_gate)."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    if oai:
        return (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
    return F.silu(gate) * up


def _clamped_reference(gate, up, alpha, limit, oai, grad_out=None):
    """Eager expression in float32 on the same inputs, rounded once to the input dtype; optionally its
    gradients (with grad_out also upcast), rounded the same way."""
    g = gate.detach().float().requires_grad_(grad_out is not None)
    u = up.detach().float().requires_grad_(grad_out is not None)
    out = _clamped_eager(g, u, alpha, limit, oai)
    if grad_out is None:
        return out.to(gate.dtype)
    out.backward(grad_out.float())
    return out.detach().to(gate.dtype), g.grad.to(gate.dtype), u.grad.to(gate.dtype)


def _max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def _assert_no_less_accurate(kernel_out, eager_out, ref, dtype):
    # The kernel must not be farther from the float32 result than the eager evaluation in the test
    # dtype is, within the same tolerance as assert_close. In float32 that eager evaluation IS the
    # reference, so without the slack the check would demand more than the tolerance allows.
    tol = _clamped_tol(dtype)
    slack = tol["atol"] + tol["rtol"] * ref.float().abs().max().item()
    assert _max_err(kernel_out, ref) <= _max_err(eager_out, ref) + slack


def _clamped_inputs(shape, dtype, scale=4.0, requires_grad=False):
    # scale 4 with limit 7 puts about 8 % of gate values and 8 % of up values past a clamp
    dev = get_accelerator().current_device_name()
    gate = (scale * torch.randn(shape, device=dev)).to(dtype).requires_grad_(requires_grad)
    up = (scale * torch.randn(shape, device=dev)).to(dtype).requires_grad_(requires_grad)
    return gate, up


@pytest.mark.parametrize("oai", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 16), (1, 1), (128, 512), (7, 13), (4, 2048), (2, 3000)])
def test_clamped_forward_matches_reference(oai, dtype, shape):
    gate, up = _clamped_inputs(shape, dtype)
    out = swiglu_clamped(gate, up, alpha=1.702, limit=7.0, oai=oai)
    ref = _clamped_reference(gate, up, 1.702, 7.0, oai)
    assert out.shape == ref.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, **_clamped_tol(dtype))
    _assert_no_less_accurate(out, _clamped_eager(gate, up, 1.702, 7.0, oai), ref, dtype)


@pytest.mark.parametrize("oai", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 16), (128, 512), (7, 13), (4, 2048), (2, 3000)])
def test_clamped_backward_matches_reference(oai, dtype, shape):
    gate, up = _clamped_inputs(shape, dtype, requires_grad=True)
    gate_eager = gate.detach().clone().requires_grad_(True)
    up_eager = up.detach().clone().requires_grad_(True)
    grad_out = torch.randn(shape, device=gate.device, dtype=dtype)

    swiglu_clamped(gate, up, alpha=1.702, limit=7.0, oai=oai).backward(grad_out)
    _clamped_eager(gate_eager, up_eager, 1.702, 7.0, oai).backward(grad_out)
    _, ref_gate, ref_up = _clamped_reference(gate, up, 1.702, 7.0, oai, grad_out=grad_out)

    torch.testing.assert_close(gate.grad, ref_gate, **_clamped_tol(dtype))
    torch.testing.assert_close(up.grad, ref_up, **_clamped_tol(dtype))
    _assert_no_less_accurate(gate.grad, gate_eager.grad, ref_gate, dtype)
    _assert_no_less_accurate(up.grad, up_eager.grad, ref_up, dtype)


@pytest.mark.parametrize("oai", [True, False])
def test_clamped_alpha_and_limit_are_honoured(oai):
    # limit 2 with scale 4 clamps about 60 % of the values; alpha 1 changes the oai form
    gate, up = _clamped_inputs((64, 256), torch.float32, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    out = swiglu_clamped(gate, up, alpha=1.0, limit=2.0, oai=oai)
    ref = _clamped_eager(gate_ref, up_ref, 1.0, 2.0, oai)
    torch.testing.assert_close(out, ref, **_clamped_tol(torch.float32))
    out.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(gate.grad, gate_ref.grad, **_clamped_tol(torch.float32))
    torch.testing.assert_close(up.grad, up_ref.grad, **_clamped_tol(torch.float32))
    # the gradient must be exactly zero where the input was clamped
    assert torch.all(up.grad[up.detach().abs() > 2.0] == 0)
    assert torch.all(gate.grad[gate.detach() > 2.0] == 0)


def test_clamped_gradient_at_the_limit_is_inclusive():
    # torch.clamp passes the gradient where input == limit; the kernel must agree
    dev = get_accelerator().current_device_name()
    gate = torch.full((8, 8), 7.0, device=dev, requires_grad=True)
    up = torch.full((8, 8), -7.0, device=dev, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    swiglu_clamped(gate, up, 1.702, 7.0, True).sum().backward()
    _clamped_eager(gate_ref, up_ref, 1.702, 7.0, True).sum().backward()
    torch.testing.assert_close(gate.grad, gate_ref.grad)
    torch.testing.assert_close(up.grad, up_ref.grad)
    assert torch.all(gate.grad != 0) and torch.all(up.grad != 0)


def test_clamped_empty_input():
    dev = get_accelerator().current_device_name()
    gate = torch.empty(0, 16, device=dev, dtype=torch.float32, requires_grad=True)
    up = torch.empty(0, 16, device=dev, dtype=torch.float32, requires_grad=True)
    out = swiglu_clamped(gate, up, 1.702, 7.0, True)
    assert out.shape == (0, 16)
    out.sum().backward()
    assert gate.grad.shape == gate.shape


def test_clamped_non_contiguous_input():
    dev = get_accelerator().current_device_name()
    gate = (4 * torch.randn(64, 32, device=dev, dtype=torch.float32)).t()
    up = (4 * torch.randn(64, 32, device=dev, dtype=torch.float32)).t()
    torch.testing.assert_close(swiglu_clamped(gate, up, 1.702, 7.0, True), _clamped_eager(gate, up, 1.702, 7.0, True),
                               **_clamped_tol(torch.float32))


def test_clamped_shape_and_dtype_mismatch_rejected():
    dev = get_accelerator().current_device_name()
    with pytest.raises(ValueError, match="same shape"):
        swiglu_clamped(torch.randn(4, 8, device=dev), torch.randn(4, 9, device=dev))
    with pytest.raises(ValueError, match="same dtype"):
        swiglu_clamped(torch.randn(4, 8, device=dev), torch.randn(4, 8, device=dev, dtype=torch.float16))
