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


@pytest.fixture(autouse=True)
def _load_swiglu_in_the_worker():
    global get_accelerator, swiglu, swiglu_packed
    # The shared Triton availability probe initializes its driver, so it must run after pytest forks its worker.
    from deepspeed.accelerator import get_accelerator
    from deepspeed.ops.triton_ops import is_triton_available
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu, swiglu_packed

    if not is_triton_available():
        pytest.skip("Triton is not available")
    if not (get_accelerator().is_available() and get_accelerator().device_name() == "cuda"):
        pytest.skip("Fused Triton SwiGLU requires a CUDA device")


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
def test_packed_matches_the_separate_kernel_exactly(dtype, shape):
    # The packed kernel reads gate and up in place from one [..., 2 * I] tensor with the same FP32 math, so
    # it must reproduce the separate kernel's values exactly, forward and backward.
    generator = torch.Generator(device=get_accelerator().current_device_name()).manual_seed(8733)
    dev = get_accelerator().current_device_name()
    gate = torch.randn(shape, device=dev, dtype=dtype, generator=generator, requires_grad=True)
    up = torch.randn(shape, device=dev, dtype=dtype, generator=generator, requires_grad=True)
    gate_up = torch.cat([gate, up], dim=-1).detach().requires_grad_(True)
    grad_out = torch.randn(shape, device=dev, dtype=dtype, generator=generator)

    out = swiglu_packed(gate_up)
    expected = swiglu(gate, up)
    assert out.shape == expected.shape and out.dtype == dtype
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(out, _ref_swiglu(gate, up), **_tol(dtype))

    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gate_up.grad, torch.cat([gate.grad, up.grad], dim=-1), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_packed_non_contiguous_input_and_gradient(dtype):
    dev = get_accelerator().current_device_name()
    generator = torch.Generator(device=dev).manual_seed(8733)
    gate_up = torch.randn(32, 64, device=dev, dtype=dtype, generator=generator).t().requires_grad_(True)
    reference = gate_up.detach().clone().requires_grad_(True)
    gate, up = reference.chunk(2, dim=-1)
    upstream = torch.randn(16, 64, device=dev, dtype=dtype, generator=generator).t()
    assert not gate_up.is_contiguous() and not upstream.is_contiguous()

    actual = swiglu_packed(gate_up)
    expected = swiglu(gate, up)
    torch.testing.assert_close(actual, _ref_swiglu(gate, up), **_tol(dtype))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(gate_up.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_create_graph_is_rejected(dtype):
    dev = get_accelerator().current_device_name()
    gate_up = torch.ones((4, 32), device=dev, dtype=dtype, requires_grad=True)
    loss = swiglu_packed(gate_up).float().sum()
    with pytest.raises(RuntimeError, match="second derivative"):
        torch.autograd.grad(loss, gate_up, create_graph=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_uses_the_input_device_and_stream(dtype):
    accelerator = get_accelerator()
    if accelerator.device_count() < 2:
        pytest.skip("cross-device launch regression needs two CUDA devices")
    with torch.cuda.device(0):  #ignore-cuda
        caller_stream = accelerator.current_stream()
        input_device = accelerator.device_name(1)
        stream = accelerator.Stream(device=1)
        with accelerator.stream(stream):
            gate_up = torch.ones((7, 26), device=input_device, dtype=dtype, requires_grad=True)
            reference = gate_up.detach().clone().requires_grad_(True)
        with accelerator.stream(stream), torch.cuda.device(0):  #ignore-cuda
            actual = swiglu_packed(gate_up)
            assert accelerator.current_device() == 0
            gate, up = reference.chunk(2, dim=-1)
            expected = _ref_swiglu(gate, up)
            actual.float().sum().backward()
            expected.float().sum().backward()
            assert accelerator.current_device() == 0
        stream.synchronize()
        assert accelerator.current_stream() == caller_stream
        torch.testing.assert_close(actual, expected, **_tol(dtype))
        torch.testing.assert_close(gate_up.grad, reference.grad, **_tol(dtype))


def test_packed_odd_width_raises():
    with pytest.raises(ValueError, match="even last dimension"):
        swiglu_packed(torch.randn(4, 7, device=get_accelerator().current_device_name()))


def test_dtype_mismatch_raises():
    dev = get_accelerator().current_device_name()
    gate = torch.randn(8, 16, device=dev, dtype=torch.float16)
    up = torch.randn(8, 16, device=dev, dtype=torch.float32)
    with pytest.raises(ValueError):
        swiglu(gate, up)
