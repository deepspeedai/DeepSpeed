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
    global get_accelerator, swiglu, swiglu_packed, swiglu_backward, swiglu_packed_backward
    global swiglu_weighted, swiglu_weighted_backward, swiglu_packed_weighted, swiglu_packed_weighted_backward
    # The shared Triton availability probe initializes its driver, so it must run after pytest forks its worker.
    from deepspeed.accelerator import get_accelerator
    from deepspeed.ops.triton_ops import is_triton_available
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu, swiglu_packed, swiglu_backward, swiglu_packed_backward
    from deepspeed.ops.triton_ops.swiglu_triton import (swiglu_weighted, swiglu_weighted_backward,
                                                        swiglu_packed_weighted, swiglu_packed_weighted_backward)

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(7, 13), (3, 768), (2, 3000), (0, 16)])
def test_packed_weighted_matches_unpacked_and_standalone_backward(dtype, shape):
    dev = get_accelerator().current_device_name()
    generator = torch.Generator(device=dev).manual_seed(8748)
    packed = torch.randn((shape[0], 2 * shape[1]), device=dev, dtype=dtype, generator=generator, requires_grad=True)
    weights = torch.rand(shape[0], device=dev, generator=generator, requires_grad=True)
    gate, up = packed.detach().chunk(2, dim=-1)
    gate, up = gate.contiguous().requires_grad_(True), up.contiguous().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)
    upstream = torch.randn((shape[1], shape[0]), device=dev, dtype=dtype, generator=generator).t()
    actual = swiglu_packed_weighted(packed, weights)
    expected = swiglu_weighted(gate, up, reference_weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, _ref_swiglu_weighted(gate, up, reference_weights).to(dtype), **_tol(dtype))
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(packed.grad, torch.cat((gate.grad, up.grad), dim=-1), rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0, atol=0)
    grad_packed, grad_weights = swiglu_packed_weighted_backward(upstream, packed.detach(), weights.detach())
    torch.testing.assert_close(grad_packed, packed.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_weights, weights.grad, rtol=0, atol=0)
    grad_gate, grad_up, grad_weights = swiglu_weighted_backward(upstream, gate.detach(), up.detach(),
                                                                reference_weights.detach())
    torch.testing.assert_close(grad_gate, gate.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_up, up.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_weights, reference_weights.grad, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_weighted_kernel_rejects_incomplete_higher_order_gradients(packed):
    dev = get_accelerator().current_device_name()
    projections = torch.ones((4, 26), device=dev, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.ones(4, device=dev, dtype=torch.float32, requires_grad=True)
    if packed:
        output = swiglu_packed_weighted(projections, weights)
    else:
        gate, up = projections.chunk(2, dim=-1)
        output = swiglu_weighted(gate, up, weights)
    with pytest.raises(RuntimeError, match="second derivative"):
        torch.autograd.grad(output.float().sum(), (projections, weights), create_graph=True)


@pytest.mark.parametrize("packed", [False, True])
def test_weighted_kernel_uses_its_input_device_and_preserves_the_caller(packed):
    accelerator = get_accelerator()
    if accelerator.device_count() < 2:
        pytest.skip("weighted cross-device kernel needs two CUDA devices")
    with torch.cuda.device(1):  #ignore-cuda
        projections = torch.ones((7, 26),
                                 device=accelerator.current_device_name(),
                                 dtype=torch.bfloat16,
                                 requires_grad=True)
        weights = torch.ones(7, device=accelerator.current_device_name(), dtype=torch.float32, requires_grad=True)
        reference = projections.detach().float().requires_grad_(True)
        reference_weights = weights.detach().clone().requires_grad_(True)
    with torch.cuda.device(0):  #ignore-cuda
        gate, up = projections.chunk(2, dim=-1)
        actual = swiglu_packed_weighted(projections, weights) if packed else swiglu_weighted(gate, up, weights)
        expected = _ref_swiglu_weighted(*reference.chunk(2, dim=-1), reference_weights)
        actual.float().sum().backward()
        expected.sum().backward()
        assert accelerator.current_device() == 0
    torch.testing.assert_close(actual, expected.to(actual.dtype), **_tol(actual.dtype))
    torch.testing.assert_close(projections.grad, reference.grad.to(projections.dtype), **_tol(projections.dtype))
    torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=1e-4, atol=1e-5)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_standalone_backward_matches_autograd_exactly(dtype):
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_standalone_backwards_select_the_input_device(dtype):
    accelerator = get_accelerator()
    if accelerator.device_count() < 2:
        pytest.skip("standalone cross-device backward needs two CUDA devices")
    with torch.cuda.device(1):  #ignore-cuda
        gate = torch.ones((7, 13), device=accelerator.current_device_name(), dtype=dtype, requires_grad=True)
        up = torch.full_like(gate, 0.5, requires_grad=True)
        packed = torch.cat((gate, up), dim=-1).detach().requires_grad_(True)
        upstream = torch.ones_like(gate)
        swiglu(gate, up).backward(upstream)
        swiglu_packed(packed).backward(upstream)
    with torch.cuda.device(0):  #ignore-cuda
        grad_gate, grad_up = swiglu_backward(upstream, gate.detach(), up.detach())
        grad_packed = swiglu_packed_backward(upstream, packed.detach())
        assert accelerator.current_device() == 0
    torch.testing.assert_close(grad_gate, gate.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_up, up.grad, rtol=0, atol=0)
    torch.testing.assert_close(grad_packed, packed.grad, rtol=0, atol=0)


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
