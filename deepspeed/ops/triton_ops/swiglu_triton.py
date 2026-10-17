# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Fused Triton SwiGLU activation for grouped-expert MLPs.

The SwiGLU gate combines two projections of the same input::

    h = silu(gate) * up

where ``silu(x) = x * sigmoid(x)``. This module fuses both into a single Triton
kernel for the forward pass and a single kernel for the backward pass, which
halves the elementwise kernel launches and the intermediate-tensor traffic
on the expert MLP hot path.

``gate`` and ``up`` are the raw outputs of the gate/up grouped GEMMs and must
share the same shape and dtype. All math is accumulated in float32 for numerical
stability and cast back to the input dtype on store.

When Triton is unavailable the public :func:`swiglu` falls back to the eager
PyTorch expression so callers on non-Triton builds keep working unchanged.

:func:`swiglu_weighted` also scales each row by one FP32 weight inside the same
kernel, ``h = silu(gate) * up * weight``, with one rounding to the input dtype.
An MoE layer can apply its routing weights there rather than to the expert
output, since the down projection that follows is linear.

:func:`swiglu_packed` takes ``gate`` and ``up`` packed side by side in one
``[..., 2 * I]`` tensor, the output of a single gate+up GEMM, and reads both
halves in place with the same math, so the packed GEMM output needs no copy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

if _TRITON_AVAILABLE:

    _BLOCK_SIZE = 2048
    _NUM_WARPS = 8

    @triton.jit
    def _swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # int64: tl.program_id is int32, so pid * BLOCK_SIZE wraps negative once
        # n_elements > 2**31, and the mask below does not reject a negative offset.
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

        silu = gate * tl.sigmoid(gate)
        out = silu * up

        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(grad_out_ptr, gate_ptr, up_ptr, grad_gate_ptr, grad_up_ptr, n_elements,
                           BLOCK_SIZE: tl.constexpr):
        # int64: tl.program_id is int32, so pid * BLOCK_SIZE wraps negative once
        # n_elements > 2**31, and the mask below does not reject a negative offset.
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        grad_out = tl.load(grad_out_ptr + offsets, mask=mask).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

        sig = tl.sigmoid(gate)
        silu = gate * sig
        # d/dgate silu(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate))).
        dsilu = sig * (1.0 + gate * (1.0 - sig))

        grad_gate = grad_out * up * dsilu
        grad_up = grad_out * silu

        tl.store(grad_gate_ptr + offsets, grad_gate.to(grad_gate_ptr.dtype.element_ty), mask=mask)
        tl.store(grad_up_ptr + offsets, grad_up.to(grad_up_ptr.dtype.element_ty), mask=mask)

    class _SwiGLUFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            gate = gate.contiguous()
            up = up.contiguous()
            out = torch.empty_like(gate)

            n_elements = gate.numel()
            if n_elements > 0:
                grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
                _swiglu_fwd_kernel[grid](gate, up, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS)

            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            gate, up = ctx.saved_tensors
            return _swiglu_backward_triton(grad_out, gate, up)

    def _swiglu_backward_triton(grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
        grad_out = grad_out.contiguous()

        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)

        n_elements = gate.numel()
        if n_elements > 0:
            grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
            with torch.cuda.device(gate.device):  #ignore-cuda
                _swiglu_bwd_kernel[grid](grad_out,
                                         gate,
                                         up,
                                         grad_gate,
                                         grad_up,
                                         n_elements,
                                         BLOCK_SIZE=_BLOCK_SIZE,
                                         num_warps=_NUM_WARPS)

        return grad_gate, grad_up

    @triton.jit
    def _swiglu_packed_fwd_kernel(gate_up_ptr, out_ptr, n_elements, INTER: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # Output element (row, col) reads gate at (row, col) and up at (row, INTER + col) of the packed row.
        gate_offsets = (offsets // INTER) * (2 * INTER) + offsets % INTER

        gate = tl.load(gate_up_ptr + gate_offsets, mask=mask).to(tl.float32)
        up = tl.load(gate_up_ptr + gate_offsets + INTER, mask=mask).to(tl.float32)

        out = gate * tl.sigmoid(gate) * up

        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_packed_bwd_kernel(grad_out_ptr, gate_up_ptr, grad_gate_up_ptr, n_elements, INTER: tl.constexpr,
                                  BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        gate_offsets = (offsets // INTER) * (2 * INTER) + offsets % INTER

        grad_out = tl.load(grad_out_ptr + offsets, mask=mask).to(tl.float32)
        gate = tl.load(gate_up_ptr + gate_offsets, mask=mask).to(tl.float32)
        up = tl.load(gate_up_ptr + gate_offsets + INTER, mask=mask).to(tl.float32)

        sig = tl.sigmoid(gate)
        silu = gate * sig
        dsilu = sig * (1.0 + gate * (1.0 - sig))

        grad_gate = grad_out * up * dsilu
        grad_up = grad_out * silu

        tl.store(grad_gate_up_ptr + gate_offsets, grad_gate.to(grad_gate_up_ptr.dtype.element_ty), mask=mask)
        tl.store(grad_gate_up_ptr + gate_offsets + INTER, grad_up.to(grad_gate_up_ptr.dtype.element_ty), mask=mask)

    class _SwiGLUPackedFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
            gate_up = gate_up.contiguous()
            inter = gate_up.shape[-1] // 2
            out = gate_up.new_empty(gate_up.shape[:-1] + (inter, ))

            n_elements = out.numel()
            if n_elements > 0:
                grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
                # Triton uses the current device and stream, which may differ from the input tensor's device.
                with torch.cuda.device(gate_up.device):  #ignore-cuda
                    _swiglu_packed_fwd_kernel[grid](gate_up,
                                                    out,
                                                    n_elements,
                                                    INTER=inter,
                                                    BLOCK_SIZE=_BLOCK_SIZE,
                                                    num_warps=_NUM_WARPS)

            ctx.save_for_backward(gate_up)
            return out

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            # The Triton backward has no autograd graph, so create_graph would silently lose the second derivative.
            if torch.is_grad_enabled():
                raise RuntimeError("The packed SwiGLU kernel has no second derivative; "
                                   "backward with create_graph=True is not supported")
            gate_up, = ctx.saved_tensors
            return _swiglu_packed_backward_triton(grad_out, gate_up)

    def _swiglu_packed_backward_triton(grad_out: torch.Tensor, gate_up: torch.Tensor) -> torch.Tensor:
        grad_out = grad_out.contiguous()
        grad_gate_up = torch.empty_like(gate_up)

        n_elements = grad_out.numel()
        if n_elements > 0:
            grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
            with torch.cuda.device(gate_up.device):  #ignore-cuda
                _swiglu_packed_bwd_kernel[grid](grad_out,
                                                gate_up,
                                                grad_gate_up,
                                                n_elements,
                                                INTER=gate_up.shape[-1] // 2,
                                                BLOCK_SIZE=_BLOCK_SIZE,
                                                num_warps=_NUM_WARPS)

        return grad_gate_up

    @triton.jit
    def _swiglu_weighted_fwd_kernel(gate_ptr, up_ptr, weight_ptr, out_ptr, n_elements, n_cols,
                                    ROW_STRIDE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        input_offsets = (offsets // n_cols) * ROW_STRIDE + offsets % n_cols
        gate = tl.load(gate_ptr + input_offsets, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + input_offsets, mask=mask).to(tl.float32)
        weight = tl.load(weight_ptr + offsets // n_cols, mask=mask)

        out = gate * tl.sigmoid(gate) * up * weight
        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_weighted_bwd_kernel(grad_out_ptr, gate_ptr, up_ptr, weight_ptr, grad_gate_ptr, grad_up_ptr,
                                    grad_weight_ptr, n_cols, ROW_STRIDE: tl.constexpr, GRAD_ROW_STRIDE: tl.constexpr,
                                    BLOCK_SIZE: tl.constexpr):
        # One program per row: the row's weight gradient is a sum over its columns.
        row = tl.program_id(axis=0).to(tl.int64)
        weight = tl.load(weight_ptr + row)
        total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            offsets = row * n_cols + cols
            grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            input_offsets = row * ROW_STRIDE + cols
            gate = tl.load(gate_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(up_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)

            sig = tl.sigmoid(gate)
            silu = gate * sig
            dsilu = sig * (1.0 + gate * (1.0 - sig))

            grad_offsets = row * GRAD_ROW_STRIDE + cols
            tl.store(grad_gate_ptr + grad_offsets, (grad_out * weight * up * dsilu).to(grad_gate_ptr.dtype.element_ty),
                     mask=mask)
            tl.store(grad_up_ptr + grad_offsets, (grad_out * weight * silu).to(grad_up_ptr.dtype.element_ty),
                     mask=mask)
            total += grad_out * silu * up
        tl.store(grad_weight_ptr + row, tl.sum(total, axis=0))

    def _swiglu_weighted_forward_triton(gate, up, weights):
        out = torch.empty_like(gate, memory_format=torch.contiguous_format)
        n_elements = gate.numel()
        if n_elements > 0:
            grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
            with torch.cuda.device(gate.device):  #ignore-cuda
                _swiglu_weighted_fwd_kernel[grid](gate,
                                                  up,
                                                  weights,
                                                  out,
                                                  n_elements,
                                                  gate.shape[1],
                                                  ROW_STRIDE=gate.stride(0),
                                                  BLOCK_SIZE=_BLOCK_SIZE,
                                                  num_warps=_NUM_WARPS)

        return out

    def _swiglu_weighted_backward_triton(grad_out, gate, up, weights, grad_gate, grad_up):
        grad_out = grad_out.contiguous()
        grad_weights = torch.zeros_like(weights)
        n_rows, n_cols = gate.shape
        if n_rows > 0 and n_cols > 0:
            block = min(triton.next_power_of_2(n_cols), _BLOCK_SIZE)
            with torch.cuda.device(gate.device):  #ignore-cuda
                _swiglu_weighted_bwd_kernel[(n_rows, )](grad_out,
                                                        gate,
                                                        up,
                                                        weights,
                                                        grad_gate,
                                                        grad_up,
                                                        grad_weights,
                                                        n_cols,
                                                        ROW_STRIDE=gate.stride(0),
                                                        GRAD_ROW_STRIDE=grad_gate.stride(0),
                                                        BLOCK_SIZE=block,
                                                        num_warps=4)

        return grad_weights

    class _SwiGLUWeightedFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate, up, weights):
            gate, up, weights = gate.contiguous(), up.contiguous(), weights.contiguous()
            ctx.save_for_backward(gate, up, weights)
            return _swiglu_weighted_forward_triton(gate, up, weights)

        @staticmethod
        def backward(ctx, grad_out):
            if torch.is_grad_enabled():
                raise RuntimeError("Weighted SwiGLU has no second derivative; create_graph=True is not supported")
            gate, up, weights = ctx.saved_tensors
            return swiglu_weighted_backward(grad_out, gate, up, weights)

    class _SwiGLUPackedWeightedFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate_up, weights):
            gate_up, weights = gate_up.contiguous(), weights.contiguous()
            ctx.save_for_backward(gate_up, weights)
            gate, up = gate_up.chunk(2, dim=-1)
            return _swiglu_weighted_forward_triton(gate, up, weights)

        @staticmethod
        def backward(ctx, grad_out):
            if torch.is_grad_enabled():
                raise RuntimeError("Weighted SwiGLU has no second derivative; create_graph=True is not supported")
            gate_up, weights = ctx.saved_tensors
            return swiglu_packed_weighted_backward(grad_out, gate_up, weights)


def _swiglu_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference used as the non-Triton fallback."""
    return F.silu(gate) * up


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU activation: ``silu(gate) * up``.

    Args:
        gate: Gate projection output, any shape, float16/bfloat16/float32.
        up: Up projection output, same shape and dtype as ``gate``.

    Returns:
        Tensor of the same shape and dtype as ``gate`` holding ``silu(gate) * up``.

    Falls back to the eager PyTorch expression when Triton is unavailable.
    """
    if gate.shape != up.shape:
        raise ValueError(f"swiglu expects gate and up to have the same shape, got {tuple(gate.shape)} "
                         f"and {tuple(up.shape)}")
    if gate.dtype != up.dtype:
        raise ValueError(f"swiglu expects gate and up to have the same dtype, got {gate.dtype} and {up.dtype}")

    if not _TRITON_AVAILABLE:
        return _swiglu_eager(gate, up)

    return _SwiGLUFn.apply(gate, up)


def _swiglu_weighted_eager(gate: torch.Tensor, up: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference used as the fallback: the same FP32 expression, rounded once."""
    return (F.silu(gate.float()) * up.float() * weights.reshape(-1, 1)).to(gate.dtype)


def swiglu_weighted(gate: torch.Tensor, up: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """SwiGLU scaled by one weight per row: ``silu(gate) * up * weights[:, None]``.

    Args:
        gate: Gate projection output, shape ``[N, I]``, float16/bfloat16/float32.
        up: Up projection output, same shape and dtype as ``gate``.
        weights: One FP32 weight per row, shape ``[N]``.

    Returns:
        Tensor of the same shape and dtype as ``gate``, computed in float32 and rounded once. Differentiable in
        all three inputs; the weight gradient is FP32.

    Runs the fused kernel on CUDA when Triton is available and the eager expression otherwise.
    """
    _validate_weighted_inputs(gate, up, weights)
    if not (_TRITON_AVAILABLE and get_accelerator().on_accelerator(gate)):
        return _swiglu_weighted_eager(gate, up, weights)
    return _SwiGLUWeightedFn.apply(gate, up, weights)


def _validate_weighted_inputs(gate, up, weights):
    if gate.dim() != 2 or gate.shape != up.shape:
        raise ValueError(f"swiglu_weighted expects gate and up of one shape [N, I], got {tuple(gate.shape)} "
                         f"and {tuple(up.shape)}")
    if gate.dtype != up.dtype:
        raise ValueError(f"swiglu_weighted expects gate and up to have the same dtype, got {gate.dtype} and "
                         f"{up.dtype}")
    if gate.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"swiglu_weighted supports FP32/FP16/BF16 projections, got {gate.dtype}")
    if weights.dtype != torch.float32 or weights.shape != (gate.shape[0], ):
        raise ValueError(f"swiglu_weighted expects FP32 weights of shape ({gate.shape[0]},), got {weights.dtype} "
                         f"{tuple(weights.shape)}")
    if gate.device != up.device or gate.device != weights.device:
        raise ValueError("swiglu_weighted needs gate, up and FP32 weights on the same device")


def _validate_weighted_gradient(grad_out, gate):
    if grad_out.shape != gate.shape or grad_out.device != gate.device:
        raise ValueError("weighted SwiGLU gradient must match the output shape and device")


def swiglu_weighted_backward(grad_out, gate, up, weights):
    """First-order gradient values of :func:`swiglu_weighted`, for custom autograd callers."""
    _validate_weighted_inputs(gate, up, weights)
    _validate_weighted_gradient(grad_out, gate)
    if not (_TRITON_AVAILABLE and get_accelerator().on_accelerator(gate)):
        with torch.enable_grad():
            gate, up, weights = (tensor.detach().requires_grad_(True) for tensor in (gate, up, weights))
            return torch.autograd.grad(_swiglu_weighted_eager(gate, up, weights), (gate, up, weights), grad_out)
    gate, up, weights = gate.contiguous(), up.contiguous(), weights.contiguous()
    grad_gate, grad_up = torch.empty_like(gate), torch.empty_like(up)
    grad_weights = _swiglu_weighted_backward_triton(grad_out, gate, up, weights, grad_gate, grad_up)
    return grad_gate, grad_up, grad_weights


def _packed_weighted_views(gate_up, weights):
    if gate_up.dim() != 2 or gate_up.shape[-1] % 2:
        raise ValueError("swiglu_packed_weighted needs shape [N, 2 * I] with an even last dimension")
    gate, up = gate_up.chunk(2, dim=-1)
    _validate_weighted_inputs(gate, up, weights)
    return gate, up


def swiglu_packed_weighted(gate_up, weights):
    """Weighted SwiGLU reading a packed [N, 2 * I] projection without copying its halves."""
    gate, up = _packed_weighted_views(gate_up, weights)
    if not (_TRITON_AVAILABLE and get_accelerator().on_accelerator(gate_up)):
        return _swiglu_weighted_eager(gate, up, weights)
    return _SwiGLUPackedWeightedFn.apply(gate_up, weights)


def swiglu_packed_weighted_backward(grad_out, gate_up, weights):
    """First-order packed projection and FP32 row-weight gradients, computed in one kernel pass."""
    gate, up = _packed_weighted_views(gate_up, weights)
    _validate_weighted_gradient(grad_out, gate)
    if not (_TRITON_AVAILABLE and get_accelerator().on_accelerator(gate_up)):
        with torch.enable_grad():
            gate_up, weights = (tensor.detach().requires_grad_(True) for tensor in (gate_up, weights))
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.autograd.grad(_swiglu_weighted_eager(gate, up, weights), (gate_up, weights), grad_out)
    gate_up, weights = gate_up.contiguous(), weights.contiguous()
    gate, up = gate_up.chunk(2, dim=-1)
    grad_gate_up = torch.empty_like(gate_up)
    grad_gate, grad_up = grad_gate_up.chunk(2, dim=-1)
    grad_weights = _swiglu_weighted_backward_triton(grad_out, gate, up, weights, grad_gate, grad_up)
    return grad_gate_up, grad_weights


def swiglu_backward(grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
    """The gradients of :func:`swiglu` with respect to ``gate`` and ``up``, as its own backward computes them.

    For callers that run the forward inside their own autograd function and keep ``gate`` and ``up``.
    """
    if not _TRITON_AVAILABLE:
        with torch.enable_grad():
            gate = gate.detach().requires_grad_(True)
            up = up.detach().requires_grad_(True)
            return torch.autograd.grad(_swiglu_eager(gate, up), (gate, up), grad_out)
    return _swiglu_backward_triton(grad_out, gate.contiguous(), up.contiguous())


def swiglu_packed_backward(grad_out: torch.Tensor, gate_up: torch.Tensor) -> torch.Tensor:
    """The gradient of :func:`swiglu_packed` with respect to ``gate_up``, as its own backward computes it."""
    if not _TRITON_AVAILABLE:
        with torch.enable_grad():
            gate_up = gate_up.detach().requires_grad_(True)
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.autograd.grad(_swiglu_eager(gate, up), gate_up, grad_out)[0]
    return _swiglu_packed_backward_triton(grad_out, gate_up.contiguous())


def swiglu_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU of a packed gate+up tensor: ``silu(gate_up[..., :I]) * gate_up[..., I:]``.

    Args:
        gate_up: The gate and up projections side by side, shape ``[..., 2 * I]``, float16/bfloat16/float32.

    Returns:
        Tensor of shape ``[..., I]`` and the dtype of ``gate_up``, matching :func:`swiglu` on the two halves.

    The Triton path supports first-order gradients only and raises for ``create_graph=True``.
    Forward and backward use the input device's current stream and restore the caller's current device.
    Falls back to the eager PyTorch expression when Triton is unavailable.
    """
    if gate_up.dim() == 0 or gate_up.shape[-1] % 2:
        raise ValueError(f"swiglu_packed expects an even last dimension, got shape {tuple(gate_up.shape)}")

    if not _TRITON_AVAILABLE:
        gate, up = gate_up.chunk(2, dim=-1)
        return _swiglu_eager(gate, up)

    return _SwiGLUPackedFn.apply(gate_up)
