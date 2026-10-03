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

:func:`swiglu_packed` takes ``gate`` and ``up`` packed side by side in one
``[..., 2 * I]`` tensor, the output of a single gate+up GEMM, and reads both
halves in place with the same math, so the packed GEMM output needs no copy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

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
            grad_out = grad_out.contiguous()

            grad_gate = torch.empty_like(gate)
            grad_up = torch.empty_like(up)

            n_elements = gate.numel()
            if n_elements > 0:
                grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
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
            gate_up, = ctx.saved_tensors
            grad_out = grad_out.contiguous()
            grad_gate_up = torch.empty_like(gate_up)

            n_elements = grad_out.numel()
            if n_elements > 0:
                grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
                _swiglu_packed_bwd_kernel[grid](grad_out,
                                                gate_up,
                                                grad_gate_up,
                                                n_elements,
                                                INTER=gate_up.shape[-1] // 2,
                                                BLOCK_SIZE=_BLOCK_SIZE,
                                                num_warps=_NUM_WARPS)

            return grad_gate_up


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


def swiglu_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU of a packed gate+up tensor: ``silu(gate_up[..., :I]) * gate_up[..., I:]``.

    Args:
        gate_up: The gate and up projections side by side, shape ``[..., 2 * I]``, float16/bfloat16/float32.

    Returns:
        Tensor of shape ``[..., I]`` and the dtype of ``gate_up``, matching :func:`swiglu` on the two halves.

    Falls back to the eager PyTorch expression when Triton is unavailable.
    """
    if gate_up.dim() == 0 or gate_up.shape[-1] % 2:
        raise ValueError(f"swiglu_packed expects an even last dimension, got shape {tuple(gate_up.shape)}")

    if not _TRITON_AVAILABLE:
        gate, up = gate_up.chunk(2, dim=-1)
        return _swiglu_eager(gate, up)

    return _SwiGLUPackedFn.apply(gate_up)
