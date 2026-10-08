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


# ---------------------------------------------------------------------------
# Clamped forms: silu(clamp(gate)) * clamp(up)            ("swiglu_clamped", DeepSeek-V4)
#                (clamp(up) + 1) * g * sigmoid(alpha * g)  ("swiglu_oai", GPT-OSS / MiniMax-M3), g = clamp(gate)
# gate is clamped from above only, up on both sides. Same fusion as above: one kernel each way, float32
# math, only gate and up saved for backward. The eager expression saves about five [rows, ffn] temporaries
# (the two clamps, the sigmoid, the products); on an expert-parallel rank that receives several times the
# average rows this is where the memory goes (MiniMax-M3, 65,536 tokens: an 11.96 GiB temporary here).
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _swiglu_clamped_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, alpha, limit, OAI: tl.constexpr,
                                   BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
        g = tl.minimum(gate, limit)
        u = tl.minimum(tl.maximum(up, -limit), limit)

        if OAI:
            out = (u + 1.0) * (g * tl.sigmoid(g * alpha))
        else:
            out = (g * tl.sigmoid(g)) * u

        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_clamped_bwd_kernel(grad_out_ptr, gate_ptr, up_ptr, grad_gate_ptr, grad_up_ptr, n_elements, alpha,
                                   limit, OAI: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0).to(tl.int64)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        grad_out = tl.load(grad_out_ptr + offsets, mask=mask).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
        g = tl.minimum(gate, limit)
        u = tl.minimum(tl.maximum(up, -limit), limit)
        # torch.clamp passes the gradient through where the input is inside the limits, inclusive.
        pass_gate = gate <= limit
        pass_up = (up >= -limit) & (up <= limit)

        if OAI:
            sig = tl.sigmoid(g * alpha)
            f = g * sig
            df = sig * (1.0 + alpha * g * (1.0 - sig))
            grad_gate = grad_out * (u + 1.0) * df
            grad_up = grad_out * f
        else:
            sig = tl.sigmoid(g)
            f = g * sig
            df = sig * (1.0 + g * (1.0 - sig))
            grad_gate = grad_out * u * df
            grad_up = grad_out * f
        grad_gate = tl.where(pass_gate, grad_gate, 0.0)
        grad_up = tl.where(pass_up, grad_up, 0.0)

        tl.store(grad_gate_ptr + offsets, grad_gate.to(grad_gate_ptr.dtype.element_ty), mask=mask)
        tl.store(grad_up_ptr + offsets, grad_up.to(grad_up_ptr.dtype.element_ty), mask=mask)

    class _SwiGLUClampedFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float, oai: bool) -> torch.Tensor:
            gate = gate.contiguous()
            up = up.contiguous()
            out = torch.empty_like(gate)
            n_elements = gate.numel()
            if n_elements > 0:
                grid = (triton.cdiv(n_elements, _BLOCK_SIZE), )
                _swiglu_clamped_fwd_kernel[grid](gate,
                                                 up,
                                                 out,
                                                 n_elements,
                                                 float(alpha),
                                                 float(limit),
                                                 OAI=oai,
                                                 BLOCK_SIZE=_BLOCK_SIZE,
                                                 num_warps=_NUM_WARPS)
            ctx.save_for_backward(gate, up)
            ctx.alpha, ctx.limit, ctx.oai = float(alpha), float(limit), bool(oai)
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
                _swiglu_clamped_bwd_kernel[grid](grad_out,
                                                 gate,
                                                 up,
                                                 grad_gate,
                                                 grad_up,
                                                 n_elements,
                                                 ctx.alpha,
                                                 ctx.limit,
                                                 OAI=ctx.oai,
                                                 BLOCK_SIZE=_BLOCK_SIZE,
                                                 num_warps=_NUM_WARPS)
            return grad_gate, grad_up, None, None, None


def _swiglu_clamped_eager(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float, oai: bool) -> torch.Tensor:
    """Pure-PyTorch reference used as the non-Triton fallback."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    if oai:
        return (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
    return F.silu(gate) * up


def swiglu_clamped(gate: torch.Tensor,
                   up: torch.Tensor,
                   alpha: float = 1.702,
                   limit: float = 7.0,
                   oai: bool = True) -> torch.Tensor:
    """Fused clamped SwiGLU.

    ``oai=True``: ``(clamp(up) + 1) * g * sigmoid(alpha * g)`` with ``g = clamp(gate, max=limit)`` and
    ``up`` clamped to ``[-limit, limit]`` (GPT-OSS, MiniMax-M3). ``oai=False``: ``silu(g) * clamp(up)``
    (DeepSeek-V4). Falls back to the eager expression when Triton is unavailable or the tensors are on
    the CPU.
    """
    if gate.shape != up.shape:
        raise ValueError(f"swiglu_clamped expects gate and up to have the same shape, got {tuple(gate.shape)} "
                         f"and {tuple(up.shape)}")
    if gate.dtype != up.dtype:
        raise ValueError(f"swiglu_clamped expects gate and up to have the same dtype, got {gate.dtype} and {up.dtype}")
    if not _TRITON_AVAILABLE or gate.device.type == "cpu":
        return _swiglu_clamped_eager(gate, up, alpha, limit, oai)
    return _SwiGLUClampedFn.apply(gate, up, alpha, limit, oai)
