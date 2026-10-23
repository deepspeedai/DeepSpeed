# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend (NPU) variant of the fused rotary position embedding.

flash-linear-attention's ``fla/modules/rotary/triton_ascend.py`` sizes the
sequence tile from the Ascend unified buffer with a conservative multiplier
(six live FP32 tiles: cos, sin, x0, x1, o0, o1), so this variant derives
``BLOCK_S`` the same way instead of the portable 16-row constant. The kernel
and the eager-rounding arithmetic (every product and sum rounds where the
eager split-half expression rounds, with ``enable_fp_fusion=False``) are the
portable module's unchanged, so outputs and gradients stay elementwise equal
to eager.
"""

import torch
from torch.autograd.function import once_differentiable

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, current_device, triton
from deepspeed.ops.triton_ops.fused_rotary_pos_emb import assert_supported

if _TRITON_AVAILABLE:
    from deepspeed.ops.triton_ops.fused_rotary_pos_emb import _HEADS_PER_PROGRAM, _rotary_kernel, _unit_stride
    from deepspeed.ops.triton_ops.npu._ub import compute_row_tile_block_size, multiprocessor_count

    # Peak live fp32 tiles in the kernel: cos, sin, x0, x1, o0, o1 (FLA's rotary budget).
    _ROTARY_MEM_MULT = 6.0
    _ROTARY_SAFETY_MARGIN = 0.90

    def _block_s(seq_len: int, block_half: int, device) -> int:
        """UB-safe sequence tile, in FLA's rotary budget.

        The desired tile fills the device (one tile per processor); the UB
        budget and FLA's per-rotary-width fallbacks cap it from above.
        """
        count = multiprocessor_count(device) or 1
        desired = triton.next_power_of_2(triton.cdiv(seq_len, count))
        fallback = 16 if block_half >= 128 else (32 if block_half >= 64 else 64)
        return compute_row_tile_block_size(
            row_dim=desired,
            fixed_dim=block_half,
            memory_multiplier=_ROTARY_MEM_MULT,
            tiling_row=True,
            safety_margin=_ROTARY_SAFETY_MARGIN,
            dtype_size=2,
            fallback=fallback,
            min_block=1,
        )

    def _launch(x, cos, sin, out, backward):
        batch, heads, seq_len, head_dim = x.shape
        if x.numel() == 0:
            return out
        block_half = max(16, triton.next_power_of_2(head_dim // 2))
        block_s = _block_s(seq_len, block_half, x.device)
        grid = (triton.cdiv(seq_len, block_s), batch, triton.cdiv(heads, _HEADS_PER_PROGRAM))
        # Triton launches on the current device and stream, which need not be the input tensor's device.
        with current_device(x):
            _rotary_kernel[grid](
                x,
                cos,
                sin,
                out,
                heads,
                seq_len,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                cos.stride(0) if cos.shape[0] > 1 else 0,
                cos.stride(1),
                HALF=head_dim // 2,
                BLOCK_HALF=block_half,
                BLOCK_S=block_s,
                HEADS_PER_PROGRAM=_HEADS_PER_PROGRAM,
                BACKWARD=backward,
                # Contracting a product and a sum into one FMA would skip the product's rounding, which eager performs.
                enable_fp_fusion=False,
            )
        return out

    class _FusedRotaryPosEmb(torch.autograd.Function):
        """Split-half RoPE on [batch, heads, sequence, head_dim] views; cos and sin are [batch, sequence, head_dim]."""

        @staticmethod
        def forward(ctx, q, k, cos, sin):
            # An unused output must leave its input without a gradient, as eager does, rather than a zero one.
            ctx.set_materialize_grads(False)
            cos, sin = cos.contiguous(), sin.contiguous()
            # For dense inputs the outputs keep their strides, as the eager expression's do.
            q_out = _launch(q, cos, sin, torch.empty_like(q), backward=False)
            k_out = _launch(k, cos, sin, torch.empty_like(k), backward=False)
            ctx.save_for_backward(cos, sin)
            # The gradients take the outputs' layout, which for dense q and k is what their producers wrote. Only the
            # strides are kept, not the outputs.
            ctx.layouts = ((q_out.shape, q_out.stride()), (k_out.shape, k_out.stride()))
            return q_out, k_out

        # Autograd cannot see inside the Triton kernels, so differentiating this backward again would silently drop
        # the second derivative; once_differentiable makes that raise instead.
        @staticmethod
        @once_differentiable
        def backward(ctx, grad_q, grad_k):
            cos, sin = ctx.saved_tensors
            grads = []
            for grad, (shape, stride) in zip((grad_q, grad_k), ctx.layouts):
                if grad is None:
                    grads.append(None)
                    continue
                out = torch.empty_strided(shape, stride, dtype=grad.dtype, device=grad.device)
                grads.append(_launch(_unit_stride(grad), cos, sin, out, backward=True))
            return grads[0], grads[1], None, None


def _run_kernels(q, k, cos, sin, unsqueeze_dim):
    if unsqueeze_dim == 2:
        q_out, k_out = _FusedRotaryPosEmb.apply(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        return q_out.transpose(1, 2), k_out.transpose(1, 2)
    return _FusedRotaryPosEmb.apply(q, k, cos, sin)


def fused_apply_rotary_pos_emb(q: torch.Tensor,
                               k: torch.Tensor,
                               cos: torch.Tensor,
                               sin: torch.Tensor,
                               unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """The portable fused RoPE with the Ascend sequence tile (see the module docstring)."""
    assert_supported(q, k, cos, sin, unsqueeze_dim)
    return _run_kernels(q, k, cos, sin, unsqueeze_dim)
