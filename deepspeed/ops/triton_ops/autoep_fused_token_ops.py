# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Fused AutoEP token restore without the eager scatter and FP32 intermediate.

The kernel reduces each token's top-k rows in FP32. Communication, routing,
expert reorder, and grouped GEMM remain unchanged.
"""

from __future__ import annotations

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

_IS_ROCM_PYTORCH = getattr(torch.version, "hip", None) is not None

SUPPORTED_ROW_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_MAX_BLOCK_HIDDEN = 512
_INVERT_INDEX_BLOCK = 256
# The kernels hold a [slots, BLOCK_H] FP32 block live, so the hidden tile shrinks
# as top-k grows to keep that block in registers rather than spilling.
_MAX_BLOCK_ELEMENTS = 2048

if _TRITON_AVAILABLE:

    @triton.jit
    def _invert_index_kernel(
        index_ptr,
        inverse_ptr,
        num_indices,
        num_inverse_rows,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        in_range = offsets < num_indices

        targets = tl.load(index_ptr + offsets, mask=in_range, other=-1).to(tl.int64)
        writable = in_range & (targets >= 0) & (targets < num_inverse_rows)
        tl.store(inverse_ptr + tl.where(writable, targets, 0), offsets.to(tl.int32), mask=writable)

    @triton.jit
    def _weighted_restore_forward_kernel(
        rows_ptr,
        inverse_ptr,
        scores_ptr,
        out_ptr,
        hidden,
        rows_stride,
        scores_stride,
        out_stride,
        TOP_K: tl.constexpr,
        K_PADDED: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < hidden

        slots = tl.arange(0, K_PADDED)
        slot_mask = slots < TOP_K

        source_rows = tl.load(inverse_ptr + token * TOP_K + slots, mask=slot_mask, other=-1).to(tl.int64)
        row_valid = slot_mask & (source_rows >= 0)
        safe_rows = tl.where(row_valid, source_rows, 0)

        scores = tl.load(scores_ptr + token * scores_stride + slots, mask=slot_mask, other=0.0).to(tl.float32)

        block_mask = row_valid[:, None] & hidden_mask[None, :]
        values = tl.load(
            rows_ptr + safe_rows[:, None] * rows_stride + hidden_offsets[None, :],
            mask=block_mask,
            other=0.0,
        ).to(tl.float32)

        # Match the eager path's FP32 product and accumulation.
        weighted = tl.sum(values * scores[:, None], axis=0)
        tl.store(
            out_ptr + token * out_stride + hidden_offsets,
            weighted.to(out_ptr.dtype.element_ty),
            mask=hidden_mask,
        )

    @triton.jit
    def _weighted_restore_backward_kernel(
        grad_out_ptr,
        rows_ptr,
        inverse_ptr,
        scores_ptr,
        grad_rows_ptr,
        grad_scores_ptr,
        hidden,
        grad_out_stride,
        rows_stride,
        scores_stride,
        grad_rows_stride,
        grad_scores_stride,
        TOP_K: tl.constexpr,
        K_PADDED: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)

        slots = tl.arange(0, K_PADDED)
        slot_mask = slots < TOP_K

        source_rows = tl.load(inverse_ptr + token * TOP_K + slots, mask=slot_mask, other=-1).to(tl.int64)
        row_valid = slot_mask & (source_rows >= 0)
        safe_rows = tl.where(row_valid, source_rows, 0)
        scores = tl.load(scores_ptr + token * scores_stride + slots, mask=slot_mask, other=0.0).to(tl.float32)

        grad_rows_dtype = grad_rows_ptr.dtype.element_ty
        # Keeping one token per program avoids a second reduction pass for scores.
        score_partials = tl.zeros([K_PADDED, BLOCK_H], dtype=tl.float32)

        for hidden_start in range(0, hidden, BLOCK_H):
            hidden_offsets = hidden_start + tl.arange(0, BLOCK_H)
            hidden_mask = hidden_offsets < hidden
            block_mask = row_valid[:, None] & hidden_mask[None, :]

            upstream = tl.load(
                grad_out_ptr + token * grad_out_stride + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)

            values = tl.load(
                rows_ptr + safe_rows[:, None] * rows_stride + hidden_offsets[None, :],
                mask=block_mask,
                other=0.0,
            ).to(tl.float32)
            score_partials += values * upstream[None, :]

            tl.store(
                grad_rows_ptr + safe_rows[:, None] * grad_rows_stride + hidden_offsets[None, :],
                (upstream[None, :] * scores[:, None]).to(grad_rows_dtype),
                mask=block_mask,
            )

        grad_scores = tl.sum(score_partials, axis=1)
        tl.store(
            grad_scores_ptr + token * grad_scores_stride + slots,
            grad_scores.to(grad_scores_ptr.dtype.element_ty),
            mask=slot_mask,
        )


def is_available() -> bool:
    """Whether this build can run the fused weighted restore at all."""
    return _TRITON_AVAILABLE and not _IS_ROCM_PYTORCH


def assert_supported(rows: torch.Tensor, *, score_apply: str) -> None:
    """Reject unsupported configurations before collectives begin."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError('combine_impl="fused_weighted_sum" needs Triton, which is not installed in this '
                           "environment. Install Triton, or leave combine_impl unset.")
    if _IS_ROCM_PYTORCH:
        raise RuntimeError('combine_impl="fused_weighted_sum" is not yet supported on ROCm. Leave combine_impl '
                           "unset to run here.")
    if rows.device.type != "cuda":
        raise RuntimeError('combine_impl="fused_weighted_sum" runs CUDA kernels but this layer is on device '
                           f'"{rows.device.type}". Leave combine_impl unset to run here.')
    if rows.dtype not in SUPPORTED_ROW_DTYPES:
        raise RuntimeError('combine_impl="fused_weighted_sum" supports bfloat16, float16, and float32 rows, got '
                           f"{rows.dtype}. Leave combine_impl unset, or use a supported floating-point dtype.")
    if score_apply != "post":
        raise RuntimeError('combine_impl="fused_weighted_sum" folds the routing weight into the top-k reduction, '
                           f'which only exists for score_apply="post", but this layer resolved '
                           f'score_apply="{score_apply}". Leave combine_impl unset.')


def _block_hidden(hidden: int, slots: int) -> int:
    """Choose a power-of-two tile within the FP32 register budget."""
    budget = max(16, _MAX_BLOCK_ELEMENTS // slots)
    return min(_MAX_BLOCK_HIDDEN, budget, max(16, triton.next_power_of_2(hidden)))


def _padded_top_k(top_k: int) -> int:
    """Round top-k up to a power of two, which ``tl.arange`` requires."""
    return max(2, triton.next_power_of_2(top_k))


def _invert_index(index: torch.Tensor, num_inverse_rows: int) -> torch.Tensor:
    """Invert the row permutation produced by sorting routed assignments."""
    inverse = torch.empty((num_inverse_rows, ), dtype=torch.int32, device=index.device)
    num_indices = index.numel()

    grid = (triton.cdiv(num_indices, _INVERT_INDEX_BLOCK), )
    _invert_index_kernel[grid](
        index.contiguous(),
        inverse,
        num_indices,
        num_inverse_rows,
        BLOCK=_INVERT_INDEX_BLOCK,
    )
    return inverse


def _differentiable_backward(grad_output, combined_rows, top_scores, inverse, top_k):
    """Build the rare higher-order backward with regular PyTorch operations."""
    n_tokens, hidden = top_scores.shape[0], combined_rows.shape[-1]
    valid = inverse >= 0
    safe_inverse = inverse.clamp_min(0).to(torch.int64)
    gathered_rows = combined_rows.index_select(0, safe_inverse).reshape(n_tokens, top_k, hidden)

    grad_by_assignment = (grad_output[:, None, :] * top_scores[:, :, None]).to(combined_rows.dtype).reshape(-1, hidden)
    grad_rows = torch.zeros_like(combined_rows)
    grad_rows = grad_rows.index_copy(0, safe_inverse[valid], grad_by_assignment[valid])

    grad_scores = (gathered_rows.float() * grad_output.float()[:, None, :]).sum(dim=-1)
    grad_scores = torch.where(valid.reshape(n_tokens, top_k), grad_scores, 0.0).to(top_scores.dtype)
    return grad_rows, grad_scores


class _FusedWeightedRestore(torch.autograd.Function):
    """Weight rows by their routing score and reduce over top-k in one pass."""

    @staticmethod
    def forward(ctx, combined_rows, top_scores, inverse, top_k):
        combined_rows = combined_rows.contiguous()
        n_tokens, hidden = top_scores.shape[0], combined_rows.shape[-1]
        output = torch.empty((n_tokens, hidden), dtype=combined_rows.dtype, device=combined_rows.device)

        ctx.save_for_backward(combined_rows, top_scores, inverse)
        ctx.top_k = top_k

        k_padded = _padded_top_k(top_k)
        block_hidden = _block_hidden(hidden, slots=k_padded)
        grid = (n_tokens, triton.cdiv(hidden, block_hidden))
        _weighted_restore_forward_kernel[grid](
            combined_rows,
            inverse,
            top_scores,
            output,
            hidden,
            combined_rows.stride(0),
            top_scores.stride(0),
            output.stride(0),
            TOP_K=top_k,
            K_PADDED=k_padded,
            BLOCK_H=block_hidden,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        combined_rows, top_scores, inverse = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        if torch.is_grad_enabled():
            grad_rows, grad_scores = _differentiable_backward(grad_output, combined_rows, top_scores, inverse,
                                                              ctx.top_k)
            return grad_rows, grad_scores, None, None

        grad_rows = torch.empty_like(combined_rows)
        grad_scores = torch.empty_like(top_scores)

        n_tokens, hidden = top_scores.shape[0], combined_rows.shape[-1]

        k_padded = _padded_top_k(ctx.top_k)
        _weighted_restore_backward_kernel[(n_tokens, )](
            grad_output,
            combined_rows,
            inverse,
            top_scores,
            grad_rows,
            grad_scores,
            hidden,
            grad_output.stride(0),
            combined_rows.stride(0),
            top_scores.stride(0),
            grad_rows.stride(0),
            grad_scores.stride(0),
            TOP_K=ctx.top_k,
            K_PADDED=k_padded,
            BLOCK_H=_block_hidden(hidden, slots=k_padded),
        )
        return grad_rows, grad_scores, None, None


def fused_weighted_restore(
    combined_rows: torch.Tensor,
    top_scores: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    top_k: int,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Restore ``[T * K, H]`` rows directly to weighted ``[B, S, H]`` output."""
    bsz, seqlen, hidden = shape
    n_tokens = bsz * seqlen
    expected_rows = n_tokens * top_k
    inverse = _invert_index(token_indices_sorted, expected_rows)
    output = _FusedWeightedRestore.apply(combined_rows, top_scores.contiguous(), inverse, top_k)
    return output.reshape(bsz, seqlen, hidden)
