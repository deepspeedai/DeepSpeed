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
SUPPORTED_ROW_WEIGHTING_DTYPES = (torch.bfloat16, torch.float16)

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
        # int64: token * out_stride overflows int32 once tokens x hidden > 2**31.
        token = tl.program_id(0).to(tl.int64)
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
        # int64: token * grad_out_stride overflows int32 once tokens x hidden > 2**31.
        token = tl.program_id(0).to(tl.int64)

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

    @triton.jit
    def _row_weighting_forward_kernel(
        rows_ptr,
        weights_ptr,
        out_ptr,
        hidden,
        rows_stride,
        weights_stride,
        out_stride,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < hidden

        values = tl.load(rows_ptr + row * rows_stride + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        weight = tl.load(weights_ptr + row * weights_stride).to(tl.float32)
        tl.store(
            out_ptr + row * out_stride + hidden_offsets,
            (values * weight).to(out_ptr.dtype.element_ty),
            mask=hidden_mask,
        )

    @triton.jit
    def _row_weighting_backward_partial_kernel(
        grad_out_ptr,
        rows_ptr,
        weights_ptr,
        grad_rows_ptr,
        partials_ptr,
        hidden,
        grad_out_stride,
        rows_stride,
        weights_stride,
        grad_rows_stride,
        num_hidden_blocks: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        hidden_block = tl.program_id(1)
        hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < hidden

        upstream = tl.load(
            grad_out_ptr + row * grad_out_stride + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            rows_ptr + row * rows_stride + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weights_ptr + row * weights_stride).to(tl.float32)

        tl.store(
            grad_rows_ptr + row * grad_rows_stride + hidden_offsets,
            (upstream * weight).to(grad_rows_ptr.dtype.element_ty),
            mask=hidden_mask,
        )
        partial = tl.sum(upstream * values, axis=0)
        tl.store(partials_ptr + row * num_hidden_blocks + hidden_block, partial)

    @triton.jit
    def _row_weighting_backward_reduce_kernel(
        partials_ptr,
        grad_weights_ptr,
        grad_weights_stride,
        NUM_HIDDEN_BLOCKS: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK_B)
        mask = offsets < NUM_HIDDEN_BLOCKS
        partials = tl.load(partials_ptr + row * NUM_HIDDEN_BLOCKS + offsets, mask=mask, other=0.0)
        grad_weight = tl.sum(partials, axis=0)
        tl.store(grad_weights_ptr + row * grad_weights_stride, grad_weight)


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


def _row_weighting_block_hidden(hidden: int) -> int:
    """Choose a power-of-two hidden tile for row-local products."""
    return min(_MAX_BLOCK_HIDDEN, max(16, triton.next_power_of_2(hidden)))


def _row_weighting_reduce_block(num_hidden_blocks: int) -> int:
    """Round the per-row partial count up to a power of two for ``tl.arange``."""
    return max(1, triton.next_power_of_2(num_hidden_blocks))


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


def assert_row_weighting_supported(rows: torch.Tensor, weights: torch.Tensor) -> None:
    """Reject unsupported per-row weighting configurations before dispatch."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError('row_weighting_impl="fused" needs Triton, which is not installed in this environment. '
                           'Install Triton, or set row_weighting_impl="eager".')
    if _IS_ROCM_PYTORCH:
        raise RuntimeError('row_weighting_impl="fused" is not yet supported on ROCm. Set '
                           'row_weighting_impl="eager" to run here.')
    if rows.device.type != "cuda":
        raise RuntimeError('row_weighting_impl="fused" runs CUDA kernels but rows are on device '
                           f'"{rows.device.type}". Set row_weighting_impl="eager" to run here.')
    if weights.device != rows.device:
        raise RuntimeError('row_weighting_impl="fused" requires rows and weights on the same CUDA device, got '
                           f"rows on {rows.device} and weights on {weights.device}.")
    if rows.dim() != 2:
        raise RuntimeError('row_weighting_impl="fused" requires rows with shape [N, H], got '
                           f"{tuple(rows.shape)}. Set row_weighting_impl=\"eager\" for unsupported shapes.")
    expected_weight_shape = (rows.shape[0], 1)
    if tuple(weights.shape) != expected_weight_shape:
        raise RuntimeError('row_weighting_impl="fused" requires weights with shape [N, 1] matching rows, got '
                           f"{tuple(weights.shape)} for rows {tuple(rows.shape)}.")
    if rows.dtype not in SUPPORTED_ROW_WEIGHTING_DTYPES:
        raise RuntimeError('row_weighting_impl="fused" supports bfloat16 and float16 rows, got '
                           f"{rows.dtype}. Set row_weighting_impl=\"eager\", or use a supported dtype.")
    if weights.dtype != torch.float32:
        raise RuntimeError('row_weighting_impl="fused" requires float32 weights, got '
                           f"{weights.dtype}. Set row_weighting_impl=\"eager\" for unsupported weights.")
    if not rows.is_contiguous():
        raise RuntimeError('row_weighting_impl="fused" requires contiguous rows. Set row_weighting_impl="eager" '
                           "for non-contiguous row layouts.")
    if not weights.is_contiguous():
        raise RuntimeError('row_weighting_impl="fused" requires contiguous weights. Set row_weighting_impl="eager" '
                           "for non-contiguous weight layouts.")


class _FusedRowWeighting(torch.autograd.Function):
    """Apply one FP32 routing weight per row, with the eager rounding point."""

    @staticmethod
    def forward(ctx, rows, weights):
        output = torch.empty_like(rows)
        ctx.save_for_backward(rows, weights)

        n_rows, hidden = rows.shape
        if n_rows == 0 or hidden == 0:
            return output

        block_hidden = _row_weighting_block_hidden(hidden)
        grid = (n_rows, triton.cdiv(hidden, block_hidden))
        _row_weighting_forward_kernel[grid](
            rows,
            weights,
            output,
            hidden,
            rows.stride(0),
            weights.stride(0),
            output.stride(0),
            BLOCK_H=block_hidden,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rows, weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_rows = torch.empty_like(rows)
        grad_weights = torch.empty_like(weights)

        n_rows, hidden = rows.shape
        if n_rows == 0:
            return grad_rows, grad_weights
        if hidden == 0:
            grad_weights.zero_()
            return grad_rows, grad_weights

        block_hidden = _row_weighting_block_hidden(hidden)
        num_hidden_blocks = triton.cdiv(hidden, block_hidden)
        partials = torch.empty((n_rows, num_hidden_blocks), dtype=torch.float32, device=rows.device)
        _row_weighting_backward_partial_kernel[(n_rows, num_hidden_blocks)](
            grad_output,
            rows,
            weights,
            grad_rows,
            partials,
            hidden,
            grad_output.stride(0),
            rows.stride(0),
            weights.stride(0),
            grad_rows.stride(0),
            num_hidden_blocks=num_hidden_blocks,
            BLOCK_H=block_hidden,
        )
        _row_weighting_backward_reduce_kernel[(n_rows, )](
            partials,
            grad_weights,
            grad_weights.stride(0),
            NUM_HIDDEN_BLOCKS=num_hidden_blocks,
            BLOCK_B=_row_weighting_reduce_block(num_hidden_blocks),
        )
        return grad_rows, grad_weights


def fused_row_weighting(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Apply ``weights`` to ``rows`` as ``(rows.float() * weights).to(rows.dtype)``."""
    assert_row_weighting_supported(rows, weights)
    return _FusedRowWeighting.apply(rows, weights)


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
