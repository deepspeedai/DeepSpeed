# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend (NPU) variant of the Triton chunked cross entropy.

flash-linear-attention's ``fla/modules/fused_cross_entropy/triton_ascend.py``
splits the vocabulary across programs - a [rows, vocab-splits] grid - instead
of looping over the whole vocabulary inside each row's program, which
parallelizes a 150k-vocabulary softmax over the device; the per-split
log-sum-exps and losses reduce host-side afterwards. This variant adopts that
launch shape with the portable module's arithmetic and semantics: no label
smoothing, no z-loss, ignored rows contribute zero loss and gradient. A
vocabulary that fits one tile degenerates to the portable per-row math, so
the small-shape contract (including the probe) is bitwise unchanged.
"""

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl
from deepspeed.runtime.chunked_cross_entropy import (_ChunkedCrossEntropy, _block_rows, _prepare_inputs,
                                                     _refuse_create_graph, _resolve_backend, _validate_block_rows)

if _TRITON_AVAILABLE:
    from deepspeed.ops.triton_ops.npu._ub import ASCEND_MAX_GRID_DIM, compute_vocab_block_size, iter_axis_launch_chunks

    # Peak live fp32 buffers along the vocabulary dimension (FLA's CE budgets):
    # the backward keeps the logits and the gradient live, so it gets more.
    _CE_FWD_MEM_MULT = 8.0
    _CE_BWD_MEM_MULT = 12.0

    def _block_and_warps(n_cols: int, n_rows: int, *, is_backward: bool) -> tuple[int, int]:
        block = compute_vocab_block_size(n_cols, n_rows, _CE_BWD_MEM_MULT if is_backward else _CE_FWD_MEM_MULT)
        return block, 2 if block <= 2048 else 4

    @triton.jit(do_not_specialize=['n_cols', 'n_rows', 'ROW_OFFSET'])
    def _log_sum_exp_split_kernel(
        logits_ptr,
        target_ptr,
        log_sum_exp_ptr,
        loss_ptr,
        n_cols,
        n_rows,
        row_stride,
        ignore_index,
        ROW_OFFSET,
        BLOCK: tl.constexpr,
        SPLIT: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_block_idx = tl.program_id(1)
        row_ptr = logits_ptr + tl.cast(row_idx + ROW_OFFSET, tl.int64) * row_stride
        offsets = col_block_idx * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_cols

        values = tl.load(row_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
        max_val = tl.max(values, axis=0)
        log_sum_exp = max_val + tl.log(tl.sum(tl.exp(values - max_val), axis=0))

        target = tl.load(target_ptr + row_idx + ROW_OFFSET)
        valid = target != ignore_index
        # A safe in-range index for the gather; the loss below zeroes it for ignored rows.
        target_logit = tl.load(row_ptr + tl.where(valid, target, 0)).to(tl.float32)
        # When the vocabulary splits across tiles, each tile stores the negative target
        # logit where the target lives (zero elsewhere) and the host adds the final
        # log-sum-exp once, so the split log-sum-exps are never summed twice (FLA's
        # reduction). A single tile stores the portable per-row loss directly.
        label_in_tile = (target >= col_block_idx * BLOCK) & (target < col_block_idx * BLOCK + BLOCK)
        tl.store(log_sum_exp_ptr + col_block_idx * n_rows + row_idx, log_sum_exp)
        if SPLIT:
            tl.store(loss_ptr + col_block_idx * n_rows + row_idx, tl.where(valid & label_in_tile, -target_logit, 0.0))
        else:
            tl.store(loss_ptr + col_block_idx * n_rows + row_idx, tl.where(valid, log_sum_exp - target_logit, 0.0))

    @triton.jit(do_not_specialize=['n_cols', 'n_rows', 'ROW_OFFSET'])
    def _cross_entropy_split_backward_kernel(
        logits_ptr,
        grad_ptr,
        target_ptr,
        log_sum_exp_ptr,
        row_scale_ptr,
        n_cols,
        row_stride,
        grad_row_stride,
        ignore_index,
        ROW_OFFSET,
        BLOCK: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_block_idx = tl.program_id(1)
        row = row_idx + ROW_OFFSET
        row_ptr = logits_ptr + tl.cast(row, tl.int64) * row_stride
        offsets = col_block_idx * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_cols

        target = tl.load(target_ptr + row)
        valid = target != ignore_index
        # Ignored rows get an all-zero gradient, as they do from cross_entropy.
        scale = tl.where(valid, tl.load(row_scale_ptr + row), 0.0)
        log_sum_exp = tl.load(log_sum_exp_ptr + row)

        values = tl.load(row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.exp(values - log_sum_exp) * scale
        grad = tl.where(offsets == target, grad - scale, grad)
        tl.store(grad_ptr + tl.cast(row, tl.int64) * grad_row_stride + offsets,
                 grad.to(grad_ptr.dtype.element_ty),
                 mask=mask)

    class _TritonCrossEntropy(torch.autograd.Function):
        """Per-row cross entropy with one read of the logits in forward and one read and write in backward."""

        @staticmethod
        def forward(ctx, logits, target, ignore_index):
            n_rows, n_cols = logits.shape
            block, num_warps = _block_and_warps(n_cols, n_rows, is_backward=False)
            n_splits = triton.cdiv(n_cols, block)
            shape = (n_splits, n_rows) if n_splits > 1 else (n_rows, )
            log_sum_exp = torch.empty(shape, dtype=torch.float32, device=logits.device)
            loss = torch.empty(shape, dtype=torch.float32, device=logits.device)
            if n_rows:
                # Host-side row chunking keeps the [rows, splits] grid within the launch limits.
                for row_off, row_len in iter_axis_launch_chunks(n_rows, n_splits, max_grid=ASCEND_MAX_GRID_DIM):
                    _log_sum_exp_split_kernel[(row_len, n_splits)](
                        logits,
                        target,
                        log_sum_exp,
                        loss,
                        n_cols,
                        n_rows,
                        logits.stride(0),
                        ignore_index,
                        row_off,
                        BLOCK=block,
                        num_warps=num_warps,
                        SPLIT=n_splits > 1,
                    )
            if n_splits > 1:
                log_sum_exp = torch.logsumexp(log_sum_exp, dim=0)
                # The split tiles stored the negative target logit; add the final
                # log-sum-exp once (FLA's split reduction), then zero the ignored
                # rows, which contribute zero loss.
                loss = (loss.sum(dim=0) + log_sum_exp).masked_fill_(target == ignore_index, 0.0)
            ctx.save_for_backward(logits, target, log_sum_exp)
            ctx.ignore_index = ignore_index
            return loss

        @staticmethod
        def backward(ctx, grad_loss):
            _refuse_create_graph()
            logits, target, log_sum_exp = ctx.saved_tensors
            grad_logits = torch.empty_like(logits)
            n_rows, n_cols = logits.shape
            if n_rows:
                row_scale = grad_loss.float().contiguous()
                block, num_warps = _block_and_warps(n_cols, n_rows, is_backward=True)
                n_splits = triton.cdiv(n_cols, block)
                for row_off, row_len in iter_axis_launch_chunks(n_rows, n_splits, max_grid=ASCEND_MAX_GRID_DIM):
                    _cross_entropy_split_backward_kernel[(row_len, n_splits)](
                        logits,
                        grad_logits,
                        target,
                        log_sum_exp,
                        row_scale,
                        n_cols,
                        logits.stride(0),
                        grad_logits.stride(0),
                        ctx.ignore_index,
                        row_off,
                        BLOCK=block,
                        num_warps=num_warps,
                    )
            return grad_logits, None, None


def chunked_cross_entropy(logits: torch.Tensor,
                          target: torch.Tensor,
                          ignore_index: int = -100,
                          reduction: str = "mean",
                          block_rows: int | None = None,
                          backend: str = "auto") -> torch.Tensor:
    """The portable chunked cross entropy with the Ascend split-vocabulary kernels (see the module docstring)."""
    block_rows = _validate_block_rows(block_rows)
    target = _prepare_inputs(logits, target, reduction, ignore_index)
    logits = logits.contiguous()
    if _resolve_backend(backend, logits) == "triton":
        loss = _TritonCrossEntropy.apply(logits, target, ignore_index)
    else:
        if block_rows is None:
            block_rows = _block_rows(logits.shape[-1])
        loss = _ChunkedCrossEntropy.apply(logits, target, ignore_index, block_rows)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / (target != ignore_index).sum()
