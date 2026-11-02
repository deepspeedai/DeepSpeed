# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Triton kernels for the vocabulary-parallel log-probability, entropy and KL ops.

Each kernel handles one row of a ``[..., local_vocab]`` logits shard and loops over the local
vocabulary in blocks, so no ``[rows, local_vocab]`` fp32 temporary is materialized. The kernels
only see the local shard; ``deepspeed.sequence.vocab_parallel_ops`` owns the all-reduces and the
autograd functions, and keeps an eager fallback with the same semantics.

Rows are the flattened leading dimensions and are addressed in place (see ``row_layout``), so a
slice such as ``logits[:, :-1]`` is read without first being copied into a ``[rows, vocab]`` view.
Gradients are returned contiguous.

``global_max`` and ``log_sum_exp`` are the global per-row softmax statistics, kept separate as in
the eager path, and a log-probability is ``(z - global_max) - log_sum_exp``.

The entropy and KL forward passes need one read of their inputs. Alongside each shard's max and
sum-exp they accumulate a probability-weighted sum of logits centred on that max, which can be
re-centred on the global max after the MAX all-reduce. Centring keeps a large common logit offset
from cancelling in fp32, as in the eager path.
"""

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

_MAX_BLOCK = 4096
_NUM_WARPS = 16
# The single-pass entropy and KL statistics keep up to six running values per lane; with a smaller
# block they stay in registers (on H100 the KL statistics ran about 1.2x faster than with 4096).
_STATS_MAX_BLOCK = 2048

if _TRITON_AVAILABLE:

    @triton.jit
    def _row_start(row, inner_rows, outer_stride, row_stride):
        return (row // inner_rows) * outer_stride + (row % inner_rows) * row_stride

    @triton.jit
    def _log_probs(ptr, cols, mask, global_max, log_sum_exp):
        z = tl.load(ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        return (z - global_max) - log_sum_exp

    @triton.jit
    def _local_softmax_stats_kernel(x_ptr, max_ptr, sum_ptr, vocab, x_inner, x_outer, x_row, BLOCK: tl.constexpr):
        # A single read computes the local max together with the sum-exp relative to it, by
        # rescaling each lane's running sum whenever its running max grows.
        row = tl.program_id(0).to(tl.int64)
        x_ptr += _row_start(row, x_inner, x_outer, x_row)
        lane_max = tl.full([BLOCK], float("-inf"), tl.float32)
        lane_sum = tl.zeros([BLOCK], tl.float32)
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            z = tl.load(x_ptr + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
            new_max = tl.maximum(lane_max, z)
            # exp(-inf - -inf) is NaN; a lane that has only seen -inf keeps a zero sum.
            empty = new_max == float("-inf")
            lane_sum = tl.where(empty, 0.0, lane_sum * tl.exp(lane_max - new_max) + tl.exp(z - new_max))
            lane_max = new_max
        row_max = tl.max(lane_max, axis=0)
        row_sum = tl.sum(tl.where(lane_max == float("-inf"), 0.0, lane_sum * tl.exp(lane_max - row_max)), axis=0)
        tl.store(max_ptr + row, row_max)
        tl.store(sum_ptr + row, row_sum)

    @triton.jit
    def _entropy_stats_kernel(x_ptr, out_ptr, num_rows, vocab, x_inner, x_outer, x_row, BLOCK: tl.constexpr):
        # Local max m, sum_j exp(z_j - m), and sum_j exp(z_j - m) * (z_j - m). Raising the max from m
        # to m' scales both sums by exp(m - m') and shifts the centred logits by -(m' - m).
        row = tl.program_id(0).to(tl.int64)
        x_ptr += _row_start(row, x_inner, x_outer, x_row)
        run_max = tl.full([BLOCK], float("-inf"), tl.float32)
        run_sum = tl.zeros([BLOCK], tl.float32)
        run_weighted = tl.zeros([BLOCK], tl.float32)
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            z = tl.load(x_ptr + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
            new_max = tl.maximum(run_max, z)
            # Sums that have only seen -inf are still 0 and must stay finite. A scale that underflows to 0
            # may come with an infinite shift when the maxima are finite extremes, so 0 * inf is skipped.
            seen = run_max != float("-inf")
            scale = tl.where(seen, tl.exp(run_max - new_max), 0.0)
            run_weighted = tl.where(scale > 0, scale * (run_weighted - (new_max - run_max) * run_sum), 0.0)
            run_sum = run_sum * scale
            live = z != float("-inf")
            e = tl.where(live, tl.exp(z - new_max), 0.0)
            run_sum += e
            run_weighted += tl.where(e > 0, e * (z - new_max), 0.0)
            run_max = new_max
        # Re-centre every lane on the row max before summing the lanes.
        row_max = tl.max(run_max, axis=0)
        seen = run_max != float("-inf")
        scale = tl.where(seen, tl.exp(run_max - row_max), 0.0)
        run_weighted = tl.where(scale > 0, scale * (run_weighted - (row_max - run_max) * run_sum), 0.0)
        run_sum = run_sum * scale
        tl.store(out_ptr + row, row_max)
        tl.store(out_ptr + num_rows + row, tl.sum(run_sum, axis=0))
        tl.store(out_ptr + 2 * num_rows + row, tl.sum(run_weighted, axis=0))

    @triton.jit
    def _entropy_backward_kernel(x_ptr, max_ptr, lse_ptr, entropy_ptr, grad_out_ptr, grad_ptr, vocab, x_inner, x_outer,
                                 x_row, BLOCK: tl.constexpr):
        # dH / dz_j = -p_j * (log p_j + H)
        row = tl.program_id(0).to(tl.int64)
        x_ptr += _row_start(row, x_inner, x_outer, x_row)
        grad_ptr += row * vocab
        global_max = tl.load(max_ptr + row)
        log_sum_exp = tl.load(lse_ptr + row)
        entropy = tl.load(entropy_ptr + row)
        grad_out = tl.load(grad_out_ptr + row)
        undefined_row = global_max == float("-inf")
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            log_p = _log_probs(x_ptr, cols, mask, global_max, log_sum_exp)
            grad = tl.where(log_p == float("-inf"), 0.0, tl.exp(log_p) * (log_p + entropy)) * -grad_out
            grad = tl.where(undefined_row, 0.0, grad)
            tl.store(grad_ptr + cols, grad.to(grad_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _logprobs_backward_kernel(x_ptr, max_ptr, lse_ptr, grad_sum_ptr, zero_row_ptr, index_ptr, index_grad_ptr,
                                  grad_ptr, vocab, x_inner, x_outer, x_row, num_index, BLOCK: tl.constexpr):
        # d logp(i) / d z_j = onehot(i)_j - softmax(z)_j, summed over the requested indices.
        row = tl.program_id(0).to(tl.int64)
        x_ptr += _row_start(row, x_inner, x_outer, x_row)
        grad_ptr += row * vocab
        index_ptr += row * num_index
        index_grad_ptr += row * num_index
        global_max = tl.load(max_ptr + row)
        log_sum_exp = tl.load(lse_ptr + row)
        grad_sum = tl.load(grad_sum_ptr + row)
        zero_row = tl.load(zero_row_ptr + row) != 0
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            log_p = _log_probs(x_ptr, cols, mask, global_max, log_sum_exp)
            grad = tl.where(zero_row, 0.0, tl.exp(log_p) * -grad_sum)
            # The one-hot terms are added in fp32 before the single cast, as in the eager path.
            for k in range(0, num_index):
                local_index = tl.load(index_ptr + k)
                if (local_index >= start) & (local_index < start + BLOCK):
                    grad += tl.where(cols == local_index, tl.load(index_grad_ptr + k), 0.0)
            tl.store(grad_ptr + cols, grad.to(grad_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _kl_stats_kernel(a_ptr, b_ptr, out_ptr, num_rows, vocab, a_inner, a_outer, a_row, b_inner, b_outer, b_row,
                         BLOCK: tl.constexpr):
        # For KL(A || B) over the local shard: both maxima and sum-exps, whether A has support that
        # B masks out, and sum_j exp(a_j - m_a) * ((a_j - m_a) - (b_j - m_b)). Raising the maxima
        # scales that sum by exp(m_a - m_a') and shifts the centred log-ratio by -(dm_a - dm_b).
        row = tl.program_id(0).to(tl.int64)
        a_ptr += _row_start(row, a_inner, a_outer, a_row)
        b_ptr += _row_start(row, b_inner, b_outer, b_row)
        a_max = tl.full([BLOCK], float("-inf"), tl.float32)
        b_max = tl.full([BLOCK], float("-inf"), tl.float32)
        a_sum = tl.zeros([BLOCK], tl.float32)
        b_sum = tl.zeros([BLOCK], tl.float32)
        weighted = tl.zeros([BLOCK], tl.float32)
        mismatch = tl.zeros([BLOCK], tl.int32)
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            za = tl.load(a_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
            zb = tl.load(b_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
            new_a_max = tl.maximum(a_max, za)
            new_b_max = tl.maximum(b_max, zb)
            a_seen = a_max != float("-inf")
            b_seen = b_max != float("-inf")
            a_scale = tl.where(a_seen, tl.exp(a_max - new_a_max), 0.0)
            b_scale = tl.where(b_seen, tl.exp(b_max - new_b_max), 0.0)
            # A sum that has seen A but no finite B holds only mismatched terms, which the final
            # +inf replaces, so dropping it keeps the -inf shift from producing NaN.
            shift = (new_a_max - a_max) - (new_b_max - b_max)
            weighted = tl.where(b_seen & (a_scale > 0), a_scale * (weighted - shift * a_sum), 0.0)
            a_sum = a_sum * a_scale
            b_sum = b_sum * b_scale
            a_live = za != float("-inf")
            b_live = zb != float("-inf")
            ea = tl.where(a_live, tl.exp(za - new_a_max), 0.0)
            a_sum += ea
            b_sum += tl.where(b_live, tl.exp(zb - new_b_max), 0.0)
            # Tokens A masks out contribute nothing; where B masks out a token A supports, KL(A || B) is
            # infinite even if p_A underflows to 0, so this is decided on the logits.
            weighted += tl.where(b_live & (ea > 0), ea * ((za - new_a_max) - (zb - new_b_max)), 0.0)
            mismatch |= (a_live & ~b_live).to(tl.int32)
            a_max = new_a_max
            b_max = new_b_max
        # Re-centre every lane on the row maxima before summing the lanes.
        row_a_max = tl.max(a_max, axis=0)
        row_b_max = tl.max(b_max, axis=0)
        a_seen = a_max != float("-inf")
        b_seen = b_max != float("-inf")
        a_scale = tl.where(a_seen, tl.exp(a_max - row_a_max), 0.0)
        b_scale = tl.where(b_seen, tl.exp(b_max - row_b_max), 0.0)
        shift = (row_a_max - a_max) - (row_b_max - b_max)
        weighted = tl.where(b_seen & (a_scale > 0), a_scale * (weighted - shift * a_sum), 0.0)
        a_sum = a_sum * a_scale
        b_sum = b_sum * b_scale
        tl.store(out_ptr + row, row_a_max)
        tl.store(out_ptr + num_rows + row, row_b_max)
        tl.store(out_ptr + 2 * num_rows + row, tl.max(mismatch, axis=0).to(tl.float32))
        tl.store(out_ptr + 3 * num_rows + row, tl.sum(a_sum, axis=0))
        tl.store(out_ptr + 4 * num_rows + row, tl.sum(b_sum, axis=0))
        tl.store(out_ptr + 5 * num_rows + row, tl.sum(weighted, axis=0))

    @triton.jit
    def _kl_backward_kernel(a_ptr, b_ptr, a_max_ptr, a_lse_ptr, b_max_ptr, b_lse_ptr, kl_ptr, grad_out_ptr, a_grad_ptr,
                            b_grad_ptr, vocab, a_inner, a_outer, a_row, b_inner, b_outer, b_row,
                            NEED_A_GRAD: tl.constexpr, NEED_B_GRAD: tl.constexpr, BLOCK: tl.constexpr):
        # For KL(A || B): dKL/dz_A = p_A * (log p_A - log p_B - KL) and dKL/dz_B = p_B - p_A.
        row = tl.program_id(0).to(tl.int64)
        a_ptr += _row_start(row, a_inner, a_outer, a_row)
        b_ptr += _row_start(row, b_inner, b_outer, b_row)
        a_grad_ptr += row * vocab
        b_grad_ptr += row * vocab
        a_max = tl.load(a_max_ptr + row)
        a_lse = tl.load(a_lse_ptr + row)
        b_max = tl.load(b_max_ptr + row)
        b_lse = tl.load(b_lse_ptr + row)
        kl = tl.load(kl_ptr + row)
        grad_out = tl.load(grad_out_ptr + row)
        # KL is undefined when either distribution is; both sides then get zero gradient.
        undefined_row = (a_max == float("-inf")) | (b_max == float("-inf"))
        for start in range(0, vocab, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < vocab
            a_log_p = _log_probs(a_ptr, cols, mask, a_max, a_lse)
            b_log_p = _log_probs(b_ptr, cols, mask, b_max, b_lse)
            a_p = tl.exp(a_log_p)
            if NEED_A_GRAD:
                a_grad = tl.where(a_log_p == float("-inf"), 0.0, a_p * (a_log_p - b_log_p - kl)) * grad_out
                a_grad = tl.where(undefined_row, 0.0, a_grad)
                tl.store(a_grad_ptr + cols, a_grad.to(a_grad_ptr.dtype.element_ty), mask=mask)
            if NEED_B_GRAD:
                b_grad = tl.where(undefined_row, 0.0, (tl.exp(b_log_p) - a_p) * grad_out)
                tl.store(b_grad_ptr + cols, b_grad.to(b_grad_ptr.dtype.element_ty), mask=mask)


def row_layout(logits):
    """``(inner_rows, outer_stride, row_stride)`` addressing every row of ``logits`` in place, or None.

    Two levels of row strides cover contiguous tensors and slices along a sequence dimension such
    as ``logits[:, :-1]``, which cannot be viewed as ``[rows, vocab]`` without a copy.
    """
    if logits.shape[-1] > 1 and logits.stride(-1) != 1:
        return None
    sizes, strides = [], []
    for size, stride in zip(logits.shape[:-1], logits.stride()[:-1]):
        if size == 1:
            continue
        if sizes and strides[-1] == stride * size:
            sizes[-1] *= size
            strides[-1] = stride
        else:
            sizes.append(size)
            strides.append(stride)
    if len(sizes) > 2:
        return None
    sizes = [1] * (2 - len(sizes)) + sizes
    strides = [0] * (2 - len(strides)) + strides
    return sizes[1], strides[0], strides[1]


def _num_rows(logits):
    return logits.shape[:-1].numel()


def _launch(kernel, logits, *args, max_block=None, **kwargs):
    # An empty local shard still launches: the loops run zero times and the statistics kernels write -inf and 0.
    if _num_rows(logits) == 0:
        return
    vocab = logits.shape[-1]
    block = min(max_block or _MAX_BLOCK, max(16, triton.next_power_of_2(vocab)))
    kernel[(_num_rows(logits), )](*args, **kwargs, BLOCK=block, num_warps=_NUM_WARPS)


def local_softmax_stats(logits):
    """Per-row max of a ``[..., local_vocab]`` shard and ``sum(exp(z - max))`` relative to it."""
    local_max = torch.empty(_num_rows(logits), dtype=torch.float32, device=logits.device)
    local_sum = torch.empty_like(local_max)
    _launch(_local_softmax_stats_kernel, logits, logits, local_max, local_sum, logits.shape[-1], *row_layout(logits))
    return local_max, local_sum


def entropy_stats(logits):
    """``[3, rows]``: the local max, the sum-exp and the exp-weighted sum of centred logits."""
    num_rows = _num_rows(logits)
    stats = torch.empty(3, num_rows, dtype=torch.float32, device=logits.device)
    _launch(_entropy_stats_kernel,
            logits,
            logits,
            stats,
            num_rows,
            logits.shape[-1],
            *row_layout(logits),
            max_block=_STATS_MAX_BLOCK)
    return stats


def entropy_backward(logits, global_max, log_sum_exp, entropy, grad_output):
    grad = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
    _launch(_entropy_backward_kernel, logits, logits, global_max, log_sum_exp, entropy, grad_output, grad,
            logits.shape[-1], *row_layout(logits))
    return grad


def logprobs_backward(logits, global_max, log_sum_exp, grad_sum, zero_grad_row, local_index, index_grad):
    grad = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
    _launch(_logprobs_backward_kernel,
            logits,
            logits,
            global_max,
            log_sum_exp,
            grad_sum,
            zero_grad_row.to(torch.int8),
            local_index.contiguous(),
            index_grad.contiguous(),
            grad,
            logits.shape[-1],
            *row_layout(logits),
            num_index=local_index.shape[-1])
    return grad


def kl_stats(a_logits, b_logits):
    """``[6, rows]`` local statistics of KL(A || B), MAX-reduced rows first, then SUM-reduced rows.

    Rows 0-2 are A's max, B's max and a support-mismatch flag; rows 3-5 are A's and B's sum-exp
    and the A-weighted sum of centred log-ratios.
    """
    num_rows = _num_rows(a_logits)
    stats = torch.empty(6, num_rows, dtype=torch.float32, device=a_logits.device)
    _launch(_kl_stats_kernel,
            a_logits,
            a_logits,
            b_logits,
            stats,
            num_rows,
            a_logits.shape[-1],
            *row_layout(a_logits),
            *row_layout(b_logits),
            max_block=_STATS_MAX_BLOCK)
    return stats


def kl_backward(a_logits, b_logits, a_max, a_lse, b_max, b_lse, kl, grad_output, need_a_grad, need_b_grad):
    a_grad = torch.empty(a_logits.shape, dtype=a_logits.dtype, device=a_logits.device) if need_a_grad else None
    b_grad = torch.empty(b_logits.shape, dtype=b_logits.dtype, device=b_logits.device) if need_b_grad else None
    # Unused gradient pointers still need a valid tensor; the constexpr flags skip their stores.
    _launch(_kl_backward_kernel,
            a_logits,
            a_logits,
            b_logits,
            a_max,
            a_lse,
            b_max,
            b_lse,
            kl,
            grad_output,
            a_logits if a_grad is None else a_grad,
            b_logits if b_grad is None else b_grad,
            a_logits.shape[-1],
            *row_layout(a_logits),
            *row_layout(b_logits),
            NEED_A_GRAD=need_a_grad,
            NEED_B_GRAD=need_b_grad)
    return a_grad, b_grad
