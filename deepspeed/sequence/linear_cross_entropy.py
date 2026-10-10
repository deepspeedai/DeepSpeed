# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Fused LM-head projection and cross entropy over a vocabulary-parallel weight.

A vocabulary-parallel LM head still materializes ``[tokens, vocab / tp]`` logits per rank.
This module fuses the projection into the loss. Forward processes tokens in row chunks and
keeps only per-token softmax statistics. Backward recomputes the logits one vocabulary block
at a time, so each weight-gradient block comes from a single GEMM over all tokens.
"""

import math

import torch
from torch.autograd.function import once_differentiable

import deepspeed.comm as dist
from deepspeed.sequence.cross_entropy import _global_sp_sum, _resolve_vocab_metadata


def _tp_active(tp_group):
    return tp_group is not None and dist.get_world_size(tp_group) > 1


def _default_chunk_rows(num_tokens, global_vocab_size, tp_world_size, hidden_size):
    # Keep each fp32 logits block about the size of the hidden states, which bounds the
    # extra peak memory independently of the vocabulary size. Derived from the global shard
    # layout rather than the local shard so that every TP rank issues the same collectives
    # even when the vocabulary is split unevenly.
    max_local_vocab_size = math.ceil(global_vocab_size / tp_world_size)
    num_chunks = max(1, math.ceil(max_local_vocab_size / hidden_size))
    return max(1, math.ceil(num_tokens / num_chunks))


# Devices without GEMMs that take low-precision inputs and produce fp32 outputs (e.g. CPU).
_MIXED_GEMM_UNSUPPORTED = set()
_MIXED_GEMM_ERRORS = (RuntimeError, TypeError, NotImplementedError)


def _accumulate_matmul(accumulator, mat1, mat2):
    """``accumulator += mat1 @ mat2`` with ``accumulator`` kept in its own (fp32) dtype.

    Accumulating block gradients in a low-precision buffer drops later contributions once the
    sum grows, so the gradient would depend on the block size.
    """
    if mat1.dtype == accumulator.dtype:
        accumulator.addmm_(mat1, mat2)
        return
    device_type = accumulator.device.type
    if device_type not in _MIXED_GEMM_UNSUPPORTED:
        try:
            torch.addmm(accumulator, mat1, mat2, out_dtype=accumulator.dtype, out=accumulator)
            return
        except _MIXED_GEMM_ERRORS:
            _MIXED_GEMM_UNSUPPORTED.add(device_type)
    accumulator.add_(torch.mm(mat1, mat2))


def _fp32_logits(hidden, weight, bias=None, row_offset=None):
    """``hidden @ weight.T + bias + row_offset`` in fp32; ``row_offset`` has shape ``[tokens, 1]``.

    Producing fp32 straight from the GEMM, with the offset applied in its epilogue, saves
    separate upcast and add passes over the ``[tokens, vocab]`` block.
    """
    device_type = hidden.device.type
    if hidden.dtype != torch.float32 and device_type not in _MIXED_GEMM_UNSUPPORTED:
        try:
            if bias is not None:
                logits = torch.addmm(bias.float(), hidden, weight.t(), out_dtype=torch.float32)
            elif row_offset is not None:
                return torch.addmm(row_offset, hidden, weight.t(), out_dtype=torch.float32)
            else:
                return torch.mm(hidden, weight.t(), out_dtype=torch.float32)
            return logits if row_offset is None else logits.add_(row_offset)
        except _MIXED_GEMM_ERRORS:
            _MIXED_GEMM_UNSUPPORTED.add(device_type)
    # Add the bias inside the GEMM as F.linear does, so a large product that the bias cancels
    # does not overflow a low-precision output first.
    logits = torch.nn.functional.linear(hidden, weight, bias).float()
    return logits if row_offset is None else logits.add_(row_offset)


class _VocabParallelLinearCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, weight, bias, target, tp_group, vocab_start_index, vocab_end_index, ignore_index,
                chunk_rows, compute_dtype):
        num_tokens = hidden.shape[0]
        local_vocab_size = weight.shape[0]
        tp_active = _tp_active(tp_group)
        # Mirror what autocast would do for ``F.linear``: run the GEMMs in one compute dtype even
        # when activations and parameters differ (e.g. bf16 activations with fp32 weights).
        hidden_compute = hidden.to(compute_dtype)
        weight_compute = weight.to(compute_dtype)
        bias_compute = None if bias is None else bias.to(compute_dtype)

        valid = target != ignore_index
        in_shard = valid & (target >= vocab_start_index) & (target < vocab_end_index)
        local_target = (target - vocab_start_index).clamp(0, local_vocab_size - 1)

        loss = torch.empty(num_tokens, dtype=torch.float32, device=hidden.device)
        row_max = torch.empty_like(loss)
        row_sum_exp = torch.empty_like(loss)
        for start in range(0, num_tokens, chunk_rows):
            end = min(start + chunk_rows, num_tokens)
            logits = _fp32_logits(hidden_compute[start:end], weight_compute, bias_compute)

            chunk_max = logits.amax(dim=-1)
            if tp_active:
                dist.all_reduce(chunk_max, op=dist.ReduceOp.MAX, group=tp_group)
            logits.sub_(chunk_max.unsqueeze(-1))
            target_logit = logits.gather(-1, local_target[start:end].unsqueeze(-1)).squeeze(-1)
            target_logit = torch.where(in_shard[start:end], target_logit, torch.zeros_like(target_logit))
            stats = torch.stack([logits.exp_().sum(dim=-1), target_logit], dim=-1)
            if tp_active:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=tp_group)
            sum_exp, target_logit = stats.unbind(-1)

            row_max[start:end] = chunk_max
            row_sum_exp[start:end] = sum_exp
            loss[start:end] = torch.where(valid[start:end], sum_exp.log() - target_logit, torch.zeros_like(sum_exp))

        # The max and the sum of exponentials are kept apart: folding them into one log-sum-exp
        # loses the small normalizer to rounding when the logits share a large offset.
        ctx.save_for_backward(hidden, weight, bias, valid, in_shard, local_target, row_max, row_sum_exp)
        ctx.tp_group = tp_group
        ctx.compute_dtype = compute_dtype
        # Backward recomputes the logits in vocabulary blocks with the same number of elements
        # as a forward chunk, so both passes have the same peak.
        num_chunks = math.ceil(num_tokens / chunk_rows) if num_tokens else 1
        ctx.block_cols = max(1, math.ceil(local_vocab_size / num_chunks))
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        hidden, weight, bias, valid, in_shard, local_target, row_max, row_sum_exp = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        needs_hidden, needs_weight, needs_bias = ctx.needs_input_grad[:3]
        local_vocab_size = weight.shape[0]

        hidden_compute = hidden.to(compute_dtype)
        token_grad = torch.where(valid, grad_loss.float(), torch.zeros_like(row_max))
        # softmax * upstream gradient = exp(logits - max) * (upstream gradient / sum_exp).
        prob_scale = (token_grad / row_sum_exp).unsqueeze(-1)
        neg_row_max = -row_max.unsqueeze(-1)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        # The hidden gradient sums over every vocabulary block, so it is accumulated in fp32.
        grad_hidden = torch.zeros_like(hidden, dtype=torch.float32) if needs_hidden else None
        grad_weight = torch.empty_like(weight) if needs_weight else None
        grad_bias = torch.empty_like(bias) if needs_bias else None

        for start in range(0, local_vocab_size, ctx.block_cols):
            end = min(start + ctx.block_cols, local_vocab_size)
            weight_block = weight[start:end].to(compute_dtype)
            bias_block = None if bias is None else bias[start:end].to(compute_dtype)
            logits = _fp32_logits(hidden_compute, weight_block, bias_block, neg_row_max)

            # d(loss) / d(logits) = (softmax - onehot) * upstream gradient, one block at a time.
            grad_logits = logits.exp_().mul_(prob_scale)
            in_block = in_shard & (local_target >= start) & (local_target < end)
            block_rows = rows[in_block]
            grad_logits[block_rows, local_target[in_block] - start] -= token_grad[in_block]

            if grad_bias is not None:
                grad_bias[start:end] = grad_logits.sum(dim=0)
            grad_logits = grad_logits.to(compute_dtype)
            if grad_weight is not None:
                grad_weight[start:end] = grad_logits.t() @ hidden_compute
            if grad_hidden is not None:
                _accumulate_matmul(grad_hidden, grad_logits, weight_block)

        if grad_hidden is not None:
            grad_hidden = grad_hidden.to(hidden.dtype)
            # Each rank only saw its vocabulary shard, so the input gradient is a partial sum, the
            # same reduction the column-parallel LM head performs in its own backward.
            if _tp_active(ctx.tp_group):
                dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None, None


def vocab_parallel_linear_cross_entropy(hidden,
                                        weight,
                                        target,
                                        bias=None,
                                        tp_group=None,
                                        sp_group=None,
                                        vocab_start_index=None,
                                        vocab_end_index=None,
                                        ignore_index=-100,
                                        reduction="mean",
                                        chunk_size=None):
    """Cross entropy of ``hidden @ weight.T (+ bias)`` without materializing the logits.

    ``weight`` (and ``bias``) hold this rank's rows of a vocabulary-parallel LM head, for
    example ``VocabParallelLinear.weight``. ``hidden`` must be identical on every TP rank, as
    for a column-parallel layer; its gradient is all-reduced over ``tp_group`` in backward.

    ``chunk_size`` sets the number of tokens per forward chunk; backward recomputes the
    logits in vocabulary blocks of the same size. By default each fp32 logits block is about
    as large as ``hidden``. ``reduction="none"`` returns the per-token losses in the shape of
    ``target``, with zeros at ignored positions, and does not reduce over ``sp_group``.
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    if hidden.shape[:-1] != target.shape:
        raise ValueError("hidden and target must have matching non-hidden dimensions")
    if weight.dim() != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("weight must have shape [local_vocab, hidden]")
    if (vocab_start_index is None) != (vocab_end_index is None):
        raise ValueError("vocab_start_index and vocab_end_index must be provided together")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    vocab_start_index, vocab_end_index, global_vocab_size = _resolve_vocab_metadata(
        weight.shape[0], vocab_start_index, vocab_end_index, tp_group, hidden.device)
    target_shape = target.shape
    target = target.reshape(-1).to(dtype=torch.long)
    invalid_target = (target != ignore_index) & ((target < 0) | (target >= global_vocab_size))
    if invalid_target.any().item():
        raise ValueError(f"Target is out of range for vocabulary size {global_vocab_size}")

    hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    weight = weight.contiguous()
    num_tokens = hidden.shape[0]

    if chunk_size is None:
        tp_world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        chunk_size = _default_chunk_rows(num_tokens, global_vocab_size, tp_world_size, hidden.shape[1])
    device_type = hidden.device.type
    if torch.is_autocast_enabled(device_type):
        compute_dtype = torch.get_autocast_dtype(device_type)
    else:
        compute_dtype = torch.promote_types(hidden.dtype, weight.dtype)

    with torch.autocast(device_type, enabled=False):
        per_token_loss = _VocabParallelLinearCrossEntropy.apply(hidden, weight, bias, target, tp_group,
                                                                vocab_start_index, vocab_end_index, ignore_index,
                                                                chunk_size, compute_dtype)
    if reduction == "none":
        return per_token_loss.view(target_shape)

    loss = per_token_loss.sum()
    if reduction == "mean":
        valid_tokens = (target != ignore_index).sum().to(torch.float32)
        if sp_group is not None and dist.get_world_size(sp_group) > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM, group=sp_group)
        loss = loss / valid_tokens.clamp(min=1.0)
    return _global_sp_sum(loss, sp_group)
