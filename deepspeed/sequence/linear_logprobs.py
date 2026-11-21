# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Masked per-token scoring from a resident vocabulary-parallel LM head."""

import math

import torch
from torch.autograd.function import once_differentiable

import deepspeed.comm as dist
from deepspeed.comm import pg_utils
from deepspeed.sequence.cross_entropy import _resolve_vocab_metadata
from deepspeed.sequence.linear_cross_entropy import _accumulate_matmul, _autocast_dtype, _fp32_logits


class _VocabParallelLinearLogProbs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, weight, bias, target, tp_group, vocab_start, temperature, chunk_size, vocab_chunk_size,
                compute_dtype, return_entropy):
        ctx.set_materialize_grads(False)
        num_tokens = hidden.shape[0]
        maxima = hidden.new_empty(num_tokens, dtype=torch.float32)
        normalizers = torch.empty_like(maxima)
        logprobs = torch.empty_like(maxima)
        entropy = torch.empty_like(maxima) if return_entropy else None
        parallel = pg_utils.is_parallel(tp_group)

        for row in range(0, num_tokens, chunk_size):
            rows = slice(row, row + chunk_size)
            h = hidden[rows].to(compute_dtype)
            m = h.new_full((h.shape[0], ), -float("inf"), dtype=torch.float32)
            s = torch.zeros_like(m)
            w = torch.zeros_like(m) if return_entropy else None
            selected = torch.zeros_like(m)
            ids = target[rows] - vocab_start
            for col in range(0, weight.shape[0], vocab_chunk_size):
                end = min(col + vocab_chunk_size, weight.shape[0])
                b = None if bias is None else bias[col:end].to(compute_dtype)
                z = _fp32_logits(h, weight[col:end].to(compute_dtype), b).div_(temperature)
                new_m = torch.maximum(m, z.amax(-1))
                seen = m != -float("inf")
                shift = torch.where(seen, m - new_m, 0.0)
                scale = torch.where(seen, shift.exp(), 0.0)
                centered = torch.where(new_m.unsqueeze(-1) != -float("inf"), z - new_m.unsqueeze(-1), -float("inf"))
                p = centered.exp()
                if return_entropy:
                    terms = torch.where(p > 0, p * centered, 0.0)
                    w = torch.where(scale > 0, scale * (w + shift * s), 0.0) + terms.sum(-1)
                s = s * scale + p.sum(-1)
                m = new_m
                in_block = (ids >= col) & (ids < end)
                values = z.gather(1, (ids - col).clamp(0, end - col - 1).unsqueeze(-1)).squeeze(-1)
                selected = torch.where(in_block, values, selected)

            global_m = m.clone()
            if parallel:
                dist.all_reduce(global_m, op=dist.ReduceOp.MAX, group=tp_group)
            seen = m != -float("inf")
            shift = torch.where(seen, m - global_m, 0.0)
            scale = torch.where(seen, shift.exp(), 0.0)
            in_shard = (ids >= 0) & (ids < weight.shape[0])
            stats = [s * scale, torch.where(in_shard, selected - global_m, 0.0)]
            if return_entropy:
                stats.append(torch.where(scale > 0, scale * (w + shift * s), 0.0))
            stats = torch.stack(stats)
            if parallel:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=tp_group)
            log_s = stats[0].log()
            maxima[rows] = global_m
            normalizers[rows] = log_s
            logprobs[rows] = stats[1] - log_s
            if return_entropy:
                entropy[rows] = log_s - stats[2] / stats[0]

        ctx.save_for_backward(hidden, weight, bias, target, maxima, normalizers, entropy)
        ctx.tp_group = tp_group
        ctx.vocab_start = vocab_start
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.vocab_chunk_size = vocab_chunk_size
        ctx.compute_dtype = compute_dtype
        return logprobs, entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_logprobs, grad_entropy):
        hidden, weight, bias, target, maxima, normalizers, entropy = ctx.saved_tensors
        need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
        grad_hidden = torch.zeros_like(hidden, dtype=torch.float32) if need_hidden else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if need_weight else None
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if need_bias else None

        for row in range(0, hidden.shape[0], ctx.chunk_size):
            rows = slice(row, row + ctx.chunk_size)
            h = hidden[rows].to(ctx.compute_dtype)
            ids = target[rows] - ctx.vocab_start
            for col in range(0, weight.shape[0], ctx.vocab_chunk_size):
                end = min(col + ctx.vocab_chunk_size, weight.shape[0])
                weight_block = weight[col:end].to(ctx.compute_dtype)
                b = None if bias is None else bias[col:end].to(ctx.compute_dtype)
                log_p = _fp32_logits(h, weight_block, b).div_(ctx.temperature)
                log_p.sub_(maxima[rows].unsqueeze(-1)).sub_(normalizers[rows].unsqueeze(-1))
                p = log_p.exp()
                grad = torch.zeros_like(log_p)
                if grad_logprobs is not None:
                    g = grad_logprobs[rows].float()
                    grad.sub_(p * g.unsqueeze(-1))
                    in_block = (ids >= col) & (ids < end)
                    block_rows = in_block.nonzero().squeeze(-1)
                    grad[block_rows, ids[in_block] - col] += g[in_block]
                if grad_entropy is not None:
                    terms = torch.where(p > 0, p * (log_p + entropy[rows].unsqueeze(-1)), 0.0)
                    grad.sub_(terms * grad_entropy[rows].float().unsqueeze(-1))
                grad.masked_fill_((maxima[rows] == -float("inf")).unsqueeze(-1), 0.0)
                grad.div_(ctx.temperature)
                if need_bias:
                    grad_bias[col:end].add_(grad.sum(0))
                grad = grad.to(ctx.compute_dtype)
                if need_weight:
                    _accumulate_matmul(grad_weight[col:end], grad.t(), h)
                if need_hidden:
                    _accumulate_matmul(grad_hidden[rows], grad, weight_block)

        if need_hidden:
            # This call replaces the column-parallel head, including its input-gradient reduction.
            if hidden.numel() and pg_utils.is_parallel(ctx.tp_group):
                dist.all_reduce(grad_hidden, group=ctx.tp_group)
            grad_hidden = grad_hidden.to(hidden.dtype)
        if need_weight:
            grad_weight = grad_weight.to(weight.dtype)
        if need_bias:
            grad_bias = grad_bias.to(bias.dtype)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None, None, None


def vocab_parallel_linear_logprobs(hidden,
                                   weight,
                                   target,
                                   bias=None,
                                   position_mask=None,
                                   tp_group=None,
                                   vocab_start_index=None,
                                   vocab_end_index=None,
                                   temperature=1.0,
                                   return_entropy=False,
                                   ignore_index=-100,
                                   chunk_size=256,
                                   vocab_chunk_size=8192):
    """Return ``(logprobs, entropy_or_None)`` without materializing full local logits.

    ``hidden`` is ``[..., H]``, ``weight`` is ``[local_vocab, H]`` and ``target`` contains global
    vocabulary IDs with shape ``hidden.shape[:-1]``. Optional boolean ``position_mask`` has that
    same shape. Masked and ``ignore_index`` positions are skipped before projection and have zero
    outputs and zero scoring gradients. Outputs are FP32, shaped like ``target``, with no reduction.
    Entropy is computed only when ``return_entropy=True``. The caller aligns hidden states, targets
    and masks for causal prediction; the function does not shift them.

    The distribution is ``softmax(linear(hidden, weight, bias) / temperature)`` across the full
    vocabulary, not a top-k/top-p or repetition-penalized sampling distribution. Only plain resident
    linear heads are supported (no ZeRO-3 parameters or implicit model-specific logit transforms).
    Projection dtype follows autocast, or the promoted hidden/weight dtype otherwise.

    ``tp_group=None`` means no TP, even in a distributed job. TP peers must provide identical
    hidden states, targets, masks, scoring options and hidden-gradient requirements, and backpropagate
    the same loss. Vocabulary shards may be uneven or empty when explicit contiguous bounds are
    supplied. Hidden gradients are summed over TP inside this primitive: pass the head's original
    input, not a tensor already wrapped in the head's TP backward-reduction mapping.

    ``chunk_size`` bounds projected token rows, ``vocab_chunk_size`` bounds local vocabulary
    columns, and backward recomputes these tiles. Only first-order derivatives are supported.
    """
    if hidden.ndim < 2 or hidden.shape[:-1] != target.shape:
        raise ValueError("hidden and target must have matching non-hidden dimensions")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("weight must have shape [local_vocab, hidden]")
    if bias is not None and bias.shape != weight.shape[:1]:
        raise ValueError("bias must have shape [local_vocab]")
    if (vocab_start_index is None) != (vocab_end_index is None):
        raise ValueError("vocab_start_index and vocab_end_index must be provided together")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
           for size in (chunk_size, vocab_chunk_size)):
        raise ValueError("chunk_size and vocab_chunk_size must be positive integers")
    if target.dtype not in (torch.int32, torch.int64):
        raise TypeError("target must contain integer vocabulary IDs")
    if position_mask is not None and (position_mask.shape != target.shape or position_mask.dtype != torch.bool):
        raise ValueError("position_mask must be boolean and have the same shape as target")
    tensors = [hidden, weight, target]
    tensors.extend(tensor for tensor in (bias, position_mask) if tensor is not None)
    if any(tensor.device != hidden.device for tensor in tensors):
        raise ValueError("all scoring inputs must be on the same device")
    if any(not tensor.is_floating_point() for tensor in (hidden, weight) + (() if bias is None else (bias, ))):
        raise TypeError("hidden, weight and bias must be floating point")
    if any(hasattr(tensor, "ds_id") for tensor in (weight, bias) if tensor is not None):
        raise ValueError("streamed scoring requires resident head parameters; ZeRO-3 is not supported")

    start, _, vocab_size = _resolve_vocab_metadata(weight.shape[0],
                                                   vocab_start_index,
                                                   vocab_end_index,
                                                   tp_group,
                                                   hidden.device,
                                                   allow_empty_shards=True)
    if vocab_size == 0:
        raise ValueError("the global vocabulary must be nonempty")
    shape = target.shape
    target = target.reshape(-1).long()
    valid = target != ignore_index
    if position_mask is not None:
        valid = valid & position_mask.reshape(-1)
    if pg_utils.is_parallel(tp_group):
        # Check lengths before comparing per-position IDs, so a mismatched mask fails on every
        # peer rather than leaving one rank waiting in a projection-chunk collective.
        n = torch.tensor(
            [target.numel(), chunk_size, vocab_chunk_size,
             int(return_entropy),
             int(hidden.requires_grad)],
            dtype=torch.long,
            device=hidden.device)
        metadata = torch.empty(dist.get_world_size(tp_group) * n.numel(), dtype=n.dtype, device=n.device)
        dist.all_gather_into_tensor(metadata, n, group=tp_group)
        if not (metadata.view(-1, n.numel()) == n).all().item():
            raise ValueError(
                "TP peers must provide identical token counts, chunk options and hidden-gradient requirements")
        invalid = (valid & ((target < 0) | (target >= vocab_size))).any().long()
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=tp_group)
        if invalid.item():
            raise ValueError(f"target is out of range for vocabulary size {vocab_size}")
        ids = torch.where(valid, target, -1)
        lower, upper = ids.clone(), ids.clone()
        dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=tp_group)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=tp_group)
        if not torch.equal(lower, upper):
            raise ValueError("TP peers must provide identical targets and position masks")
    if (valid & ((target < 0) | (target >= vocab_size))).any().item():
        raise ValueError(f"target is out of range for vocabulary size {vocab_size}")

    indices = valid.nonzero().squeeze(-1)
    hidden = hidden.reshape(target.numel(), hidden.shape[-1]).index_select(0, indices)
    target = target.index_select(0, indices)
    compute_dtype = _autocast_dtype(hidden.device.type) or torch.promote_types(hidden.dtype, weight.dtype)
    with torch.autocast(hidden.device.type, enabled=False):
        logprobs, entropy = _VocabParallelLinearLogProbs.apply(hidden, weight, bias, target, tp_group, start,
                                                               temperature, chunk_size, vocab_chunk_size,
                                                               compute_dtype, return_entropy)
    logprobs = logprobs.new_zeros(shape.numel()).index_copy(0, indices, logprobs).view(shape)
    if entropy is not None:
        entropy = entropy.new_zeros(shape.numel()).index_copy(0, indices, entropy).view(shape)
    return logprobs, entropy
