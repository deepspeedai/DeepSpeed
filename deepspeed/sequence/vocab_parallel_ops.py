# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-token log-probability, entropy, and KL divergence ops over vocabulary-sharded logits.

These ops let RL and on-policy distillation losses run directly on the local logits shard
produced by a vocabulary-parallel LM head, without all-gathering the full ``[..., vocab]``
logits on every tensor-parallel rank.

Every op returns a per-token tensor that is replicated across the tensor-parallel group, so
masking and reduction are left to the caller. As with vocabulary-parallel cross entropy, the
backward pass assumes every tensor-parallel rank back-propagates the same loss, which lets the
gradient of each local logit be computed without communication.

Teacher and student logits passed to the KL op must be sharded identically: same
tensor-parallel group and same vocabulary range on every rank.

A row whose logits are all ``-inf`` has no defined distribution: entropy and KL return NaN for
it, but its logits get zero gradient so that callers can mask such rows (e.g. padding) out.
"""

import torch
from torch.autograd.function import once_differentiable

import deepspeed.comm as dist
from deepspeed.sequence.cross_entropy import _resolve_vocab_metadata

# Elementwise work is done on row chunks of at most this many logits so that fp32 temporaries
# stay bounded no matter how many tokens are processed; only per-token statistics span all rows.
_CHUNK_NUMEL = 1 << 24


def _tp_active(tp_group):
    return tp_group is not None and dist.get_world_size(tp_group) > 1


def _all_reduce(tensor, op, tp_group):
    if _tp_active(tp_group):
        dist.all_reduce(tensor, op=op, group=tp_group)
    return tensor


def _row_chunks(logits_2d):
    rows_per_chunk = max(1, _CHUNK_NUMEL // max(1, logits_2d.shape[-1]))
    for start in range(0, logits_2d.shape[0], rows_per_chunk):
        yield slice(start, start + rows_per_chunk)


def _softmax_stats(logits_2d_list, tp_group):
    """Global fp32 softmax statistics over the sharded vocabulary of each ``[rows, local_vocab]`` tensor.

    All tensors share one MAX and one SUM all-reduce. Returns ``(global_max, log_sum_exp)``, each
    ``[len(logits_2d_list), rows]``. They are kept separate rather than summed into a single
    logsumexp because adding a small normalizer to a large max rounds it away in fp32.
    """
    num_rows = logits_2d_list[0].shape[0]
    device = logits_2d_list[0].device
    global_max = torch.empty(len(logits_2d_list), num_rows, dtype=torch.float32, device=device)
    for i, logits in enumerate(logits_2d_list):
        # An empty local shard still joins both collectives, contributing -inf and 0, so no TP
        # rank is left waiting; amax itself rejects an empty dimension.
        if logits.shape[-1] == 0:
            global_max[i].fill_(float("-inf"))
            continue
        for rows in _row_chunks(logits):
            global_max[i, rows] = logits[rows].amax(dim=-1).float()
    _all_reduce(global_max, dist.ReduceOp.MAX, tp_group)

    sum_exp = torch.empty_like(global_max)
    for i, logits in enumerate(logits_2d_list):
        for rows in _row_chunks(logits):
            sum_exp[i, rows] = torch.exp(logits[rows].float() - global_max[i, rows].unsqueeze(-1)).sum(dim=-1)
    _all_reduce(sum_exp, dist.ReduceOp.SUM, tp_group)
    return global_max, torch.log(sum_exp)


def _log_softmax(logits, global_max, log_sum_exp):
    return torch.sub(logits.float(), global_max.unsqueeze(-1)).sub_(log_sum_exp.unsqueeze(-1))


def _weighted_by_probs(log_probs, values, probs=None):
    # Tokens masked to -inf contribute nothing, even where ``values`` is infinite; without this,
    # ``0 * -inf`` turns the whole row into NaN. The test is on the log-probability, not the
    # probability, because a finite log-probability may still underflow to 0 after ``exp``.
    if probs is None:
        probs = torch.exp(log_probs)
    return (probs * values).masked_fill_(log_probs == float("-inf"), 0.0)


def _kl_terms(a_log_probs, b_log_probs):
    terms = _weighted_by_probs(a_log_probs, a_log_probs - b_log_probs)
    # Where A has support that B masks out, KL(A || B) is infinite even if p_A underflowed to 0,
    # which would otherwise give 0 * inf = NaN.
    support_mismatch = (b_log_probs == float("-inf")) & (a_log_probs != float("-inf"))
    return terms.masked_fill_(support_mismatch, float("inf"))


def _undefined_rows(global_max):
    # A fully -inf row has no softmax; its gradient is zeroed instead of propagating NaN.
    return (global_max == float("-inf")).unsqueeze(-1)


def _flatten(vocab_parallel_logits):
    # reshape(-1, 0) is ambiguous for an empty local shard, so the row count is given explicitly.
    num_rows = vocab_parallel_logits.shape[:-1].numel()
    return vocab_parallel_logits.reshape(num_rows, vocab_parallel_logits.shape[-1])


class _VocabParallelLogProbs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, index, tp_group, vocab_start_index, ignore_index):
        logits = _flatten(vocab_parallel_logits)
        index = index.reshape(logits.shape[0], index.shape[-1])
        local_vocab_size = logits.shape[-1]
        global_max, log_sum_exp = (stat[0] for stat in _softmax_stats([logits], tp_group))

        valid_index = index != ignore_index
        local_index = index - vocab_start_index
        index_in_partition = valid_index & (local_index >= 0) & (local_index < local_vocab_size)
        local_index = local_index.clamp(min=0, max=local_vocab_size - 1)
        shifted_logits = logits.gather(-1, local_index).float() - global_max.unsqueeze(-1)
        shifted_logits = torch.where(index_in_partition, shifted_logits, torch.zeros_like(shifted_logits))
        _all_reduce(shifted_logits, dist.ReduceOp.SUM, tp_group)

        logprobs = shifted_logits - log_sum_exp.unsqueeze(-1)
        logprobs = torch.where(valid_index, logprobs, torch.zeros_like(logprobs))

        ctx.save_for_backward(vocab_parallel_logits, global_max, log_sum_exp, local_index, index_in_partition,
                              valid_index)
        return logprobs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        vocab_parallel_logits, global_max, log_sum_exp, local_index, index_in_partition, valid_index = ctx.saved_tensors
        logits = _flatten(vocab_parallel_logits)
        grad_output = grad_output.reshape(valid_index.shape).float() * valid_index
        index_grad = grad_output * index_in_partition
        grad_sum = grad_output.sum(dim=-1, keepdim=True)
        # A row whose indices are all ignored has a constant output, so its gradient is exactly 0.
        # Multiplying its softmax by 0 is not enough: a fully masked row's softmax is NaN.
        zero_grad_row = ~valid_index.any(dim=-1, keepdim=True) | _undefined_rows(global_max)

        # d logp(i) / d z_j = onehot(i)_j - softmax(z)_j, summed over the requested indices.
        grad_logits = torch.empty_like(logits)
        for rows in _row_chunks(logits):
            grad_chunk = torch.exp(_log_softmax(logits[rows], global_max[rows], log_sum_exp[rows])) * -grad_sum[rows]
            grad_chunk.masked_fill_(zero_grad_row[rows], 0.0)
            grad_chunk.scatter_add_(-1, local_index[rows], index_grad[rows])
            grad_logits[rows] = grad_chunk
        return grad_logits.view_as(vocab_parallel_logits), None, None, None, None


class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, tp_group):
        logits = _flatten(vocab_parallel_logits)
        global_max, log_sum_exp = (stat[0] for stat in _softmax_stats([logits], tp_group))
        entropy = torch.empty_like(global_max)
        for rows in _row_chunks(logits):
            log_probs = _log_softmax(logits[rows], global_max[rows], log_sum_exp[rows])
            entropy[rows] = -_weighted_by_probs(log_probs, log_probs).sum(dim=-1)
        _all_reduce(entropy, dist.ReduceOp.SUM, tp_group)

        ctx.save_for_backward(vocab_parallel_logits, global_max, log_sum_exp, entropy)
        # Callers may mask the result in place, so it must not alias the saved entropy.
        return entropy.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        vocab_parallel_logits, global_max, log_sum_exp, entropy = ctx.saved_tensors
        logits = _flatten(vocab_parallel_logits)
        grad_output = grad_output.reshape(-1, 1).float()
        undefined_row = _undefined_rows(global_max)

        # dH / dz_j = -p_j * (log p_j + H)
        grad_logits = torch.empty_like(logits)
        for rows in _row_chunks(logits):
            log_probs = _log_softmax(logits[rows], global_max[rows], log_sum_exp[rows])
            grad_chunk = _weighted_by_probs(log_probs, log_probs + entropy[rows].unsqueeze(-1))
            grad_logits[rows] = grad_chunk.mul_(-grad_output[rows]).masked_fill_(undefined_row[rows], 0.0)
        return grad_logits.view_as(vocab_parallel_logits), None


def _order_kl_arguments(student_log_probs, teacher_log_probs, reverse):
    # KL(A || B) weights the log-ratio by A: the teacher for forward KL, the student for reverse KL.
    if reverse:
        return student_log_probs, teacher_log_probs
    return teacher_log_probs, student_log_probs


class _VocabParallelKLDiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, student_logits, teacher_logits, reverse, tp_group):
        student = _flatten(student_logits)
        teacher = _flatten(teacher_logits)
        global_max, log_sum_exp = _softmax_stats([student, teacher], tp_group)

        kl = torch.empty_like(global_max[0])
        for rows in _row_chunks(student):
            student_log_probs = _log_softmax(student[rows], global_max[0, rows], log_sum_exp[0, rows])
            teacher_log_probs = _log_softmax(teacher[rows], global_max[1, rows], log_sum_exp[1, rows])
            a_log_probs, b_log_probs = _order_kl_arguments(student_log_probs, teacher_log_probs, reverse)
            kl[rows] = _kl_terms(a_log_probs, b_log_probs).sum(dim=-1)
        _all_reduce(kl, dist.ReduceOp.SUM, tp_group)

        ctx.reverse = reverse
        ctx.save_for_backward(student_logits, teacher_logits, global_max, log_sum_exp, kl)
        # Callers may mask the result in place, so it must not alias the saved KL.
        return kl.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        student_logits, teacher_logits, global_max, log_sum_exp, kl = ctx.saved_tensors
        student = _flatten(student_logits)
        teacher = _flatten(teacher_logits)
        grad_output = grad_output.reshape(-1, 1).float()
        need_student_grad, need_teacher_grad = ctx.needs_input_grad[:2]
        grad_student = torch.empty_like(student) if need_student_grad else None
        grad_teacher = torch.empty_like(teacher) if need_teacher_grad else None
        a_grad, b_grad = _order_kl_arguments(grad_student, grad_teacher, ctx.reverse)
        # KL is undefined when either distribution is; both sides then get zero gradient.
        undefined_row = _undefined_rows(global_max).any(dim=0)

        for rows in _row_chunks(student):
            student_log_probs = _log_softmax(student[rows], global_max[0, rows], log_sum_exp[0, rows])
            teacher_log_probs = _log_softmax(teacher[rows], global_max[1, rows], log_sum_exp[1, rows])
            a_log_probs, b_log_probs = _order_kl_arguments(student_log_probs, teacher_log_probs, ctx.reverse)
            a_probs = torch.exp(a_log_probs)
            # For KL(A || B): dKL/dz_A = p_A * (log p_A - log p_B - KL) and dKL/dz_B = p_B - p_A.
            if a_grad is not None:
                log_ratio = a_log_probs - b_log_probs - kl[rows].unsqueeze(-1)
                a_grad_chunk = _weighted_by_probs(a_log_probs, log_ratio, a_probs).mul_(grad_output[rows])
                a_grad[rows] = a_grad_chunk.masked_fill_(undefined_row[rows], 0.0)
            if b_grad is not None:
                b_grad_chunk = (torch.exp(b_log_probs) - a_probs).mul_(grad_output[rows])
                b_grad[rows] = b_grad_chunk.masked_fill_(undefined_row[rows], 0.0)

        if grad_student is not None:
            grad_student = grad_student.view_as(student_logits)
        if grad_teacher is not None:
            grad_teacher = grad_teacher.view_as(teacher_logits)
        return grad_student, grad_teacher, None, None


def vocab_parallel_logprobs(vocab_parallel_logits,
                            index,
                            tp_group=None,
                            vocab_start_index=None,
                            vocab_end_index=None,
                            global_vocab_size=None,
                            ignore_index=-100):
    """Log-probabilities of the requested token ids under vocabulary-sharded logits.

    ``index`` either matches ``vocab_parallel_logits.shape[:-1]`` (one token per position, e.g.
    sampled tokens for policy-gradient losses) or adds a trailing ``k`` dimension (e.g. a
    teacher's top-k token ids for sparse distillation). Ids equal to ``ignore_index`` yield 0.
    ``index`` must be identical on every tensor-parallel rank. The result has the same shape as
    ``index`` and is in fp32.

    Each call collectively validates the vocabulary shard layout. A caller that calls this
    repeatedly for a fixed shard can validate it once and pass the resulting ``global_vocab_size``
    with explicit shard bounds to skip that collective; the value is trusted as-is, as in
    ``vocab_parallel_cross_entropy``. ``index`` is still range-checked on every call.
    """
    single_index = index.shape == vocab_parallel_logits.shape[:-1]
    if not single_index and index.shape[:-1] != vocab_parallel_logits.shape[:-1]:
        raise ValueError("index must match the non-vocabulary dimensions of vocab_parallel_logits, "
                         "optionally followed by a top-k dimension")
    if (vocab_start_index is None) != (vocab_end_index is None):
        raise ValueError("vocab_start_index and vocab_end_index must be provided together")

    if global_vocab_size is not None:
        if vocab_start_index is None:
            raise ValueError("global_vocab_size requires explicit vocab_start_index and vocab_end_index")
    else:
        vocab_start_index, _, global_vocab_size = _resolve_vocab_metadata(vocab_parallel_logits.shape[-1],
                                                                          vocab_start_index, vocab_end_index, tp_group,
                                                                          vocab_parallel_logits.device)
    index = index.to(dtype=torch.long)
    invalid_index = (index != ignore_index) & ((index < 0) | (index >= global_vocab_size))
    if invalid_index.any().item():
        raise ValueError(f"index is out of range for vocabulary size {global_vocab_size}")

    if single_index:
        index = index.unsqueeze(-1)
    logprobs = _VocabParallelLogProbs.apply(vocab_parallel_logits, index, tp_group, vocab_start_index, ignore_index)
    return logprobs.view(index.shape).squeeze(-1) if single_index else logprobs.view(index.shape)


def vocab_parallel_entropy(vocab_parallel_logits, tp_group=None):
    """Per-token entropy of the softmax over vocabulary-sharded logits, in fp32."""
    entropy = _VocabParallelEntropy.apply(vocab_parallel_logits, tp_group)
    return entropy.view(vocab_parallel_logits.shape[:-1])


def vocab_parallel_kl_div(student_logits, teacher_logits, reverse=False, tp_group=None):
    """Per-token KL divergence between identically sharded student and teacher logits, in fp32.

    Returns the forward ``KL(teacher || student)`` by default, or the reverse
    ``KL(student || teacher)`` (mode-seeking, the usual on-policy distillation objective) when
    ``reverse=True``. Apply any temperature to both logits first. Gradients flow to the teacher
    logits too unless they are detached. As with ``torch.nn.functional.kl_div`` in fp32, terms whose
    probability underflows to 0 contribute nothing unless the other distribution masks that token
    to ``-inf``, in which case the KL is ``+inf``.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape and vocabulary sharding")
    kl = _VocabParallelKLDiv.apply(student_logits, teacher_logits, bool(reverse), tp_group)
    return kl.view(student_logits.shape[:-1])
