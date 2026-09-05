# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch import nn

import deepspeed.comm as dist
from deepspeed.utils.logging import logger


class _VocabParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group, vocab_start_index, vocab_end_index, ignore_index):
        target = target.to(dtype=torch.long)
        logits = vocab_parallel_logits.float()
        local_vocab_size = logits.shape[-1]

        local_max = logits.amax(dim=-1)
        global_max = local_max.clone()
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

        exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
        global_sum_exp = exp_logits.sum(dim=-1)
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            dist.all_reduce(global_sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

        valid_target = target != ignore_index
        target_in_partition = valid_target & (target >= vocab_start_index) & (target < vocab_end_index)
        local_target = (target - vocab_start_index).clamp(min=0, max=local_vocab_size - 1)
        target_logits = logits.gather(-1, local_target.unsqueeze(-1)).squeeze(-1)
        target_logits = torch.where(target_in_partition, target_logits, torch.zeros_like(target_logits))
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_group)

        loss = torch.log(global_sum_exp) + global_max - target_logits
        loss = torch.where(valid_target, loss, torch.zeros_like(loss))

        ctx.save_for_backward(exp_logits, global_sum_exp, local_target, target_in_partition, valid_target)
        ctx.logits_dtype = vocab_parallel_logits.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        exp_logits, global_sum_exp, local_target, target_in_partition, valid_target = ctx.saved_tensors

        grad_logits = exp_logits / global_sum_exp.unsqueeze(-1)
        grad_logits.scatter_add_(-1, local_target.unsqueeze(-1),
                                 -target_in_partition.unsqueeze(-1).to(dtype=grad_logits.dtype))
        grad_logits *= valid_target.unsqueeze(-1).to(dtype=grad_logits.dtype)
        grad_logits *= grad_output.to(dtype=grad_logits.dtype).unsqueeze(-1)

        return grad_logits.to(dtype=ctx.logits_dtype), None, None, None, None, None


class _GatherSequenceLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, local_loss, sp_group):
        ctx.sp_group = sp_group
        ctx.local_sequence_size = local_loss.shape[0]

        output_shape = (ctx.local_sequence_size * dist.get_world_size(sp_group), *local_loss.shape[1:])
        gathered_loss = torch.empty(output_shape, dtype=local_loss.dtype, device=local_loss.device)
        dist.all_gather_into_tensor(gathered_loss, local_loss.contiguous(), group=sp_group)
        return gathered_loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty((ctx.local_sequence_size, *grad_output.shape[1:]),
                                 dtype=grad_output.dtype,
                                 device=grad_output.device)
        dist.reduce_scatter_fn(grad_input, grad_output.contiguous(), group=ctx.sp_group)
        return grad_input, None


def _global_sp_sum(local_value, sp_group):
    if sp_group is None or dist.get_world_size(sp_group) == 1:
        return local_value

    global_value = local_value.detach().clone()
    dist.all_reduce(global_value, op=dist.ReduceOp.SUM, group=sp_group)
    return local_value + (global_value - local_value.detach())


def _validate_vocab_shard_bounds(local_vocab_size, vocab_start_index, vocab_end_index, tp_group, device):
    tp_world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    if tp_world_size == 1:
        if vocab_start_index != 0 or vocab_end_index != local_vocab_size:
            raise ValueError("Vocabulary shard bounds must cover the complete local vocabulary when TP is disabled")
        return vocab_end_index

    local_metadata = torch.tensor([vocab_start_index, vocab_end_index, local_vocab_size],
                                  dtype=torch.long,
                                  device=device)
    gathered_metadata = torch.empty(tp_world_size * local_metadata.numel(), dtype=local_metadata.dtype, device=device)
    dist.all_gather_into_tensor(gathered_metadata, local_metadata, group=tp_group)

    expected_start = 0
    for rank, (shard_start, shard_end, shard_size) in enumerate(gathered_metadata.view(tp_world_size, 3).tolist()):
        if shard_end - shard_start != shard_size:
            raise ValueError(f"Vocabulary shard bounds for TP rank {rank} do not match its local vocabulary size")
        if shard_size <= 0:
            raise ValueError(f"TP rank {rank} received an empty vocabulary shard; the vocabulary must be at least "
                             f"as large as the tensor-parallel size")
        if shard_start != expected_start:
            raise ValueError("Vocabulary shard bounds must form a contiguous, non-overlapping partition starting at 0")
        expected_start = shard_end

    return expected_start


_vocab_metadata_cache = {}


def _resolve_vocab_metadata(local_vocab_size, vocab_start_index, vocab_end_index, tp_group, device):
    """Collectively validate the vocabulary shard layout once and cache the result.

    The shard geometry is fixed for the lifetime of the layers, so the repeated calls a
    training loop makes every micro-batch must not re-run the validation collectives or
    their host synchronizations. Decisions are made from identical all-gathered data, so
    every TP rank raises together instead of diverging into a collective hang.
    """
    key = (tp_group, local_vocab_size, vocab_start_index, vocab_end_index)
    cached = _vocab_metadata_cache.get(key)
    if cached is not None:
        return cached

    if vocab_start_index is None:
        tp_world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        if tp_world_size > 1:
            local_size = torch.tensor(local_vocab_size, device=device, dtype=torch.long)
            min_local_size = local_size.clone()
            max_local_size = local_size.clone()
            dist.all_reduce(min_local_size, op=dist.ReduceOp.MIN, group=tp_group)
            dist.all_reduce(max_local_size, op=dist.ReduceOp.MAX, group=tp_group)
            if min_local_size.item() != max_local_size.item():
                raise ValueError("Explicit vocabulary shard bounds are required for uneven tensor-parallel shards")
        vocab_start_index = tp_rank * local_vocab_size
        vocab_end_index = vocab_start_index + local_vocab_size
    global_vocab_size = _validate_vocab_shard_bounds(local_vocab_size, vocab_start_index, vocab_end_index, tp_group,
                                                     device)

    metadata = (vocab_start_index, vocab_end_index, global_vocab_size)
    _vocab_metadata_cache[key] = metadata
    return metadata


def vocab_parallel_cross_entropy(vocab_parallel_logits,
                                 target,
                                 tp_group=None,
                                 sp_group=None,
                                 vocab_start_index=None,
                                 vocab_end_index=None,
                                 ignore_index=-100,
                                 reduction="mean",
                                 gather_sequence_loss=False):
    """Compute cross entropy over vocabulary-sharded logits.

    Tensor parallel ranks collectively own the last (vocabulary) dimension. Sequence
    parallel ranks may independently own shards of the leading sequence dimension.
    """
    if vocab_parallel_logits.shape[:-1] != target.shape:
        raise ValueError("vocab_parallel_logits and target must have matching non-vocabulary dimensions")
    # With tensor parallelism an empty shard is rejected from the all-gathered shard
    # metadata so every rank fails together; only the single-process case can raise here.
    if vocab_parallel_logits.shape[-1] == 0 and (tp_group is None or dist.get_world_size(tp_group) == 1):
        raise ValueError("vocab_parallel_logits must contain at least one local vocabulary entry")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    if gather_sequence_loss and reduction != "none":
        raise ValueError("gather_sequence_loss is only supported with reduction='none'")

    local_vocab_size = vocab_parallel_logits.shape[-1]
    if (vocab_start_index is None) != (vocab_end_index is None):
        raise ValueError("vocab_start_index and vocab_end_index must be provided together")

    vocab_start_index, vocab_end_index, global_vocab_size = _resolve_vocab_metadata(
        local_vocab_size, vocab_start_index, vocab_end_index, tp_group, vocab_parallel_logits.device)
    # Data-dependent, so it cannot be hoisted out of the training loop: an out-of-range
    # target belongs to no shard and would otherwise silently contribute a wrong, finite loss.
    invalid_target = (target != ignore_index) & ((target < 0) | (target >= global_vocab_size))
    if invalid_target.any().item():
        raise ValueError(f"Target is out of range for vocabulary size {global_vocab_size}")

    loss = _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, tp_group, vocab_start_index,
                                            vocab_end_index, ignore_index)
    if reduction == "none":
        if gather_sequence_loss:
            if sp_group is None:
                raise ValueError("sp_group is required when gather_sequence_loss=True")
            loss = _GatherSequenceLoss.apply(loss, sp_group)
        return loss

    loss_sum = _global_sp_sum(loss.sum(), sp_group)
    if reduction == "sum":
        return loss_sum

    valid_tokens = (target != ignore_index).sum().to(dtype=loss.dtype)
    if sp_group is not None and dist.get_world_size(sp_group) > 1:
        dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM, group=sp_group)
    return loss_sum / valid_tokens.clamp_min(1)


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits,
                                          target,
                                          sp_group,
                                          tp_group=None,
                                          vocab_start_index=None,
                                          vocab_end_index=None,
                                          ignore_index=-100,
                                          reduction="none",
                                          gather_sequence_loss=True):
    """Sequence-parallel wrapper over :func:`vocab_parallel_cross_entropy`.

    Backward reduce-scatters the gradient over ``sp_group``, so each rank receives the
    gradient of its own sequence shard with the other ranks' contributions already
    summed in. Downstream code must therefore treat the returned loss as replicated
    across the SP group and must not average SP gradients a second time.
    """
    return vocab_parallel_cross_entropy(vocab_parallel_logits,
                                        target,
                                        tp_group=tp_group,
                                        sp_group=sp_group,
                                        vocab_start_index=vocab_start_index,
                                        vocab_end_index=vocab_end_index,
                                        ignore_index=ignore_index,
                                        reduction=reduction,
                                        gather_sequence_loss=gather_sequence_loss)


class VocabParallelCrossEntropyLoss(nn.Module):

    def __init__(self,
                 tp_group=None,
                 sp_group=None,
                 vocab_start_index=None,
                 vocab_end_index=None,
                 ignore_index=-100,
                 reduction="mean",
                 gather_sequence_loss=False):
        super().__init__()
        self.tp_group = tp_group
        self.sp_group = sp_group
        self.vocab_start_index = vocab_start_index
        self.vocab_end_index = vocab_end_index
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.gather_sequence_loss = gather_sequence_loss

    def forward(self, vocab_parallel_logits, target):
        return vocab_parallel_cross_entropy(vocab_parallel_logits,
                                            target,
                                            tp_group=self.tp_group,
                                            sp_group=self.sp_group,
                                            vocab_start_index=self.vocab_start_index,
                                            vocab_end_index=self.vocab_end_index,
                                            ignore_index=self.ignore_index,
                                            reduction=self.reduction,
                                            gather_sequence_loss=self.gather_sequence_loss)


class VocabParallelCausalLMLoss(nn.Module):
    """Distributed causal-LM loss for a vocabulary-sharded (no-gather) LM head.

    ``sp_group`` must stay ``None`` under DeepSpeed's Ulysses sequence-parallel engine:
    that engine aggregates the per-shard means itself, weighted by each shard's
    valid-token count, so an additional SP reduction here would double-count tokens.
    Pass ``sp_group`` only when this loss is the sole aggregation over a manually
    constructed TP x SP process-group mesh.
    """

    def __init__(self, tp_group=None, sp_group=None, vocab_start_index=None, vocab_end_index=None, ignore_index=-100):
        super().__init__()
        self.tp_group = tp_group
        self.sp_group = sp_group
        self.vocab_start_index = vocab_start_index
        self.vocab_end_index = vocab_end_index
        self.ignore_index = ignore_index

    def forward(self, logits, labels=None, vocab_size=None, shift_labels=None, num_items_in_batch=None, **kwargs):
        if shift_labels is None:
            if labels is None:
                raise ValueError("labels or shift_labels must be provided")
            shift_labels = labels[..., 1:].contiguous()
            logits = logits[..., :-1, :].contiguous()
        else:
            shift_labels = shift_labels.contiguous()

        if vocab_size is not None and self.vocab_start_index is not None and self.vocab_end_index is not None:
            # The LM head's shard metadata is the source of truth; a mismatch usually means
            # the embedding was resized, which gathered loss implementations tolerated.
            _, _, global_vocab_size = _resolve_vocab_metadata(self.vocab_end_index - self.vocab_start_index,
                                                              self.vocab_start_index, self.vocab_end_index,
                                                              self.tp_group, logits.device)
            if vocab_size != global_vocab_size:
                logger.warning_once(f"Vocab-parallel LM head holds vocab_size={global_vocab_size}, but the caller "
                                    f"described vocab_size={vocab_size}; the LM head's weights win")

        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = vocab_parallel_cross_entropy(logits,
                                            shift_labels,
                                            tp_group=self.tp_group,
                                            sp_group=self.sp_group,
                                            vocab_start_index=self.vocab_start_index,
                                            vocab_end_index=self.vocab_end_index,
                                            ignore_index=self.ignore_index,
                                            reduction=reduction)
        if num_items_in_batch is not None:
            denominator = torch.as_tensor(num_items_in_batch, device=loss.device, dtype=loss.dtype)
            loss = loss / denominator.clamp_min(1)
        return loss


def configure_vocab_parallel_loss(model, vocab_parallel_head, sp_group=None, ignore_index=-100):
    """Install the causal-LM loss required by a no-gather vocabulary projection.

    Leave ``sp_group`` as ``None`` when running under DeepSpeed's Ulysses
    sequence-parallel engine: the engine performs the token-count-weighted aggregation
    across SP ranks itself and expects this loss to return the local shard's mean.
    """
    if not hasattr(model, "loss_function"):
        raise ValueError("A no-gather vocab-parallel LM head requires a writable loss_function hook; "
                         "use gather_output=True for models without one")

    loss_fn = VocabParallelCausalLMLoss(tp_group=vocab_parallel_head.mp_group,
                                        sp_group=sp_group,
                                        vocab_start_index=vocab_parallel_head.vocab_start_index,
                                        vocab_end_index=vocab_parallel_head.vocab_end_index,
                                        ignore_index=ignore_index)
    # Keep the stock loss reachable so callers can restore it when tearing the head down.
    if not hasattr(model, "_deepspeed_original_loss_function"):
        model._deepspeed_original_loss_function = model.loss_function

    # Some model classes expose loss_function as a read-only property, in which case the
    # assignment raises; the identity check below turns that into an actionable error
    # instead of leaving the model silently computing loss on rank-local logits.
    try:
        model.loss_function = loss_fn
    except AttributeError:
        pass

    if model.loss_function is not loss_fn:
        raise ValueError("Unable to install the vocab-parallel loss_function hook; use gather_output=True")
    return model
