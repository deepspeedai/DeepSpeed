# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch import nn
from functools import lru_cache

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.logging import logger


@lru_cache(maxsize=1)
def _load_liger_vocab_ce():
    try:
        from liger_kernel.ops.vocab_parallel_cross_entropy import LigerVocabParallelCEFunction
    except ImportError as error:
        raise ImportError("Liger vocabulary-parallel CE requires liger-kernel>=0.8.1; "
                          "install it or select backend='torch'.") from error
    return LigerVocabParallelCEFunction


def _liger_vocab_cross_entropy(logits, target, tp_group, ignore_index):
    kernel = _load_liger_vocab_ce()
    loss = kernel.apply(logits.reshape(-1, 1, logits.shape[-1]), target.reshape(-1, 1), tp_group, ignore_index, 0.0,
                        "pytorch", "global").reshape_as(target)
    if loss.requires_grad:
        consumed = False

        def guard_backward(gradient):
            nonlocal consumed
            # Liger overwrites its saved exp buffer with the first gradient, so a retained-graph
            # second backward must fail rather than silently reuse that buffer.
            if consumed or torch.is_grad_enabled():
                raise RuntimeError("Liger CE supports one first-order backward per forward; "
                                   "use backend='torch' for retained-graph or higher-order gradients.")
            consumed = True
            return gradient

        loss.register_hook(guard_backward)
    return loss


def _liger_can_run(vocab_parallel_logits, target, local_vocab_size, global_vocab_size, tp_group):
    # A ``None`` group is ambiguous to Liger: depending on its version it may resolve to the
    # default world, which would fold data-parallel replicas into the vocabulary dimension, so
    # the kernel only runs behind a real tensor-parallel group. Liger's kernel all-gathers
    # fixed-width shards while the PyTorch path trims uneven ones, so the kernel stays
    # restricted to equal shards.
    if tp_group is None:
        return False
    tp_world_size = dist.get_world_size(tp_group)
    return (get_accelerator().is_triton_supported() and get_accelerator().on_accelerator(vocab_parallel_logits)
            and vocab_parallel_logits.dtype in (torch.float32, torch.float16, torch.bfloat16) and target.numel() > 0
            and local_vocab_size * tp_world_size == global_vocab_size)


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
    def forward(ctx, local_loss, sp_group, sum_gradients):
        ctx.sp_group = sp_group
        ctx.local_sequence_size = local_loss.shape[0]
        ctx.sum_gradients = sum_gradients
        ctx.sp_rank = dist.get_rank(sp_group)

        output_shape = (ctx.local_sequence_size * dist.get_world_size(sp_group), *local_loss.shape[1:])
        gathered_loss = torch.empty(output_shape, dtype=local_loss.dtype, device=local_loss.device)
        dist.all_gather_into_tensor(gathered_loss, local_loss.contiguous(), group=sp_group)
        return gathered_loss

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.sum_gradients:
            start = ctx.sp_rank * ctx.local_sequence_size
            return grad_output.narrow(0, start, ctx.local_sequence_size), None, None

        grad_input = torch.empty((ctx.local_sequence_size, *grad_output.shape[1:]),
                                 dtype=grad_output.dtype,
                                 device=grad_output.device)
        dist.reduce_scatter_fn(grad_input, grad_output.contiguous(), group=ctx.sp_group)
        return grad_input, None, None


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


def _resolve_vocab_metadata(local_vocab_size, vocab_start_index, vocab_end_index, tp_group, device):
    """Infer (if needed) and collectively validate the vocabulary shard layout.

    This always re-runs the validation collective; it does not cache across calls. A
    process-wide cache keyed on ``tp_group`` would hold a strong reference to that group
    for as long as the cache entry lives, which is arbitrarily longer than any single
    caller's own lifetime and can keep a destroyed process group's resources alive.
    Callers that invoke this repeatedly for a fixed shard (e.g. ``VocabParallelCausalLMLoss``)
    should resolve and cache the result once on an object they already own instead.
    """
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
    return vocab_start_index, vocab_end_index, global_vocab_size


def vocab_parallel_cross_entropy(vocab_parallel_logits,
                                 target,
                                 tp_group=None,
                                 sp_group=None,
                                 vocab_start_index=None,
                                 vocab_end_index=None,
                                 global_vocab_size=None,
                                 ignore_index=-100,
                                 reduction="mean",
                                 gather_sequence_loss=False,
                                 backend="torch"):
    """Compute cross entropy over vocabulary-sharded logits.

    Tensor parallel ranks collectively own the last (vocabulary) dimension. Sequence
    parallel ranks may independently own shards of the leading sequence dimension.

    ``global_vocab_size`` lets a caller that already validated the shard layout once (e.g.
    ``VocabParallelCausalLMLoss``) skip the collective re-validation on every call; it
    requires explicit ``vocab_start_index``/``vocab_end_index`` and is trusted as-is. Bare
    calls that omit it always re-run the validation collective.
    """
    if vocab_parallel_logits.shape[:-1] != target.shape:
        raise ValueError("vocab_parallel_logits and target must have matching non-vocabulary dimensions")
    # With tensor parallelism an empty shard is rejected from the all-gathered shard
    # metadata so every rank fails together; only the single-process case can raise here.
    if vocab_parallel_logits.shape[-1] == 0 and (tp_group is None or dist.get_world_size(tp_group) == 1):
        raise ValueError("vocab_parallel_logits must contain at least one local vocabulary entry")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    if backend not in ("torch", "liger"):
        raise ValueError(f"Unsupported vocabulary-parallel CE backend: {backend}")
    if gather_sequence_loss and reduction != "none":
        raise ValueError("gather_sequence_loss is only supported with reduction='none'")

    local_vocab_size = vocab_parallel_logits.shape[-1]
    if (vocab_start_index is None) != (vocab_end_index is None):
        raise ValueError("vocab_start_index and vocab_end_index must be provided together")

    if global_vocab_size is not None:
        if vocab_start_index is None:
            raise ValueError("global_vocab_size requires explicit vocab_start_index and vocab_end_index")
    else:
        vocab_start_index, vocab_end_index, global_vocab_size = _resolve_vocab_metadata(
            local_vocab_size, vocab_start_index, vocab_end_index, tp_group, vocab_parallel_logits.device)
    # Data-dependent, so it cannot be hoisted out of the training loop: an out-of-range
    # target belongs to no shard and would otherwise silently contribute a wrong, finite loss.
    invalid_target = (target != ignore_index) & ((target < 0) | (target >= global_vocab_size))
    if invalid_target.any().item():
        raise ValueError(f"Target is out of range for vocabulary size {global_vocab_size}")

    use_liger = backend == "liger" and _liger_can_run(vocab_parallel_logits, target, local_vocab_size,
                                                      global_vocab_size, tp_group)
    if use_liger:
        loss = _liger_vocab_cross_entropy(vocab_parallel_logits, target.to(dtype=torch.long), tp_group, ignore_index)
    else:
        if backend == "liger":
            logger.warning_once("Liger CE requires an explicit tensor-parallel group, a Triton-supported "
                                "accelerator, nonempty accelerator tokens, a supported dtype, and equal vocabulary "
                                "shards; using the PyTorch reference backend for this layout.")
        loss = _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, tp_group, vocab_start_index,
                                                vocab_end_index, ignore_index)
    if reduction == "none":
        if gather_sequence_loss:
            if sp_group is None:
                raise ValueError("sp_group is required when gather_sequence_loss=True")
            loss = _GatherSequenceLoss.apply(loss, sp_group, True)
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
                                          gather_sequence_loss=True,
                                          backend="torch"):
    """Sequence-parallel wrapper preserving the legacy local-slice gradient."""
    if backend not in ("torch", "liger"):
        raise ValueError(f"Unsupported vocabulary-parallel CE backend: {backend}")
    if gather_sequence_loss and reduction != "none":
        raise ValueError("gather_sequence_loss is only supported with reduction='none'")
    if gather_sequence_loss and sp_group is None:
        raise ValueError("sp_group is required when gather_sequence_loss=True")

    loss = vocab_parallel_cross_entropy(vocab_parallel_logits,
                                        target,
                                        tp_group=tp_group,
                                        sp_group=sp_group,
                                        vocab_start_index=vocab_start_index,
                                        vocab_end_index=vocab_end_index,
                                        ignore_index=ignore_index,
                                        reduction=reduction,
                                        backend=backend)
    if not gather_sequence_loss:
        return loss
    return _GatherSequenceLoss.apply(loss, sp_group, False)


class VocabParallelCrossEntropyLoss(nn.Module):

    def __init__(self,
                 tp_group=None,
                 sp_group=None,
                 vocab_start_index=None,
                 vocab_end_index=None,
                 ignore_index=-100,
                 reduction="mean",
                 gather_sequence_loss=False,
                 backend="torch"):
        super().__init__()
        self.tp_group = tp_group
        self.sp_group = sp_group
        self.vocab_start_index = vocab_start_index
        self.vocab_end_index = vocab_end_index
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.gather_sequence_loss = gather_sequence_loss
        self.backend = backend

    def forward(self, vocab_parallel_logits, target):
        return vocab_parallel_cross_entropy(vocab_parallel_logits,
                                            target,
                                            tp_group=self.tp_group,
                                            sp_group=self.sp_group,
                                            vocab_start_index=self.vocab_start_index,
                                            vocab_end_index=self.vocab_end_index,
                                            ignore_index=self.ignore_index,
                                            reduction=self.reduction,
                                            gather_sequence_loss=self.gather_sequence_loss,
                                            backend=self.backend)


class VocabParallelCausalLMLoss:
    """Distributed causal-LM loss for a vocabulary-sharded (no-gather) LM head.

    ``sp_group`` must stay ``None`` under DeepSpeed's Ulysses sequence-parallel engine:
    that engine aggregates the per-shard means itself, weighted by each shard's
    valid-token count, so an additional SP reduction here would double-count tokens.
    Pass ``sp_group`` only when this loss is the sole aggregation over a manually
    constructed TP x SP process-group mesh.

    Each instance is bound to a single, fixed vocabulary shard: the first call
    collectively validates ``tp_group``/``vocab_start_index``/``vocab_end_index`` once and
    caches the result on ``self``, so the cache's lifetime is exactly this instance's
    lifetime and never outlives (or is shared beyond) the object that already holds
    ``tp_group``. Create a separate instance per shard -- e.g. one each for a teacher and a
    student model, even if they happen to share a shard layout -- instead of mutating
    ``vocab_start_index``/``vocab_end_index`` on a shared instance; reusing one instance
    for a different shard raises rather than silently computing loss against stale
    metadata.
    """

    def __init__(self,
                 tp_group=None,
                 sp_group=None,
                 vocab_start_index=None,
                 vocab_end_index=None,
                 ignore_index=-100,
                 backend="torch"):
        self.tp_group = tp_group
        self.sp_group = sp_group
        self.vocab_start_index = vocab_start_index
        self.vocab_end_index = vocab_end_index
        self.ignore_index = ignore_index
        self.backend = backend
        self._resolved_vocab_key = None
        self._resolved_global_vocab_size = None

    def _resolve_metadata_once(self, local_vocab_size, device):
        if self._resolved_vocab_key is None:
            vocab_start_index, vocab_end_index, global_vocab_size = _resolve_vocab_metadata(
                local_vocab_size, self.vocab_start_index, self.vocab_end_index, self.tp_group, device)
            self.vocab_start_index = vocab_start_index
            self.vocab_end_index = vocab_end_index
            self._resolved_global_vocab_size = global_vocab_size
            self._resolved_vocab_key = (self.tp_group, device, local_vocab_size, vocab_start_index, vocab_end_index)
            return global_vocab_size

        key = (self.tp_group, device, local_vocab_size, self.vocab_start_index, self.vocab_end_index)
        if key != self._resolved_vocab_key:
            raise RuntimeError("VocabParallelCausalLMLoss shard metadata changed after its first call; create a "
                               "new instance instead of reusing one across different vocabulary shards")
        return self._resolved_global_vocab_size

    def __call__(self, logits, labels=None, vocab_size=None, shift_labels=None, num_items_in_batch=None, **kwargs):
        if shift_labels is None:
            if labels is None:
                raise ValueError("labels or shift_labels must be provided")
            shift_labels = labels[..., 1:].contiguous()
            logits = logits[..., :-1, :].contiguous()
        else:
            shift_labels = shift_labels.contiguous()

        # Resolved and validated once per instance; see the class docstring.
        global_vocab_size = self._resolve_metadata_once(logits.shape[-1], logits.device)
        if vocab_size is not None and vocab_size != global_vocab_size:
            # The LM head's shard metadata is the source of truth; a mismatch usually means
            # the embedding was resized, which gathered loss implementations tolerated.
            logger.warning_once(f"Vocab-parallel LM head holds vocab_size={global_vocab_size}, but the caller "
                                f"described vocab_size={vocab_size}; the LM head's weights win")

        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = vocab_parallel_cross_entropy(logits,
                                            shift_labels,
                                            tp_group=self.tp_group,
                                            sp_group=self.sp_group,
                                            vocab_start_index=self.vocab_start_index,
                                            vocab_end_index=self.vocab_end_index,
                                            global_vocab_size=global_vocab_size,
                                            ignore_index=self.ignore_index,
                                            reduction=reduction,
                                            backend=self.backend)
        if num_items_in_batch is not None:
            denominator = torch.as_tensor(num_items_in_batch, device=loss.device, dtype=loss.dtype)
            loss = loss / denominator.clamp_min(1)
        return loss


def _set_vocab_parallel_loss(model, loss_fn):
    if not hasattr(model, "loss_function"):
        raise ValueError("A no-gather vocab-parallel LM head requires a writable loss_function hook; "
                         "use gather_output=True for models without one")

    original_loss_function = model.loss_function
    registered_loss_module = getattr(model, "_modules", {}).pop("loss_function", None)

    # Read-only properties and setters that ignore the new hook cannot be used with
    # rank-local logits; reject both before the caller commits to vocabulary sharding.
    try:
        model.loss_function = loss_fn
    except (AttributeError, TypeError) as exc:
        if registered_loss_module is not None:
            model.loss_function = registered_loss_module
        raise ValueError("Unable to install the vocab-parallel loss_function hook; use gather_output=True") from exc

    if model.loss_function is not loss_fn:
        model.loss_function = original_loss_function
        raise ValueError("Unable to install the vocab-parallel loss_function hook; use gather_output=True")
    return original_loss_function


def validate_vocab_parallel_loss(model):
    """Check hook replacement before partitioning weights, then restore the original loss."""
    original_loss_function = _set_vocab_parallel_loss(model, VocabParallelCausalLMLoss())
    model.loss_function = original_loss_function


def configure_vocab_parallel_loss(model, vocab_parallel_head, sp_group=None, ignore_index=-100, backend="torch"):
    """Install the causal-LM loss required by a no-gather vocabulary projection.

    Leave ``sp_group`` as ``None`` when running under DeepSpeed's Ulysses
    sequence-parallel engine: the engine performs the token-count-weighted aggregation
    across SP ranks itself and expects this loss to return the local shard's mean.
    """
    loss_fn = VocabParallelCausalLMLoss(tp_group=vocab_parallel_head.mp_group,
                                        sp_group=sp_group,
                                        vocab_start_index=vocab_parallel_head.vocab_start_index,
                                        vocab_end_index=vocab_parallel_head.vocab_end_index,
                                        ignore_index=ignore_index,
                                        backend=backend)
    original_loss_function = _set_vocab_parallel_loss(model, loss_fn)

    # Keep the stock loss reachable so callers can restore it when tearing the head down.
    if not hasattr(model, "_deepspeed_original_loss_function"):
        model._deepspeed_original_loss_function = original_loss_function
    return model
