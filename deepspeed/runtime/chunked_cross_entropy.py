# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Causal-LM cross entropy that never materializes a full-vocabulary FP32 tensor.

Hugging Face's ``ForCausalLMLoss`` upcasts the whole ``[tokens, vocab]`` logits to FP32 and calls
``cross_entropy``, which saves an FP32 log-softmax for backward; backward then allocates two more
FP32 ``[tokens, vocab]`` tensors. At S8192 with Qwen3's 151,936-token vocabulary each of those is
about 5 GB, and all three coexist at the start of backward, where training memory peaks.

This computes the same loss in FP32 one block of rows at a time. Forward saves only the logits it
was given and one FP32 log-sum-exp per token; backward recomputes each block's softmax and writes
the gradient straight into a tensor of the logits' own dtype, which is the dtype the eager path
casts its FP32 gradient to anyway.
"""

import torch

# Rows are processed in blocks of about this many FP32 elements, so a block's temporaries stay near
# 256 MiB whatever the vocabulary size.
_BLOCK_ELEMENTS = 1 << 26


def _block_rows(vocab_size: int) -> int:
    return max(1, _BLOCK_ELEMENTS // vocab_size)


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Per-row cross entropy over ``[rows, vocab]`` logits; ignored rows contribute zero loss and gradient."""

    @staticmethod
    def forward(ctx, logits, target, ignore_index, block_rows):
        n_rows = logits.shape[0]
        valid = target != ignore_index
        # Ignored rows still need an in-range index to gather from; their result is zeroed below.
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        log_sum_exp = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        target_logits = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        for start in range(0, n_rows, block_rows):
            end = min(start + block_rows, n_rows)
            block = logits[start:end].float()
            log_sum_exp[start:end] = torch.logsumexp(block, dim=-1)
            target_logits[start:end] = block.gather(-1, safe_target[start:end].unsqueeze(-1)).squeeze(-1)
        loss = torch.where(valid, log_sum_exp - target_logits, torch.zeros_like(log_sum_exp))
        ctx.save_for_backward(logits, safe_target, valid, log_sum_exp)
        ctx.block_rows = block_rows
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        logits, safe_target, valid, log_sum_exp = ctx.saved_tensors
        # d loss_i / d logit_ij = softmax_ij - [j == target_i], scaled by the incoming gradient of row i.
        row_scale = torch.where(valid, grad_loss.float(), torch.zeros_like(log_sum_exp))
        grad_logits = torch.empty_like(logits)
        n_rows = logits.shape[0]
        for start in range(0, n_rows, ctx.block_rows):
            end = min(start + ctx.block_rows, n_rows)
            probabilities = torch.exp(logits[start:end].float() - log_sum_exp[start:end].unsqueeze(-1))
            probabilities.scatter_add_(-1, safe_target[start:end].unsqueeze(-1),
                                       -torch.ones_like(probabilities[:, :1]))
            probabilities.mul_(row_scale[start:end].unsqueeze(-1))
            grad_logits[start:end] = probabilities.to(grad_logits.dtype)
        return grad_logits, None, None, None


def chunked_cross_entropy(logits: torch.Tensor,
                          target: torch.Tensor,
                          ignore_index: int = -100,
                          reduction: str = "mean",
                          block_rows: int | None = None) -> torch.Tensor:
    """``torch.nn.functional.cross_entropy(logits.float(), target, ...)`` without FP32 ``[rows, vocab]`` tensors.

    ``logits`` is ``[rows, vocab]`` and ``target`` is ``[rows]``. As with ``cross_entropy``, "mean"
    divides by the number of non-ignored rows, so it is NaN when every row is ignored.
    """
    if logits.dim() != 2 or target.shape != logits.shape[:1]:
        raise ValueError(f"Expected [rows, vocab] logits and [rows] targets, got {tuple(logits.shape)} and "
                         f"{tuple(target.shape)}")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    # An out-of-range target fails in the gather, as it does in cross_entropy; checking it here would
    # add a host synchronization to every training step.
    target = target.to(device=logits.device, dtype=torch.long)
    if block_rows is None:
        block_rows = _block_rows(logits.shape[-1])
    loss = _ChunkedCrossEntropy.apply(logits.contiguous(), target, ignore_index, block_rows)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / (target != ignore_index).sum()


class ChunkedCausalLMLoss:
    """Drop-in for Hugging Face's ``ForCausalLMLoss`` with the same shifting, ignoring and normalization."""

    def __init__(self, block_rows: int | None = None):
        self.block_rows = block_rows

    def __call__(self,
                 logits,
                 labels,
                 vocab_size,
                 num_items_in_batch=None,
                 ignore_index=-100,
                 shift_labels=None,
                 **kwargs):
        if shift_labels is None:
            # Shift so that tokens < n predict n; the last position has no next token and is ignored.
            labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
        rows = logits.reshape(-1, vocab_size)
        targets = shift_labels.reshape(-1)
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = chunked_cross_entropy(rows,
                                     targets,
                                     ignore_index=ignore_index,
                                     reduction=reduction,
                                     block_rows=self.block_rows)
        if reduction == "sum":
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss


def install_chunked_causal_lm_loss(model, block_rows: int | None = None) -> ChunkedCausalLMLoss:
    """Make a Hugging Face causal LM compute its training loss with :class:`ChunkedCausalLMLoss`.

    Only models whose ``loss_function`` is the stock ``ForCausalLMLoss`` are accepted, since this
    reproduces exactly that loss; anything else raises rather than silently changing the objective.
    """
    from transformers.loss.loss_utils import ForCausalLMLoss

    if getattr(model, "loss_function", None) is not ForCausalLMLoss:
        raise ValueError("install_chunked_causal_lm_loss only replaces the stock Hugging Face ForCausalLMLoss, "
                         f"but this model's loss_function is {getattr(model, 'loss_function', None)!r}")
    loss_function = ChunkedCausalLMLoss(block_rows=block_rows)
    model.loss_function = loss_function
    if model.loss_function is not loss_function:
        raise ValueError("Unable to install the chunked causal-LM loss: the model's loss_function is not writable")
    return loss_function
