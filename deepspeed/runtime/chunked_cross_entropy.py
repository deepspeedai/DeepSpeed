# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Causal-LM cross entropy that never materializes a full-vocabulary FP32 tensor.

Hugging Face's ``ForCausalLMLoss`` upcasts the whole ``[tokens, vocab]`` logits to FP32 and calls
``cross_entropy``, which saves an FP32 log-softmax for backward; backward then allocates two more
FP32 ``[tokens, vocab]`` tensors. At S8192 with Qwen3's 151,936-token vocabulary each of those is
about 5 GB, and all three coexist at the start of backward, where training memory peaks.

This computes the same loss in FP32 without those tensors. Forward saves only the logits it was
given and one FP32 log-sum-exp per token; backward recomputes the softmax and writes the gradient
straight into a tensor of the logits' own dtype, which is the dtype the eager path casts its FP32
gradient to anyway. On CUDA a Triton kernel does each pass in one read of the logits; the PyTorch
backend does the same arithmetic one block of rows at a time and runs anywhere.
"""

from __future__ import annotations

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

_IS_ROCM_PYTORCH = getattr(torch.version, "hip", None) is not None
BACKENDS = ("auto", "triton", "torch")
_TRITON_BLOCK = 4096

if _TRITON_AVAILABLE:

    @triton.jit
    def _log_sum_exp_forward_kernel(logits_ptr, target_ptr, log_sum_exp_ptr, loss_ptr, n_cols, row_stride,
                                    ignore_index, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        row_ptr = logits_ptr + row * row_stride
        running_max = tl.full((), float("-inf"), tl.float32)
        running_sum = tl.zeros((), dtype=tl.float32)
        for start in range(0, n_cols, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            values = tl.load(row_ptr + offsets, mask=offsets < n_cols, other=float("-inf")).to(tl.float32)
            new_max = tl.maximum(running_max, tl.max(values, axis=0))
            running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(tl.exp(values - new_max), axis=0)
            running_max = new_max
        log_sum_exp = running_max + tl.log(running_sum)
        target = tl.load(target_ptr + row)
        valid = target != ignore_index
        target_logit = tl.load(row_ptr + tl.where(valid, target, 0)).to(tl.float32)
        tl.store(log_sum_exp_ptr + row, log_sum_exp)
        tl.store(loss_ptr + row, tl.where(valid, log_sum_exp - target_logit, 0.0))

    @triton.jit
    def _cross_entropy_backward_kernel(logits_ptr, grad_ptr, target_ptr, log_sum_exp_ptr, row_scale_ptr, n_cols,
                                       row_stride, grad_row_stride, ignore_index, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        target = tl.load(target_ptr + row)
        valid = target != ignore_index
        # Ignored rows get an all-zero gradient, as they do from cross_entropy.
        scale = tl.where(valid, tl.load(row_scale_ptr + row), 0.0)
        log_sum_exp = tl.load(log_sum_exp_ptr + row)
        for start in range(0, n_cols, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            mask = offsets < n_cols
            values = tl.load(logits_ptr + row * row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
            grad = tl.exp(values - log_sum_exp) * scale
            grad = tl.where(offsets == target, grad - scale, grad)
            tl.store(grad_ptr + row * grad_row_stride + offsets, grad.to(grad_ptr.dtype.element_ty), mask=mask)


def triton_backend_available() -> bool:
    return _TRITON_AVAILABLE and not _IS_ROCM_PYTORCH


def _resolve_backend(backend: str, logits: torch.Tensor) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported chunked cross-entropy backend {backend!r}; expected one of {BACKENDS}")
    on_cuda = logits.device.type == "cuda"
    if backend == "auto":
        return "triton" if on_cuda and triton_backend_available() else "torch"
    if backend == "triton" and not (on_cuda and triton_backend_available()):
        raise RuntimeError("The Triton chunked cross-entropy backend needs CUDA logits and Triton; use "
                           'backend="torch" to run elsewhere')
    return backend


# Rows are processed in blocks of about this many FP32 elements, so a block's temporaries stay near
# 256 MiB whatever the vocabulary size.
_BLOCK_ELEMENTS = 1 << 26


def _block_rows(vocab_size: int) -> int:
    return max(1, _BLOCK_ELEMENTS // vocab_size)


def _refuse_create_graph():
    # The gradient is computed outside autograd, by the Triton kernel or from a log-sum-exp saved as a constant,
    # so a graph built through it would silently miss the loss's second derivative.
    if torch.is_grad_enabled():
        raise RuntimeError("The chunked cross entropy has no second derivative; backward with create_graph=True is "
                           "not supported")


class _TritonCrossEntropy(torch.autograd.Function):
    """Per-row cross entropy with one read of the logits in forward and one read and write in backward."""

    @staticmethod
    def forward(ctx, logits, target, ignore_index):
        n_rows, n_cols = logits.shape
        log_sum_exp = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        if n_rows:
            _log_sum_exp_forward_kernel[(n_rows, )](logits,
                                                    target,
                                                    log_sum_exp,
                                                    loss,
                                                    n_cols,
                                                    logits.stride(0),
                                                    ignore_index,
                                                    BLOCK=_TRITON_BLOCK,
                                                    num_warps=8)
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
            _cross_entropy_backward_kernel[(n_rows, )](logits,
                                                       grad_logits,
                                                       target,
                                                       log_sum_exp,
                                                       row_scale,
                                                       n_cols,
                                                       logits.stride(0),
                                                       grad_logits.stride(0),
                                                       ctx.ignore_index,
                                                       BLOCK=_TRITON_BLOCK,
                                                       num_warps=8)
        return grad_logits, None, None


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
        _refuse_create_graph()
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
                          block_rows: int | None = None,
                          backend: str = "auto") -> torch.Tensor:
    """``torch.nn.functional.cross_entropy(logits.float(), target, ...)`` without FP32 ``[rows, vocab]`` tensors.

    ``logits`` is ``[rows, vocab]`` and ``target`` is ``[rows]``. As with ``cross_entropy``, "mean"
    divides by the number of non-ignored rows, so it is NaN when every row is ignored. ``backend``
    "auto" uses Triton for CUDA logits when it is available and PyTorch otherwise; naming a backend
    that cannot run raises instead of substituting the other. ``block_rows`` applies to PyTorch only.
    """
    if logits.dim() != 2 or target.shape != logits.shape[:1]:
        raise ValueError(f"Expected [rows, vocab] logits and [rows] targets, got {tuple(logits.shape)} and "
                         f"{tuple(target.shape)}")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    # The Triton kernels index both tensors by row with unit stride.
    target = target.to(device=logits.device, dtype=torch.long).contiguous()
    logits = logits.contiguous()
    # The Triton kernel reads the target's logit without a bounds check, so an out-of-range target would
    # silently read another row. Asserting on the device fails the way cross_entropy does, without the host
    # synchronization a Python-side check would add to every step.
    out_of_range = (target != ignore_index) & ((target < 0) | (target >= logits.shape[-1]))
    torch._assert_async(~out_of_range.any(), f"Target is out of range for vocabulary size {logits.shape[-1]}")
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


class ChunkedCausalLMLoss:
    """Drop-in for Hugging Face's ``ForCausalLMLoss`` with the same shifting, ignoring and normalization."""

    def __init__(self, block_rows: int | None = None, backend: str = "auto"):
        if backend not in BACKENDS:
            raise ValueError(f"Unsupported chunked cross-entropy backend {backend!r}; expected one of {BACKENDS}")
        self.block_rows = block_rows
        self.backend = backend

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
                                     block_rows=self.block_rows,
                                     backend=self.backend)
        if reduction == "sum":
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss


def install_chunked_causal_lm_loss(model, block_rows: int | None = None, backend: str = "auto") -> ChunkedCausalLMLoss:
    """Make a Hugging Face causal LM compute its training loss with :class:`ChunkedCausalLMLoss`.

    Only models whose ``loss_function`` is the stock ``ForCausalLMLoss`` are accepted, since this
    reproduces exactly that loss; anything else raises rather than silently changing the objective.
    """
    from transformers.loss.loss_utils import ForCausalLMLoss

    if getattr(model, "loss_function", None) is not ForCausalLMLoss:
        raise ValueError("install_chunked_causal_lm_loss only replaces the stock Hugging Face ForCausalLMLoss, "
                         f"but this model's loss_function is {getattr(model, 'loss_function', None)!r}")
    loss_function = ChunkedCausalLMLoss(block_rows=block_rows, backend=backend)
    model.loss_function = loss_function
    if model.loss_function is not loss_function:
        raise ValueError("Unable to install the chunked causal-LM loss: the model's loss_function is not writable")
    return loss_function
