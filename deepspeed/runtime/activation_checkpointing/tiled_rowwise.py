# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Token-axis chunking for functions that act on each token row independently.

``tiled_rowwise_compute(fn, *tensors, shared=..., token_budget=...)`` runs ``fn`` over the rows of ``tensors`` in chunks
of at most ``token_budget`` rows along ``token_dim``, in forward and in backward. Forward runs under ``no_grad`` and
saves only the inputs; backward recomputes and differentiates one chunk at a time. Peak memory for ``fn``'s
temporaries is therefore one chunk instead of the whole sequence.

Use it for row-wise ops whose autograd graph would otherwise hold several full-sequence temporaries, e.g. a norm
evaluated in fp32: each fp32 ``[T, hidden]`` temporary of a pure-PyTorch norm backward is ``4 * T * hidden`` bytes.

Requirements on ``fn``:
  * ``fn(*chunks, *shared)`` returns a tensor or a tuple of tensors, each with the chunk's extent on ``token_dim``.
  * Row ``i`` of every output depends only on row ``i`` of every chunked input (plus ``shared``). Statistics over the
    token axis (e.g. a batch norm) are not row-wise and give wrong results.
  * ``shared`` tensors (weights) are passed whole to every chunk; their gradients are summed over chunks.

Forward outputs are bit-identical to ``fn`` on the whole sequence for row-wise ``fn`` (each row runs the same ops).
Gradients of ``shared`` tensors are sums over chunks, so they match to rounding, not bit-for-bit;
``shared_grad_dtype=torch.float32`` accumulates them in fp32 (only for an ``fn`` whose result does not change when its
shared tensors are given in fp32, e.g. one that upcasts them itself). ZeRO-3 partitioned shared parameters are held
gathered for the forward and for every backward chunk.
"""

import torch

from deepspeed.runtime.activation_checkpointing.tiled_rematerialization import _hold_gathered_params


def plan_row_chunks(num_rows, token_budget):
    """Contiguous ``(start, end)`` row ranges of at most ``token_budget`` rows covering ``num_rows``."""
    budget = int(token_budget)
    if budget <= 0:
        raise ValueError(f"tiled_rowwise_compute: token_budget must be positive, got {token_budget}")
    return [(start, min(start + budget, num_rows)) for start in range(0, num_rows, budget)]


def _as_tuple(result):
    if torch.is_tensor(result):
        return (result, ), True
    if isinstance(result, tuple) and result and all(torch.is_tensor(r) for r in result):
        return result, False
    raise TypeError(f"tiled_rowwise_compute: fn must return a tensor or a tuple of tensors, got {type(result)}")


class _TiledRowwise(torch.autograd.Function):

    @staticmethod
    def forward(ctx, fn, token_budget, token_dim, n_chunked, shared_grad_dtype, *args):
        chunked, shared = args[:n_chunked], args[n_chunked:]
        num_rows = chunked[0].shape[token_dim]
        for t in chunked:
            if t.shape[token_dim] != num_rows:
                raise ValueError("tiled_rowwise_compute: every chunked tensor needs the same extent on token_dim, "
                                 f"got {[x.shape[token_dim] for x in chunked]}")
        bounds = plan_row_chunks(num_rows, token_budget)
        params = [s for s in shared if isinstance(s, torch.nn.Parameter)]
        outputs = None
        single = True
        with _hold_gathered_params(params), torch.no_grad():
            for start, end in bounds:
                parts = [t.narrow(token_dim, start, end - start) for t in chunked]
                result, single = _as_tuple(fn(*parts, *shared))
                if outputs is None:
                    outputs = []
                    for r in result:
                        shape = list(r.shape)
                        shape[token_dim] = num_rows
                        outputs.append(r.new_empty(shape))
                for out, r in zip(outputs, result):
                    out.narrow(token_dim, start, end - start).copy_(r)
        ctx.fn = fn
        ctx.token_dim = token_dim
        ctx.bounds = bounds
        ctx.n_chunked = n_chunked
        ctx.shared_grad_dtype = shared_grad_dtype
        ctx.save_for_backward(*chunked, *shared)
        ctx.single = single
        return outputs[0] if single else tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved = ctx.saved_tensors
        chunked, shared = saved[:ctx.n_chunked], saved[ctx.n_chunked:]
        needs = ctx.needs_input_grad[5:]
        needs_chunked, needs_shared = needs[:ctx.n_chunked], needs[ctx.n_chunked:]
        dim = ctx.token_dim

        shared_leaves = []
        for s, need in zip(shared, needs_shared):
            if need and s.is_floating_point():
                leaf = s.detach()
                if ctx.shared_grad_dtype is not None:
                    leaf = leaf.to(ctx.shared_grad_dtype)
                shared_leaves.append(leaf.requires_grad_(True))
            else:
                shared_leaves.append(s)
        input_grads = [torch.zeros_like(t) if need else None for t, need in zip(chunked, needs_chunked)]

        params = [s for s in shared if isinstance(s, torch.nn.Parameter)]
        with _hold_gathered_params(params), torch.enable_grad():
            for start, end in ctx.bounds:
                length = end - start
                parts = [
                    t.narrow(dim, start, length).detach().requires_grad_(need)
                    for t, need in zip(chunked, needs_chunked)
                ]
                result, _ = _as_tuple(ctx.fn(*parts, *shared_leaves))
                pairs = [(r, g.narrow(dim, start, length)) for r, g in zip(result, grad_outputs)
                         if g is not None and r.requires_grad]
                if pairs:
                    torch.autograd.backward([r for r, _ in pairs], [g for _, g in pairs])
                for part, grad in zip(parts, input_grads):
                    if grad is not None and part.grad is not None:
                        grad.narrow(dim, start, length).copy_(part.grad)
                del parts, result, pairs

        shared_grads = []
        for s, leaf, need in zip(shared, shared_leaves, needs_shared):
            if need and leaf is not s and leaf.grad is not None:
                shared_grads.append(leaf.grad.to(s.dtype))
            else:
                shared_grads.append(None)
        return (None, None, None, None, None, *input_grads, *shared_grads)


def tiled_rowwise_compute(fn, *tensors, shared=(), token_budget, token_dim=-2, shared_grad_dtype=None):
    """Run row-wise ``fn(*tensors, *shared)`` in chunks of at most ``token_budget`` rows along ``token_dim``.

    Returns what ``fn`` returns (a tensor or a tuple of tensors) for the whole sequence. See the module docstring for
    the contract on ``fn``. With a single chunk (``token_budget`` at least the row count) ``fn`` is called directly.
    """
    if not tensors:
        raise ValueError("tiled_rowwise_compute: need at least one chunked tensor")
    if tensors[0].shape[token_dim] <= int(token_budget):
        return fn(*tensors, *shared)
    token_dim = token_dim % tensors[0].dim()
    return _TiledRowwise.apply(fn, int(token_budget), token_dim, len(tensors), shared_grad_dtype, *tensors, *shared)
