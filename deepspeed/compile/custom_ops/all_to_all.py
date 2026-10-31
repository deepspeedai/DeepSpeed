# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple

import torch
import deepspeed.comm as dist
from .sp_dp_registry import get_group, is_setup, sp_size


@torch.library.custom_op("autosp::all_to_all", mutates_args=())
def all_to_all(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    name: str,
) -> torch.Tensor:
    """
    All-to-all collective for SDPA tensors [B, N, S, H].

    For QKV (scatter_idx=1, gather_idx=2):
        [B, N, S/P, H] -> [B, N/P, S, H]
    For O (scatter_idx=2, gather_idx=1):
        [B, N/P, S, H] -> [B, N, S/P, H]
    """
    assert is_setup(), 'Incorrect initialization of SP/DP mesh.'
    B, dim1, dim2, H = input.shape
    gid = dist.get_rank() // sp_size()
    group = get_group(gid)

    if scatter_idx == 1:
        N, local_S = dim1, dim2
        if N % sp_size() != 0:
            raise ValueError(f"AutoSP requires the number of attention heads ({N}) to be divisible by "
                             f"sequence_parallel_size ({sp_size()})")
        input_t = input.reshape(B, sp_size(), N // sp_size(), local_S, H)
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()

        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)

        output = output.permute(1, 2, 0, 3, 4).contiguous()
        output = output.reshape(B, N // sp_size(), sp_size() * local_S, H)
    else:
        local_N, S = dim1, dim2
        input_t = input.reshape(B, local_N, sp_size(), S // sp_size(), H)
        input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()

        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)

        output = output.permute(1, 0, 2, 3, 4).contiguous()
        output = output.reshape(B, sp_size() * local_N, S // sp_size(), H)

    return output


def maybe_restore_sharded_dim(dim: torch.SymInt, factor: int):
    # Torch 2.9 may keep `P * (s // P)` distinct from the original `s` during
    # fake shape propagation. When the local dim is exactly `FloorDiv(s, P)`,
    # restore the original symbol so downstream ops see a consistent sequence dim.
    node = getattr(dim, "node", None)
    if node is None:
        return dim * factor

    # Imported here because torch.utils._sympy imports sympy, which slows down import deepspeed.
    from torch.utils._sympy.functions import FloorDiv

    expr = node.expr
    if isinstance(expr, FloorDiv) and expr.args[1] == factor:
        hint = node.hint * factor if node.has_hint() else None
        return node.shape_env.create_symintnode(expr.args[0], hint=hint)

    return dim * factor


@torch.library.register_fake("autosp::all_to_all")
def all_to_all_fake(input: torch.Tensor, scatter_idx: int, gather_idx: int, name: str):
    B, dim1, dim2, H = input.shape
    if scatter_idx == 1:
        return input.new_empty(B, dim1 // sp_size(), maybe_restore_sharded_dim(dim2, sp_size()), H)
    else:
        return input.new_empty(B, dim1 * sp_size(), dim2 // sp_size(), H)


def _all_to_all_backward_setup(ctx, inputs, output):
    _, scatter_idx, gather_idx, name = inputs
    ctx.scatter_idx = gather_idx
    ctx.gather_idx = scatter_idx
    ctx.name = name + "_grad"


def _all_to_all_backward(ctx, grad):
    return (all_to_all(grad, ctx.scatter_idx, ctx.gather_idx, ctx.name), None, None, None)


torch.library.register_autograd("autosp::all_to_all", _all_to_all_backward, setup_context=_all_to_all_backward_setup)


@torch.library.custom_op("autosp::aggregate_loss", mutates_args=())
def aggregate_loss(local_loss_sum: torch.Tensor, local_valid_tokens: torch.Tensor,
                   average: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine the per-shard summed loss of a sequence-parallel group into the full-sequence loss.

    Returns the full-sequence loss (summed, or averaged over the valid tokens of the group)
    and the factor that backward applies to the incoming gradient.
    """
    assert is_setup(), 'Incorrect initialization of SP/DP mesh.'
    group = get_group(dist.get_rank() // sp_size())

    loss = local_loss_sum.clone()
    dist.all_reduce(loss, group=group)
    # The engine sums gradients over the SP group, which adds up the contribution of every shard.
    grad_scale = torch.ones_like(loss)
    if average:
        valid_tokens = local_valid_tokens.clone()
        dist.all_reduce(valid_tokens, group=group)
        valid_tokens = valid_tokens.to(loss.dtype)
        loss = loss / valid_tokens
        grad_scale = grad_scale / valid_tokens
    return loss, grad_scale


@torch.library.register_fake("autosp::aggregate_loss")
def aggregate_loss_fake(local_loss_sum: torch.Tensor, local_valid_tokens: torch.Tensor, average: bool):
    return torch.empty_like(local_loss_sum), torch.empty_like(local_loss_sum)


def _aggregate_loss_backward_setup(ctx, inputs, output):
    _, grad_scale = output
    ctx.mark_non_differentiable(grad_scale)
    ctx.save_for_backward(grad_scale)


def _aggregate_loss_backward(ctx, grad_loss, grad_scale_grad):
    (grad_scale, ) = ctx.saved_tensors
    return grad_loss * grad_scale, None, None


torch.library.register_autograd("autosp::aggregate_loss",
                                _aggregate_loss_backward,
                                setup_context=_aggregate_loss_backward_setup)


@torch.library.custom_op("autosp::all_gather_sequence", mutates_args=())
def all_gather_sequence(input: torch.Tensor, dim: int) -> torch.Tensor:
    """Concatenate the local sequence shards of `input` along `dim` across the SP group."""
    assert is_setup(), 'Incorrect initialization of SP/DP mesh.'
    gid = dist.get_rank() // sp_size()
    group = get_group(gid)

    input = input.contiguous()
    shards = [torch.empty_like(input) for _ in range(sp_size())]
    dist.all_gather(shards, input, group=group)
    return torch.cat(shards, dim=dim)


@torch.library.register_fake("autosp::all_gather_sequence")
def all_gather_sequence_fake(input: torch.Tensor, dim: int):
    output_shape = list(input.shape)
    output_shape[dim] = maybe_restore_sharded_dim(output_shape[dim], sp_size())
    return input.new_empty(output_shape)


@torch.library.custom_op("autosp::reduce_scatter_sequence", mutates_args=())
def reduce_scatter_sequence(input: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum `input` over the SP group and keep this rank's shard along `dim`; the adjoint of all_gather_sequence."""
    assert is_setup(), 'Incorrect initialization of SP/DP mesh.'
    group = get_group(dist.get_rank() // sp_size())

    dim = dim % input.dim()
    input = input.movedim(dim, 0).contiguous()
    output = input.new_empty((input.shape[0] // sp_size(), ) + tuple(input.shape[1:]))
    dist.reduce_scatter_tensor(output, input, group=group)
    return output.movedim(0, dim).contiguous()


@torch.library.register_fake("autosp::reduce_scatter_sequence")
def reduce_scatter_sequence_fake(input: torch.Tensor, dim: int):
    output_shape = list(input.shape)
    output_shape[dim] = output_shape[dim] // sp_size()
    return input.new_empty(output_shape)


def _all_gather_sequence_backward_setup(ctx, inputs, output):
    _, dim = inputs
    ctx.dim = dim


def _all_gather_sequence_backward(ctx, grad):
    # Every rank consumes the whole gathered tensor, so the gradient of a local shard is the sum of
    # the matching slice over the SP group.
    return reduce_scatter_sequence(grad, ctx.dim), None


torch.library.register_autograd("autosp::all_gather_sequence",
                                _all_gather_sequence_backward,
                                setup_context=_all_gather_sequence_backward_setup)
