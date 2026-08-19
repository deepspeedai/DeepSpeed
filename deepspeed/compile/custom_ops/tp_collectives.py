# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist
from deepspeed.utils import groups


def get_tp_group():
    """Return the tensor-parallel group created by the existing AutoTP setup."""
    return groups.get_tensor_model_parallel_group()


@torch.library.custom_op("autotp::copy_to_tp_region", mutates_args=())
def copy_to_tp_region(input: torch.Tensor) -> torch.Tensor:
    """Identity in the forward pass, all-reduce in the backward pass.

    Inserted before a column-parallel matmul: the activation is
    already replicated across the tensor-parallel group, so nothing has to happen in the forward
    pass, while each rank contributes a partial gradient that must be summed in the backward pass.
    """
    return input.clone()


@torch.library.register_fake("autotp::copy_to_tp_region")
def copy_to_tp_region_fake(input: torch.Tensor):
    return torch.empty_like(input)


@torch.library.custom_op("autotp::reduce_from_tp_region", mutates_args=())
def reduce_from_tp_region(input: torch.Tensor) -> torch.Tensor:
    """All-reduce in the forward pass, identity in the backward pass.

    Inserted after a row-parallel matmul, whose output is only a
    partial sum because each rank holds a slice of the input dimension.
    """
    output = input.contiguous().clone()
    dist.all_reduce(output, group=get_tp_group())
    return output


@torch.library.register_fake("autotp::reduce_from_tp_region")
def reduce_from_tp_region_fake(input: torch.Tensor):
    return torch.empty_like(input)


@torch.library.custom_op("autotp::gather_from_tp_region", mutates_args=())
def gather_from_tp_region(input: torch.Tensor) -> torch.Tensor:
    """All-gather the last dimension in the forward pass, take this rank's slice in the backward.

    Inserted after a column-parallel matmul whose layer asks for gather_output, so that
    every rank leaves the layer holding the full output rather than its own shard.
    """
    group = get_tp_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input.clone()

    local_shard = input.contiguous()
    flat_gathered = torch.empty((world_size * local_shard.shape[0], *local_shard.shape[1:]),
                                dtype=local_shard.dtype,
                                device=local_shard.device)
    dist.all_gather_into_tensor(flat_gathered, local_shard, group=group)
    shards = flat_gathered.view(world_size, *local_shard.shape)
    return torch.cat(shards.unbind(0), dim=-1)


@torch.library.register_fake("autotp::gather_from_tp_region")
def gather_from_tp_region_fake(input: torch.Tensor):
    world_size = dist.get_world_size(group=get_tp_group())
    return input.new_empty((*input.shape[:-1], input.shape[-1] * world_size))


def _copy_to_tp_region_backward(ctx, grad):
    # copy_to_tp_region and reduce_from_tp_region are duals,
    # so copy_to_tp_region's backward is simply reduce_from_tp_region.
    return reduce_from_tp_region(grad.contiguous())


def _reduce_from_tp_region_backward(ctx, grad):
    return grad


def _gather_from_tp_region_backward(ctx, grad):
    group = get_tp_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return grad
    shard_width = grad.shape[-1] // world_size
    shard_start = dist.get_rank(group=group) * shard_width
    return grad.narrow(-1, shard_start, shard_width).contiguous()


def _setup_context_without_saved_tensors(ctx, inputs, output):
    pass


torch.library.register_autograd("autotp::copy_to_tp_region",
                                _copy_to_tp_region_backward,
                                setup_context=_setup_context_without_saved_tensors)
torch.library.register_autograd("autotp::reduce_from_tp_region",
                                _reduce_from_tp_region_backward,
                                setup_context=_setup_context_without_saved_tensors)
torch.library.register_autograd("autotp::gather_from_tp_region",
                                _gather_from_tp_region_backward,
                                setup_context=_setup_context_without_saved_tensors)
