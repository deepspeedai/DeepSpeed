# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist
from deepspeed.utils import groups


def get_tp_group():
    """Return the tensor-parallel group created by the existing AutoTP setup.

    The AutoTP pass reuses the groups that ``TpTrainingManager`` already builds, so the compiled
    collectives always communicate over the same group as the module-level ones they replace.
    """
    return groups.get_tensor_model_parallel_group()


@torch.library.custom_op("autotp::copy_to_tp_region", mutates_args=())
def copy_to_tp_region(input: torch.Tensor) -> torch.Tensor:
    """Identity in the forward pass, all-reduce in the backward pass.

    This is Megatron's ``f``. It is inserted before a column-parallel matmul: the activation is
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

    This is Megatron's ``g``. It is inserted after a row-parallel matmul, whose output is only a
    partial sum because each rank holds a slice of the input dimension.
    """
    output = input.contiguous().clone()
    dist.all_reduce(output, group=get_tp_group())
    return output


@torch.library.register_fake("autotp::reduce_from_tp_region")
def reduce_from_tp_region_fake(input: torch.Tensor):
    return torch.empty_like(input)


def _copy_to_tp_region_backward(ctx, grad):
    # f and g are duals, so f's backward is simply g.
    return reduce_from_tp_region(grad.contiguous())


def _reduce_from_tp_region_backward(ctx, grad):
    return grad


def _setup_context_without_saved_tensors(ctx, inputs, output):
    # Both collectives are shape-preserving and stateless, so their backwards need nothing saved.
    pass


torch.library.register_autograd("autotp::copy_to_tp_region",
                                _copy_to_tp_region_backward,
                                setup_context=_setup_context_without_saved_tensors)
torch.library.register_autograd("autotp::reduce_from_tp_region",
                                _reduce_from_tp_region_backward,
                                setup_context=_setup_context_without_saved_tensors)
