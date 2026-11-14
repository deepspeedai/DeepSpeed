# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Sequence sharding for AutoTP training (``tensor_parallel.sequence_sharding``), the scheme Megatron-LM calls
sequence parallelism. It is unrelated to DeepSpeed's Ulysses sequence parallelism (``sequence_parallel_size``),
which splits attention over ranks of its own.

Without it, every tensor-parallel rank runs the norms, residual adds and other token-wise ops of a decoder layer
on the full sequence and keeps their activations at full size. With it, the hidden states between the
tensor-parallel blocks (the attention and MLP modules) are split over the tensor-parallel ranks, 1/tp of the tokens
each (one contiguous chunk of the flattened batch and sequence):

    block input   [b, s/tp, h]  -- all-gather -->      [b, s, h]  -> column-parallel linears -> ...
    row output    [b, s, h] partial sums  -- reduce-scatter -->  [b, s/tp, h]

In the forward pass the all-gather and the reduce-scatter replace the row-parallel all-reduce. In the backward
pass, the reduce-scatter of the all-gather sums the input gradients of all the column-parallel linears of a block
at once, instead of one all-reduce per linear (Qwen3: q, k and v share one; gate and up share one).

Requirements, checked when it is applied:
  - every tensor-parallel linear inside the decoder layers belongs to a block: the parent module of a row-parallel
    linear, whose first tensor argument (or ``hidden_states``) is the input of its column-parallel linears;
  - every decoder layer of the layer list has at least one block;
  - the sequence length is divisible by the tensor-parallel size.
Every replicated parameter inside the decoder layers (norm weights, row-parallel biases) then sees only this rank's
part of the sequence or of the heads, so its gradient is summed over the tensor-parallel group.
"""

import torch
from torch import nn

from deepspeed import comm as dist
from deepspeed.comm import pg_utils
from deepspeed.runtime.zero.linear import autocast_custom_bwd, autocast_custom_fwd
from deepspeed.runtime.zero.utils import is_zero_param
from deepspeed.utils.logging import print_dist
from deepspeed.utils.torch import register_grad_hook

SEQUENCE_SHARDING_ATTR = "_ds_autotp_sequence_sharding"


def _shape_with(x, dim, size):
    return (*x.shape[:dim], size, *x.shape[dim + 1:])


def _check_divisible(x, dim, world_size):
    if x.shape[dim] % world_size != 0:
        raise ValueError(f"AutoTP sequence sharding: sequence length {x.shape[dim]} (dim {dim}) is not divisible by "
                         f"the tensor-parallel size {world_size}")


# The ops between blocks act on each token alone, so a rank may hold any 1/tp of the tokens. Each rank holds one
# contiguous chunk of the flattened [batch * sequence] tokens: the collectives then work on contiguous buffers and
# need no reordering copy for any batch size (splitting each sequence instead needs two copies per collective at
# batch > 1). The local tensor keeps the shape [..., sequence / tp, ...].


def _split(x, dim, world_size, rank):
    _check_divisible(x, dim, world_size)
    flat = x.contiguous().view(-1, x.shape[-1])
    chunk = flat.shape[0] // world_size
    return flat[rank * chunk:(rank + 1) * chunk].view(_shape_with(x, dim, x.shape[dim] // world_size))


def _all_gather(x, dim, group, world_size):
    x = x.contiguous()
    hidden = x.shape[-1]
    gathered = x.new_empty((world_size * (x.numel() // hidden), hidden))
    dist.all_gather_into_tensor(gathered, x.view(-1, hidden), group=group)
    return gathered.view(_shape_with(x, dim, x.shape[dim] * world_size))


def _reduce_scatter(x, dim, group, world_size, async_op=False):
    _check_divisible(x, dim, world_size)
    x = x.contiguous()
    hidden = x.shape[-1]
    out = x.new_empty((x.numel() // hidden // world_size, hidden))
    handle = dist.reduce_scatter_tensor(out, x.view(-1, hidden), group=group, async_op=async_op)
    out = out.view(_shape_with(x, dim, x.shape[dim] // world_size))
    return (out, handle) if async_op else out


class ScatterToSequenceShards(torch.autograd.Function):
    """Enter the sequence-sharded region: keep this rank's part of a replicated tensor; backward all-gathers."""

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        return _split(x, dim, ctx.world_size, dist.get_rank(group=group))

    @staticmethod
    def backward(ctx, grad):
        return _all_gather(grad, ctx.dim, ctx.group, ctx.world_size), None, None


class GatherFromSequenceShards(torch.autograd.Function):
    """Leave the sequence-sharded region: all-gather into a replicated tensor; backward keeps this rank's part.

    The ops after it (final norm, output head, loss) are replicated, so every rank receives the same full gradient.
    """

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        return _all_gather(x, dim, group, ctx.world_size)

    @staticmethod
    def backward(ctx, grad):
        return _split(grad, ctx.dim, ctx.world_size, dist.get_rank(group=ctx.group)), None, None


class GatherToTensorParallelBlock(torch.autograd.Function):
    """All-gather a block input for its column-parallel linears; backward reduce-scatters their partial gradients."""

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        return _all_gather(x, dim, group, ctx.world_size)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter(grad, ctx.dim, ctx.group, ctx.world_size), None, None


class ReduceScatterFromTensorParallelBlock(torch.autograd.Function):
    """Sum the row-parallel partial outputs and keep this rank's part of the sequence; backward all-gathers."""

    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        return _reduce_scatter(x, dim, group, ctx.world_size)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather(grad, ctx.dim, ctx.group, ctx.world_size), None, None


def reduce_scatter_from_tensor_parallel_block(x, group, dim):
    return ReduceScatterFromTensorParallelBlock.apply(x, group, dim)


class ColumnGroupFromSequenceShards(torch.autograd.Function):
    """All-gather a block input and run every column-parallel linear that reads it, in one autograd node.

    Each linear keeps its own GEMM, so its output stays contiguous for the reshapes that follow (one GEMM over the
    concatenated weights gives strided outputs that later kernels copy). Backward sums the input gradients of the
    linears and reduce-scatters them while the weight-gradient GEMMs run, instead of after them. Returns the
    gathered input (for the block's own use of its shape) and one output per weight.

    Only this rank's part of the input is saved for backward, which gathers it again while the input-gradient GEMMs
    run: saving the gathered input would keep tp times as much activation memory per block.
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x_shard, group, dim, *weights):
        ctx.set_materialize_grads(False)
        ctx.group, ctx.dim = group, dim
        ctx.world_size = dist.get_world_size(group=group)
        x_full = _all_gather(x_shard, dim, group, ctx.world_size)
        outs = [torch.matmul(x_full, w.t()) for w in weights]
        ctx.save_for_backward(x_shard, *weights)
        return (x_full, *outs)

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, grad_x_full, *grad_outs):
        x_shard, *weights = ctx.saved_tensors
        hidden = x_shard.shape[-1]
        x_local = x_shard.contiguous().view(-1, hidden)
        full_shape = _shape_with(x_shard, ctx.dim, x_shard.shape[ctx.dim] * ctx.world_size)
        x_2d = x_local.new_empty((ctx.world_size * x_local.shape[0], hidden))
        gather = dist.all_gather_into_tensor(x_2d, x_local, group=ctx.group, async_op=True)
        # Sum of the input gradients of all the linears; addmm_ accumulates inside each GEMM.
        grad_in_2d = None
        for grad_out, weight in zip(grad_outs, weights):
            if grad_out is None:
                continue
            grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1])
            if grad_in_2d is None:
                grad_in_2d = torch.matmul(grad_out_2d, weight)
            else:
                # In-place ops are not autocast: match dtypes by hand (fp32 weights under torch_autocast).
                grad_in_2d.addmm_(grad_out_2d.to(grad_in_2d.dtype), weight.to(grad_in_2d.dtype))
        if grad_in_2d is None:
            grad_in_2d = x_2d.new_zeros(x_2d.shape)
        grad_in = grad_in_2d.view(full_shape)
        if grad_x_full is not None:
            grad_in = grad_in + grad_x_full
        grad_shard, handle = _reduce_scatter(grad_in, ctx.dim, ctx.group, ctx.world_size, async_op=True)
        gather.wait()
        grad_weights = []
        for grad_out in grad_outs:
            if grad_out is None:
                grad_weights.append(None)
            else:
                grad_weights.append(torch.matmul(grad_out.reshape(-1, grad_out.shape[-1]).t(), x_2d))
        handle.wait()
        return (grad_shard, None, None, *grad_weights)


class _BlockState:
    """Which column-parallel linears of a block read the block input, learned in the model's first forward pass."""

    def __init__(self, columns, passes):
        self.columns = columns
        self.readers = []
        self.passes = passes  # one-element list: the model's forward passes so far
        self.learned_in = None  # the pass that recorded the readers

    def can_fuse(self):
        weights = [layer.weight for layer in self.readers]
        if not weights or any(is_zero_param(w) or w.dim() != 2 for w in weights):
            return False
        return len({(w.dtype, w.device, w.shape[1]) for w in weights}) == 1


def take_precomputed_column_output(layer, input):
    """The output the block pre-hook already computed for ``layer`` from ``input``, or None.

    In the model's first forward pass nothing is precomputed; a layer that reads the gathered block input records
    itself instead, so that later passes compute it inside ColumnGroupFromSequenceShards.
    """
    pending = getattr(layer, "_ds_sequence_sharding_pending", None)
    if pending is None:
        return None
    layer._ds_sequence_sharding_pending = None
    state, x_full, out = pending
    if x_full is not input:
        return None
    if out is None:
        state.readers.append(layer)
    return out


def _replace_hidden_states(args, kwargs, fn, where):
    if args and torch.is_tensor(args[0]):
        return (fn(args[0]), *args[1:]), kwargs
    if torch.is_tensor(kwargs.get("hidden_states")):
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = fn(kwargs["hidden_states"])
        return args, kwargs
    raise RuntimeError(f"AutoTP sequence sharding: {where} was called without a tensor as its first argument or "
                       "as hidden_states")


def _layer_list_of(block_name):
    """'model.layers.3.self_attn' -> ('model.layers', 3)."""
    parts = block_name.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return ".".join(parts[:i]), int(parts[i])
    return None, None


def apply_autotp_sequence_sharding(model, group, seq_dim=1, overlap=True, zero_stage=0):
    """Split the hidden states between AutoTP blocks along ``seq_dim`` over ``group``. Call after AutoTP has
    replaced the linears. ``overlap``: run the column-parallel linears that read a block input in one autograd node
    and overlap their backward reduce-scatter (ColumnGroupFromSequenceShards). ``zero_stage``: the ZeRO stage, which
    decides how the replicated parameters' gradients are summed. Returns the number of blocks."""
    from deepspeed.module_inject.layers import (DS_TENSOR_MODEL_PARALLEL, LinearAllreduce, LinearLayer,
                                                TensorParallel_Layer)

    if not pg_utils.is_parallel(group) or getattr(model, SEQUENCE_SHARDING_ATTR, False):
        return 0

    row_layers = {name: m for name, m in model.named_modules() if isinstance(m, LinearAllreduce)}
    if not row_layers:
        raise ValueError("AutoTP sequence sharding found no row-parallel linear in the model")
    blocks = {}
    for name in row_layers:
        if "." not in name:
            raise ValueError(f"AutoTP sequence sharding: row-parallel linear {name!r} has no parent module")
        parent = name.rsplit(".", 1)[0]
        blocks[parent] = model.get_submodule(parent)

    list_names = {_layer_list_of(name)[0] for name in blocks}
    if len(list_names) != 1 or None in list_names:
        raise ValueError("AutoTP sequence sharding needs all tensor-parallel blocks inside one numbered layer list; "
                         f"found blocks {sorted(blocks)}")
    list_name = list_names.pop()
    layers = model.get_submodule(list_name)
    if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
        raise ValueError(f"AutoTP sequence sharding: {list_name!r} is not a non-empty nn.ModuleList")
    layers_with_blocks = {_layer_list_of(name)[1] for name in blocks}
    missing = sorted(set(range(len(layers))) - layers_with_blocks)
    if missing:
        raise ValueError(f"AutoTP sequence sharding: decoder layers {missing} of {list_name!r} have no "
                         "tensor-parallel block; their ops would see a split sequence")

    # Every tensor-parallel layer inside the decoder layers must belong to a block, and column outputs must stay
    # sharded (a gathered column output would mix a split sequence and a full one).
    for name, m in layers.named_modules(prefix=list_name):
        if not isinstance(m, TensorParallel_Layer):
            continue
        if not any(name.startswith(block + ".") for block in blocks):
            raise ValueError(f"AutoTP sequence sharding: tensor-parallel layer {name!r} is outside every block")
        if getattr(m, "defer_collectives_to_compiler", False):
            raise NotImplementedError("AutoTP sequence sharding does not support the DeepCompile 'autotp' pass")
        if isinstance(m, LinearLayer) and getattr(m, "gather_output", False):
            raise NotImplementedError(f"AutoTP sequence sharding: column-parallel layer {name!r} gathers its output")
        # The sequence-sharding paths live in LinearLayer.forward and LinearAllreduce.forward; a subclass with its
        # own forward would keep its all-reduce and reduce the gradients twice.
        if type(m).forward not in (LinearLayer.forward, LinearAllreduce.forward):
            raise NotImplementedError(f"AutoTP sequence sharding does not support {type(m).__name__} ({name!r})")
        m.sequence_sharding = True
        m.sequence_sharding_dim = seq_dim

    def make_gather_hook(state):

        def gather(x):
            # Fused only from the pass after the one that recorded the readers: activation checkpointing recomputes
            # the block in backward within the same pass, and the recomputation must save what the forward saved.
            if overlap and state.learned_in is not None and state.passes[0] > state.learned_in and state.can_fuse():
                readers = state.readers
                weights = [layer.weight for layer in readers]
                x_full, *outs = ColumnGroupFromSequenceShards.apply(x, group, seq_dim, *weights)
                for layer, layer_out in zip(readers, outs):
                    layer._ds_sequence_sharding_pending = (state, x_full, layer_out)
                return x_full
            x_full = GatherToTensorParallelBlock.apply(x, group, seq_dim)
            if overlap and state.learned_in is None:
                state.learned_in = state.passes[0]
                for layer in state.columns:
                    layer._ds_sequence_sharding_pending = (state, x_full, None)
            return x_full

        def hook(module, args, kwargs):
            return _replace_hidden_states(args, kwargs, gather, type(module).__name__)

        return hook

    def make_release_hook(state):
        # A column-parallel linear that never read the gathered block input (it reads another tensor of the block,
        # or the block uses its weight without calling it) still holds that input; drop it with the block.
        def hook(module, args, output):
            for layer in state.columns:
                layer._ds_sequence_sharding_pending = None

        return hook

    passes = [0]

    def count_pass(module, args):
        passes[0] += 1

    model.register_forward_pre_hook(count_pass)

    for block_name, block in blocks.items():
        columns = [
            m for name, m in block.named_modules(prefix=block_name)
            if isinstance(m, LinearLayer) and getattr(m, "sequence_sharding", False)
        ]
        state = _BlockState(columns, passes)
        block.register_forward_pre_hook(make_gather_hook(state), with_kwargs=True)
        block.register_forward_hook(make_release_hook(state))

    def enter(module, args, kwargs):
        return _replace_hidden_states(args, kwargs, lambda x: ScatterToSequenceShards.apply(x, group, seq_dim),
                                      type(module).__name__)

    def leave(module, args, output):
        if torch.is_tensor(output):
            return GatherFromSequenceShards.apply(output, group, seq_dim)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (GatherFromSequenceShards.apply(output[0], group, seq_dim), *output[1:])
        raise RuntimeError(f"AutoTP sequence sharding: {type(module).__name__} returned {type(output).__name__}")

    layers[0].register_forward_pre_hook(enter, with_kwargs=True)
    layers[-1].register_forward_hook(leave)

    # Replicated parameters in the decoder layers see this rank's part of the sequence (norms, row biases) or of
    # the heads (q_norm/k_norm, which AutoTP may already reduce), so their gradients are partial sums.
    def make_hook():

        def hook(grad):
            grad = grad.contiguous()
            dist.all_reduce(grad, group=group)
            return grad

        return hook

    # ZeRO-3 replaces torch.nn.functional.linear with a custom autograd function that saves the weight, and under
    # non-reentrant activation checkpointing PyTorch then skips the weight's tensor hooks (torch 2.8): a replicated
    # linear weight (router, shared expert, unsharded MLP) would keep its partial gradient. ZeRO-3 reduces and
    # frees every micro-step's gradient in its own post-accumulate hook, so a post-accumulate hook registered before
    # it (the same way, with register_grad_hook) sees exactly this micro-step's gradient.
    def make_accumulated_grad_hook(param):

        def hook(*unused):
            dist.all_reduce(param.grad, group=group)

        return hook

    reduced = []
    for name, param in layers.named_parameters(prefix=list_name):
        if not param.requires_grad or getattr(param, DS_TENSOR_MODEL_PARALLEL, False):
            continue
        if getattr(param, "_ds_grad_allreduce_registered", False):
            continue
        if zero_stage == 3:
            register_grad_hook(param, make_accumulated_grad_hook(param))
        else:
            param.register_hook(make_hook())
        param._ds_grad_allreduce_registered = True
        reduced.append(name)

    setattr(model, SEQUENCE_SHARDING_ATTR, True)
    print_dist(
        f"AutoTP sequence sharding: {len(blocks)} blocks in {len(layers)} layers of {list_name!r}, sequence dim "
        f"{seq_dim}, overlap={overlap}; tensor-parallel gradient sum for {len(reduced)} replicated parameters"
        f"{', e.g. ' + repr(reduced[0]) if reduced else ''}",
        ranks=[0])
    return len(blocks)
