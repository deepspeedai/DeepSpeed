# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
from typing import Optional, List, Callable

import torch
import deepspeed.comm as dist
from torch._subclasses.fake_tensor import FakeTensorMode, maybe_get_fake_mode
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.nn._reduction import legacy_get_string

from deepspeed.compile import constants

from ..custom_ops import all_to_all, sp_dp_registry  # noqa: F401
from ..fx import find_node_by_name, get_node_shape_meta
from ..util import (get_autosp_seq_dim, get_input_id_node, get_label_id_node, get_position_id_node, shard_tensor_node,
                    get_sdpa_nodes, insert_sequence_shard)


def prepare_autosp_inputs(input_id: torch.Tensor,
                          label_id: torch.Tensor,
                          position_id: torch.Tensor = None,
                          attention_mask: torch.Tensor = None,
                          seq_dim: int = 1):
    """
    Prepare inputs for AutoSP by marking dynamic dimensions and tagging tensors.

    Args:
        input_id: Token IDs tensor (required)
        label_id: Label IDs tensor (required)
        position_id: Position IDs tensor (optional)
        attention_mask: Attention mask tensor (optional)
        seq_dim: Sequence dimension index to mark as dynamic (default: 1)
    """

    if input_id is None:
        raise ValueError("input_id is required")
    if label_id is None:
        raise ValueError("label_id is required")

    if seq_dim < 0 or seq_dim >= input_id.ndim:
        raise ValueError(f"seq_dim {seq_dim} must be a valid index for input_id with shape {input_id.shape}")
    if seq_dim >= label_id.ndim:
        raise ValueError(f"seq_dim {seq_dim} is out of bounds for label_id with shape {label_id.shape}")

    if position_id is not None:
        if seq_dim >= position_id.ndim:
            raise ValueError(f"seq_dim {seq_dim} is out of bounds for position_id with shape {position_id.shape}")

    if attention_mask is not None:
        if seq_dim >= attention_mask.ndim:
            raise ValueError(
                f"seq_dim {seq_dim} is out of bounds for attention_mask with shape {attention_mask.shape}")

    torch._dynamo.decorators.mark_dynamic(input_id, seq_dim)
    torch._dynamo.decorators.mark_dynamic(label_id, seq_dim)
    if position_id is not None:
        torch._dynamo.decorators.mark_dynamic(position_id, seq_dim)
    if attention_mask is not None:
        torch._dynamo.decorators.mark_dynamic(attention_mask, seq_dim)

    # Dynamo only carries the `tag` attribute into graph metadata, so seq_dim must travel inside it.
    input_id.tag = (constants.AUTOSP_INPUT_ID_KEY, seq_dim)
    label_id.tag = (constants.AUTOSP_LABEL_ID_KEY, seq_dim)
    if position_id is not None:
        position_id.tag = (constants.AUTOSP_POSITION_ID_KEY, seq_dim)

    return input_id, label_id, position_id, attention_mask


def pass_shard_seq_dim(gm: GraphModule, example_inputs):
    """
    Finds all direct and indirect consumers of the input sequence, label and position ids.
    Shard the sequence dimension used by all such consumers.
    """
    sp_size = sp_dp_registry.sp_size()

    input_ids_node = get_input_id_node(gm)
    val = get_node_shape_meta(input_ids_node)
    seq_symint = val.shape[get_autosp_seq_dim(input_ids_node)]
    assert isinstance(
        seq_symint,
        torch.SymInt), f"expected sequence dimension to be of type {torch.SymInt!r} but found {type(seq_symint)!r}"

    sym_seq_dim_node = find_node_by_name(gm, str(seq_symint))
    if sym_seq_dim_node is None:
        print(f"WARNING: Could not find the symbolic node for the sequence dimension")
        return

    with gm.graph.inserting_after(sym_seq_dim_node):
        sharded_node = gm.graph.call_function(operator.floordiv, args=(sym_seq_dim_node, sp_size))

    # Labels stay whole: pass_shard_label_ids gives each cross entropy the shard of its target.
    sharded_input_nodes = set()
    position_ids_node = get_position_id_node(gm)

    if input_ids_node is not None:
        sharded_input_nodes.add(input_ids_node)
    if position_ids_node is not None:
        sharded_input_nodes.add(position_ids_node)

    # find all consumers of the sharded inputs
    consumer_nodes = set()
    worklist = list(sharded_input_nodes)
    visited = set()

    while worklist:
        node = worklist.pop(0)
        if node in visited:
            continue
        visited.add(node)
        consumer_nodes.add(node)

        for user in node.users:
            if user not in visited:
                worklist.append(user)

    to_replace = []
    for node in consumer_nodes:
        if sym_seq_dim_node in node.all_input_nodes:
            to_replace.append(node)

    for user in to_replace:
        user.replace_input_with(sym_seq_dim_node, sharded_node)


def pass_shard_input_ids(gm: GraphModule, example_inputs):
    input_ids_node = get_input_id_node(gm)
    shard_tensor_node(gm, input_ids_node)


def _get_cross_entropy_arg(loss_node: Node, name: str, position: int, default):
    if name in loss_node.kwargs:
        return loss_node.kwargs[name]
    if len(loss_node.args) > position:
        return loss_node.args[position]
    return default


_RANK_PRESERVING_METHODS = {"contiguous", "float", "to", "long", "int", "half", "bfloat16"}
_RESHAPE_METHODS = {"view", "reshape", "flatten"}
_RESHAPE_FUNCTIONS = {torch.reshape, torch.flatten}


def _is_rank_preserving(node: Node) -> bool:
    return node.op == "call_method" and node.target in _RANK_PRESERVING_METHODS


def _is_reshape(node: Node) -> bool:
    if node.op == "call_method":
        return node.target in _RESHAPE_METHODS
    return node.op == "call_function" and node.target in _RESHAPE_FUNCTIONS


def _is_tensor_getitem(node: Node) -> bool:
    if node.op != "call_function" or node.target is not operator.getitem:
        return False
    return isinstance(get_node_shape_meta(node.args[0]), torch.Tensor)


def _get_seq_slice(node: Node, seq_dim: int):
    """
    Return ``(start, stop)`` of a getitem that only narrows ``seq_dim`` by static amounts, where
    ``start >= 0`` counts from the front and ``stop <= 0`` counts from the end.
    """
    ndim = get_node_shape_meta(node.args[0]).ndim
    index = node.args[1] if isinstance(node.args[1], tuple) else (node.args[1], )
    if index.count(Ellipsis) == 1:
        position = index.index(Ellipsis)
        index = index[:position] + (slice(None), ) * (ndim - len(index) + 1) + index[position + 1:]
    index = index + (slice(None), ) * (ndim - len(index))

    unsupported = RuntimeError(f"AutoSP cannot shard cross entropy logits indexed by {node.args[1]!r}; only static "
                               f"slices of the sequence dimension such as logits[..., :-1, :] are supported")
    if len(index) != ndim or not all(isinstance(i, slice) for i in index):
        raise unsupported
    for dim, dim_slice in enumerate(index):
        if dim_slice.step not in (None, 1):
            raise unsupported
        if dim != seq_dim and (dim_slice.start not in (None, 0) or dim_slice.stop is not None):
            raise unsupported

    start = index[seq_dim].start or 0
    stop = 0 if index[seq_dim].stop is None else index[seq_dim].stop
    if not isinstance(start, int) or not isinstance(stop, int) or start < 0 or stop > 0:
        raise unsupported
    if index[seq_dim].stop == 0:
        raise unsupported
    return start, stop


def _find_seq_dim(node: Node, length: torch.SymInt) -> int:
    """Return the only dimension of ``node`` whose symbolic size is ``length``."""
    val = get_node_shape_meta(node)
    dims = [
        dim for dim, size in enumerate(val.shape)
        if isinstance(size, torch.SymInt) and size.node.expr == length.node.expr
    ]
    if len(dims) != 1:
        raise RuntimeError(f"AutoSP expected exactly one dimension of size {length} in {node.format_node()} "
                           f"with shape {tuple(val.shape)} to line up with the sequence")
    return dims[0]


def _undo_logits_seq_slices(input_node: Node, seq_len: torch.SymInt, sharded_nodes):
    """
    Remove the sequence slices between the logits and a cross entropy, as each rank must keep the
    logits of its whole shard. Return the window ``[start, seq_len + stop)`` of the full sequence that
    the cross entropy input covered before the slices were removed.
    """
    chain = []
    slices = []
    node = input_node
    while _is_rank_preserving(node) or _is_reshape(node) or _is_tensor_getitem(node):
        if _is_tensor_getitem(node):
            slices.append(node)
        elif _is_reshape(node) and slices:
            # A reshape that runs before a slice would let the slice cut across the sequence dimension.
            raise RuntimeError("AutoSP only supports slicing the cross entropy logits before they are reshaped")
        chain.append(node)
        node = node.args[0]

    if node not in sharded_nodes:
        raise RuntimeError(f"AutoSP could not trace the cross entropy input back to sequence-sharded logits; "
                           f"stopped at {node.format_node()}")

    seq_dim = _find_seq_dim(node, seq_len)
    start, stop = 0, 0
    for slice_node in slices:
        slice_start, slice_stop = _get_seq_slice(slice_node, seq_dim)
        start += slice_start
        stop += slice_stop
    if slices and any(len(chain_node.users) != 1 for chain_node in chain):
        raise RuntimeError("AutoSP cannot remove the sequence slice of cross entropy logits used elsewhere")
    for slice_node in slices:
        slice_node.replace_all_uses_with(slice_node.args[0])
        slice_node.graph.erase_node(slice_node)
    return start, stop


def _localize_cross_entropy_target(gm: GraphModule, loss_node: Node, window, seq_len: torch.SymInt, ignore_index):
    """
    Replace the full-sequence target of a cross entropy with the part that lines up with this rank's
    logits. Positions outside the logits window get ``ignore_index``.
    """
    consumer = loss_node
    target = _get_cross_entropy_arg(loss_node, "target", 1, None)
    while _is_rank_preserving(target) or _is_reshape(target):
        if _is_reshape(target) and len(target.all_input_nodes) > 1:
            # A size taken from the full sequence would not fit this rank's shorter target.
            raise RuntimeError(f"AutoSP cannot reshape a cross entropy target with explicit symbolic sizes in "
                               f"{target.format_node()}; flatten it with view(-1) instead")
        consumer = target
        target = target.args[0]

    start, stop = window
    # The target pairs with the logits position by position, so it covers as many positions as the window.
    seq_dim = _find_seq_dim(target, seq_len + stop - start)
    target_val = get_node_shape_meta(target)

    padded = target
    if start or stop:
        anchor = target
        if target.op == "placeholder":
            anchor = [node for node in gm.graph.nodes if node.op == "placeholder"][-1]
        # Pad with cat rather than F.pad, which would specialize the symbolic sequence length.
        pieces = [target]
        for width, at_front in ((start, True), (-stop, False)):
            if width == 0:
                continue
            with gm.graph.inserting_after(anchor):
                anchor = gm.graph.call_function(torch.narrow, args=(target, seq_dim, 0, width))
            with gm.graph.inserting_after(anchor):
                anchor = gm.graph.call_function(torch.full_like, args=(anchor, ignore_index))
            pieces.insert(0 if at_front else len(pieces), anchor)
        with gm.graph.inserting_after(anchor):
            padded = gm.graph.call_function(torch.cat, args=(pieces, seq_dim))
    local_target = insert_sequence_shard(gm, padded, target_val.ndim, seq_dim, find_node_by_name(gm, str(seq_len)))
    consumer.replace_input_with(target, local_target)


def _check_label_consumers(label_ids_node: Node, sharded_nodes):
    """Labels stay unsharded, so they may only flow into cross entropy targets."""
    worklist = [label_ids_node]
    label_nodes = set()
    while worklist:
        node = worklist.pop()
        if node in label_nodes:
            continue
        label_nodes.add(node)
        if any(input_node in sharded_nodes for input_node in node.all_input_nodes):
            raise RuntimeError(f"AutoSP labels cannot be combined with sequence-sharded tensors in "
                               f"{node.format_node()}")
        for user in node.users:
            if user.target is torch.nn.functional.cross_entropy:
                if _get_cross_entropy_arg(user, "target", 1, None) is not node:
                    raise RuntimeError(f"AutoSP labels may only be the target of a cross entropy, got "
                                       f"{user.format_node()}")
            elif user.op == "output":
                raise RuntimeError("AutoSP labels cannot be returned from the compiled module")
            else:
                worklist.append(user)


def pass_shard_label_ids(gm: GraphModule, example_inputs):
    """
    Make every cross entropy on the labels return the loss of the full sequence.

    The labels stay whole, so any shift the model applies to them sees the full sequence. Each cross
    entropy then takes the part of its target that lines up with this rank's logits, sums its loss and
    combines the sums over the SP group, which weights shards with different numbers of valid tokens
    correctly.
    """
    label_ids_node = get_label_id_node(gm)
    input_ids_node = get_input_id_node(gm)
    seq_len = get_node_shape_meta(input_ids_node).shape[get_autosp_seq_dim(input_ids_node)]

    sharded_nodes = set()
    worklist = [node for node in (input_ids_node, get_position_id_node(gm)) if node is not None]
    while worklist:
        node = worklist.pop()
        if node not in sharded_nodes:
            sharded_nodes.add(node)
            worklist.extend(node.users)

    _check_label_consumers(label_ids_node, sharded_nodes)

    label_users = set()
    worklist = [label_ids_node]
    while worklist:
        node = worklist.pop()
        if node not in label_users:
            label_users.add(node)
            worklist.extend(node.users)

    loss_nodes = [
        node for node in gm.graph.nodes if node.target is torch.nn.functional.cross_entropy
        and _get_cross_entropy_arg(node, "target", 1, None) in label_users
    ]

    for loss_node in loss_nodes:
        weight = _get_cross_entropy_arg(loss_node, "weight", 2, None)
        if weight is not None:
            raise RuntimeError("AutoSP does not support class-weighted cross entropy on sequence-sharded labels")
        reduction = _get_cross_entropy_arg(loss_node, "reduction", 6, "mean")
        size_average = _get_cross_entropy_arg(loss_node, "size_average", 3, None)
        reduce = _get_cross_entropy_arg(loss_node, "reduce", 5, None)
        if size_average is not None or reduce is not None:
            # The deprecated arguments override reduction inside cross entropy.
            reduction = legacy_get_string(size_average, reduce, emit_warning=False)
        if reduction not in ("mean", "sum"):
            raise RuntimeError(f"AutoSP supports cross entropy with reduction 'mean' or 'sum', got {reduction!r}")
        ignore_index = _get_cross_entropy_arg(loss_node, "ignore_index", 4, -100)
        if get_node_shape_meta(_get_cross_entropy_arg(loss_node, "target", 1, None)).dtype.is_floating_point:
            # Class probability targets have no ignore_index, so counting valid tokens does not apply.
            raise RuntimeError("AutoSP supports cross entropy with class index targets only, not class probabilities")

        window = _undo_logits_seq_slices(_get_cross_entropy_arg(loss_node, "input", 0, None), seq_len, sharded_nodes)
        _localize_cross_entropy_target(gm, loss_node, window, seq_len, ignore_index)

        target_node = _get_cross_entropy_arg(loss_node, "target", 1, None)
        # Under autocast, cross entropy returns FP32 even for half-precision logits.
        loss_dtype = get_node_shape_meta(loss_node).dtype
        with gm.graph.inserting_after(target_node):
            valid_mask = gm.graph.call_function(operator.ne, args=(target_node, ignore_index))
        with gm.graph.inserting_after(valid_mask):
            valid_tokens = gm.graph.call_function(torch.sum, args=(valid_mask, ))

        # Sum the per-token losses in FP32: a half-precision sum over a long shard overflows even
        # where the mean the model asked for would not.
        if "reduction" in loss_node.kwargs or len(loss_node.args) <= 6:
            loss_node.update_kwarg("reduction", "none")
        else:
            loss_node.update_arg(6, "none")
        for name, position in (("size_average", 3), ("reduce", 5)):
            if name in loss_node.kwargs:
                loss_node.update_kwarg(name, None)
            elif len(loss_node.args) > position:
                loss_node.update_arg(position, None)

        with gm.graph.inserting_after(loss_node):
            token_losses = gm.graph.call_method("float", args=(loss_node, ))
        with gm.graph.inserting_after(token_losses):
            local_loss_sum = gm.graph.call_function(torch.sum, args=(token_losses, ))
        with gm.graph.inserting_after(local_loss_sum):
            aggregated = gm.graph.call_function(torch.ops.autosp.aggregate_loss.default,
                                                args=(local_loss_sum, valid_tokens, reduction == "mean"))
        with gm.graph.inserting_after(aggregated):
            global_loss = gm.graph.call_function(operator.getitem, args=(aggregated, 0))
        with gm.graph.inserting_after(global_loss):
            # Return the loss in the dtype cross entropy would have produced.
            output_loss = gm.graph.call_method("to", args=(global_loss, loss_dtype))
        loss_node.replace_all_uses_with(output_loss)
        token_losses.update_arg(0, loss_node)


def pass_shard_position_ids(gm: GraphModule, example_inputs):
    position_ids_node = get_position_id_node(gm)
    if position_ids_node is None:
        print("[WARNING] position id node not found. Skipping sharding of position ids.")
        return
    shard_tensor_node(gm, position_ids_node)


def get_sdpa_attn_mask(attn_node: Node):
    """Return SDPA's attn_mask argument and whether it was passed as a keyword."""
    if "attn_mask" in attn_node.kwargs:
        return attn_node.kwargs["attn_mask"], True
    if len(attn_node.args) > 3:
        return attn_node.args[3], False
    return None, True


def set_sdpa_attn_mask(attn_node: Node, attn_mask, as_kwarg: bool):
    if as_kwarg:
        attn_node.update_kwarg("attn_mask", attn_mask)
    else:
        attn_node.update_arg(3, attn_mask)


def gather_attn_mask_queries(gm: GraphModule, attn_mask: Node, q: Node, k: Node) -> Node:
    """
    Rewrite an SDPA mask for the layout after the Q/K/V all-to-all, where each rank attends with
    the full query sequence and its own block of heads.

    A mask computed from the local sequence shard only holds this rank's query rows, so the rows
    are gathered across the SP group. A mask with a full head dimension must also be reduced to
    this rank's heads.
    """
    mask_meta = get_node_shape_meta(attn_mask)
    if mask_meta is None or mask_meta.ndim < 2:
        return attn_mask
    # Without sharding every rank already holds the full mask, and its full key dimension is
    # indistinguishable from a sharded one.
    if sp_dp_registry.sp_size() == 1:
        return attn_mask

    if mask_meta.ndim == 3 and mask_meta.shape[0] != 1:
        # SDPA aligns a 3D mask as [heads, queries, keys]; add the batch dimension so the head
        # redistribution below sees heads in dimension 1.
        with gm.graph.inserting_after(attn_mask):
            attn_mask = gm.graph.call_function(torch.unsqueeze, args=(attn_mask, 0))
        attn_mask.meta["val"] = mask_meta.unsqueeze(0)
        mask_meta = attn_mask.meta["val"]

    q_meta = get_node_shape_meta(q)
    mask_keys = mask_meta.shape[-1]
    # A key dimension of 1 is broadcast by SDPA, not sharded.
    if mask_keys != 1 and str(mask_keys) == str(get_node_shape_meta(k).shape[2]):
        # The other ranks' key columns were never computed here, so they cannot be gathered.
        raise RuntimeError("AutoSP cannot rebuild an attention mask whose key dimension is sharded along the "
                           f"sequence (mask shape {tuple(mask_meta.shape)}). Build the mask from the full, "
                           "unsharded sequence length instead, for example from the 2D padding attention_mask.")
    query_sharded = str(mask_meta.shape[-2]) == str(q_meta.shape[2])
    per_head = mask_meta.ndim == 4 and mask_meta.shape[1] != 1

    if per_head:
        if str(mask_meta.shape[1]) != str(q_meta.shape[1]):
            raise RuntimeError(f"AutoSP expects the head dimension of the attention mask {tuple(mask_meta.shape)} "
                               f"to be 1 or the number of query heads ({q_meta.shape[1]})")
        if query_sharded:
            # Same redistribution as the query: [B, N, S/P, S] -> [B, N/P, S, S].
            with gm.graph.inserting_after(attn_mask):
                return gm.graph.call_function(torch.ops.autosp.all_to_all.default, args=(attn_mask, 1, 2, "attn_mask"))
        # The query all-to-all hands rank r the r-th contiguous block of heads.
        sp_size = sp_dp_registry.sp_size()
        local_heads = q_meta.shape[1] // sp_size
        sp_rank = dist.get_rank() % sp_size
        with gm.graph.inserting_after(attn_mask):
            return gm.graph.call_function(torch.narrow, args=(attn_mask, 1, sp_rank * local_heads, local_heads))

    if not query_sharded:
        return attn_mask
    with gm.graph.inserting_after(attn_mask):
        return gm.graph.call_function(torch.ops.autosp.all_gather_sequence.default, args=(attn_mask, -2))


def pass_insert_attention_all_to_all(gm: GraphModule, real_inputs):

    def insert_a2a(node: Node, scatter_idx: int, gather_idx: int, name: str) -> Node:
        with gm.graph.inserting_after(node):
            a2a_node = gm.graph.call_function(
                torch.ops.autosp.all_to_all.default,
                args=(node, scatter_idx, gather_idx, name),
            )
            a2a_node.name = f"a2a_{name}"
            node.replace_all_uses_with(a2a_node)
            a2a_node.update_arg(0, node)
        return a2a_node

    sp_size = sp_dp_registry.sp_size()
    attention_nodes = get_sdpa_nodes(gm)
    if len(attention_nodes) == 0:
        raise RuntimeError("AutoSP currently supports torch.nn.functional.scaled_dot_product_attention as the "
                           "attention backend. No SDPA attention operations were found in the compiled graph. "
                           "Please ensure your model uses torch.nn.functional.scaled_dot_product_attention "
                           "for AutoSP to work as expected.")

    # Layers commonly share one mask node, so gather it once.
    gathered_masks = {}
    for idx, attn_node in enumerate(attention_nodes):
        q, k, v = attn_node.args[:3]
        suffix = f"_{idx}" if len(attention_nodes) > 1 else ""

        # Q/K/V are checked separately: with GQA, K/V can have fewer heads than Q.
        for role, tensor in (("query", q), ("key", k), ("value", v)):
            meta = get_node_shape_meta(tensor) if isinstance(tensor, Node) else None
            if meta is None:
                continue
            if meta.ndim != 4:
                raise RuntimeError(f"AutoSP expects 4D [batch, heads, seq, head_dim] SDPA {role} tensors, "
                                   f"but got shape {tuple(meta.shape)}")
            num_heads = meta.shape[1]
            if isinstance(num_heads, int) and num_heads % sp_size != 0:
                raise ValueError(f"AutoSP requires the number of {role} heads ({num_heads}) to be divisible by "
                                 f"sequence_parallel_size ({sp_size})")

        attn_mask, mask_as_kwarg = get_sdpa_attn_mask(attn_node)
        if isinstance(attn_mask, Node):
            if attn_mask not in gathered_masks:
                gathered_masks[attn_mask] = gather_attn_mask_queries(gm, attn_mask, q, k)
            set_sdpa_attn_mask(attn_node, gathered_masks[attn_mask], mask_as_kwarg)

        # QKV: [B, N, S/P, H] -> [B, N/P, S, H]
        insert_a2a(q, scatter_idx=1, gather_idx=2, name=f"q{suffix}")
        insert_a2a(k, scatter_idx=1, gather_idx=2, name=f"k{suffix}")
        insert_a2a(v, scatter_idx=1, gather_idx=2, name=f"v{suffix}")

        # O: [B, N/P, S, H] -> [B, N, S/P, H]
        insert_a2a(attn_node, scatter_idx=2, gather_idx=1, name=f"o{suffix}")


def pass_canonicalize(gm: GraphModule, real_inputs):
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def pass_propagate_shapes(gm: torch.fx.GraphModule, real_inputs):
    fake_mode = None
    for node in gm.graph.nodes:
        # Reuse the graph's existing fake mode when metadata is already present.
        # Its ShapeEnv owns the symbolic dims captured during tracing, so using a
        # fresh mode here can desynchronize fake inputs from graph metadata.
        if node.op == "placeholder" and "val" in node.meta:
            fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_mode = maybe_get_fake_mode(fake_val)
        elif fake_mode is None:
            fake_val = node.meta.get("example_value", node.meta.get("val"))
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_mode = maybe_get_fake_mode(fake_val)
        if fake_mode is not None:
            break

    if fake_mode is None:
        # Some graphs do not carry fake tensor metadata yet; create a fallback
        # mode so FakeTensorProp can still run shape-only execution.
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    fake_inputs = []
    for t in real_inputs:
        if isinstance(t, torch.Tensor):
            fake_inputs.append(fake_mode.from_tensor(t))
        else:
            fake_inputs.append(t)

    # Torch 2.9 can fail fake propagation through SDPA's masked fake-CUDA path,
    # even though this pass only needs output metadata. Temporarily clear
    # attn_mask so shape propagation can proceed, then restore it immediately;
    # SDPA output shapes are still determined by Q/K/V shapes, not mask values.
    saved_sdpa_masks = []
    for attn_node in get_sdpa_nodes(gm):
        attn_mask, mask_as_kwarg = get_sdpa_attn_mask(attn_node)
        if attn_mask is not None:
            saved_sdpa_masks.append((attn_node, attn_mask, mask_as_kwarg))
            set_sdpa_attn_mask(attn_node, None, mask_as_kwarg)

    try:
        # fake_inputs are already created under fake_mode above, so run
        # propagation without reconverting them into a different fake mode.
        FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)
    finally:
        for attn_node, attn_mask, mask_as_kwarg in saved_sdpa_masks:
            set_sdpa_attn_mask(attn_node, attn_mask, mask_as_kwarg)


def apply_autosp(gm: GraphModule,
                 real_inputs,
                 debug: bool = False,
                 passes: Optional[List[Callable]] = None,
                 sp_size: int = 2,
                 dp_size: int = 1):
    """
    Apply AutoSP (Ulysses) transformation passes to the graph and setup either DP/SP (2D) or SP (1D) mesh.

    Args:
        gm: GraphModule to transform
        real_inputs: Example inputs for shape propagation
        debug: If True, print graph before/after each pass
        passes: Optional custom list of passes (default: DEFAULT_PASSES)
    """
    sp_dp_registry.populate_registry(sp_size, dp_size)

    AUTOSP_PASSES = [
        pass_shard_seq_dim,
        pass_shard_input_ids,
        pass_shard_label_ids,
        pass_shard_position_ids,
        # The attention pass compares mask and Q/K/V shapes after sharding, so refresh them first.
        pass_propagate_shapes,
        pass_insert_attention_all_to_all,
        pass_propagate_shapes,
        pass_canonicalize,
    ]

    passes = passes or AUTOSP_PASSES
    rank = dist.get_rank()

    for p in passes:
        if debug and rank == 0:
            print(f"\n{'='*60}")
            print(f" BEFORE: {p.__name__}")
            print(f"{'='*60}\n")
            print(gm.print_readable(print_output=False))

        p(gm, real_inputs)

        if debug and rank == 0:
            print(f"\n{'='*60}")
            print(f" AFTER: {p.__name__}")
            print(f"{'='*60}\n")
            print(gm.print_readable(print_output=False))
