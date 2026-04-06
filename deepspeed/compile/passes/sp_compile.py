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

from deepspeed.compile import constants

from ..custom_ops import all_to_all, sp_dp_registry  # noqa: F401
from ..fx import find_node_by_name, get_node_shape_meta
from ..util import get_input_id_node, get_label_id_node, get_position_id_node, shard_tensor_node, get_sdpa_nodes


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

    input_id.tag = constants.AUTOSP_INPUT_ID_KEY
    label_id.tag = constants.AUTOSP_LABEL_ID_KEY
    if position_id is not None:
        position_id.tag = constants.AUTOSP_POSITION_ID_KEY

    return input_id, label_id, position_id, attention_mask


def pass_shard_seq_dim(gm: GraphModule, example_inputs):
    """
    Finds all direct and indirect consumers of the input sequence, label and position ids.
    Shard the sequence dimension used by all such consumers.
    """
    sp_size = sp_dp_registry.sp_size()

    input_ids_node = get_input_id_node(gm)
    val = get_node_shape_meta(input_ids_node)
    seq_symint = val.shape[1]
    assert isinstance(
        seq_symint,
        torch.SymInt), f"expected sequence dimension to be of type {torch.SymInt!r} but found {type(seq_symint)!r}"

    sym_seq_dim_node = find_node_by_name(gm, str(seq_symint))
    if sym_seq_dim_node is None:
        print(f"WARNING: Could not find the symbolic node for the sequence dimension")
        return

    with gm.graph.inserting_after(sym_seq_dim_node):
        sharded_node = gm.graph.call_function(operator.floordiv, args=(sym_seq_dim_node, sp_size))

    sharded_input_nodes = set()
    label_ids_node = get_label_id_node(gm)
    position_ids_node = get_position_id_node(gm)

    if input_ids_node is not None:
        sharded_input_nodes.add(input_ids_node)
    if label_ids_node is not None:
        sharded_input_nodes.add(label_ids_node)
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


def pass_shard_label_ids(gm: GraphModule, example_inputs):
    label_ids_node = get_label_id_node(gm)
    shard_tensor_node(gm, label_ids_node)


def pass_shard_position_ids(gm: GraphModule, example_inputs):
    position_ids_node = get_position_id_node(gm)
    if position_ids_node is None:
        print("[WARNING] position id node not found. Skipping sharding of position ids.")
        return
    shard_tensor_node(gm, position_ids_node)


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

    attention_nodes = get_sdpa_nodes(gm)
    if len(attention_nodes) == 0:
        raise RuntimeError("AutoSP currently supports torch.nn.functional.scaled_dot_product_attention as the "
                           "attention backend. No SDPA attention operations were found in the compiled graph. "
                           "Please ensure your model uses torch.nn.functional.scaled_dot_product_attention "
                           "for AutoSP to work as expected.")

    for idx, attn_node in enumerate(attention_nodes):
        q, k, v = attn_node.args[:3]
        suffix = f"_{idx}" if len(attention_nodes) > 1 else ""

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
        attn_mask = attn_node.kwargs.get("attn_mask")
        if attn_mask is not None:
            saved_sdpa_masks.append((attn_node, attn_mask))
            attn_node.update_kwarg("attn_mask", None)

    try:
        # fake_inputs are already created under fake_mode above, so run
        # propagation without reconverting them into a different fake mode.
        FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)
    finally:
        for attn_node, attn_mask in saved_sdpa_masks:
            attn_node.update_kwarg("attn_mask", attn_mask)


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
    assert sp_size * dp_size <= dist.get_world_size(), 'Insufficient device count for mesh size'

    sp_dp_registry.populate_registry(sp_size, dp_size)

    AUTOSP_PASSES = [
        pass_shard_seq_dim,
        pass_shard_input_ids,
        pass_shard_label_ids,
        pass_shard_position_ids,
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
