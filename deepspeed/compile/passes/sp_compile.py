"""AutoSP: Automatic Sequence Parallel (Ulysses) pass for graph modules.

Ulysses Transformation:
    Input:  [B, N, S/P, H]  (all heads, partitioned sequence)
    After A2A on QKV: [B, N/P, S, H]  (partitioned heads, full sequence)
    After SDPA: [B, N/P, S, H]
    After A2A on O: [B, N, S/P, H]  (all heads, partitioned sequence)

Where:
    B = batch size, N = num heads, S = full sequence length, H = head dim, P = world size
"""

import operator
from typing import Optional, List, Callable

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.experimental.symbolic_shapes import ShapeEnv


from ..custom_ops import all_to_all
from ..fx import find_node_by_name, get_node_shape_meta
from ..util import get_input_id_node, get_label_id_node, get_position_id_node, shard_tensor_node, get_sdpa_nodes, ShardingConfig

def pass_shard_seq_dim(gm: GraphModule, example_inputs):
    """
    Finds all direct and indirect consumers of the input sequence, label and position ids.
    Shard the sequence dimension used by all such consumers.
    """
    world_size = dist.get_world_size()
    
    input_ids_node = get_input_id_node(gm)
    val = get_node_shape_meta(input_ids_node)
    seq_symint = val.shape[1]
    assert isinstance(seq_symint, torch.SymInt), f"expected sequence dimension to be of type `torch.SymInt` but found `{type(seq_symint)}`"
    
    sym_seq_dim_node = find_node_by_name(gm, str(seq_symint))
    if sym_seq_dim_node is None:
        print(f"WARNING: Could not find the symbolic node for the sequence dimension")
        return
    
    with gm.graph.inserting_after(sym_seq_dim_node):
        sharded_node = gm.graph.call_function(
            operator.floordiv, 
            args=(sym_seq_dim_node, world_size)
        )
    
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
    config = ShardingConfig.from_distributed()
    input_ids_node = get_input_id_node(gm)
    shard_tensor_node(gm, input_ids_node, config)


def pass_shard_label_ids(gm: GraphModule, example_inputs):
    config = ShardingConfig.from_distributed()
    label_ids_node = get_label_id_node(gm)
    shard_tensor_node(gm, label_ids_node, config)

def pass_shard_position_ids(gm: GraphModule, example_inputs):
    config = ShardingConfig.from_distributed()
    position_ids_node = get_position_id_node(gm)
    if position_ids_node is None:
        print("[WARNING] position id node not found. Skipping sharding of position ids.")
        return
    shard_tensor_node(gm, position_ids_node, config)


def pass_insert_attention_all_to_all(gm: GraphModule, real_inputs):
    """
    Insert all-to-all collectives around SDPA for Ulysses parallelism.
    
    For each SDPA:
        - Before Q, K, V: scatter heads (dim=1), gather sequence (dim=2)
        - After O: scatter sequence (dim=2), gather heads (dim=1)
    """
    world_size = dist.get_world_size()
    attention_nodes = get_sdpa_nodes(gm)
    
    def insert_a2a(node: Node, scatter_idx: int, gather_idx: int, name: str) -> Node:
        with gm.graph.inserting_after(node):
            a2a_node = gm.graph.call_function(
                torch.ops.autosp.all_to_all.default,
                args=(node, scatter_idx, gather_idx, world_size, name),
            )
            a2a_node.name = f"a2a_{name}"
            node.replace_all_uses_with(a2a_node)
            a2a_node.update_arg(0, node)
        return a2a_node
    
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
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    fake_inputs = []
    for t in real_inputs:
        if isinstance(t, torch.Tensor):
            fake_inputs.append(fake_mode.from_tensor(t))
        else:
            fake_inputs.append(t)
    FakeTensorProp(gm).propagate(*fake_inputs)


def apply_autosp(
    gm: GraphModule, 
    real_inputs, 
    debug: bool = False,
    passes: Optional[List[Callable]] = None,
):
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
    
