# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple, List

import torch
from torch.fx import GraphModule, Graph, Node

try:
    from torch.utils.checkpoint import CheckpointPolicy
    from torch._functorch.partitioners import _is_primal
except ImportError:
    pass

from .util import get_no_copy_ops, is_cast_op


def _recompute_param_aliases(joint_graph: Graph, param_indices: List[Tuple[int, int, torch.Size]]):
    """Recompute nodes aliasing or downcasting any parameter

    In ZeRO3, sharded parameters are gathered before use and the gathered
    parameters should be freed once they are no longer needed to save GPU
    memory.

    When DeepCompile is active for ZeRO3, parameter gathering is done by custom
    passes after the joint graph captured by Dynamo and AOT Autograd is
    partitioned into fwd and bwd parts. Since the partitioner has no clue about
    parameter sharding now, the partitioned graphs will save for backward all
    intermediate activations including those aliasing the gathered parameters.
    That essentially nullifies the memory reduction that ZeRO3 is designed to
    bring.

    The solution is to recompute the parameter-aliasing activations in the
    backward. It is done by marking such nodes as MUST_RECOMPUTE and reusing the
    min-cut partitioner originally designed for checkpointing. If autocast is
    enabled, parameter downcasts are also recomputed.

    This cannot be converted to a standalone pass because it must be applied
    before partitioning the joint graph, but passes run after the partitioning.

    TODO(eternalNight) `min_cut_rematerialization_partition` may recompute more
    nodes than required for ZeRO3. Need investigate its performance
    implications.
    """
    no_copy_ops = get_no_copy_ops()

    def need_recompute(n: Node) -> bool:
        if n.op == "call_function":
            is_cast, _ = is_cast_op(n)
            return n.target in no_copy_ops or is_cast
        return False

    primal_inputs = list(filter(_is_primal, joint_graph.nodes))
    ds_param_inputs = set([primal_inputs[arg_idx] for arg_idx, _, _ in param_indices])
    recomputed_nodes = set()

    for node in joint_graph.nodes:
        # The `ac_graph_id` tag tracks the checkpoint module that a node belongs
        # to, and is for enforcing the saving of activations at the boundary of
        # consecutive checkpointed blocks. It starts from 1 and increments by 1
        # each time a graph module is checkpointed.
        #
        # `min_cut_rematerialization_partition` requires every node to have
        # `ac_graph_id`. If this graph is not checkpointed (and thus
        # `ac_graph_id` is missing), we tag all nodes to 1 to prevent the
        # partition function from modifying the recompute tag.
        node.meta.setdefault("ac_graph_id", 1)

        # Arguments can be non-tensor types some of which are not hashable. So
        # we must inspect the type of an argument before checking if it is in
        # any set.
        if need_recompute(node) and \
            any([(isinstance(a, Node) and (a in ds_param_inputs or a in recomputed_nodes)) for a in node.args]):
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
            recomputed_nodes.add(node)
        else:
            # If checkpointing is not enabled for this graph, assume all
            # activations required by the backward pass should be saved.
            node.meta.setdefault("recompute", CheckpointPolicy.MUST_SAVE)


def get_wrapped_partitioner(
    z3_partition: bool,
    param_indices: List[Tuple[int, int, torch.Size]],
    partition_fn,
):

    def partition_recompute_ds_params(joint_module: GraphModule, _joint_inputs, *, num_fwd_outputs,
                                      **kwargs) -> Tuple[GraphModule, GraphModule]:
        if z3_partition:
            _recompute_param_aliases(joint_module.graph, param_indices)
        return partition_fn(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs, **kwargs)

    return partition_recompute_ds_params
