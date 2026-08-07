# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict, List

import torch
from torch.fx import GraphModule, Node

from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer

from ..custom_ops import tp_collectives  # noqa: F401

COLUMN_PARALLEL_OP = torch.ops.autotp.copy_to_tp_region.default
ROW_PARALLEL_OP = torch.ops.autotp.reduce_from_tp_region.default

# AutoTP replaces nn.Linear with these layers and shards their weights, so the injected layer type
# already records the partitioning decision the pass needs. Reading it back is more robust than
# re-deriving column/row from parameter-name patterns.
COLUMN_PARALLEL_LAYER = LinearLayer
ROW_PARALLEL_LAYER = LinearAllreduce

# The injected layers compute their matmul with torch.matmul; the plain nn.Linear spelling is
# accepted too so the pass keeps working if a layer is lowered differently.
_MATMUL_TARGETS = {
    torch.matmul,
    torch.ops.aten.matmul.default,
    torch.ops.aten.linear.default,
    torch._C._nn.linear,
}


def defer_collectives_to_compiler(model) -> int:
    """Suppress the module-level TP collectives on layers this pass will handle in the graph.

    Returns the number of layers handed over to the pass. Layers the pass does not rewrite (the
    fused sub-param variants, conv and embedding layers) keep their module-level collectives and
    stay correct as-is.
    """
    deferred = 0
    for name, module in model.named_modules():
        is_row_parallel = type(module) is ROW_PARALLEL_LAYER
        is_column_parallel = type(module) is COLUMN_PARALLEL_LAYER
        if not (is_row_parallel or is_column_parallel):
            continue
        if module.mp_group is None:
            continue
        if type(module).tp_overlap_comm:
            raise NotImplementedError("AutoTP compile pass does not support tp_overlap_comm. Set "
                                      "'tp_overlap_comm': false to emit the collectives into the graph.")
        # GatherFromTensorParallelRegion reads the gathered shard sizes back into Python, which the
        # full graph this pass needs cannot capture. Leaving such a layer on the module-level path
        # is not an option either: the pass identifies column-parallel layers by type, so it would
        # add a second collective on top of the module's own and reduce the input gradient twice.
        if is_column_parallel and module.gather_output:
            raise NotImplementedError(
                f"AutoTP compile pass does not support gather_output layers, but '{name}' is one. Partition it "
                "without gather_output, or drop 'autotp' from the DeepCompile passes for this model.")
        module.defer_collectives_to_compiler = True
        deferred += 1
    return deferred


def _originating_layer_type(node: Node):
    """Return the innermost nn.Module type a node was traced from, or None."""
    module_stack = node.meta.get("nn_module_stack")
    if not module_stack:
        return None
    _, module_type = list(module_stack.values())[-1]
    return module_type


def _insert_row_collective(gm: GraphModule, matmul: Node) -> Node:
    """Insert g after a row-parallel matmul.

    Every consumer has to read the reduced value, which is also what the module-level
    RowParallel.apply this replaces produces.
    """
    with gm.graph.inserting_after(matmul):
        collective_node = gm.graph.call_function(ROW_PARALLEL_OP, args=(matmul, ))
    collective_node.meta["val"] = matmul.meta.get("val")
    matmul.replace_all_uses_with(collective_node)
    collective_node.update_arg(0, matmul)
    return collective_node


def _insert_column_collective(gm: GraphModule, activation: Node, consumers: List[Node]) -> Node:
    """
    Insert f in front of the column-parallel matmuls that share activation.
    """
    with gm.graph.inserting_before(consumers[0]):
        collective_node = gm.graph.call_function(COLUMN_PARALLEL_OP, args=(activation, ))
    collective_node.meta["val"] = activation.meta.get("val")
    for consumer in consumers:
        consumer.replace_input_with(activation, collective_node)
    return collective_node


def pass_insert_tp_collectives(gm: GraphModule, real_inputs):
    """Insert the tensor-parallel collectives around the matmuls of the injected AutoTP layers."""
    column_consumers: Dict[Node, List[Node]] = {}

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _MATMUL_TARGETS:
            continue

        layer_type = _originating_layer_type(node)
        if layer_type is ROW_PARALLEL_LAYER:
            _insert_row_collective(gm, node)
        elif layer_type is COLUMN_PARALLEL_LAYER:
            activation = node.args[0]
            column_consumers.setdefault(activation, []).append(node)

    for activation, consumers in column_consumers.items():
        _insert_column_collective(gm, activation, consumers)


def pass_canonicalize(gm: GraphModule, real_inputs):
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


AUTOTP_PASSES = [
    pass_insert_tp_collectives,
    pass_canonicalize,
]


def apply_autotp(gm: GraphModule, real_inputs, passes=None):
    """Apply the AutoTP transformation passes to the graph.

    The collectives are shape-preserving, so unlike AutoSP this needs no shape re-propagation.
    """
    for opt_pass in passes or AUTOTP_PASSES:
        opt_pass(gm, real_inputs)
    return gm
