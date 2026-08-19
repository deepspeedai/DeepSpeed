# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict, List

import torch
from torch.fx import GraphModule, Node

from deepspeed.module_inject.layers import (LinearAllreduce, LinearLayer, LmHeadLinearAllreduce,
                                            SubParamLinearAllreduce, SubParamLinearLayer, TensorParallel_Layer)

from ..custom_ops import tp_collectives  # noqa: F401

COLUMN_PARALLEL_OP = torch.ops.autotp.copy_to_tp_region.default
ROW_PARALLEL_OP = torch.ops.autotp.reduce_from_tp_region.default
GATHER_OUTPUT_OP = torch.ops.autotp.gather_from_tp_region.default

# AutoTP replaces nn.Linear with these layers and shards their weights, so the injected layer type
# already records the partitioning decision the pass needs.

COLUMN_PARALLEL_LAYERS = (LinearLayer, SubParamLinearLayer)
ROW_PARALLEL_LAYERS = (LinearAllreduce, SubParamLinearAllreduce)
# LmHeadLinearAllreduce subclasses LinearAllreduce but slices its own input and reduces with
# inference_all_reduce rather than going through RowParallel, so the pass cannot stand in for it.
UNSUPPORTED_LAYERS = (LmHeadLinearAllreduce, )

_MATMUL_TARGETS = {
    torch.matmul,
    torch.ops.aten.matmul.default,
    torch.ops.aten.linear.default,
    torch._C._nn.linear,
}


def _is_column_parallel(layer_type) -> bool:
    return _in_family(layer_type, COLUMN_PARALLEL_LAYERS)


def _is_row_parallel(layer_type) -> bool:
    return _in_family(layer_type, ROW_PARALLEL_LAYERS)


def _in_family(layer_type, family) -> bool:
    if not isinstance(layer_type, type) or issubclass(layer_type, UNSUPPORTED_LAYERS):
        return False
    return issubclass(layer_type, family)


def defer_collectives_to_compiler(model) -> None:
    """Suppress the module-level TP collectives on layers this pass will handle in the graph."""
    tp_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, TensorParallel_Layer) or module.mp_group is None:
            continue

        layer_type = type(module)
        is_column_parallel = _is_column_parallel(layer_type)
        if not (is_column_parallel or _is_row_parallel(layer_type)):
            raise NotImplementedError(
                f"AutoTP compile pass cannot rewrite '{name}' ({layer_type.__name__}), and leaving it on the "
                "module-level path under a full graph would silently drop its backward collective. Drop "
                "'autotp' from the DeepCompile passes for this model.")
        if layer_type.tp_overlap_comm:
            raise NotImplementedError("AutoTP compile pass does not support tp_overlap_comm. Set "
                                      "'tp_overlap_comm': false to emit the collectives into the graph.")

        tp_modules.append(module)

    for module in tp_modules:
        module.defer_collectives_to_compiler = True


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
    """Insert f in front of the column-parallel matmuls that share activation."""
    with gm.graph.inserting_before(consumers[0]):
        collective_node = gm.graph.call_function(COLUMN_PARALLEL_OP, args=(activation, ))
    collective_node.meta["val"] = activation.meta.get("val")
    for consumer in consumers:
        consumer.replace_input_with(activation, collective_node)
    return collective_node


def pass_insert_tp_collectives(gm: GraphModule, real_inputs):
    """Insert the tensor-parallel collectives around the matmuls of the injected AutoTP layers.

    Only f and g are inserted here. The output gather of a gather_output layer is emitted by the
    layer's own forward (see LinearLayer.forward): it changes the activation's width, so the ops
    downstream of it only trace correctly if it is already present during graph capture.
    """
    column_consumers: Dict[Node, List[Node]] = {}

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _MATMUL_TARGETS:
            continue

        layer_type = _originating_layer_type(node)
        if _is_row_parallel(layer_type):
            _insert_row_collective(gm, node)
        elif _is_column_parallel(layer_type):
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
    """Apply the AutoTP transformation passes to the graph."""
    for opt_pass in passes or AUTOTP_PASSES:
        opt_pass(gm, real_inputs)
    return gm
