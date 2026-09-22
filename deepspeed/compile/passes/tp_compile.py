# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
import re
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.fx import Graph, GraphModule, Node

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

# Set on the model by defer_collectives_to_compiler so the graph pass can check that every layer
# whose module-level collective it switched off actually received a graph-level one.
DEFERRED_NAMES_ATTR = "_ds_autotp_deferred_module_names"


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
    deferred_names = set()
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
        deferred_names.add(name)

    for module in tp_modules:
        module.defer_collectives_to_compiler = True
    setattr(model, DEFERRED_NAMES_ATTR, deferred_names)


def _originating_layer(node: Node) -> Tuple[Optional[str], Optional[type]]:
    """Return the (qualified name, type) of the innermost nn.Module a node was traced from."""
    module_stack = node.meta.get("nn_module_stack")
    if not module_stack:
        return None, None
    fqn, module_type = list(module_stack.values())[-1]
    return fqn, module_type


def _originating_layer_type(node: Node):
    return _originating_layer(node)[1]


_MODULES_INDEX = re.compile(r"_modules\['([^']*)'\]")
_ROOT_PREFIX = re.compile(r"^(?:L\['[^']*'\]|self)(?:\.|$)")


def _normalize_fqn(fqn: Optional[str]) -> Optional[str]:
    """Turn Dynamo's nn_module_stack path into the dotted name model.named_modules() uses.

    Dynamo records the access expression, e.g. "L['self']._modules['layers']._modules['0'].down_proj",
    which has to become "layers.0.down_proj" before it can be compared against the set of layers
    whose collectives were deferred.
    """
    if fqn is None:
        return None
    name = _MODULES_INDEX.sub(r"\1", fqn)
    name = _ROOT_PREFIX.sub("", name)
    return name.strip(".")


def iter_graphs(gm: GraphModule) -> Iterator[Graph]:
    """Yield the graph of gm and of every nested GraphModule.

    Uses modules() rather than recursing through named_children(): a GraphModule can sit under a
    plain nn.Module, which a GraphModule-only recursion would never reach. nn.Module.modules()
    already deduplicates by identity.

    Activation checkpointing (and any other higher-order op) lifts its body into a child
    GraphModule referenced by a get_attr node. A pass that walks only gm.graph never sees those
    nodes, so the parallel matmuls inside a checkpointed block would keep their module-level
    collectives switched off while never receiving a graph-level replacement.
    """
    for module in gm.modules():
        if isinstance(module, GraphModule):
            yield module.graph


def _insert_row_collective(graph: Graph, matmul: Node) -> Node:
    """Insert g after a row-parallel matmul.

    Every consumer has to read the reduced value, which is also what the module-level
    RowParallel.apply this replaces produces.
    """
    with graph.inserting_after(matmul):
        collective_node = graph.call_function(ROW_PARALLEL_OP, args=(matmul, ))
    collective_node.meta["val"] = matmul.meta.get("val")
    collective_node.meta["nn_module_stack"] = matmul.meta.get("nn_module_stack")
    matmul.replace_all_uses_with(collective_node)
    collective_node.update_arg(0, matmul)
    return collective_node


def _insert_column_collective(graph: Graph, activation: Node, consumers: List[Node]) -> Node:
    """Insert f in front of the column-parallel matmuls that share activation."""
    with graph.inserting_before(consumers[0]):
        collective_node = graph.call_function(COLUMN_PARALLEL_OP, args=(activation, ))
    collective_node.meta["val"] = activation.meta.get("val")
    collective_node.meta["nn_module_stack"] = consumers[0].meta.get("nn_module_stack")
    for consumer in consumers:
        consumer.replace_input_with(activation, collective_node)
    return collective_node


def _rewrite_graph(graph: Graph) -> Tuple[Set[str], Set[str]]:
    """Insert f/g into one graph. Returns (layer names reached, layer names given a collective)."""
    reached: Set[str] = set()
    handled: Set[str] = set()
    column_consumers: Dict[Node, List[Node]] = {}
    row_matmuls: List[Node] = []

    for node in list(graph.nodes):
        fqn, layer_type = _originating_layer(node)
        name = _normalize_fqn(fqn)
        # Deliberately computed over EVERY node op, not just call_function. When a layer sits inside
        # a checkpointed region, the only trace of it left in the root graph is the placeholder for
        # its lifted weight -- so a call_function-only scan reports nothing reached, `missed` comes
        # out empty, and the check cannot see the very failure it exists to catch.
        if name is not None and (_is_column_parallel(layer_type) or _is_row_parallel(layer_type)):
            reached.add(name)

        if node.op != "call_function" or node.target not in _MATMUL_TARGETS:
            continue

        if _is_row_parallel(layer_type):
            row_matmuls.append(node)
            handled.add(name)
        elif _is_column_parallel(layer_type):
            activation = node.args[0]
            column_consumers.setdefault(activation, []).append(node)
            handled.add(name)

    for matmul in row_matmuls:
        _insert_row_collective(graph, matmul)

    for activation, consumers in column_consumers.items():
        _insert_column_collective(graph, activation, consumers)

    return reached, handled


def pass_insert_tp_collectives(gm: GraphModule, real_inputs, deferred_names=None, **kwargs):
    """Insert the tensor-parallel collectives around the matmuls of the injected AutoTP layers.

    Only f and g are inserted here. The output gather of a gather_output layer is emitted by the
    layer's own forward (see LinearLayer.forward): it changes the activation's width, so the ops
    downstream of it only trace correctly if it is already present during graph capture.

    Every graph is walked, nested ones included, and the layers reached are checked against the
    layers that received a collective. Silently leaving a layer out is not a missed optimization:
    its module-level collective is already switched off, so the layer would simply stop
    communicating and the run would produce wrong numbers without any error.
    """
    reached: Set[str] = set()
    handled: Set[str] = set()

    for graph in iter_graphs(gm):
        graph_reached, graph_handled = _rewrite_graph(graph)
        reached |= graph_reached
        handled |= graph_handled

    missed = reached - handled
    if deferred_names and (reached & set(deferred_names)):
        # Only layers whose module-level collective was actually switched off can go silently wrong.
        # The filter is applied only when the two naming schemes demonstrably line up: if they do
        # not, intersecting would quietly empty the set and turn this check into a no-op, which is
        # the exact failure mode it exists to catch.
        missed &= set(deferred_names)
    if missed:
        raise RuntimeError(
            f"AutoTP compile pass reached {len(missed)} tensor-parallel layer(s) in the graph without "
            f"inserting a collective for them: {sorted(missed)[:8]}. Their module-level collectives are "
            "already switched off, so continuing would silently produce wrong results. This usually means "
            "the layer's matmul was traced to an op the pass does not recognize, or into a nested graph "
            "it did not walk.")


def pass_canonicalize(gm: GraphModule, real_inputs, **kwargs):
    for module in gm.modules():
        if isinstance(module, GraphModule):
            module.graph.eliminate_dead_code()
            module.graph.lint()
            module.recompile()


AUTOTP_PASSES = [
    pass_insert_tp_collectives,
    pass_canonicalize,
]


def _run_pass(opt_pass, gm: GraphModule, real_inputs, **context):
    """Call a pass, handing it only the extra context it actually accepts.

    `passes` is a caller-supplied extension point whose contract is (gm, real_inputs). Passing a new
    keyword to every pass unconditionally would raise TypeError on any two-argument pass written
    against that contract, before the pass even runs.
    """
    try:
        parameters = inspect.signature(opt_pass).parameters
    except (TypeError, ValueError):  # builtins and some C callables expose no signature
        return opt_pass(gm, real_inputs)

    takes_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    accepted = {k: v for k, v in context.items() if takes_var_keyword or k in parameters}
    return opt_pass(gm, real_inputs, **accepted)


def apply_autotp(gm: GraphModule, real_inputs, passes=None, deferred_names=None):
    """Apply the AutoTP transformation passes to the graph."""
    for opt_pass in passes or AUTOTP_PASSES:
        _run_pass(opt_pass, gm, real_inputs, deferred_names=deferred_names)
    return gm
