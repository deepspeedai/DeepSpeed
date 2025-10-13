# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Tuple

import torch
from torch.fx import GraphModule

from ..util import get_deepcompile_handle
from ..fx import add_postprocess, move_primals_to_head, _make_node_meta, get_output_node

NAME = "zero1_compile"


def add_z1_reduce_fw(gm: GraphModule, graph_id: int, profiling_results, param_manager, use_z2=False) -> GraphModule:

    dc = get_deepcompile_handle()
    param_indices = profiling_results[graph_id].param_indices
    # Need this before profiling
    if use_z2:
        dc.register_graph_z2(graph_id, [v[1] for v in param_indices])
    else:
        dc.register_graph_z1(graph_id, [v[1] for v in param_indices])

    return gm


def add_z1_reduce_bw(gm: GraphModule, graph_id: int, param_manager) -> GraphModule:

    graph = gm.graph
    pm = param_manager[graph_id]
    _, param_name_to_grad = pm.get_bwd_mapping(graph)

    for param_name in pm.param_names:

        grad_node = param_name_to_grad[param_name]

        assert param_name in pm.ds_ids, f"param_name={param_name} not in ds_ids"
        ds_id = pm.ds_ids[param_name]

        new_node = add_postprocess(graph,
                                   grad_node,
                                   torch.ops.dc.reduce_grad.default,
                                   extra_args=[graph_id, ds_id],
                                   name=f"reduce_param_{param_name}",
                                   meta=_make_node_meta(grad_node, param_name, True))
        new_node.meta["val"] = None

    gm.graph = move_primals_to_head(graph)

    with gm.graph.inserting_before(get_output_node(gm.graph)):
        gm.graph.create_node("call_function", torch.ops.dc.end_backward.default, (graph_id, ))

    return gm


def add_z1_reduce(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                  create_inputs_fn, mem_budget: float, param_manager, bwd: bool) -> GraphModule:
    if bwd:
        return add_z1_reduce_bw(gm, graph_id, param_manager)
    return add_z1_reduce_fw(gm, graph_id, profiling_results, param_manager, use_z2=False)


def add_z2_reduce(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                  create_inputs_fn, mem_budget: float, param_manager, bwd: bool) -> GraphModule:
    if bwd:
        return add_z1_reduce_bw(gm, graph_id, param_manager)
    return add_z1_reduce_fw(gm, graph_id, profiling_results, param_manager, use_z2=True)
