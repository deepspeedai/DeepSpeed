# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Tuple

import torch
from torch.fx import Node, GraphModule
from deepspeed.compile.util import get_last_uses
from ..graph_param import DSGraphParamManager


def add_offload_parameter(graph_id: int, gm: GraphModule, node: Node, ds_id: int):
    new_node = None
    with gm.graph.inserting_after(node):
        args = (node, )
        for a in [graph_id, ds_id]:  # To add ds_id
            args += (a, )
        new_node = gm.graph.create_node('call_function',
                                        torch.ops.dc.offload_parameter.default,
                                        args, {},
                                        name="offload_parameter")

    return new_node


def add_reload_parameter(graph_id: int, gm: GraphModule, node: Node, ds_id: int):
    new_node = None
    with gm.graph.inserting_after(node):
        args = (node, )
        for a in [graph_id, ds_id]:  # To add ds_id
            args += (a, )
        new_node = gm.graph.create_node('call_function',
                                        torch.ops.dc.reload_parameter.default,
                                        args, {},
                                        name="reload_parameter")
    return new_node


def get_ds_id(node: Node):
    assert node.target == torch.ops.dc.allgather_param.default
    return node.args[2]


def offload_parameter_fwd(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                          create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                          bwd: bool) -> GraphModule:
    node_to_last_use, user_to_last_uses = get_last_uses(gm.graph)
    for node in gm.graph.nodes:
        if (isinstance(node, Node) and node.target == torch.ops.dc.allgather_param.default):
            add_reload_parameter(graph_id, gm, node.args[0], get_ds_id(node))
            add_offload_parameter(graph_id, gm, node_to_last_use[node], get_ds_id(node))
    gm.graph.lint()
    return gm
