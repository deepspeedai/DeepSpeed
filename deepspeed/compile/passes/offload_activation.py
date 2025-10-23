# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Dict, Set, Tuple
import random
from collections import defaultdict

import torch
from torch.fx import Graph, Node

from ..fx import get_output_node, move_primals_to_head
from ..graph_param import DSGraphParamManager

value_to_id: Dict[int, Dict[str, int]] = defaultdict(dict)
used_ids: Set[int] = set()


def get_random_id() -> int:

    def _gen():
        # generate random int
        return random.randint(10000, 2**31)

    global used_ids
    v = _gen()
    while v in used_ids:
        v = _gen()
    used_ids.add(v)
    return v


def _should_offload(node: Node) -> bool:
    if not hasattr(node, "meta"):
        return False
    if "tensor_meta" not in node.meta:
        return False

    return True


def offload_activation_fwd(graph: Graph, graph_id: int, nodes_to_offload_with_names: List[Tuple[str, Node]],
                           graph_order: List[Tuple[int, bool]], mem_budget: float,
                           param_manager: DSGraphParamManager) -> Graph:
    param_names = set(param_manager.param_names)

    import copy
    cl_graph = copy.deepcopy(graph)
    cl_graph.erase_node(get_output_node(cl_graph))

    global value_to_id
    for name, node in nodes_to_offload_with_names:
        if node.name in param_names:
            continue

        if not _should_offload(node):
            continue

        val_id = get_random_id()
        with graph.inserting_after(node):
            offload_node = graph.create_node('call_function',
                                             torch.ops.dc.offload_tensor.default, (node, graph_id, val_id), {},
                                             name=f"offload_{node.name}_{val_id}")
        with graph.inserting_after(offload_node):
            wait_node = graph.create_node('call_function',
                                          torch.ops.dc.wait_offload.default, (offload_node, graph_id, val_id), {},
                                          name=f"wait_copy_{node.name}_{val_id}")

        output_node = get_output_node(graph)
        output_node.replace_input_with(node, wait_node)

        value_to_id[graph_id][name] = val_id

    graph = move_primals_to_head(graph)

    graph.lint()
    return graph


def reload_activation_bwd(graph: Graph, graph_id: int, graph_order: List[Tuple[int, bool]], mem_budget: float,
                          param_manager: DSGraphParamManager) -> Graph:

    graph_value_to_id = value_to_id[graph_id]
    name_to_node = {n.name: n for n in graph.nodes}
    act_nodes = [name_to_node[n] for n in graph_value_to_id.keys()]

    node_to_first_user = {}
    for act in act_nodes:
        for node in graph.nodes:
            if act in node.args:
                node_to_first_user[act] = node
                break

    for node in act_nodes:
        val_id = graph_value_to_id[node.name]

        with graph.inserting_before(node_to_first_user[node]):
            reload_node = graph.create_node('call_function',
                                            torch.ops.dc.reload_tensor.default, (node, graph_id, val_id), {},
                                            name=f"reload_{node.name}_{val_id}")
        with graph.inserting_after(reload_node):
            wait_node = graph.create_node('call_function',
                                          torch.ops.dc.wait_reload.default, (reload_node, graph_id, val_id), {},
                                          name=f"wait_copy_{reload_node.name}_{val_id}")

        # replace all uses of node with wait_node
        users = {}
        for u in node.users.keys():
            if u != reload_node:
                users[u] = (node, wait_node)
        for u, (old_in, new_in) in users.items():
            u.replace_input_with(old_in, new_in)

    graph = move_primals_to_head(graph)
    graph.lint()
    return graph
