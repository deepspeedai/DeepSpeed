# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List, Dict
from copy import copy
from dataclasses import dataclass

import torch
from torch.fx import Graph, Node
from torch.fx.node import map_arg

try:
    from torch.utils._pytree import tree_iter
except ImportError:
    pass

from .util import get_last_uses, is_release_node
from .fx import get_output_node


def make_graph_from_schedule(scheduled: List[Node]):
    new_graph = Graph()
    env = {}
    for node in scheduled:
        new_node = new_graph.node_copy(node, lambda n: env[n.name])
        env[node.name] = new_node

    return new_graph


def get_original_args_num(node: Node):
    if node.name.startswith("allgather_ds_param") \
        or node.name.startswith("release_ds_param") \
        or node.name.startswith("wait_allgather_ds_param") \
        or node.name.startswith("reduce_ds_param"):
        return 1

    return len(node.args)


def flat_nodes_in_args(args: List[Node]):
    return [a for a in tree_iter(args) if isinstance(a, Node)]


def filter_args(node: Node):
    args = node.args[:get_original_args_num(node)]
    return flat_nodes_in_args(args)


def init_schedule(graph: Graph):
    mem_table = create_mem_table(graph)
    remaining_users = defaultdict(set)
    user_to_producer = {}

    scheduled = []
    unscheduled = []
    edges = defaultdict(list)
    for node in graph.nodes:
        filtered_args = filter_args(node)
        # print(f"Node: {node} args: {node.args}")
        if len(filtered_args) == 0:
            scheduled.append(node)

            remaining_users[node] = set(node.users.keys())
            for user in node.users.keys():
                user_to_producer[user] = node
        else:
            unscheduled.append(node)
        for a in filtered_args:
            for elem_a in tree_iter(a):
                if isinstance(elem_a, Node):
                    if node not in edges[elem_a]:
                        edges[elem_a].append(node)

    return scheduled, unscheduled, edges, mem_table, remaining_users, user_to_producer


def get_runnable_nodes(scheduled: List[Node], unscheduled: List[Node]):
    scheduled = set(scheduled)
    return [node for node in unscheduled if all(arg in scheduled for arg in filter_args(node))]


def choose_next_node(scheduled: List[Node], unscheduled: List[Node], mem_table: Dict[str, int]):
    runnable_nodes = get_runnable_nodes(scheduled, unscheduled)

    # sort by memory usage
    runnable_nodes = sorted(runnable_nodes, key=lambda n: mem_table[n.name])
    return runnable_nodes[0]


def create_mem_table(graph: Graph) -> Dict[str, int]:
    mem_table = {}
    for node in graph.nodes:
        if node.name.startswith("allgather_ds_param"):
            mem_table[node.name] = node.meta["tensor_size"]
        elif node.name.startswith("release_ds_param") or node.name.startswith("reduce_ds_param"):
            mem_table[node.name] = -node.meta["tensor_size"]
        else:
            mem_table[node.name] = 0

    return mem_table


def list_schedule(graph: Graph) -> Graph:

    scheduled, unscheduled, mem_table = init_schedule(graph)

    while len(unscheduled) > 0:
        next_node = choose_next_node(scheduled, unscheduled, mem_table)
        scheduled.append(next_node)
        unscheduled.remove(next_node)

    return make_graph_from_schedule(scheduled)


###############################


def get_new_runnable_nodes_with(scheduled: List[Node], edges: Dict[Node, List[Node]], new_scheduled: Node):
    scheduled = set(scheduled)
    new_runnables = []
    for node in edges[new_scheduled]:
        if all(arg in scheduled for arg in filter_args(node) if arg != new_scheduled):
            new_runnables.append(node)

    return new_runnables


def _do_schedule_without_allgather(scheduled: List[Node], unscheduled: List[Node], edges: Dict[Node, List[Node]],
                                   non_ag_runnable: List[Node]):

    while len(non_ag_runnable) > 0:
        next_node = non_ag_runnable.pop()

        new_runnables = get_new_runnable_nodes_with(scheduled, edges, next_node)
        non_ag_runnable += [n for n in new_runnables if not n.name.startswith("allgather_ds_param")]

        scheduled.append(next_node)
        unscheduled.remove(next_node)

    return scheduled, unscheduled


def schedule_without_allgather(scheduled: List[Node], unscheduled: List[Node], edges: Dict[Node, List[Node]]):
    runnable = get_runnable_nodes(scheduled, unscheduled)
    non_ag_runnable = [n for n in runnable if not n.name.startswith("allgather_ds_param")]

    tmp_scheduled = copy(scheduled)
    tmp_unscheduled = copy(unscheduled)

    return _do_schedule_without_allgather(tmp_scheduled, tmp_unscheduled, edges, non_ag_runnable)


def try_schedule_with_new_allgather(scheduled: List[Node], unscheduled: List[Node], edges: Dict[Node, List[Node]],
                                    new_scheduled: Node):
    new_runnables = get_new_runnable_nodes_with(scheduled, edges, new_scheduled)
    non_ag_runnable = [n for n in new_runnables if not n.name.startswith("allgather_ds_param")]

    tmp_scheduled = copy(scheduled)
    tmp_unscheduled = copy(unscheduled)

    tmp_scheduled.append(new_scheduled)
    tmp_unscheduled.remove(new_scheduled)

    return _do_schedule_without_allgather(tmp_scheduled, tmp_unscheduled, edges, non_ag_runnable)


def simple_prefetch(graph: Graph, available_mem: int, output_size: int, debug_log: bool) -> Graph:

    scheduled, unscheduled, edges, mem_table, remaining_users, user_to_producer = init_schedule(graph)
    tmp_scheduled, tmp_unscheduled = schedule_without_allgather(scheduled, unscheduled, edges)

    while len(tmp_unscheduled) > 0:

        runnable = get_runnable_nodes(tmp_scheduled, tmp_unscheduled)
        ag_with_unblock_time = []

        for ag_node in runnable:
            ag_scheduled, ag_unscheduled = try_schedule_with_new_allgather(tmp_scheduled, tmp_unscheduled, edges,
                                                                           ag_node)
            unblock_time = sum(n.meta["device_time"] for n in ag_scheduled[len(tmp_scheduled) + 1:])
            ag_with_unblock_time.append((ag_node, unblock_time, ag_scheduled, ag_unscheduled))

        ag_with_unblock_time = sorted(ag_with_unblock_time, key=lambda x: x[1], reverse=True)
        best_ag_node = ag_with_unblock_time[0][0]
        best_ag_scheduled = ag_with_unblock_time[0][2]

        no_ag_runnables = tmp_scheduled[len(scheduled):]
        after_ag_runnables = best_ag_scheduled[len(tmp_scheduled) + 1:]

        scheduled.append(best_ag_node)
        unscheduled.remove(best_ag_node)
        for n in no_ag_runnables:
            scheduled.append(n)
            unscheduled.remove(n)

        tmp_scheduled = copy(scheduled)
        tmp_unscheduled = copy(unscheduled)
        for n in after_ag_runnables:
            tmp_scheduled.append(n)
            tmp_unscheduled.remove(n)

    return make_graph_from_schedule(tmp_scheduled)


###############################


def init_schedule_with_placeholders(graph: Graph):
    mem_table = create_mem_table(graph)
    remaining_users = defaultdict(set)
    user_to_producer = {}

    scheduled = []
    unscheduled = []
    edges = defaultdict(list)
    for node in graph.nodes:
        if node.op == 'placeholder':
            scheduled.append(node)

            remaining_users[node] = set(node.users.keys())
            for user in node.users.keys():
                user_to_producer[user] = node
        else:
            unscheduled.append(node)

    return scheduled, unscheduled, edges, mem_table, remaining_users, user_to_producer


def get_node_requirements(target_node: Node, scheduled: List[Node]):
    scheduled = set(scheduled)
    visited = set()
    ordered_nodes = []

    def dfs(node: Node):
        if node in scheduled:
            return
        if node in visited:
            return
        visited.add(node)

        args = []

        def register_arg(n: Node):
            args.append(n)

        map_arg(node.args, register_arg)

        for arg in args:
            dfs(arg)
        ordered_nodes.append(node)

    dfs(target_node)

    return ordered_nodes


@dataclass
class AllgatherTask:
    node: Node
    allgather_cost: float
    free_cost: float
    allgathered_mem: int
    allgather_acc_mem: int
    free_acc_mem: int
    last_use: Node
    n_scheduled_ags: int
    schedule_until_ag: List[Node]
    schedule_until_free: List[Node]


def fast_free_schedule(graph: Graph, available_mem: int, output_size: int, debug_log: bool) -> Graph:
    node_to_last_use, user_to_last_uses = get_last_uses(graph)

    # check tensor size
    for node in graph.nodes:
        if "tensor_size" not in node.meta:
            # Our profiler may not visit all nodes because of the control flow.
            node.meta["tensor_size"] = 0

    scheduled, unscheduled, edges, mem_table, remaining_users, user_to_producer = init_schedule_with_placeholders(
        graph)

    unscheduled_ags = [n for n in unscheduled if n.target == torch.ops.dc.allgather_param.default]

    release_nodes = defaultdict(list)
    for n in unscheduled:
        if is_release_node(n):
            release_nodes[n.args[2]].append(n)

    ag_nodes_in_path = {}
    for ag_node in unscheduled_ags:
        last_use = node_to_last_use[ag_node]
        required_nodes = get_node_requirements(last_use, scheduled)
        ag_nodes_in_path[ag_node] = set(n for n in required_nodes if n.target == torch.ops.dc.allgather_param.default)

    reduce_nodes = [n for n in unscheduled if n.target == torch.ops.dc.reduce_grad.default]
    ag_nodes_in_path_to_reduce_nodes = {}
    for reduce_node in reduce_nodes:
        ag_nodes_in_path_to_reduce_nodes[reduce_node] = set(n for n in get_node_requirements(reduce_node, scheduled)
                                                            if n.target == torch.ops.dc.allgather_param.default)

    output_nodes = [
        n for n in get_output_node(graph).args[0]
        if isinstance(n, Node) and n.target != torch.ops.dc.reduce_grad.default
    ]
    ag_nodes_in_path_to_output_nodes = {}
    for output_node in output_nodes:
        ag_nodes_in_path_to_output_nodes[output_node] = set(n for n in get_node_requirements(output_node, scheduled)
                                                            if n.target == torch.ops.dc.allgather_param.default)

    while len(unscheduled_ags) > 0:

        ag_nodes_count = {ag_node: len(nodes) for ag_node, nodes in ag_nodes_in_path.items()}
        count_list = sorted(set(ag_nodes_count.values()))

        runnable_ags = []
        for ag_count in count_list:

            target_unscheduled_ags = [ag for ag in unscheduled_ags if ag_nodes_count[ag] == ag_count]

            for node in target_unscheduled_ags:
                ds_id = node.args[2]

                schedule_until_ag = get_node_requirements(node, scheduled)
                if schedule_until_ag is None:
                    continue

                last_use = node_to_last_use[node]

                diff_required_nodes = get_node_requirements(last_use, scheduled + schedule_until_ag)

                allgather_cost = sum(n.meta["device_time"] for n in schedule_until_ag)
                free_cost = sum(n.meta["device_time"] for n in diff_required_nodes)
                allgathered_mem = node.meta["tensor_size"]
                allgather_acc_mem = sum(n.meta["tensor_size"] for n in schedule_until_ag
                                        if n.target == torch.ops.dc.allgather_param.default)
                free_acc_mem = sum(n.meta["tensor_size"] for n in diff_required_nodes
                                   if n.target == torch.ops.dc.allgather_param.default)

                schedule_until_free = schedule_until_ag + diff_required_nodes
                for release_node in release_nodes[ds_id]:
                    for release_dep_node in get_node_requirements(release_node, scheduled + schedule_until_free):
                        if release_dep_node not in schedule_until_free:
                            schedule_until_free.append(release_dep_node)

                n_scheduled_ags = len(
                    [n for n in schedule_until_free if n.target == torch.ops.dc.allgather_param.default])

                task = AllgatherTask(node, allgather_cost, free_cost, allgathered_mem, allgather_acc_mem, free_acc_mem,
                                     last_use, n_scheduled_ags, schedule_until_ag, schedule_until_free)

                # print(f" ag_count {ag_count} allgather runnable {i}: {node} last_use: {node_to_last_use[node]} t: {t2-t1:.2f}")
                runnable_ags.append(task)

            if len(runnable_ags) > 0:
                break

        assert len(runnable_ags) > 0, "No runnable allgather nodes"

        # Criteria of the choice:
        # We want to choose allgather that does not require additional allgather until releasing the param.
        # When we can find such a node, free_acc_mem will be zero. In that case, we choose the one with the smallest cost until free to minimize the period of occupying memory for the gathered param.
        # If there is no such node, we choose the one with the smallest free_cost to minimize the period of occupying memory for the gathered param.
        ags_with_no_additional_ag = [ag for ag in runnable_ags if ag.free_acc_mem == 0]
        if len(ags_with_no_additional_ag) > 0:
            sorted_ags = sorted(runnable_ags, key=lambda x: x.free_cost)
            next_ag = sorted_ags[0]
            nodes_to_schedule = next_ag.schedule_until_free
        else:
            # sorted_ags = sorted(runnable_ags, key=lambda x: x.allgathered_mem)
            sorted_ags = sorted(runnable_ags, key=lambda x: x.free_acc_mem)
            next_ag = sorted_ags[0]
            nodes_to_schedule = next_ag.schedule_until_ag

        # print(f" next_ag {next_ag}")
        for n in nodes_to_schedule:
            scheduled.append(n)
            unscheduled.remove(n)

        unscheduled_ags.remove(next_ag.node)

        ag_nodes_in_path.pop(next_ag.node)
        for ag_node, nodes in ag_nodes_in_path.items():
            if next_ag.node in nodes:
                nodes.remove(next_ag.node)

        # Schedule reduce nodes when possible to free memory earlier
        reduces_to_schedule = []
        for reduce_node in reduce_nodes:
            if next_ag.node in ag_nodes_in_path_to_reduce_nodes[reduce_node]:
                ag_nodes_in_path_to_reduce_nodes[reduce_node].remove(next_ag.node)
                if len(ag_nodes_in_path_to_reduce_nodes[reduce_node]) == 0:
                    reduces_to_schedule.append(reduce_node)

        for n in reduces_to_schedule:
            need_to_schedule = get_node_requirements(n, scheduled)
            for nn in need_to_schedule:
                scheduled.append(nn)
                unscheduled.remove(nn)

        # Do the same for output nodes
        outputs_to_schedule = []
        for output_node in output_nodes:
            if next_ag.node in ag_nodes_in_path_to_output_nodes[output_node]:
                ag_nodes_in_path_to_output_nodes[output_node].remove(next_ag.node)
                if len(ag_nodes_in_path_to_output_nodes[output_node]) == 0:
                    outputs_to_schedule.append(output_node)

        for n in outputs_to_schedule:
            need_to_schedule = get_node_requirements(n, scheduled)
            for nn in need_to_schedule:
                scheduled.append(nn)
                unscheduled.remove(nn)

    # print(f"After ag scheduled: scheduled: {scheduled}")

    scheduled_set = set(scheduled)
    for node in graph.nodes:
        if node in scheduled_set:
            continue
        scheduled.append(node)
        unscheduled.remove(node)

    assert len(unscheduled) == 0, f"There are unscheduled nodes: {unscheduled}"

    ret_graph = make_graph_from_schedule(scheduled)
    ret_graph.lint()
    return ret_graph
