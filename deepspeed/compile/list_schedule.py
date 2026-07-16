# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List, Dict, Optional
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


SCHEDULER_MEMORY_MARGIN = 0.1
SCHEDULER_BUDGET_DIAGNOSTICS_ATTR = "_deepcompile_scheduler_budget_diagnostics"


@dataclass(frozen=True)
class SchedulerMemoryBudget:
    """Rank-consistent allowance for scheduler-managed gathered parameter buffers.

    ``max_gathered_bytes`` is the headroom remaining after any profiled
    non-gathered peak, output reservation, and safety margin are removed.
    """

    max_gathered_bytes: int
    source: str
    available_mem: int = 0
    output_size: int = 0
    safety_margin: int = 0
    total_mem: int = 0
    profiled_non_gathered_peak_mem: int = 0

    @classmethod
    def from_profiled_non_gathered_peak(cls, total_mem: int, profiled_non_gathered_peak_mem: int, output_size: int):
        """Reserve profiled non-gather memory before budgeting transient gathers."""
        if total_mem is None or total_mem <= 0 or profiled_non_gathered_peak_mem is None or profiled_non_gathered_peak_mem <= 0:
            return None
        total_mem = int(total_mem)
        profiled_non_gathered_peak_mem = int(profiled_non_gathered_peak_mem)
        output_size = max(0, int(output_size or 0))
        safety_margin = int(total_mem * SCHEDULER_MEMORY_MARGIN)
        max_gathered_bytes = max(0, total_mem - profiled_non_gathered_peak_mem - output_size - safety_margin)
        return cls(max_gathered_bytes=max_gathered_bytes,
                   source="profiled_non_gathered_peak_memory",
                   available_mem=max_gathered_bytes,
                   output_size=output_size,
                   safety_margin=safety_margin,
                   total_mem=total_mem,
                   profiled_non_gathered_peak_mem=profiled_non_gathered_peak_mem)


class _GatheredParamTracker:
    """Simulate gathered-buffer residency for a committed schedule prefix.

    A gathered buffer remains live until the final release for its ``ds_id``;
    repeated gathers while it is live only increase the tracked allocation.
    """

    def __init__(self,
                 release_expected: Dict[int, int],
                 live_bytes_by_ds_id: Optional[Dict[int, int]] = None,
                 release_seen_by_ds_id: Optional[Dict[int, int]] = None,
                 live_bytes: int = 0,
                 peak_bytes: int = 0):
        self.release_expected = release_expected
        self.live_bytes_by_ds_id = dict(live_bytes_by_ds_id or {})
        self.release_seen_by_ds_id = dict(release_seen_by_ds_id or {})
        self.live_bytes = live_bytes
        self.peak_bytes = peak_bytes

    def copy(self, *, reset_peak: bool = False):
        """Copy residency state, optionally starting a candidate-local peak."""
        return _GatheredParamTracker(
            self.release_expected,
            self.live_bytes_by_ds_id,
            self.release_seen_by_ds_id,
            self.live_bytes,
            self.live_bytes if reset_peak else self.peak_bytes,
        )

    def apply(self, node: Node):
        """Advance residency through one node without treating reduce_grad as a release."""
        if node.target == torch.ops.dc.allgather_param.default:
            ds_id = _get_ds_id(node)
            size = _allgather_allocation_bytes(node)
            current_size = self.live_bytes_by_ds_id.get(ds_id)
            if current_size is None:
                self.live_bytes_by_ds_id[ds_id] = size
                self.live_bytes += size
            elif size > current_size:
                self.live_bytes_by_ds_id[ds_id] = size
                self.live_bytes += size - current_size
            self.release_seen_by_ds_id[ds_id] = 0
        elif is_release_node(node):
            ds_id = _get_ds_id(node)
            if ds_id in self.live_bytes_by_ds_id:
                release_seen = self.release_seen_by_ds_id.get(ds_id, 0) + 1
                release_expected = max(1, self.release_expected.get(ds_id, _release_n_users(node)))
                if release_seen >= release_expected:
                    self.live_bytes -= self.live_bytes_by_ds_id.pop(ds_id)
                    self.release_seen_by_ds_id.pop(ds_id, None)
                else:
                    self.release_seen_by_ds_id[ds_id] = release_seen

        self.peak_bytes = max(self.peak_bytes, self.live_bytes)


def _get_ds_id(node: Node):
    return node.args[2]


def _release_n_users(node: Node):
    if len(node.args) > 3:
        return int(node.args[3])
    return 1


def _dtype_element_size(dtype):
    if dtype is None:
        return None
    return torch.empty((), dtype=dtype).element_size()


def allgather_allocation_bytes(tensor_size: int, dtype, world_size: int):
    """Return the padded allocation size used by a process-group all-gather."""
    element_size = _dtype_element_size(dtype)
    tensor_size = int(tensor_size)
    if tensor_size <= 0 or element_size is None or element_size <= 0 or world_size <= 1:
        return tensor_size
    true_numel, remainder = divmod(tensor_size, element_size)
    if remainder != 0:
        return tensor_size
    padded_per_rank = (true_numel + world_size - 1) // world_size
    return padded_per_rank * world_size * element_size


def _allgather_allocation_bytes(node: Node):
    return int(node.meta.get("allgather_allocation_bytes", node.meta.get("tensor_size", 0)))


def _allgather_schedule_bytes(node: Node, scheduler_budget: Optional[SchedulerMemoryBudget]):
    """Use logical bytes for legacy ranking and padded bytes for an enabled memory gate."""
    if scheduler_budget is None:
        return int(node.meta.get("tensor_size", 0))
    return _allgather_allocation_bytes(node)


def max_possible_gathered_bytes(graph: Graph):
    """Upper-bound simultaneous gather residency, counting each ``ds_id`` once."""
    gathered_bytes_by_ds_id = {}
    for node in graph.nodes:
        if node.target != torch.ops.dc.allgather_param.default:
            continue
        ds_id = _get_ds_id(node)
        gathered_bytes_by_ds_id[ds_id] = max(gathered_bytes_by_ds_id.get(ds_id, 0), _allgather_allocation_bytes(node))
    return sum(gathered_bytes_by_ds_id.values())


def _build_release_expected(nodes: List[Node]):
    """Derive the final-release count for every gathered parameter interval."""
    release_expected = defaultdict(int)
    release_counts = defaultdict(int)
    for node in nodes:
        if is_release_node(node):
            ds_id = _get_ds_id(node)
            release_expected[ds_id] = max(release_expected[ds_id], _release_n_users(node))
            release_counts[ds_id] += 1

    for ds_id, release_count in release_counts.items():
        release_expected[ds_id] = max(release_expected[ds_id], release_count)

    return dict(release_expected)


def _simulate_path(tracker: _GatheredParamTracker, nodes: List[Node]):
    peak_bytes, _ = _simulate_path_stats(tracker, nodes)
    return peak_bytes


def _simulate_path_stats(tracker: _GatheredParamTracker, nodes: List[Node]):
    """Return this candidate's peak and ending residency without mutating the prefix.

    The committed tracker keeps a cumulative diagnostic peak, but an earlier
    overflow must not make every later candidate appear over budget after its
    buffers have been released.
    """
    candidate_tracker = tracker.copy(reset_peak=True)
    for node in nodes:
        candidate_tracker.apply(node)
    return candidate_tracker.peak_bytes, candidate_tracker.live_bytes


def profiled_non_gathered_peak(graph: Graph, mem_records):
    """Return a conservative peak for scheduler-budget headroom.

    Operator profiling invalidates gathered parameters between nodes, so a
    source-order residency replay is not guaranteed to be present in each
    recorded peak.  Keep the observed peak intact rather than subtracting
    hypothetical gathered residency and risking an unsafe budget.
    """
    return max((int(peak_mem) for _, _, _, peak_mem in mem_records), default=0)


def _fits_budget(scheduler_budget: Optional[SchedulerMemoryBudget], peak_bytes: int):
    return scheduler_budget is None or peak_bytes <= scheduler_budget.max_gathered_bytes


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
    schedule_until_ag_peak_mem: int
    schedule_until_free_peak_mem: int
    schedule_until_ag_live_mem: int
    schedule_until_free_live_mem: int


def _free_path_allgather_key(task: AllgatherTask):
    return (task.n_scheduled_ags, task.allgather_acc_mem, task.free_cost, task.node.name)


def _fallback_allgather_key(task: AllgatherTask):
    return (task.free_acc_mem, task.n_scheduled_ags, task.allgather_acc_mem, task.free_cost, task.node.name)


def _select_allgather_task(runnable_ags: List[AllgatherTask], scheduler_budget: Optional[SchedulerMemoryBudget]):
    """Select the best fitting release path, then the best fitting gather-only path."""
    ags_with_no_additional_ag = [
        ag for ag in runnable_ags
        if ag.free_acc_mem == 0 and _fits_budget(scheduler_budget, ag.schedule_until_free_peak_mem)
    ]
    if len(ags_with_no_additional_ag) > 0:
        next_ag = sorted(ags_with_no_additional_ag, key=_free_path_allgather_key)[0]
        return next_ag, next_ag.schedule_until_free

    fallback_ags = [ag for ag in runnable_ags if _fits_budget(scheduler_budget, ag.schedule_until_ag_peak_mem)]
    if len(fallback_ags) == 0:
        return None, None
    next_ag = sorted(fallback_ags, key=_fallback_allgather_key)[0]
    return next_ag, next_ag.schedule_until_ag


def _rejected_budget_candidates(runnable_ags: List[AllgatherTask], scheduler_budget: Optional[SchedulerMemoryBudget]):
    """Describe tasks for which neither scheduler-eligible path fits the budget."""
    if scheduler_budget is None:
        return []
    rejected = []
    for task in runnable_ags:
        path_options = list(_over_budget_path_options(task, scheduler_budget))
        if any(_fits_budget(scheduler_budget, peak_bytes) for _, _, peak_bytes, _ in path_options):
            continue
        _, _, peak_bytes, _ = min(path_options,
                                  key=lambda option:
                                  (max(0, option[2] - scheduler_budget.max_gathered_bytes), option[2]))
        rejected.append({
            "node": task.node.name,
            "schedule_until_ag_peak_mem": task.schedule_until_ag_peak_mem,
            "schedule_until_free_peak_mem": task.schedule_until_free_peak_mem,
            "over_budget_bytes": max(0, peak_bytes - scheduler_budget.max_gathered_bytes),
        })
    return rejected


def _over_budget_path_options(task: AllgatherTask, scheduler_budget: SchedulerMemoryBudget):
    if task.free_acc_mem == 0:
        yield ("until_free", task.schedule_until_free, task.schedule_until_free_peak_mem,
               task.schedule_until_free_live_mem)
        return
    yield ("until_ag", task.schedule_until_ag, task.schedule_until_ag_peak_mem, task.schedule_until_ag_live_mem)
    yield ("until_free", task.schedule_until_free, task.schedule_until_free_peak_mem,
           task.schedule_until_free_live_mem)


def _select_over_budget_allgather_task(runnable_ags: List[AllgatherTask], scheduler_budget: SchedulerMemoryBudget):
    """Choose the smallest deterministic peak overage when no candidate fits."""
    options = []
    for task in runnable_ags:
        for path, nodes, peak_bytes, live_bytes in _over_budget_path_options(task, scheduler_budget):
            overage_bytes = max(0, peak_bytes - scheduler_budget.max_gathered_bytes)
            options.append((overage_bytes, peak_bytes, live_bytes, task.free_acc_mem, task.n_scheduled_ags,
                            task.allgather_acc_mem, task.free_cost, task.node.name, path, task, nodes))
    *_, task, nodes = min(options)
    return task, nodes


def _budget_overflow_diagnostic(scheduler_budget: SchedulerMemoryBudget, task: AllgatherTask, path: str,
                                live_gathered_bytes: int):
    peak_bytes = task.schedule_until_free_peak_mem if path == "until_free" else task.schedule_until_ag_peak_mem
    return {
        "source": scheduler_budget.source,
        "max_gathered_bytes": scheduler_budget.max_gathered_bytes,
        "live_gathered_bytes": live_gathered_bytes,
        "selected_candidate": task.node.name,
        "path": path,
        "candidate_allgather_bytes": task.allgathered_mem,
        "candidate_peak_bytes": peak_bytes,
        "over_budget_bytes": max(0, peak_bytes - scheduler_budget.max_gathered_bytes),
    }


def fast_free_schedule(graph: Graph,
                       available_mem: int,
                       output_size: int,
                       debug_log: bool,
                       *,
                       scheduler_budget: Optional[SchedulerMemoryBudget] = None) -> Graph:
    """Order a ZeRO-3 graph while limiting scheduler-managed gather residency.

    The optional budget is already reduced across ranks by the caller.  With no
    budget, this preserves the legacy candidate ordering.  With a budget, only
    candidate simulations that fit are considered until all dependency levels
    are exhausted; a deterministic least-over-budget path guarantees progress.
    """
    node_to_last_use, user_to_last_uses = get_last_uses(graph)
    diagnostics = {
        "budget": scheduler_budget,
        "budget_rejections": 0,
        "budget_rejected_candidates": [],
        "budget_overflows": [],
        "selected": [],
    }

    # check tensor size
    for node in graph.nodes:
        if "tensor_size" not in node.meta:
            # Our profiler may not visit all nodes because of the control flow.
            node.meta["tensor_size"] = 0

    scheduled, unscheduled, edges, mem_table, remaining_users, user_to_producer = init_schedule_with_placeholders(
        graph)

    unscheduled_ags = [n for n in unscheduled if n.target == torch.ops.dc.allgather_param.default]
    gathered_tracker = (_GatheredParamTracker(_build_release_expected(list(graph.nodes)))
                        if scheduler_budget is not None else None)

    release_nodes = defaultdict(list)
    for n in unscheduled:
        if is_release_node(n):
            release_nodes[n.args[2]].append(n)

    ag_nodes_in_path = {}
    for ag_node in unscheduled_ags:
        last_use = node_to_last_use[ag_node]
        required_nodes = get_node_requirements(last_use, scheduled)
        # Dependency gather count is the primary scheduling tier.  Searching
        # higher tiers is allowed only when every candidate in a lower tier is
        # infeasible under the shared memory budget.
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

        over_budget_ags = []
        next_ag = None
        nodes_to_schedule = None
        for ag_count in count_list:

            runnable_ags = []
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
                allgathered_mem = _allgather_schedule_bytes(node, scheduler_budget)
                allgather_acc_mem = sum(
                    _allgather_schedule_bytes(n, scheduler_budget) for n in schedule_until_ag
                    if n.target == torch.ops.dc.allgather_param.default)
                free_acc_mem = sum(
                    _allgather_schedule_bytes(n, scheduler_budget) for n in diff_required_nodes
                    if n.target == torch.ops.dc.allgather_param.default)

                schedule_until_free = schedule_until_ag + diff_required_nodes
                for release_node in release_nodes[ds_id]:
                    for release_dep_node in get_node_requirements(release_node, scheduled + schedule_until_free):
                        if release_dep_node not in schedule_until_free:
                            schedule_until_free.append(release_dep_node)

                n_scheduled_ags = len(
                    [n for n in schedule_until_free if n.target == torch.ops.dc.allgather_param.default])

                if scheduler_budget is not None:
                    # Candidate simulation starts from the residency committed
                    # by earlier iterations and never mutates that prefix.
                    schedule_until_ag_peak_mem, schedule_until_ag_live_mem = _simulate_path_stats(
                        gathered_tracker, schedule_until_ag)
                    schedule_until_free_peak_mem, schedule_until_free_live_mem = _simulate_path_stats(
                        gathered_tracker, schedule_until_free)
                else:
                    schedule_until_ag_peak_mem = 0
                    schedule_until_free_peak_mem = 0
                    schedule_until_ag_live_mem = 0
                    schedule_until_free_live_mem = 0

                task = AllgatherTask(node, allgather_cost, free_cost, allgathered_mem, allgather_acc_mem, free_acc_mem,
                                     last_use, n_scheduled_ags, schedule_until_ag, schedule_until_free,
                                     schedule_until_ag_peak_mem, schedule_until_free_peak_mem,
                                     schedule_until_ag_live_mem, schedule_until_free_live_mem)

                # print(f" ag_count {ag_count} allgather runnable {i}: {node} last_use: {node_to_last_use[node]} t: {t2-t1:.2f}")
                runnable_ags.append(task)

            if len(runnable_ags) > 0:
                if scheduler_budget is None:
                    next_ag, nodes_to_schedule = _select_allgather_task(runnable_ags, None)
                    break
                else:
                    rejected = _rejected_budget_candidates(runnable_ags, scheduler_budget)
                    diagnostics["budget_rejections"] += len(rejected)
                    diagnostics["budget_rejected_candidates"].extend(rejected)
                    next_ag, nodes_to_schedule = _select_allgather_task(runnable_ags, scheduler_budget)
                    if next_ag is not None:
                        assert not debug_log or nodes_to_schedule is not next_ag.schedule_until_free or next_ag.free_acc_mem == 0
                        break
                    over_budget_ags.extend(runnable_ags)

        if next_ag is None:
            if scheduler_budget is not None and len(over_budget_ags) > 0:
                # Failing compilation cannot improve memory pressure. Commit
                # the deterministic path with the smallest peak overage and
                # retain the overflow in diagnostics instead.
                next_ag, nodes_to_schedule = _select_over_budget_allgather_task(over_budget_ags, scheduler_budget)
            else:
                raise AssertionError("No runnable allgather nodes")

        # print(f" next_ag {next_ag}")
        for n in nodes_to_schedule:
            # Only the selected path advances the committed residency model;
            # rejected simulations operated on tracker copies above.
            scheduled.append(n)
            unscheduled.remove(n)
            if gathered_tracker is not None:
                gathered_tracker.apply(n)

        scheduled_ags = [n for n in nodes_to_schedule if n.target == torch.ops.dc.allgather_param.default]
        for scheduled_ag in scheduled_ags:
            if scheduled_ag in unscheduled_ags:
                unscheduled_ags.remove(scheduled_ag)

            ag_nodes_in_path.pop(scheduled_ag, None)
            for ag_node, nodes in ag_nodes_in_path.items():
                if scheduled_ag in nodes:
                    nodes.remove(scheduled_ag)

        selected_diagnostic = {
            "node": next_ag.node.name,
            "path": "until_free" if nodes_to_schedule is next_ag.schedule_until_free else "until_ag",
        }
        if scheduler_budget is not None:
            selected_diagnostic.update({
                "live_gathered_bytes": gathered_tracker.live_bytes,
                "peak_gathered_bytes": gathered_tracker.peak_bytes,
                "schedule_until_ag_peak_mem": next_ag.schedule_until_ag_peak_mem,
                "schedule_until_free_peak_mem": next_ag.schedule_until_free_peak_mem,
                "schedule_until_ag_live_mem": next_ag.schedule_until_ag_live_mem,
                "schedule_until_free_live_mem": next_ag.schedule_until_free_live_mem,
            })
        diagnostics["selected"].append(selected_diagnostic)
        if scheduler_budget is not None:
            selected_peak_bytes = (next_ag.schedule_until_free_peak_mem if nodes_to_schedule
                                   is next_ag.schedule_until_free else next_ag.schedule_until_ag_peak_mem)
            if selected_peak_bytes > scheduler_budget.max_gathered_bytes:
                diagnostics["budget_overflows"].append(
                    _budget_overflow_diagnostic(scheduler_budget, next_ag, diagnostics["selected"][-1]["path"],
                                                diagnostics["selected"][-1]["live_gathered_bytes"]))

        # Schedule reduce nodes when possible to free memory earlier
        reduces_to_schedule = []
        for reduce_node in reduce_nodes:
            for scheduled_ag in scheduled_ags:
                if scheduled_ag in ag_nodes_in_path_to_reduce_nodes[reduce_node]:
                    ag_nodes_in_path_to_reduce_nodes[reduce_node].remove(scheduled_ag)
            if len(ag_nodes_in_path_to_reduce_nodes[reduce_node]) == 0:
                reduces_to_schedule.append(reduce_node)

        for n in reduces_to_schedule:
            need_to_schedule = get_node_requirements(n, scheduled)
            for nn in need_to_schedule:
                scheduled.append(nn)
                unscheduled.remove(nn)
                if gathered_tracker is not None:
                    gathered_tracker.apply(nn)

        # Do the same for output nodes
        outputs_to_schedule = []
        for output_node in output_nodes:
            for scheduled_ag in scheduled_ags:
                if scheduled_ag in ag_nodes_in_path_to_output_nodes[output_node]:
                    ag_nodes_in_path_to_output_nodes[output_node].remove(scheduled_ag)
            if len(ag_nodes_in_path_to_output_nodes[output_node]) == 0:
                outputs_to_schedule.append(output_node)

        for n in outputs_to_schedule:
            need_to_schedule = get_node_requirements(n, scheduled)
            for nn in need_to_schedule:
                scheduled.append(nn)
                unscheduled.remove(nn)
                if gathered_tracker is not None:
                    gathered_tracker.apply(nn)

    # print(f"After ag scheduled: scheduled: {scheduled}")

    scheduled_set = set(scheduled)
    for node in graph.nodes:
        if node in scheduled_set:
            continue
        scheduled.append(node)
        unscheduled.remove(node)
        if gathered_tracker is not None:
            gathered_tracker.apply(node)

    assert len(unscheduled) == 0, f"There are unscheduled nodes: {unscheduled}"

    ret_graph = make_graph_from_schedule(scheduled)
    setattr(ret_graph, SCHEDULER_BUDGET_DIAGNOSTICS_ATTR, diagnostics)
    ret_graph.lint()
    return ret_graph
