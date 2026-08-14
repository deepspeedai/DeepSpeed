# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import hashlib
import math
from typing import Dict, List, Tuple

import torch
from torch.fx import Graph, Node, GraphModule

from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist

from ..profilers.comm_profile import create_predictor
from ..profilers.graph_profile import is_profile_incomplete
from ..graph_param import DSGraphParamManager
from ..util import get_deepcompile_handle
from .contract import PassContract, CAP_Z3_GATHER_RELEASE

NAME = "prefetch"
# Reorders the all-gathers that zero3_compile emits, so it must run after that pass.
CONTRACT = PassContract(requires=frozenset({CAP_Z3_GATHER_RELEASE}))

FUSE_FACTOR = 0.8
MARGIN = 0.1
MAX_FUSE_SIZE = 1e9
MAX_BUFFERED_SIZE = 4e9

run_prefetch_pass = False

PREFETCH_ARENA_ALIGNMENT = 256


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _release_n_users(node: Node) -> int:
    return int(node.args[3]) if len(node.args) > 3 else 1


def _pack_prefetch_intervals(intervals: List[Dict], alignment: int = PREFETCH_ARENA_ALIGNMENT):
    """Pack fixed scheduled lifetimes into aligned slices of one stable backing."""
    if alignment <= 0:
        return None, "invalid_alignment"

    packed = []
    active = []
    free_blocks = []
    capacity = 0
    live_bytes = 0
    max_live_bytes = 0

    def add_free_block(offset, size):
        free_blocks.append((offset, size))
        free_blocks.sort()
        merged = []
        for block_offset, block_size in free_blocks:
            if merged and merged[-1][0] + merged[-1][1] == block_offset:
                prev_offset, prev_size = merged[-1]
                merged[-1] = (prev_offset, prev_size + block_size)
            else:
                merged.append((block_offset, block_size))
        free_blocks[:] = merged

    for interval in sorted(intervals, key=lambda item: (item["start"], item["end"], item["ds_id"])):
        size = _align_up(interval["bytes"], alignment)
        still_active = []
        for active_interval in active:
            if active_interval["end"] < interval["start"]:
                add_free_block(active_interval["offset"], active_interval["size"])
                live_bytes -= active_interval["size"]
            else:
                still_active.append(active_interval)
        active = still_active

        offset = None
        fitting_blocks = [(block_size, block_offset, index)
                          for index, (block_offset, block_size) in enumerate(free_blocks) if block_size >= size]
        if fitting_blocks:
            block_size, block_offset, index = min(fitting_blocks)
            offset = block_offset
            if block_size == size:
                free_blocks.pop(index)
            else:
                free_blocks[index] = (block_offset + size, block_size - size)
        if offset is None:
            offset = capacity
            capacity += size

        packed_interval = dict(interval, offset=offset, size=size)
        packed.append(packed_interval)
        active.append(packed_interval)
        live_bytes += size
        max_live_bytes = max(max_live_bytes, live_bytes)

    return {
        "capacity": capacity,
        "max_live_bytes": max_live_bytes,
        "internal_fragmentation": capacity - max_live_bytes,
        "intervals": packed,
    }, None


def _build_prefetch_arena_plan(graph: Graph,
                               param_manager: DSGraphParamManager,
                               world_size: int,
                               bwd: bool,
                               alignment: int = PREFETCH_ARENA_ALIGNMENT):
    """Derive attempt-0 arena slices solely from final fused-prefetch lifetimes."""
    if world_size <= 0:
        return None, "invalid_world_size"

    params_by_ds_id = {}
    for name, param in param_manager.params.items():
        ds_id = int(param_manager.ds_ids[name])
        previous = params_by_ds_id.get(ds_id)
        if previous is not None and (previous.shape != param.shape or previous.dtype != param.dtype):
            return None, "repeated_id_metadata_mismatch"
        params_by_ds_id[ds_id] = param

    nodes = list(graph.nodes)
    intervals = []
    active = {}
    for index, node in enumerate(nodes):
        if node.target == torch.ops.dc.prefetch_params_fused.default:
            plan_id = int(node.args[4]) if len(node.args) > 4 else -1
            eligible = set(node.meta.get("prefetch_arena_eligible_ds_ids", ()))
            scheduled_bytes = dict(node.meta.get("prefetch_arena_bytes_by_ds_id", ()))
            seen_in_prefetch = set()
            for ds_id_value in node.args[2]:
                ds_id = int(ds_id_value)
                if plan_id < 0 or ds_id not in eligible or ds_id in seen_in_prefetch:
                    continue
                seen_in_prefetch.add(ds_id)
                param = params_by_ds_id.get(ds_id)
                if param is None:
                    return None, "missing_param_metadata"
                try:
                    true_numel = math.prod(int(dim) for dim in param.shape)
                except (TypeError, ValueError):
                    return None, "dynamic_shape"
                if true_numel <= 0:
                    return None, "dynamic_shape"
                element_size = torch.empty((), dtype=param.dtype).element_size()
                padded_numel = ((true_numel + world_size - 1) // world_size) * world_size
                computed_bytes = padded_numel * element_size
                request_bytes = int(scheduled_bytes.get(ds_id, 0))
                if request_bytes <= 0:
                    return None, "missing_scheduled_bytes"
                if request_bytes != computed_bytes:
                    return None, "scheduled_bytes_mismatch"

                interval = active.get(ds_id)
                if interval is None:
                    interval = {
                        "ds_id": ds_id,
                        "start": index,
                        "end": None,
                        "bytes": request_bytes,
                        "requests": [],
                        "release_expected": None,
                        "release_seen": 0,
                    }
                    active[ds_id] = interval
                    intervals.append(interval)
                elif interval["bytes"] != request_bytes:
                    return None, "repeated_id_size_mismatch"
                interval["requests"].append((plan_id, ds_id))

        elif node.target == torch.ops.dc.release_param.default:
            ds_id = int(node.args[2])
            if ds_id not in active:
                continue
            interval = active[ds_id]
            if interval["release_expected"] is None:
                interval["release_expected"] = max(1, _release_n_users(node))
            interval["release_seen"] += 1
            if interval["release_seen"] >= interval["release_expected"]:
                active.pop(ds_id)["end"] = index

    if active:
        return None, "incomplete_release"
    if not intervals:
        return None, "no_eligible_prefetch"

    packed, reason = _pack_prefetch_intervals(intervals, alignment)
    if packed is None:
        return None, reason

    entries = []
    for interval in packed["intervals"]:
        for plan_id, ds_id in interval["requests"]:
            entries.append({
                "plan_id": plan_id,
                "ds_id": ds_id,
                "offset": interval["offset"],
                "bytes": interval["bytes"],
            })
    entries.sort(key=lambda entry: (entry["plan_id"], entry["ds_id"]))
    packed["entries"] = entries
    packed["phase"] = 1 if bwd else 0
    digest_payload = repr((packed["phase"], packed["capacity"], packed["max_live_bytes"], entries)).encode()
    packed["digest"] = int.from_bytes(hashlib.sha256(digest_payload).digest()[:8], "big") & ((1 << 63) - 1)
    return packed, None


def _add_prefetch_arena_release_dependencies(graph: Graph) -> None:
    """Keep arena-reusing prefetches after releases that make slices free.

    ``prefetch_params_fused`` otherwise consumes no value produced by a
    preceding release, so Inductor may advance it ahead of the FX order used
    by the arena lifetime planner. The native prefetch ignores its ``params``
    values, allowing release outputs to act as ordering-only inputs without
    changing the collective payload.
    """
    arena_ds_ids = {
        int(ds_id)
        for node in graph.nodes if node.target == torch.ops.dc.prefetch_params_fused.default
        for ds_id in node.meta.get("prefetch_arena_eligible_ds_ids", ())
    }
    pending_releases = []
    for node in graph.nodes:
        if node.target == torch.ops.dc.release_param.default:
            if int(node.args[2]) in arena_ds_ids:
                pending_releases.append(node)
            continue
        if node.target != torch.ops.dc.prefetch_params_fused.default or len(node.args) < 5:
            continue
        if int(node.args[4]) < 0 or not pending_releases:
            continue

        args = list(node.args)
        args[1] = [*args[1], *pending_releases]
        node.args = tuple(args)
        node.meta["prefetch_arena_ordering_dependencies"] = tuple(release.name for release in pending_releases)
        pending_releases.clear()


def _prefetch_arena_budget_admission(graph: Graph,
                                     plan: Dict,
                                     mem_dict: Dict[str, Tuple[int, int]],
                                     reserved_mem_dict: Dict[str, int],
                                     max_mem: int,
                                     pool_reclaimable_dict: Dict[str, int] = None):
    """Check the fixed backing against the same profiled peaks as prefetch.

    The existing scheduler charges each gathered tensor only while it is live.
    A flat arena instead keeps its full aligned capacity from the first arena
    prefetch through graph completion.  Admission therefore replaces the live
    eligible-gather charge with the fixed capacity; it must not add capacity on
    top of those same live bytes. Native execution also retires the old gather
    pool and flushes allocator cache before allocating the backing, so the
    projected allocated peak removes the pool charge measured at each node.
    """
    intervals = plan["intervals"]
    previous_peak = 0
    original_allocated_peak = 0
    arena_allocated_peak = 0
    original_reserved_peak = 0
    arena_reserved_peak = 0
    original_modeled_peak = 0
    arena_modeled_peak = 0
    limiting_node = None
    limiting_metric = None
    limiting_pool_reclaimable_bytes = 0
    previous_reserved_peak = 0
    previous_pool_reclaimable = 0
    pool_reclaimable_dict = pool_reclaimable_dict or {}

    for index, node in enumerate(graph.nodes):
        if node.name in mem_dict:
            previous_peak = int(mem_dict[node.name][1])
        if node.name in reserved_mem_dict:
            previous_reserved_peak = int(reserved_mem_dict[node.name])
        if node.name in pool_reclaimable_dict:
            previous_pool_reclaimable = int(pool_reclaimable_dict[node.name])
        live_gather_bytes = sum(
            int(interval["bytes"]) for interval in intervals if interval["start"] <= index <= interval["end"])
        # Keep the scheduler's existing conservative model: the profiled peak
        # is treated as non-gather residency and scheduled gathers are charged
        # separately according to the final prefetch lifetimes.
        original_at_node = previous_peak + live_gather_bytes
        # Native backing survives executor reuse and is shared by forward and
        # backward, so after warmup it is resident for the complete graph, not
        # merely between the first and last interval in this invocation.
        # The optimized graph inserts fused-prefetch nodes that do not exist in
        # the profiling graph. Carry the most recent profiled pool charge over
        # such nodes just as the existing memory projection carries its peak.
        pool_reclaimable_at_node = min(previous_peak, previous_pool_reclaimable)
        arena_at_node = previous_peak - pool_reclaimable_at_node + int(plan["capacity"])
        original_allocated_peak = max(original_allocated_peak, original_at_node)
        original_reserved_at_node = previous_reserved_peak
        # Keep the pre-transition reserved projection for diagnostics. Arena
        # execution has a stronger runtime contract: it retires the old gather
        # pool and performs one allocator-cache flush before allocating the
        # fixed backing. Admission must therefore not add the arena to unused
        # cache that the transition has guaranteed to release. The existing
        # 10% max_mem margin remains available for non-releasable fragmentation
        # and unrelated allocator growth.
        arena_reserved_at_node = previous_reserved_peak + int(plan["capacity"])
        original_reserved_peak = max(original_reserved_peak, original_reserved_at_node)
        arena_reserved_peak = max(arena_reserved_peak, arena_reserved_at_node)
        original_modeled_peak = max(original_modeled_peak, original_at_node, original_reserved_at_node)
        arena_allocated_peak = max(arena_allocated_peak, arena_at_node)
        effective_arena_peak = arena_at_node
        if effective_arena_peak > arena_modeled_peak:
            arena_modeled_peak = effective_arena_peak
            limiting_node = node.name
            limiting_metric = "allocated_after_pool_drain_and_empty_cache"
            limiting_pool_reclaimable_bytes = pool_reclaimable_at_node

    return {
        "accepted": arena_modeled_peak <= int(max_mem),
        "max_mem": int(max_mem),
        "capacity": int(plan["capacity"]),
        "original_allocated_peak": original_allocated_peak,
        "arena_allocated_peak": arena_allocated_peak,
        "original_modeled_peak": original_modeled_peak,
        "arena_modeled_peak": arena_modeled_peak,
        "original_reserved_peak": original_reserved_peak,
        "arena_reserved_peak": arena_reserved_peak,
        "incremental_peak": arena_modeled_peak - original_modeled_peak,
        "limiting_node": limiting_node,
        "limiting_metric": limiting_metric,
        "limiting_pool_reclaimable_bytes": limiting_pool_reclaimable_bytes,
    }


def _training_session_budget_admission(forward_graph: Graph,
                                       forward_plan: Dict,
                                       forward_mem_dict: Dict[str, Tuple[int, int]],
                                       forward_reserved_mem_dict: Dict[str, int],
                                       backward_graph: Graph,
                                       backward_plan: Dict,
                                       backward_mem_dict: Dict[str, Tuple[int, int]],
                                       backward_reserved_mem_dict: Dict[str, int],
                                       max_mem: int,
                                       forward_pool_reclaimable_dict: Dict[str, int] = None,
                                       backward_pool_reclaimable_dict: Dict[str, int] = None):
    """Admit the final shared backing against both actual phase plans.

    The forward plan is metadata-only until AOTAutograd lazily invokes the
    backward compiler.  At that point both interval packs and both original
    memory profiles are known, so the shared capacity can be charged to each
    phase without either a forward-only allocation or a pessimistic estimate.
    """
    capacity_bound = max(int(forward_plan["capacity"]), int(backward_plan["capacity"]))
    bounded_forward_plan = dict(forward_plan, capacity=capacity_bound)
    bounded_backward_plan = dict(backward_plan, capacity=capacity_bound)
    forward = _prefetch_arena_budget_admission(forward_graph, bounded_forward_plan, forward_mem_dict,
                                               forward_reserved_mem_dict, max_mem, forward_pool_reclaimable_dict)
    backward = _prefetch_arena_budget_admission(backward_graph, bounded_backward_plan, backward_mem_dict,
                                                backward_reserved_mem_dict, max_mem, backward_pool_reclaimable_dict)
    return {
        "accepted": forward["accepted"] and backward["accepted"],
        "max_mem": int(max_mem),
        "capacity_bound": capacity_bound,
        "forward_arena_peak": int(forward["arena_modeled_peak"]),
        "backward_arena_peak": int(backward["arena_modeled_peak"]),
        "forward_original_peak": int(forward["original_modeled_peak"]),
        "backward_original_peak": int(backward["original_modeled_peak"]),
        "forward_pool_reclaimable_bytes": int(forward["limiting_pool_reclaimable_bytes"]),
        "backward_pool_reclaimable_bytes": int(backward["limiting_pool_reclaimable_bytes"]),
    }


def _disable_prefetch_arena(graph: Graph) -> None:
    """Restore the exact original fused-prefetch call shape before lowering."""
    for node in graph.nodes:
        if node.target != torch.ops.dc.prefetch_params_fused.default or len(node.args) < 5:
            continue
        node.args = tuple(node.args[:3])
        node.meta.pop("prefetch_arena_eligible_ds_ids", None)
        node.meta.pop("prefetch_arena_bytes_by_ds_id", None)
        node.meta.pop("prefetch_arena_ordering_dependencies", None)


def _configure_prefetch_arena(graph: Graph, graph_id: int, plan: Dict, require_backward: bool) -> None:
    entries = plan["entries"]
    # Configure the rank-local immutable plan here, but defer cross-rank
    # consensus until the first fused-prefetch execution.  Compilation can run
    # concurrently and in a different order on each rank, so a blocking
    # process-group collective in this pass can cross graph-compilation
    # lifetimes and deadlock.  The native executor verifies the complete plan
    # digest on its communication stream before admitting any arena slice.
    get_deepcompile_handle().configure_z3_prefetch_arena(graph_id, plan["phase"], require_backward, plan["capacity"],
                                                         plan["max_live_bytes"], plan["digest"],
                                                         [entry["plan_id"]
                                                          for entry in entries], [entry["ds_id"] for entry in entries],
                                                         [entry["offset"]
                                                          for entry in entries], [entry["bytes"] for entry in entries])
    ordering_dependencies = sum(
        len(node.meta.get("prefetch_arena_ordering_dependencies", ())) for node in graph.nodes
        if node.target == torch.ops.dc.prefetch_params_fused.default)
    print_rank_0(f"prefetch_arena graph_id={graph_id} phase={'bwd' if plan['phase'] else 'fwd'} "
                 f"capacity={plan['capacity']} "
                 f"max_live_bytes={plan['max_live_bytes']} internal_fragmentation={plan['internal_fragmentation']} "
                 f"entries={len(entries)} ordering_dependencies={ordering_dependencies} digest={plan['digest']}")


def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)


def get_ds_id(node: Node):
    assert node.target == torch.ops.dc.allgather_param.default
    return node.args[2]


def schedule_prefetch(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                      create_inputs_fn, mem_budget: float, param_manager: Dict[int, DSGraphParamManager],
                      bwd: bool) -> GraphModule:

    profile = profiling_results[graph_id]
    require_backward = bool(profile.needs_backward)
    process_group = getattr(profile, "process_group", None)
    profile_graph = profile.bwd_graph if bwd else profile.fwd_graph
    mem_complete = profile.bwd_mem_complete if bwd else profile.fwd_mem_complete
    if is_profile_incomplete(profile_graph) or not mem_complete:
        print_rank_0(f"schedule_prefetch graph_id={graph_id} incomplete profiling data; skipping prefetch")
        return gm

    max_mem = get_accelerator().total_memory() * (1 - MARGIN)
    vals_to_bcast = torch.tensor([max_mem], device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN, group=process_group)
    max_mem = vals_to_bcast[0].item()

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    op_time = profiling_results[graph_id].bwd_time if bwd else profiling_results[graph_id].fwd_time
    tensor_sizes = profiling_results[graph_id].bwd_tensor_sizes if bwd else profiling_results[graph_id].fwd_tensor_sizes

    mem_dict = {name: (alloc_mem, peak) for name, alloc_mem, delta, peak in mem}
    reserved_mem_dict = {
        node.name: int(node.meta["profile_reserved_peak"])
        for node in profile_graph.nodes if "profile_reserved_peak" in node.meta
    }
    pool_reclaimable_dict = {
        node.name: int(node.meta.get("profile_gather_pool_charged", 0))
        for node in profile_graph.nodes
    }
    time_dict = {name: (device_time, wall_time) for name, device_time, wall_time in op_time}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

    graph = gm.graph
    total_param_size = sum(
        [tensor_size_dict[n.name] for n in graph.nodes if n.target == torch.ops.dc.allgather_param.default])

    print_rank_0(
        f"schedule_prefetch graph_id={graph_id} max_mem={max_mem} available_memory={get_accelerator().available_memory()} memory_allocated={get_accelerator().memory_allocated()} max_allocated={get_accelerator().max_memory_allocated()} total_param_size={total_param_size} margin={MARGIN}"
    )

    # Fill missing values
    prev_mem = 0
    prev_peak = 0
    for node in graph.nodes:
        if node.name in mem_dict:
            prev_mem = mem_dict[node.name][0]
            prev_peak = mem_dict[node.name][1]
        else:
            print_rank_0(f"node {node.name} not in mem_dict")
            mem_dict[node.name] = (prev_mem, prev_peak)

    comm_predictor = create_predictor()

    order_rev = list(reversed(graph.nodes))
    new_order_rev = []
    prefetch_ags = []
    prefetch_ag_groups = []
    ag_tensor_size_sum = 0
    for i, node in enumerate(order_rev):
        # print_rank_0(
        #     f"Checking node reverse order {node.name} {node.target} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}"
        # )

        if node.op != "placeholder":
            assert i < len(order_rev) - 1
            assert node.name in mem_dict
            next_node = order_rev[i + 1]
            next_alloc_mem, next_peak = mem_dict[next_node.name]

            # Free up memory
            while next_peak + ag_tensor_size_sum > max_mem or ag_tensor_size_sum > MAX_BUFFERED_SIZE:
                if len(prefetch_ag_groups) > 0:
                    # launch prefetch
                    fused_ag_nodes = prefetch_ag_groups.pop(0)
                    total_ag_tensor_size = sum([tensor_size_dict[ag_node.name] for ag_node in fused_ag_nodes])
                    ag_tensor_size_sum -= total_ag_tensor_size
                    new_order_rev.append(fused_ag_nodes)
                    assert len(fused_ag_nodes) > 0
                    # print_rank_0(
                    #     f"Free up memory fused_ag_nodes={fused_ag_nodes} next_alloc_mem={next_alloc_mem} total_ag_tensor_size={total_ag_tensor_size} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}"
                    # )
                elif len(prefetch_ags) > 0:
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                    # print_rank_0(
                    #     f"Free up memory prefetch_ags={prefetch_ag_groups} next_alloc_mem={next_alloc_mem} ag_tensor_size_sum={ag_tensor_size_sum} max_mem={max_mem}"
                    # )
                else:
                    break

            if node.target == torch.ops.dc.allgather_param.default:

                current_ag_size = sum([tensor_size_dict[ag_node.name] for ag_node in prefetch_ags])
                pred_time_current = comm_predictor(current_ag_size)
                pred_time_next = comm_predictor(tensor_size_dict[node.name])
                pred_time_fused = comm_predictor(current_ag_size + tensor_size_dict[node.name])

                do_fuse = max(pred_time_current, pred_time_next) * 1.2 > pred_time_fused and (
                    current_ag_size + tensor_size_dict[node.name]) < MAX_FUSE_SIZE
                # print_rank_0(
                #     f"found allgather_param do_fuse={do_fuse} current_ag_size={current_ag_size} tensor_size_dict[node.name]={tensor_size_dict[node.name]} pred_time_current={pred_time_current} pred_time_next={pred_time_next} pred_time_fused={pred_time_fused}"
                # )

                if len(prefetch_ags) > 0 and not do_fuse:
                    # stop fusing here
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                #     print_rank_0(
                #         f"stop fusing prefetch_ags={prefetch_ag_groups} ag_tensor_size_sum={ag_tensor_size_sum}")
                # else:
                #     print_rank_0(
                #         f"continue fusing ag_tensor_size_sum={ag_tensor_size_sum} ag_size={tensor_size_dict[node.name]} prefetch_ags={prefetch_ags} prefetch_ag_groups={prefetch_ag_groups}"
                #     )
                prefetch_ags.append(node)
                ag_tensor_size_sum += tensor_size_dict[node.name]

        new_order_rev.append(node)

        if (node.op != "placeholder"
                and node.target != torch.ops.dc.reload_parameter) and order_rev[i + 1].op == "placeholder":
            for ag_group in prefetch_ag_groups:
                assert len(ag_group) > 0
                new_order_rev.append(ag_group)
                total_ag_tensor_size = sum([tensor_size_dict[ag_node.name] for ag_node in ag_group])
                ag_tensor_size_sum -= total_ag_tensor_size
            if len(prefetch_ags) > 0:
                new_order_rev.append(prefetch_ags)
                ag_tensor_size_sum -= sum([tensor_size_dict[ag_node.name] for ag_node in prefetch_ags])
            assert ag_tensor_size_sum == 0

        # print_rank_0(
        #     f"node={node} next_alloc_mem={next_alloc_mem} pending_ags={len(prefetch_ags)} ag_tensor_size_sum={ag_tensor_size_sum}"
        # )

        assert ag_tensor_size_sum >= 0

    graph_param_manager = param_manager[graph_id]
    phase_plan_base = 1_000_000 if bwd else 0
    fused_group_index = 0
    new_graph = Graph()
    env = {}
    for node in reversed(new_order_rev):
        if isinstance(node, Node):
            #print(f"reconstruct {node.name} {node.target}")
            new_node = new_graph.node_copy(node, lambda n: env[n.name])
            env[node.name] = new_node
        else:
            param_nodes = [ag_node.args[0] for ag_node in node]
            param_nodes_copy = [env[param_node.name] for param_node in param_nodes]

            ds_ids = [get_ds_id(ag_node) for ag_node in node]
            plan_id = phase_plan_base + fused_group_index
            prefetch_node = new_graph.call_function(torch.ops.dc.prefetch_params_fused.default,
                                                    args=(graph_id, param_nodes_copy, ds_ids, None, plan_id))
            eligible_ds_ids = []
            arena_bytes_by_ds_id = []
            for ag_node, param_node, ds_id in zip(node, param_nodes, ds_ids):
                param = graph_param_manager.params.get(param_node.name)
                requested_dtype = ag_node.kwargs.get("dtype")
                persistent = param is None or bool(getattr(param.param, "ds_persist", False))
                if param is not None and not persistent and (requested_dtype is None
                                                             or requested_dtype == param.dtype):
                    eligible_ds_ids.append(ds_id)
                    arena_bytes_by_ds_id.append((ds_id, int(ag_node.meta.get("allgather_allocation_bytes", 0))))
            prefetch_node.meta["prefetch_arena_eligible_ds_ids"] = tuple(eligible_ds_ids)
            prefetch_node.meta["prefetch_arena_bytes_by_ds_id"] = tuple(arena_bytes_by_ds_id)
            fused_group_index += 1
    arena_plan = None
    if not dist.is_initialized():
        arena_reason = "distributed_uninitialized"
    else:
        arena_plan, arena_reason = _build_prefetch_arena_plan(new_graph, graph_param_manager,
                                                              dist.get_world_size(group=process_group), bwd)
    if arena_plan is not None:
        if len(reserved_mem_dict) != len(list(profile_graph.nodes)):
            arena_plan = None
            arena_reason = "incomplete_reserved_profile"
        else:
            admission = _prefetch_arena_budget_admission(new_graph, arena_plan, mem_dict, reserved_mem_dict, max_mem,
                                                         pool_reclaimable_dict)
    if arena_plan is not None:
        print_rank_0(f"prefetch_arena_admission graph_id={graph_id} "
                     f"phase={'bwd' if bwd else 'fwd'} accepted={int(admission['accepted'])} "
                     f"max_mem={admission['max_mem']} capacity={admission['capacity']} "
                     f"original_allocated_peak={admission['original_allocated_peak']} "
                     f"arena_allocated_peak={admission['arena_allocated_peak']} "
                     f"original_modeled_peak={admission['original_modeled_peak']} "
                     f"arena_modeled_peak={admission['arena_modeled_peak']} "
                     f"original_reserved_peak={admission['original_reserved_peak']} "
                     f"arena_reserved_peak={admission['arena_reserved_peak']} "
                     f"incremental_peak={admission['incremental_peak']} "
                     f"limiting_node={admission['limiting_node']} "
                     f"limiting_metric={admission['limiting_metric']} "
                     f"pool_reclaimable_bytes={admission['limiting_pool_reclaimable_bytes']}")
        if not admission["accepted"]:
            if require_backward and not bwd:
                profile.prefetch_arena_session_accepted = False
                profile.prefetch_arena_session_reason = "shared_budget"
            arena_plan = None
            arena_reason = "shared_budget"
    if arena_plan is not None and require_backward:
        if not bwd:
            # The optimized backward does not exist yet.  Keep the final
            # forward plan as native metadata, but native execution will
            # deliberately use the normal gather pool until the backward
            # compiler registers its plan and decides the shared budget.
            profile.prefetch_arena_forward_graph = new_graph
            profile.prefetch_arena_forward_plan = arena_plan
            profile.prefetch_arena_forward_mem = dict(mem_dict)
            profile.prefetch_arena_forward_reserved_mem = dict(reserved_mem_dict)
            profile.prefetch_arena_forward_pool_reclaimable = dict(pool_reclaimable_dict)
            profile.prefetch_arena_session_accepted = None
            profile.prefetch_arena_session_capacity_bound = 0
            profile.prefetch_arena_session_reason = None
            print_rank_0(f"prefetch_arena_session_pending graph_id={graph_id} "
                         f"forward_capacity={arena_plan['capacity']}")
        elif profile.prefetch_arena_session_accepted is False:
            arena_plan = None
            arena_reason = profile.prefetch_arena_session_reason or "session_not_admitted"
        elif (profile.prefetch_arena_forward_graph is None or profile.prefetch_arena_forward_plan is None
              or profile.prefetch_arena_forward_mem is None or profile.prefetch_arena_forward_reserved_mem is None
              or profile.prefetch_arena_forward_pool_reclaimable is None):
            arena_plan = None
            arena_reason = "incomplete_forward_arena_profile"
        else:
            session_admission = _training_session_budget_admission(
                profile.prefetch_arena_forward_graph, profile.prefetch_arena_forward_plan,
                profile.prefetch_arena_forward_mem, profile.prefetch_arena_forward_reserved_mem, new_graph, arena_plan,
                mem_dict, reserved_mem_dict, max_mem, profile.prefetch_arena_forward_pool_reclaimable,
                pool_reclaimable_dict)
            profile.prefetch_arena_session_accepted = bool(session_admission["accepted"])
            profile.prefetch_arena_session_capacity_bound = int(session_admission["capacity_bound"])
            profile.prefetch_arena_session_reason = (None if session_admission["accepted"] else "shared_budget")
            print_rank_0(f"prefetch_arena_session_admission graph_id={graph_id} "
                         f"accepted={int(session_admission['accepted'])} "
                         f"max_mem={session_admission['max_mem']} "
                         f"capacity_bound={session_admission['capacity_bound']} "
                         f"forward_original_peak={session_admission['forward_original_peak']} "
                         f"forward_arena_peak={session_admission['forward_arena_peak']} "
                         f"backward_original_peak={session_admission['backward_original_peak']} "
                         f"backward_arena_peak={session_admission['backward_arena_peak']} "
                         f"forward_pool_reclaimable_bytes={session_admission['forward_pool_reclaimable_bytes']} "
                         f"backward_pool_reclaimable_bytes={session_admission['backward_pool_reclaimable_bytes']}")
            if not session_admission["accepted"]:
                arena_plan = None
                arena_reason = "shared_budget"
    if arena_plan is None:
        print_rank_0(f"prefetch_arena graph_id={graph_id} phase={'bwd' if bwd else 'fwd'} fallback={arena_reason}")
        _disable_prefetch_arena(new_graph)
        # Forward may already have registered a pending training-session plan. A
        # later backward rejection must invalidate that native metadata as one
        # session decision, not leave a forward-only arena available on a
        # subsequent compile/execution.
        if bwd and require_backward:
            get_deepcompile_handle().disable_z3_prefetch_arena(graph_id, arena_reason)
    else:
        _add_prefetch_arena_release_dependencies(new_graph)
    new_graph.lint()
    gm.graph = new_graph

    if arena_plan is not None:
        _configure_prefetch_arena(new_graph, graph_id, arena_plan, require_backward)

    return gm
