# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
import hashlib
import os
from typing import List, Dict, Tuple
import _operator

import torch
from torch.fx import Graph, Node, GraphModule

from ..util import get_input_nodes, get_param_nodes, get_index_by_graph_id, get_deepcompile_handle, get_real_uses, is_cast_op
from ..util import all_reduce, get_rank
from ..fx import (add_postprocess, _make_node_meta, get_output_node, move_primals_to_head, add_end_backward,
                  replace_reduce_outputs_with_none, should_release_reduce_buckets)
from ..profilers.graph_profile import ProfilingInterpreter, is_profile_incomplete
from ..list_schedule import (SCHEDULER_BUDGET_DIAGNOSTICS_ATTR, SchedulerMemoryBudget, allgather_allocation_bytes,
                             fast_free_schedule, max_possible_gathered_bytes, profiled_non_gathered_peak)

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

NAME = "zero3_compile"
SCHEDULER_DEBUG_ENV = "DEEPSPEED_COMPILE_SCHEDULER_BUDGET_DEBUG"
SCHEDULER_DEBUG_ENV_LEGACY = "DEEPSPEED_DEEPCOMPILE_SCHEDULER_DEBUG"


def _reduce_int(value: int, op, process_group=None):
    """Reduce an integer scheduling input, or preserve it outside distributed mode."""
    if not dist.is_initialized():
        return int(value)

    value_tensor = torch.tensor([int(value)],
                                device=torch.device(get_accelerator().current_device()),
                                dtype=torch.int64)
    all_reduce(value_tensor, op, process_group)
    return int(value_tensor.item())


def _rank_min_total_memory(process_group=None):
    return _reduce_int(get_accelerator().total_memory(), dist.ReduceOp.MIN, process_group)


def _world_size(process_group=None):
    if dist.is_initialized():
        if process_group is None:
            return dist.get_world_size()
        return dist.get_world_size(group=process_group)
    return 1


def _sync_profile_complete(profile_complete: bool, process_group=None):
    """Require every rank to have a complete profile before using it for scheduling."""
    if not dist.is_initialized():
        return profile_complete

    complete = torch.tensor([1 if profile_complete else 0],
                            device=torch.device(get_accelerator().current_device()),
                            dtype=torch.int)
    all_reduce(complete, dist.ReduceOp.MIN, process_group)
    return bool(complete.item())


def _operator_profile_complete(graph: Graph):
    """Require the graph marker plus per-node absolute start and peak memory."""
    return not is_profile_incomplete(graph) and all(
        "profile_mem_start" in node.meta and "profile_mem_peak" in node.meta for node in graph.nodes)


def _rank_max_operator_profiled_non_gathered_peak(graph: Graph, process_group=None):
    """Return the worst absolute peak after removing profiled gather residency."""
    records = [(node.name, int(node.meta.get("profile_mem_start", 0)
                               or 0), 0, int(node.meta.get("profile_mem_peak", 0) or 0)) for node in graph.nodes]
    return _reduce_int(profiled_non_gathered_peak(graph, records), dist.ReduceOp.MAX, process_group)


def _build_scheduler_budget_from_operator_profile(graph: Graph, output_size: int = 0, process_group=None):
    """Build a budget only when every node has trustworthy operator profile data."""
    if not _operator_profile_complete(graph):
        return None

    return SchedulerMemoryBudget.from_profiled_non_gathered_peak(
        _rank_min_total_memory(process_group), _rank_max_operator_profiled_non_gathered_peak(graph, process_group),
        output_size)


def _scheduler_debug_enabled():
    return any(
        os.environ.get(env_name, "").lower() not in ("", "0", "false", "no")
        for env_name in (SCHEDULER_DEBUG_ENV, SCHEDULER_DEBUG_ENV_LEGACY))


def _print_scheduler_debug(message: str, process_group=None):
    if not _scheduler_debug_enabled():
        return
    if not dist.is_initialized() or get_rank(process_group) == 0:
        print(message, flush=True)


def _stable_schedule_target(target):
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return f"{type(target).__module__}.{type(target).__qualname__}"


def _final_schedule_fingerprint(graph: Graph):
    """Hash stable node order and dependencies without process-local graph identifiers."""
    entries = []
    for node in graph.nodes:
        inputs = ",".join(input_node.name for input_node in node.all_input_nodes)
        entries.append(f"{node.op}|{node.name}|{_stable_schedule_target(node.target)}|{inputs}")
    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def _validate_final_schedule_fingerprint(graph: Graph, graph_id: int, bwd: bool, process_group=None):
    """In scheduler debug mode, fail every rank when final graph order diverges."""
    if not _scheduler_debug_enabled():
        return None

    fingerprint = _final_schedule_fingerprint(graph)
    if not dist.is_initialized():
        _print_scheduler_debug(
            f"DeepCompile ZeRO-3 final_schedule_fingerprint graph_id={graph_id} bwd={bwd} value={fingerprint}",
            process_group)
        return fingerprint

    device = torch.device(get_accelerator().current_device())
    min_fingerprint = torch.tensor([fingerprint], device=device, dtype=torch.int64)
    max_fingerprint = min_fingerprint.clone()
    all_reduce(min_fingerprint, dist.ReduceOp.MIN, process_group)
    all_reduce(max_fingerprint, dist.ReduceOp.MAX, process_group)
    if min_fingerprint.item() != max_fingerprint.item():
        raise RuntimeError(
            f"DeepCompile ZeRO-3 final schedule fingerprint mismatch for graph_id={graph_id} bwd={bwd}: "
            f"min={min_fingerprint.item()} max={max_fingerprint.item()}")
    _print_scheduler_debug(
        f"DeepCompile ZeRO-3 final_schedule_fingerprint graph_id={graph_id} bwd={bwd} value={fingerprint}",
        process_group)
    return fingerprint


def _set_allgather_allocation_metadata(graph: Graph, process_group=None):
    """Stamp padded gather allocation bytes without discarding a more precise estimate."""
    world_size = None
    for node in graph.nodes:
        if node.target == torch.ops.dc.allgather_param.default:
            if world_size is None:
                world_size = _world_size(process_group)
            dtype = node.kwargs.get("dtype") if isinstance(node.kwargs, dict) else None
            profiled_bytes = allgather_allocation_bytes(node.meta.get("tensor_size", 0), dtype, world_size)
            node.meta["allgather_allocation_bytes"] = max(int(node.meta.get("allgather_allocation_bytes", 0) or 0),
                                                          profiled_bytes)


def _scheduler_budget_disabled_reason(graph: Graph, scheduler_budget):
    if scheduler_budget is not None:
        return None
    if not _operator_profile_complete(graph):
        return "incomplete_operator_profile"
    return "invalid_profiled_non_gathered_peak"


def _scheduler_budget_from_operator_profile(gm: GraphModule, process_group=None):
    """Derive a rank-consistent budget and explain why a non-constraining gate is disabled."""
    if not dist.is_initialized():
        return None, "non_distributed"

    _set_allgather_allocation_metadata(gm.graph, process_group)
    operator_profile_complete = _sync_profile_complete(_operator_profile_complete(gm.graph), process_group)
    if not operator_profile_complete:
        # An unvisited suffix can exceed every observed partial peak, so no
        # absolute headroom estimate is safe until every rank completes.
        return None, "incomplete_operator_profile"

    scheduler_budget = _build_scheduler_budget_from_operator_profile(gm.graph, process_group=process_group)
    # A gate larger than every gather combined cannot affect ordering, so keep
    # legacy behavior and make the disabled state explicit in diagnostics.
    if scheduler_budget is not None and scheduler_budget.max_gathered_bytes >= max_possible_gathered_bytes(gm.graph):
        return None, "budget_not_constraining"
    return scheduler_budget, _scheduler_budget_disabled_reason(gm.graph, scheduler_budget)


def _log_scheduler_result(graph_id: int,
                          bwd: bool,
                          scheduler_budget,
                          disabled_reason,
                          graph: Graph,
                          process_group=None):
    diagnostics = getattr(graph, SCHEDULER_BUDGET_DIAGNOSTICS_ATTR, {})
    selected = diagnostics.get("selected", [])
    max_live_gathered_bytes = max((entry.get("peak_gathered_bytes", 0) for entry in selected), default=0)
    if scheduler_budget is None:
        _print_scheduler_debug(
            f"DeepCompile ZeRO-3 scheduler graph_id={graph_id} bwd={bwd} budget_enabled=False "
            f"disabled_reason={disabled_reason} selected_count={len(selected)} "
            f"max_live_gathered_bytes={max_live_gathered_bytes}", process_group)
        return

    _print_scheduler_debug(
        f"DeepCompile ZeRO-3 scheduler graph_id={graph_id} bwd={bwd} budget_enabled=True "
        f"budget_source={scheduler_budget.source} max_gathered_bytes={scheduler_budget.max_gathered_bytes} "
        f"safety_margin={scheduler_budget.safety_margin} "
        f"profiled_non_gathered_peak_mem={scheduler_budget.profiled_non_gathered_peak_mem} "
        f"budget_rejections={diagnostics.get('budget_rejections', 0)} "
        f"over_budget_fallbacks={len(diagnostics.get('budget_overflows', []))} "
        f"max_live_gathered_bytes={max_live_gathered_bytes}", process_group)


def _dtype_element_size(dtype: torch.dtype):
    return torch.empty((), dtype=dtype).element_size()


def _param_allgather_allocation_bytes(param, dtype: torch.dtype):
    """Return the registered parameter size in its target gather dtype."""
    return int(param.numel) * _dtype_element_size(dtype)


def add_allgather(graph_id: int,
                  graph: Graph,
                  node: Node,
                  ds_id: int,
                  dtype: torch.dtype,
                  allgather_allocation_bytes: int = None):
    """Insert gather and wait nodes while preserving the original graph output edge."""
    new_ag_node = add_postprocess(graph,
                                  node,
                                  torch.ops.dc.allgather_param.default,
                                  extra_args=[graph_id, ds_id],
                                  extra_kwargs={"dtype": dtype},
                                  name=f"allgather_ds_param_{node.target}_{ds_id}",
                                  meta=_make_node_meta(node, ds_id, True))
    if allgather_allocation_bytes is not None:
        new_ag_node.meta["allgather_allocation_bytes"] = int(allgather_allocation_bytes)
    new_ag_node.meta["val"] = node.meta["val"].to(dtype)

    # Set the previous node back to output
    # We don't want to change the output node to allgather
    output_node = get_output_node(graph)
    output_node.replace_input_with(new_ag_node, node)

    # Add wait as well
    new_wait_node = add_postprocess(graph,
                                    new_ag_node,
                                    torch.ops.dc.wait_allgather.default,
                                    extra_args=[graph_id, ds_id],
                                    name=f"wait_allgather_ds_param__{node.target}_{ds_id}",
                                    meta=_make_node_meta(node, ds_id, False))
    new_wait_node.meta["val"] = new_ag_node.meta["val"]

    return new_ag_node


def add_release(graph_id: int, graph: Graph, node: Node, release_node: Node, ds_id: int, n_users: int):
    new_node = add_postprocess(graph,
                               node,
                               torch.ops.dc.release_param.default,
                               extra_args=[graph_id, ds_id, n_users],
                               name=f"release_ds_param_{release_node.target}_{node.name}_{ds_id}",
                               meta=_make_node_meta(node, ds_id, False))
    new_node.meta["val"] = None


def add_reduce(graph_id: int, graph: Graph, grad_node: Node, param_name: str, ds_id: int):
    new_node = add_postprocess(graph,
                               grad_node,
                               torch.ops.dc.reduce_grad.default,
                               extra_args=[graph_id, ds_id],
                               name=f"reduce_ds_param_{param_name}",
                               meta=_make_node_meta(grad_node, ds_id, True))
    new_node.meta["val"] = None


def add_gather_and_release(graph_id: int, graph: Graph, param_manager, param_nodes: List[Node]) -> Graph:
    """Insert gather/wait lifetimes and attach releases to ordinary parameter consumers."""

    node_to_uses = get_real_uses(graph)
    for pn in param_nodes:
        if len(pn.users) == 0:
            continue

        # If the only use of the parameter is a type-cast to a smaller type, fuse it with all-gather.
        fuse_typecast = False
        target_dtype = param_manager.params[pn.name].dtype
        if len([user for user in pn.users if user.op != "output"]) == 1:
            typecast_node = next(iter(pn.users))

            is_cast, casted_dtype = is_cast_op(typecast_node)
            if is_cast and casted_dtype.itemsize < target_dtype.itemsize:
                fuse_typecast = True
                target_dtype = casted_dtype

        param = param_manager.params[pn.name]
        allgather_node = add_allgather(graph_id,
                                       graph,
                                       pn,
                                       param_manager.ds_ids[pn.name],
                                       target_dtype,
                                       allgather_allocation_bytes=_param_allgather_allocation_bytes(
                                           param, target_dtype))
        if fuse_typecast:
            users = node_to_uses[typecast_node]
            wait_node = typecast_node.args[0]
            for user in list(typecast_node.users.keys()):
                if user.op == "output":
                    wait_node.meta["original_output_name"] = typecast_node.name
                user.replace_input_with(typecast_node, wait_node)
            graph.erase_node(typecast_node)
        else:
            users = node_to_uses[pn]
            if len(users) == 0:
                # Parameters returned directly by the graph have no ordinary
                # consumer to trigger gathering, so make the waited gather the
                # output while retaining its original output name.
                output_node = get_output_node(graph)
                wait_node = next(user for user in allgather_node.users
                                 if user.target == torch.ops.dc.wait_allgather.default)
                wait_node.meta["original_output_name"] = pn.name
                output_node.replace_input_with(pn, wait_node)

        ds_id = param_manager.ds_ids[pn.name]
        for user in users:
            # release_param() only accepts tensors as its first argument. If
            # `user` is a tuple, we should release the param after any of
            # operator.getitem of that tuple.
            #
            # Since no torch op takes a tuple as an input, we simply walk
            # through users of `user` and check if there is any call to
            # operator.getitem.
            for secondary_user in user.users:
                if secondary_user.op == "call_function" and secondary_user.target == _operator.getitem:
                    add_release(graph_id, graph, secondary_user, pn, ds_id, len(users))
                    break
            else:
                add_release(graph_id, graph, user, pn, ds_id, len(users))

    return move_primals_to_head(graph)


def add_gather_and_reduce(graph_id: int, graph: Graph, param_manager, param_nodes_bw: List[Node],
                          param_name_to_grad: Dict[str, Node]) -> Graph:
    """Add parameter lifetimes and gradient reductions to a backward graph."""

    add_gather_and_release(graph_id, graph, param_manager, param_nodes_bw)

    for param_name in param_manager.param_names:
        if param_name_to_grad[param_name] is None:
            continue
        add_reduce(graph_id, graph, param_name_to_grad[param_name], param_name, param_manager.ds_ids[param_name])

    return move_primals_to_head(graph)


def add_z3_gather_release_fw(gm: GraphModule,
                             graph_id: int,
                             graph_order: List[Tuple[int, bool]],
                             profiling_results,
                             create_inputs_fn,
                             param_manager,
                             debug_log=False) -> GraphModule:
    """Profile, budget, and schedule ZeRO-3 parameter lifetimes for a forward graph."""

    nz3 = get_deepcompile_handle()

    real_inputs = create_inputs_fn()
    param_indices = profiling_results[graph_id].param_indices
    process_group = getattr(profiling_results[graph_id], "process_group", None)

    gm.graph = add_gather_and_release(graph_id, gm.graph, param_manager[graph_id],
                                      get_param_nodes(gm.graph, param_indices))

    nz3.register_graph_z3(graph_id, [v[1] for v in param_indices])  # Need this before profiling

    profiler = ProfilingInterpreter(gm, debug_log=debug_log, process_group=process_group)
    profiler.run(*real_inputs)
    del profiler
    gc.collect()
    get_accelerator().empty_cache()
    # Build the shared scheduling budget after the operator profile is complete
    # but before the scheduler rewrites graph order and Inductor metadata.
    scheduler_budget, disabled_reason = _scheduler_budget_from_operator_profile(gm, process_group)

    rank = get_rank(process_group)
    graph_index = get_index_by_graph_id(graph_order, graph_id)
    if rank == 0 and debug_log:
        print(f"Fwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    for n in gm.graph.nodes:
        is_ds_param = n.name in param_manager[graph_id].ds_ids
        if "val" in n.meta and is_ds_param:
            # Used for Inductor's validation
            n.meta["val"] = torch.empty([0], dtype=n.meta['val'].dtype, device=n.meta['val'].device)

    gm.graph = fast_free_schedule(
        gm.graph,
        get_accelerator().available_memory(),
        0,  # unused
        debug_log=debug_log,
        scheduler_budget=scheduler_budget)
    _log_scheduler_result(graph_id,
                          bwd=False,
                          scheduler_budget=scheduler_budget,
                          disabled_reason=disabled_reason,
                          graph=gm.graph,
                          process_group=process_group)
    _validate_final_schedule_fingerprint(gm.graph, graph_id, bwd=False, process_group=process_group)

    if rank == 0 and debug_log:
        print(f"Fwd after scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    return gm


def add_z3_gather_release_bw(gm: GraphModule,
                             graph_id: int,
                             graph_order: List[Tuple[int, bool]],
                             profiling_results,
                             create_inputs_fn,
                             param_manager,
                             debug_log=False) -> GraphModule:
    """Profile, budget, and schedule gathers, releases, and reductions for backward."""

    param_nodes_bw, param_name_to_grad = param_manager[graph_id].get_bwd_mapping(gm.graph)
    gm.graph = add_gather_and_reduce(graph_id, gm.graph, param_manager[graph_id], param_nodes_bw, param_name_to_grad)

    input_nodes = get_input_nodes(gm.graph)
    real_inputs = create_inputs_fn()
    process_group = getattr(profiling_results[graph_id], "process_group", None)
    assert len(input_nodes) == len(real_inputs), f"Expected {len(real_inputs)} inputs, got {len(input_nodes)}"

    real_outputs = ProfilingInterpreter(gm, debug_log=debug_log, process_group=process_group).run(*real_inputs)

    del real_outputs
    gc.collect()
    get_accelerator().empty_cache()
    # The scheduler consumes only DP-group-reduced inputs, ensuring every group
    # rank emits collectives in the same order even when allocator state differs.
    scheduler_budget, disabled_reason = _scheduler_budget_from_operator_profile(gm, process_group)

    rank = get_rank(process_group)
    graph_index = get_index_by_graph_id(graph_order, graph_id)
    if rank == 0 and debug_log:
        print(f"Bwd before scheduling graph {graph_index} graph_id={graph_id} {gm.graph}")

    gm.graph = fast_free_schedule(
        gm.graph,
        get_accelerator().available_memory(),
        0,  # unused
        debug_log=debug_log,
        scheduler_budget=scheduler_budget)
    _log_scheduler_result(graph_id,
                          bwd=True,
                          scheduler_budget=scheduler_budget,
                          disabled_reason=disabled_reason,
                          graph=gm.graph,
                          process_group=process_group)

    add_end_backward(gm.graph, graph_id, should_release_reduce_buckets(graph_order, graph_id))
    replace_reduce_outputs_with_none(gm.graph)
    _validate_final_schedule_fingerprint(gm.graph, graph_id, bwd=True, process_group=process_group)

    return gm


def add_z3_gather_release(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                          create_inputs_fn, mem_budget: float, param_manager, bwd: bool) -> GraphModule:
    if bwd:
        return add_z3_gather_release_bw(gm,
                                        graph_id,
                                        graph_order,
                                        profiling_results,
                                        create_inputs_fn,
                                        param_manager,
                                        debug_log=False)
    return add_z3_gather_release_fw(gm,
                                    graph_id,
                                    graph_order,
                                    profiling_results,
                                    create_inputs_fn,
                                    param_manager,
                                    debug_log=False)
