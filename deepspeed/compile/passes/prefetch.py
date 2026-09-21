# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
from typing import Iterable, List, Tuple

import torch
from torch.fx import Graph, Node, GraphModule

from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist

from ..executor_arena import (DEFAULT_FUSE_BUDGET, DEFAULT_LIVE_BUDGET, admit_executor_arena,
                              plan_graph_executor_arena, register_executor_arena)
from ..profilers.comm_profile import create_predictor
from ..profilers.graph_profile import is_profile_incomplete
from ..graph_param import DSGraphParamManager
from ..util import get_deepcompile_handle
from .contract import PassContract, CAP_Z3_GATHER_RELEASE

NAME = "prefetch"
# Reorders the all-gathers that zero3_compile emits, so it must run after that pass.
CONTRACT = PassContract(requires=frozenset({CAP_Z3_GATHER_RELEASE}), conflicts_with=frozenset({"selective_gather"}))

FUSE_FACTOR = 0.8
MARGIN = 0.1
MAX_FUSE_SIZE = DEFAULT_FUSE_BUDGET
MAX_BUFFERED_SIZE = DEFAULT_LIVE_BUDGET

run_prefetch_pass = False


def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)


def get_ds_id(node: Node):
    assert node.target == torch.ops.dc.allgather_param.default
    return node.args[2]


def _fused_group_dtypes(ag_nodes: List[Node]):
    dtypes = [node.kwargs.get("dtype") for node in ag_nodes]
    if all(dtype is None for dtype in dtypes):
        return None
    if any(dtype is None for dtype in dtypes):
        return False
    return dtypes


def _prefetch_size_admissible(nbytes: int) -> bool:
    return 0 < nbytes <= MAX_BUFFERED_SIZE and nbytes <= MAX_FUSE_SIZE


def _rewrite_fused_prefetch(ordered_nodes: Iterable, graph_id: int) -> Graph:
    """Replace scheduled all-gathers with Tensor-producing fused prefetch edges."""
    new_graph = Graph()
    env = {}
    replaced_allgathers = set()

    for node in ordered_nodes:
        if isinstance(node, Node):
            if node.name in replaced_allgathers:
                continue
            new_node = new_graph.node_copy(node, lambda old_node: env[old_node.name])
            env[node.name] = new_node
            continue

        ag_nodes = list(node)
        if not ag_nodes:
            continue
        dtypes = _fused_group_dtypes(ag_nodes)
        if dtypes is False:
            # The fused schema cannot represent a mixture of explicit and implicit
            # dtypes, so preserve the independent demand all-gathers.
            continue

        param_nodes = [ag_node.args[0] for ag_node in ag_nodes]
        if any(param_node.name not in env for param_node in param_nodes):
            raise RuntimeError("fused prefetch was scheduled before its parameter inputs")
        param_nodes_copy = [env[param_node.name] for param_node in param_nodes]
        ds_ids = [get_ds_id(ag_node) for ag_node in ag_nodes]
        if len(set(ds_ids)) != len(ds_ids):
            continue
        fused_node = new_graph.call_function(torch.ops.dc.prefetch_params_fused.default,
                                             args=(graph_id, param_nodes_copy, ds_ids, dtypes))
        fused_node.meta["deepcompile_fused_ds_ids"] = tuple(ds_ids)

        for index, ag_node in enumerate(ag_nodes):
            output = new_graph.call_function(operator.getitem, args=(fused_node, index))
            output.meta.update(ag_node.meta)
            output.meta["deepcompile_arena_ds_id"] = get_ds_id(ag_node)
            output.meta["deepcompile_arena_dtype"] = ag_node.kwargs.get("dtype")
            env[ag_node.name] = output
            replaced_allgathers.add(ag_node.name)

    new_graph.lint()
    return new_graph


def schedule_prefetch(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                      create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                      bwd: bool) -> GraphModule:

    profile = profiling_results[graph_id]
    process_group = getattr(profile, "process_group", None)
    profile_graph = profile.bwd_graph if bwd else profile.fwd_graph
    mem_complete = profile.bwd_mem_complete if bwd else profile.fwd_mem_complete
    if is_profile_incomplete(profile_graph) or not mem_complete:
        print_rank_0(f"schedule_prefetch graph_id={graph_id} incomplete profiling data; skipping prefetch")
        return gm

    max_mem = get_accelerator().total_memory() * (1 - MARGIN)
    vals_to_bcast = torch.tensor([max_mem], device=torch.device(get_accelerator().current_device_name()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN, group=process_group)
    max_mem = vals_to_bcast[0].item()

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    op_time = profiling_results[graph_id].bwd_time if bwd else profiling_results[graph_id].fwd_time
    tensor_sizes = profiling_results[graph_id].bwd_tensor_sizes if bwd else profiling_results[graph_id].fwd_tensor_sizes

    mem_dict = {name: (alloc_mem, peak) for name, alloc_mem, delta, peak in mem}
    time_dict = {name: (device_time, wall_time) for name, device_time, wall_time in op_time}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

    graph = gm.graph
    demand_plan = plan_graph_executor_arena(graph)
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

                node_size = tensor_size_dict[node.name]
                if not _prefetch_size_admissible(node_size):
                    new_order_rev.append(node)
                    continue

                while ag_tensor_size_sum + node_size > MAX_BUFFERED_SIZE:
                    if prefetch_ag_groups:
                        bounded_group = prefetch_ag_groups.pop(0)
                    elif prefetch_ags:
                        bounded_group = prefetch_ags
                        prefetch_ags = []
                    else:
                        break
                    new_order_rev.append(bounded_group)
                    ag_tensor_size_sum -= sum(tensor_size_dict[ag_node.name] for ag_node in bounded_group)

                current_ag_size = sum([tensor_size_dict[ag_node.name] for ag_node in prefetch_ags])
                pred_time_current = comm_predictor(current_ag_size)
                pred_time_next = comm_predictor(node_size)
                pred_time_fused = comm_predictor(current_ag_size + node_size)

                do_fuse = max(pred_time_current, pred_time_next) * 1.2 > pred_time_fused and (
                    current_ag_size + node_size) <= MAX_FUSE_SIZE
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
                ag_tensor_size_sum += node_size

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

    fused_graph = _rewrite_fused_prefetch(reversed(new_order_rev), graph_id)
    final_plan = plan_graph_executor_arena(fused_graph)
    admission = admit_executor_arena(final_plan.packed,
                                     demand_profile_bytes=demand_plan.packed.capacity,
                                     live_budget=int(MAX_BUFFERED_SIZE))
    if admission.accepted:
        gm.graph = fused_graph
    else:
        gm.graph = graph
        final_plan = demand_plan
        admission = admit_executor_arena(final_plan.packed,
                                         demand_profile_bytes=final_plan.packed.capacity,
                                         live_budget=int(MAX_BUFFERED_SIZE))
    gm._deepcompile_executor_arena_plan = final_plan
    gm._deepcompile_executor_arena_admission = admission
    gm._deepcompile_executor_arena_registration = register_executor_arena(get_deepcompile_handle(),
                                                                          graph_id,
                                                                          final_plan,
                                                                          process_group=process_group,
                                                                          bwd=bwd,
                                                                          admission=admission)

    return gm
