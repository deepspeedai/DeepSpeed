# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.fx import GraphModule

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from ..util import get_deepcompile_handle
from ..graph_param import DSGraphParamManager

NAME = "selective_gather"

max_alloc_mem = 0
last_optimize_step = 0
MEM_MARGIN = 0.1


def print_rank_0(message):
    if dist.get_rank() == 0:
        print(message)


def _compute_persistence_budget(all_graph_mem_records: List[List[Tuple[str, int, int, int]]], total_mem: int,
                                mem_margin: float) -> Dict[str, int]:
    usable_mem = int(total_mem * (1 - mem_margin))
    non_empty_records = [mem_records for mem_records in all_graph_mem_records if mem_records]

    if not non_empty_records:
        return {
            "usable_mem": usable_mem,
            "peak_resident_alloc": 0,
            "transient_peak": 0,
            "available_mem": 0,
            "profiled_list_count": 0,
        }

    # Persistent parameters add to live allocations that remain resident past an op boundary.
    peak_resident_alloc = max(record[1] for mem_records in non_empty_records for record in mem_records)
    transient_peak = max(record[3] for mem_records in non_empty_records for record in mem_records)

    return {
        "usable_mem": usable_mem,
        "peak_resident_alloc": peak_resident_alloc,
        "transient_peak": transient_peak,
        "available_mem": max(0, usable_mem - peak_resident_alloc),
        "profiled_list_count": len(non_empty_records),
    }


def selective_gather(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                     create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                     bwd: bool) -> GraphModule:
    target_graph_id = graph_id

    if not bwd:
        return gm

    last_backward_graph_id = None
    for g_id, needs_bwd in graph_order:
        if needs_bwd:
            last_backward_graph_id = g_id
            break

    # Run only on the last backward graph
    if last_backward_graph_id is None or graph_id != last_backward_graph_id:
        return gm

    all_graph_mem_records = []
    for profile_graph_id, prof in profiling_results.items():
        all_graph_mem_records.extend([prof.fwd_mem, prof.bwd_mem])

        fwd_peak_resident = max((m[1] for m in prof.fwd_mem), default=0)
        fwd_transient_peak = max((m[3] for m in prof.fwd_mem), default=0)
        bwd_peak_resident = max((m[1] for m in prof.bwd_mem), default=0)
        bwd_transient_peak = max((m[3] for m in prof.bwd_mem), default=0)

        print_rank_0(f"selective_gather graph_id={profile_graph_id} "
                     f"fwd_peak_resident={fwd_peak_resident} fwd_transient_peak={fwd_transient_peak} "
                     f"bwd_peak_resident={bwd_peak_resident} bwd_transient_peak={bwd_transient_peak}")

    persistent_ds_ids = set()
    for param_graph_id, pm in param_manager.items():
        for name, ds_param in pm.params.items():
            if ds_param.param.ds_persist:
                persistent_ds_ids.add(pm.ds_ids[name])

    ds_id_to_size = {}
    ds_id_to_time = defaultdict(float)
    ds_id_to_prof_dtime = defaultdict(float)
    ds_id_to_prof_wtime = defaultdict(float)

    for param_graph_id, pm in param_manager.items():
        params = pm.params
        for param_name, param in params.items():
            ds_id = pm.ds_ids[param_name]
            ds_id_to_size[ds_id] = param.numel * param.dtype.itemsize

        profile = profiling_results[param_graph_id]
        for n in profile.fwd_graph.nodes:
            if n.target == torch.ops.dc.allgather_param.default:
                assert "tensor_size" in n.meta
                ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
                assert "device_time" in n.meta
                ds_id_to_time[n.args[2]] += n.meta["device_time"]

                ds_id_to_prof_dtime[n.args[2]] = n.meta["device_time"]
                ds_id_to_prof_wtime[n.args[2]] = n.meta["wall_time"]

        if profile.bwd_graph is not None:
            for n in profile.bwd_graph.nodes:
                if n.target == torch.ops.dc.allgather_param.default:
                    assert "tensor_size" in n.meta
                    ds_id_to_size[n.args[2]] = n.meta["tensor_size"]
                    assert "device_time" in n.meta
                    ds_id_to_time[n.args[2]] += n.meta["device_time"]

    ds_ids = [ds_id for ds_id in ds_id_to_size if ds_id not in persistent_ds_ids]
    ds_ids.sort(key=lambda ds_id: ds_id_to_time[ds_id] / ds_id_to_size[ds_id], reverse=True)

    # print(f"ds_id_to_size={ds_id_to_size}")
    # print(f"ds_id_to_time={ds_id_to_time}")

    # if dist.get_rank() == 0:
    #     for ds_id in ds_ids:
    #         dtime_in_sec = ds_id_to_prof_dtime[ds_id]
    #         wtime_in_sec = ds_id_to_prof_wtime[ds_id]
    #         size_in_mb = ds_id_to_size[ds_id] / 1024 / 1024
    #         print(
    #             f"ds_id={ds_id} time_per_size={ds_id_to_time[ds_id] / ds_id_to_size[ds_id]:.5f} dtime={dtime_in_sec:.3f} wtime={wtime_in_sec:.3f} size={size_in_mb:.2f}MB bw={size_in_mb/dtime_in_sec:.2f}MB/s"
    #         )

    accelerator = get_accelerator()
    total_mem = accelerator.total_memory()
    current_available_mem = accelerator.available_memory()
    vals_to_bcast = torch.tensor([total_mem, current_available_mem],
                                 device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    total_mem = vals_to_bcast[0].item()
    current_available_mem = vals_to_bcast[1].item()

    budget = _compute_persistence_budget(all_graph_mem_records, total_mem, MEM_MARGIN)
    available_mem = int(current_available_mem * (1 - MEM_MARGIN))

    ds_id_to_param = {}
    for g_id, g_pm in param_manager.items():
        for name, ds_param in g_pm.params.items():
            ds_id_to_param[g_pm.ds_ids[name]] = ds_param.param

    candidate_bytes = sum(ds_id_to_size[ds_id] for ds_id in ds_ids)
    persistent_bytes = sum(ds_id_to_size.get(ds_id, 0) for ds_id in persistent_ds_ids)

    print_rank_0(
        f"selective_gather target_graph_id={target_graph_id} profiled_mem_lists={budget['profiled_list_count']} "
        f"total_mem={total_mem} usable_mem={budget['usable_mem']} peak_resident_alloc={budget['peak_resident_alloc']} "
        f"transient_peak={budget['transient_peak']} current_available_mem={current_available_mem} "
        f"usable_available_mem={available_mem} "
        f"persistent_count={len(persistent_ds_ids)} persistent_bytes={persistent_bytes} "
        f"candidate_count={len(ds_ids)} candidate_bytes={candidate_bytes}")

    if budget["profiled_list_count"] == 0:
        print_rank_0("selective_gather no profiling data; skipping persistence update")
        return gm

    if len(ds_ids) == 0:
        print_rank_0("selective_gather no candidates to persist")
        return gm

    if available_mem == 0:
        print_rank_0("selective_gather no currently available memory for new persistent params")
        return gm

    persistent_mem = 0
    selected_count = 0
    nz3 = get_deepcompile_handle()
    for ds_id in ds_ids:
        size = ds_id_to_size[ds_id]
        if persistent_mem + size > available_mem:
            break
        persistent_mem += size
        selected_count += 1

        param_obj = ds_id_to_param[ds_id]

        nz3.set_persistent(ds_id)
        print_rank_0(
            f"Set persistent: {ds_id} size: {size} persistent_mem: {persistent_mem} shape: {param_obj.ds_shape}")

    if selected_count == 0:
        smallest_candidate = min(ds_id_to_size[ds_id] for ds_id in ds_ids)
        print_rank_0(f"selective_gather selected no new params: available_mem={available_mem} "
                     f"smallest_candidate={smallest_candidate}")
    else:
        print_rank_0(f"selective_gather selected_count={selected_count} selected_bytes={persistent_mem}")

    return gm


# def make_selective_gather(z3_optimizer, nz3):

#     def selective_gather_wrapper(graph: Graph, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
#                                  mem_budget: float, param_manager, bwd: bool) -> Graph:
#         return selective_gather(graph, graph_id, graph_order, profiling_results, mem_budget, param_manager, bwd,
#                                 z3_optimizer, nz3)

#     return selective_gather_wrapper
