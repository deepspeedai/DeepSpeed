# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict
from typing import List, Tuple

import torch
from torch.fx import GraphModule

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from ..util import get_deepcompile_handle
from ..graph_param import DSGraphParamManager

NAME = "selective_gather"

max_alloc_mem = 0
last_optimize_step = 0


def selective_gather(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                     create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                     bwd: bool) -> GraphModule:

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

    peak_mem = 0
    for graph_id, prof in profiling_results.items():
        # Use peak memory
        fwd_max_mem = max(m[3] for m in prof.fwd_mem)
        bwd_max_mem = max(m[3] for m in prof.bwd_mem) if len(prof.bwd_mem) > 0 else 0
        peak_mem = max(peak_mem, fwd_max_mem, bwd_max_mem)
        if dist.get_rank() == 0:
            print(
                f"selective_gather graph_id={graph_id} max_mem={peak_mem} fwd_max_mem={fwd_max_mem} bwd_max_mem={bwd_max_mem}"
            )

    persistent_ds_ids = set()
    for graph_id, pm in param_manager.items():
        for name, ds_param in pm.params.items():
            if ds_param.param.ds_persist:
                persistent_ds_ids.add(pm.ds_ids[name])

    ds_id_to_size = {}
    ds_id_to_time = defaultdict(float)
    ds_id_to_prof_dtime = defaultdict(float)
    ds_id_to_prof_wtime = defaultdict(float)

    for graph_id, pm in param_manager.items():
        params = pm.params
        for param_name, param in params.items():
            ds_id = pm.ds_ids[param_name]
            ds_id_to_size[ds_id] = param.numel * param.dtype.itemsize

        profile = profiling_results[graph_id]
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

    sorted_ds_ids = {ds_id: ds_id_to_size[ds_id] for ds_id in ds_ids}

    accelerator = get_accelerator()
    total_mem = accelerator.total_memory()
    vals_to_bcast = torch.tensor([total_mem], device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    total_mem = vals_to_bcast[0].item()

    MEM_MARGIN = 0.1
    available_mem = total_mem * (1 - MEM_MARGIN) - peak_mem

    if dist.get_rank() == 0:
        print(
            f"selective_gather max_mem={peak_mem} total_mem={total_mem} MEM_MARGIN={MEM_MARGIN} available_mem={available_mem}"
        )

    ds_id_to_param = {}
    for g_id, g_pm in param_manager.items():
        for name, ds_param in g_pm.params.items():
            ds_id_to_param[g_pm.ds_ids[name]] = ds_param.param

    persistent_mem = 0
    nz3 = get_deepcompile_handle()
    for ds_id, size in sorted_ds_ids.items():
        if persistent_mem + size > available_mem:
            break
        persistent_mem += size

        param_obj = ds_id_to_param[ds_id]

        nz3.set_persistent(ds_id)
        if dist.get_rank() == 0:
            print(f"Set persistent: {ds_id} size: {size} persistent_mem: {persistent_mem} shape: {param_obj.ds_shape}")

    return gm


# def make_selective_gather(z3_optimizer, nz3):

#     def selective_gather_wrapper(graph: Graph, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
#                                  mem_budget: float, param_manager, bwd: bool) -> Graph:
#         return selective_gather(graph, graph_id, graph_order, profiling_results, mem_budget, param_manager, bwd,
#                                 z3_optimizer, nz3)

#     return selective_gather_wrapper
