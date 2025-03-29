# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ..profilers.graph_profile import MemoryProfilingInterpreter

import deepspeed.comm as dist


def run_opt_passes(nz3,
                   graph_index,
                   graph_id,
                   gm,
                   create_inputs_fn,
                   opt_passes,
                   graph_order,
                   profiling_results,
                   param_manager,
                   bwd,
                   debug_log=False):
    profile = profiling_results[graph_id]
    rank = dist.get_rank()

    for i, opt_pass in enumerate(opt_passes):

        opt_pass_fn, mem_budget = opt_pass

        graph = opt_pass_fn(gm.graph, graph_id, graph_order, profiling_results, mem_budget, param_manager, bwd)
        graph.lint()
        gm.graph = graph
        gm.recompile()

        if debug_log:
            print(f"Prefetching enabled for {'bwd' if bwd else 'fwd'} graph_id={graph_id} {graph}")

        mem_prof = MemoryProfilingInterpreter(nz3, gm)
        mem_prof.run(*create_inputs_fn())
        if debug_log and rank == 0:
            mem_prof.dump(f"mem_prof_r{rank}_{'bwd' if bwd else 'fwd'}_{graph_index}_{graph_id}_pass_{i}.csv")

        mem = [(name, current_alloc, delta, peak) for name, current_alloc, delta, peak in mem_prof.mem_record]
        if bwd:
            profile.bwd_mem = mem
        else:
            profile.fwd_mem = mem

    return gm
