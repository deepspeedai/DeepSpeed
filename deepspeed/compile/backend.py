# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict, List, Callable
import time
import gc

import torch
from torch.fx import Graph, GraphModule

try:
    import torch.utils._pytree as pytree
    import torch._dynamo
    import torch._inductor.scheduler
    from functorch.compile import make_boxed_func
    from torch._functorch.aot_autograd import aot_module_simplified
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    pass

from deepspeed.accelerator import get_accelerator

from .fx import add_free_activations
from .graph_param import DSGraphParamManager
from .profilers import ProfilingResult
from .profilers.graph_profile import MemoryProfilingInterpreter
from .patch_compiled_func import patch_compiled_func, unpatch_compiled_func, get_backward_inputs
from .util import get_input_nodes, get_activation_node_names, get_index_by_graph_id, get_deepcompile_handle, log_rank0
from .partitioner import get_wrapped_partitioner
from .inductor import register_custom_ops, patch_create_aot_dispatcher_function

remaining_schedule = None
next_pass_step = -1
next_passes = None
current_passes = None

param_manager: Dict[int, DSGraphParamManager] = {}
graph_order = []
profiling_results: Dict[int, ProfilingResult] = {}
opt_pass_times = []

opt_passes = {}

fwd_real_inputs = []
remaining_bwd_compile_count = 0


def register_compile_pass(name: str, opt_pass_fn):
    opt_passes[name] = opt_pass_fn


def init_schedule(schedule):

    assert isinstance(schedule, list), f"schedule should be a list, but got {type(schedule)}"

    for step, passes in schedule:
        assert isinstance(step, int), f"Each step in schedule should be an integer, but got {type(step)}"
        assert isinstance(passes, list), f"Passes at a certain step should be a list, but got {type(passes)}"

    global remaining_schedule
    remaining_schedule = schedule


def launch_compile_passes(global_steps: int):
    global next_pass_step, next_passes

    if len(remaining_schedule) > 0 and global_steps == remaining_schedule[0][0]:
        _, next_passes = remaining_schedule.pop(0)
        log_rank0(f"Launching compile passes: global_steps={global_steps} passes={next_passes}", True)

        torch._dynamo.reset()
        get_deepcompile_handle().reset()
        patch_compiled_func()
        graph_order.clear()
        profiling_results.clear()
        param_manager.clear()


def set_time_and_tensor_size(graph_id, graph: Graph, mem, bwd, profiling_results):
    node_time = []
    tensor_sizes = []

    for n in graph.nodes:
        node_time.append((n.name, n.meta["device_time"] if "device_time" in n.meta else 0.0,
                          n.meta["wall_time"] if "wall_time" in n.meta else 0.0))
        tensor_sizes.append((n.name, n.meta["tensor_size"] if "tensor_size" in n.meta else 0))

    if bwd:
        profiling_results[graph_id].bwd_graph = graph
        profiling_results[graph_id].bwd_time = node_time
        profiling_results[graph_id].bwd_tensor_sizes = tensor_sizes
        profiling_results[graph_id].bwd_mem = mem
    else:
        profiling_results[graph_id].fwd_graph = graph
        profiling_results[graph_id].fwd_time = node_time
        profiling_results[graph_id].fwd_tensor_sizes = tensor_sizes
        profiling_results[graph_id].fwd_mem = mem


def run_opt_passes(opt_passes: List[Callable],
                   gm: GraphModule,
                   graph_id: int,
                   graph_order: List[int],
                   profiling_results,
                   create_inputs_fn,
                   mem_budget: float,
                   param_manager,
                   bwd: bool,
                   debug_log=False) -> None:

    with unset_fake_temporarily():
        get_accelerator().synchronize()
        gc.collect()
        get_accelerator().empty_cache()

    for i, opt_pass_fn in enumerate(opt_passes):
        log_rank0(f"Running opt pass {i} for graph {graph_id}. bwd={bwd}", enable=debug_log)

        gm_new = opt_pass_fn(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager,
                             bwd)
        if gm_new is not None:
            gm = gm_new
            gm.graph.lint()
            gm.recompile()

            mem_prof = MemoryProfilingInterpreter(gm, debug_log=debug_log)
            mem_prof.run(*create_inputs_fn())
            mem = [(name, current_alloc, delta, peak) for name, current_alloc, delta, peak in mem_prof.mem_record]

            set_time_and_tensor_size(graph_id, gm.graph, mem, bwd, profiling_results)

        with unset_fake_temporarily():
            get_accelerator().synchronize()
            gc.collect()
            get_accelerator().empty_cache()


def make_backend(backend, compile_kwargs={}, free_activation=False, debug_log=False):

    register_custom_ops()

    def backend_fn(gm: GraphModule, real_inputs):
        graph_id = id(gm.graph)
        needs_backward = pytree.tree_any(lambda x: x.requires_grad if torch.is_tensor(x) else False, real_inputs)

        global graph_order
        graph_order.append((graph_id, needs_backward))

        z3_partition = any(hasattr(v, "ds_id") for v in real_inputs)
        if z3_partition:
            param_indices = [(i, input_val.ds_id, input_val.ds_shape) for i, input_val in enumerate(real_inputs)
                             if isinstance(input_val, torch.nn.Parameter)]
        else:
            assert all(hasattr(v, "param_id") for v in real_inputs
                       if isinstance(v, torch.nn.Parameter)), "All param inputs should have param_id"
            param_indices = [(i, input_val.param_id, input_val.shape) for i, input_val in enumerate(real_inputs)
                             if isinstance(input_val, torch.nn.Parameter)]

        global fwd_real_inputs
        fwd_real_inputs.append(real_inputs)

        global profiling_results
        if graph_id not in profiling_results:
            profiling_results[graph_id] = ProfilingResult()
            profiling_results[graph_id].param_indices = param_indices
            profiling_results[graph_id].needs_backward = needs_backward

        def make_fw_graph(gm, sample_inputs):
            time_start = time.time()
            graph_index = len(graph_order) - 1
            real_inputs = fwd_real_inputs.pop(0)

            param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)

            real_inputs_with_rng = real_inputs + sample_inputs[len(real_inputs):]
            run_opt_passes(
                opt_passes=next_passes,
                gm=gm,
                graph_id=graph_id,
                graph_order=graph_order,
                profiling_results=profiling_results,
                create_inputs_fn=lambda: real_inputs_with_rng,
                mem_budget=.0,  # unused
                param_manager=param_manager,
                bwd=False,
                debug_log=debug_log)

            if needs_backward:
                global remaining_bwd_compile_count
                remaining_bwd_compile_count += 1

            opt_pass_times.append(("fwd", graph_index, graph_id, time.time() - time_start))

            log_rank0(
                f"Fwd end {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()} graph={gm.graph}",
                enable=debug_log)

            return gm.graph

        def make_bw_graph(gm, sample_inputs):
            time_start = time.time()

            graph_index = get_index_by_graph_id(graph_order, graph_id)
            log_rank0(
                f"Bwd start {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()} graph={gm.graph}",
                enable=debug_log)

            bwd_inputs_stack = get_backward_inputs()

            if len(bwd_inputs_stack) == 0:
                # dynamo calls bw compiler ahead of time when symints are saved for backward. See the details for aot_dispatch_autograd in jit_compile_runtime_wrappers.
                # As we currently use actually bwd input values in bw compiler, we return None to skip the compilation there.
                # This would need be handled properly in the future.
                return None

            bwd_real_inputs = bwd_inputs_stack.pop()
            run_opt_passes(
                opt_passes=next_passes,
                gm=gm,
                graph_id=graph_id,
                graph_order=graph_order,
                profiling_results=profiling_results,
                create_inputs_fn=lambda: tuple(bwd_real_inputs),
                mem_budget=.0,  # unused
                param_manager=param_manager,
                bwd=True,
                debug_log=debug_log)

            # assert graph_id in param_manager, f"Graph {graph_id} not found in param_manager"

            if free_activation:
                param_nodes_bw, _ = param_manager[graph_id].get_bwd_mapping(gm.graph)
                param_names = [n.name for n in param_nodes_bw]
                non_param_input_names = [n.name for n in get_input_nodes(gm.graph) if n.name not in param_names]
                add_free_activations(graph_id, gm.graph,
                                     get_activation_node_names(gm.graph, param_nodes_bw, non_param_input_names))

            global remaining_bwd_compile_count
            remaining_bwd_compile_count -= 1
            if remaining_bwd_compile_count == 0:
                unpatch_compiled_func()

            log_rank0(
                f"Bwd end {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()} graph={gm.graph}",
                enable=debug_log)

            gm.recompile()
            opt_pass_times.append(("bwd", graph_index, graph_id, time.time() - time_start))

            return gm.graph

        if backend == "eager":

            def make_compiler_fn(make_graph_fn):

                def compiler_fn(gm, sample_inputs):
                    return None if make_graph_fn(gm, sample_inputs) is None else make_boxed_func(gm.forward)

                return compiler_fn

            aot_mod = aot_module_simplified(gm,
                                            real_inputs,
                                            fw_compiler=make_compiler_fn(make_fw_graph),
                                            bw_compiler=make_compiler_fn(make_bw_graph),
                                            partition_fn=get_wrapped_partitioner(param_indices))
            return torch._dynamo.optimize(**compile_kwargs)(aot_mod)
        elif backend == "inductor":
            patch_create_aot_dispatcher_function(graph_id, z3_partition, make_fw_graph, make_bw_graph, real_inputs,
                                                 param_indices, param_manager)
            from .partitioner import get_wrapped_choose_saved_values_set
            torch._functorch.partitioners.choose_saved_values_set = get_wrapped_choose_saved_values_set(param_indices)

            return torch._inductor.compile(gm, real_inputs)

        raise ValueError(f"Unsupported backend {backend}")

    return backend_fn
