# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict, List, Callable, Tuple
import time
import gc
from collections import OrderedDict

import torch
from torch.fx import Graph, GraphModule

try:
    import torch.utils._pytree as pytree
    import torch._dynamo
    from functorch.compile import make_boxed_func
    from torch._functorch.aot_autograd import aot_module_simplified
    from torch._functorch.partitioners import min_cut_rematerialization_partition
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch._subclasses.fake_tensor import is_fake
except ImportError:
    pass

from deepspeed.accelerator import get_accelerator

from .fx import add_free_activations
from .graph_param import DSGraphParamManager
from .profilers import ProfilingResult
from .profilers.graph_profile import MemoryProfilingInterpreter
from .patch_compiled_func import patch_compiled_func, unpatch_compiled_func, get_backward_inputs
from .util import get_input_nodes, get_activation_node_names, get_index_by_graph_id, get_deepcompile_handle, log_rank0, is_backend_inductor
from .partitioner import get_wrapped_partitioner
from .inductor import register_custom_ops, patch_create_aot_dispatcher_function
from .input_storage import InputStorage

remaining_schedule = None
next_pass_step = -1
next_passes = None
current_passes = None

param_manager: Dict[int, DSGraphParamManager] = {}


class GraphOrder:

    def __init__(self):
        self.frames = OrderedDict()

    def add_graph(self, graph_id: int, frame_id: int, needs_backward: bool):
        if frame_id not in self.frames:
            self.frames[frame_id] = (graph_id, needs_backward)

    def get_graph_order(self) -> List[Tuple[int, bool]]:
        return list(self.frames.values())

    def clear(self):
        self.frames.clear()


graph_order_with_frame_id = GraphOrder()

frames_needing_bwd = set()
profiling_results: Dict[int, ProfilingResult] = {}
opt_pass_times = []
opt_passes = {}

fwd_real_inputs = []


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
        graph_order_with_frame_id.clear()
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


def evaluate_symint_from_shape_env(sym_int_v):
    assert isinstance(sym_int_v, torch.SymInt)
    # shape_env = sym_int_v.node.shape_env
    # v = shape_env.evaluate_sym_node(sym_int_v.node)
    return sym_int_v.node.hint


def set_example_values_to_symints(real_inputs, param_indices=None):
    real_inputs_ret = []

    # Create a set of parameter indices for quick lookup
    param_idx_set = set()
    if param_indices is not None:
        param_idx_set = {i for i, _, _ in param_indices}

    for i, v in enumerate(real_inputs):
        if isinstance(v, torch.Tensor):
            if is_fake(v):
                shape = []
                for fs in v.shape:
                    if isinstance(fs, torch.SymInt):
                        shape.append(evaluate_symint_from_shape_env(fs))
                    else:
                        shape.append(fs)
                stride = []
                for fs in v.stride():
                    if isinstance(fs, torch.SymInt):
                        stride.append(evaluate_symint_from_shape_env(fs))
                    else:
                        stride.append(fs)
                with unset_fake_temporarily():
                    dummy_v = torch.empty_strided(shape,
                                                  stride,
                                                  dtype=v.dtype,
                                                  layout=v.layout,
                                                  device=v.device,
                                                  requires_grad=v.requires_grad).zero_()

                    # Create Parameter if this input index corresponds to a parameter
                    if i in param_idx_set:
                        dummy_v = torch.nn.Parameter(dummy_v)
                        # Copy any additional attributes from the original if they exist
                        if hasattr(v, 'ds_id'):
                            dummy_v.ds_id = v.ds_id

                    real_inputs_ret.append(dummy_v)
            else:
                real_inputs_ret.append(v)
        else:
            if isinstance(v, torch.SymInt):
                real_inputs_ret.append(evaluate_symint_from_shape_env(v))
            else:
                real_inputs_ret.append(v)

    return tuple(real_inputs_ret)


def run_opt_passes(opt_passes: List[Callable],
                   gm: GraphModule,
                   graph_id: int,
                   graph_order: List[Tuple[int, bool]],
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


def make_backend(backend, compile_config, compile_kwargs={}):

    register_custom_ops()

    # Extract values from compile_config
    debug_log = compile_config.debug_log
    free_activation = compile_config.free_activation and not is_backend_inductor(backend)

    def backend_fn(gm: GraphModule, real_inputs):
        graph_id = id(gm.graph)

        needs_backward = pytree.tree_any(lambda x: x.requires_grad if torch.is_tensor(x) else False, real_inputs)
        frame_id = gm.meta["dynamo_compile_id"].frame_id
        graph_order_with_frame_id.add_graph(graph_id, frame_id, needs_backward)

        graph_order = graph_order_with_frame_id.get_graph_order()

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

        # Create an InputStorage instance for this specific graph
        # It will be captured by the make_fw_graph closure, eliminating the need for graph ID management
        input_storage = InputStorage(keep_int_input_tensors=compile_config.keep_int_input_tensors,
                                     keep_all_input_tensors=compile_config.keep_all_input_tensors)

        # Store in both list (for backward compatibility) and storage (for persistence)
        # The input_storage keeps tensor metadata to handle cases where
        # backend_fn is called once but make_fw_graph is called multiple times
        fwd_real_inputs.append(real_inputs)
        input_storage.put(real_inputs)

        global profiling_results
        if graph_id not in profiling_results:
            profiling_results[graph_id] = ProfilingResult()
            profiling_results[graph_id].param_indices = param_indices
            profiling_results[graph_id].needs_backward = needs_backward

        def make_fw_graph(gm, sample_inputs):
            time_start = time.time()
            graph_index = len(graph_order) - 1

            if needs_backward:
                if len(frames_needing_bwd) == 0:
                    patch_compiled_func()
                frames_needing_bwd.add(frame_id)

            # Try to get real_inputs from the list first, then from storage
            if fwd_real_inputs:
                real_inputs = fwd_real_inputs.pop(0)
            elif input_storage.has_data():
                # Note: input_storage is captured from the enclosing backend_fn scope
                # Materialize tensors from storage when list is empty
                log_rank0(f"Retrieving real inputs from storage for graph_id={graph_id}", enable=debug_log)
                real_inputs = input_storage.get()
            else:
                raise RuntimeError(f"No real inputs available for graph_id {graph_id}. "
                                   f"List size: {len(fwd_real_inputs)}, Storage has data: {input_storage.has_data()}")

            real_inputs = set_example_values_to_symints(real_inputs)

            param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)

            real_inputs_with_rng = real_inputs + tuple(sample_inputs[len(real_inputs):])
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

            opt_pass_times.append(("fwd", graph_index, graph_id, time.time() - time_start))

            log_rank0(f"Fwd end {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()}",
                      enable=debug_log)

            return gm.graph

        def make_bw_graph(gm, sample_inputs):
            time_start = time.time()

            graph_index = get_index_by_graph_id(graph_order, graph_id)
            log_rank0(
                f"Bwd start {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()} graph={gm.graph}",
                enable=debug_log)

            bwd_inputs_stack = get_backward_inputs()

            param_nodes_bw, _ = param_manager[graph_id].get_bwd_mapping(gm.graph)
            if len(bwd_inputs_stack) == 0:
                # dynamo calls bw compiler ahead of time when symints are saved for backward. See the details for aot_dispatch_autograd in jit_compile_runtime_wrappers.
                # As we currently use actually bwd input values in bw compiler, we make dummy data for profiling.
                # Replace fake tensors with real parameters before calling set_example_values_to_symints
                log_rank0(f"Generating dummy backward inputs for profiling. graph_id={graph_id}", enable=True)
                sample_inputs_with_real_params = param_manager[graph_id].replace_fake_tensors_with_real_params(
                    sample_inputs, gm.graph)
                bwd_real_inputs = set_example_values_to_symints(sample_inputs_with_real_params)
            else:
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
                param_names = [n.name for n in param_nodes_bw]
                non_param_input_names = [n.name for n in get_input_nodes(gm.graph) if n.name not in param_names]
                add_free_activations(graph_id, gm.graph,
                                     get_activation_node_names(gm.graph, param_nodes_bw, non_param_input_names))

            frames_needing_bwd.remove(frame_id)
            if len(frames_needing_bwd) == 0:
                unpatch_compiled_func()

            log_rank0(
                f"Bwd end {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()} graph={gm.graph}",
                enable=debug_log)

            opt_pass_times.append(("bwd", graph_index, graph_id, time.time() - time_start))

            return gm.graph

        if backend == "eager":

            def make_compiler_fn(make_graph_fn):

                def compiler_fn(gm, sample_inputs):
                    return None if make_graph_fn(gm, sample_inputs) is None else make_boxed_func(gm.forward)

                return compiler_fn

            partition_fn = get_wrapped_partitioner(z3_partition, param_indices, min_cut_rematerialization_partition)
            aot_mod = aot_module_simplified(gm,
                                            real_inputs,
                                            fw_compiler=make_compiler_fn(make_fw_graph),
                                            bw_compiler=make_compiler_fn(make_bw_graph),
                                            partition_fn=partition_fn)
            return torch._dynamo.optimize(**compile_kwargs)(aot_mod)
        elif backend == "inductor":
            patch_create_aot_dispatcher_function(graph_id, z3_partition, make_fw_graph, make_bw_graph, real_inputs,
                                                 param_indices, param_manager)

            return torch._inductor.compile(gm, real_inputs)

        raise ValueError(f"Unsupported backend {backend}")

    return backend_fn
