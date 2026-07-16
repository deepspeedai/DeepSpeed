# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict, List, Callable, Tuple, Set
import time
import gc
from collections import OrderedDict, deque

import torch
from torch.fx import Graph, GraphModule

try:
    import torch._dynamo
    from functorch.compile import make_boxed_func
    from torch._functorch.aot_autograd import aot_module_simplified
    from torch._functorch.partitioners import min_cut_rematerialization_partition
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch._subclasses.fake_tensor import is_fake
except ImportError:
    pass

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .fx import add_free_activations
from .graph_param import DSGraphParamManager
from .profilers import ProfilingResult
from .profilers.graph_profile import MemoryProfilingInterpreter, is_profile_incomplete
from .patch_compiled_func import patch_compiled_func, unpatch_compiled_func, get_backward_inputs
from .util import get_input_nodes, get_activation_node_names, get_index_by_graph_id, get_deepcompile_handle, log_rank0, is_backend_inductor
from .partitioner import get_wrapped_partitioner
from .inductor import register_custom_ops, patch_create_aot_dispatcher_function, deepcompile_z3_inductor_config_patch
from .input_storage import InputStorage

remaining_schedule = None
next_pass_step = -1
next_passes = None
current_passes = None

param_manager: Dict[int, DSGraphParamManager] = {}


class GraphOrder:

    def __init__(self):
        self.frames = OrderedDict()

    def __len__(self):
        return len(self.frames)

    def add_graph(self, graph_id: int, frame_id: int):
        if frame_id not in self.frames:
            self.frames[frame_id] = (graph_id, None)

    def set_needs_backward(self, frame_id: int, needs_backward: bool):
        if frame_id in self.frames:
            self.frames[frame_id] = (self.frames[frame_id][0], needs_backward)

    def get_graph_order(self) -> List[Tuple[int, bool]]:
        assert all(isinstance(needs_backward, bool) for _, needs_backward in self.frames.values())
        return list(self.frames.values())

    def clear(self):
        self.frames.clear()


graph_order_with_frame_id = GraphOrder()

frames_needing_bwd = set()
frames_partitioned: Set[int] = set()
profiling_results: Dict[int, ProfilingResult] = {}
opt_pass_times = []
opt_passes = {}

fwd_real_inputs = []


def cleanup_compiled_backward_state(frame_id=None, owned_frames=None):
    """Release process-global compiled-backward state after completion or failure."""
    if frame_id is None:
        if owned_frames is None:
            frames_needing_bwd.clear()
        else:
            frames_needing_bwd.difference_update(owned_frames)
            owned_frames.clear()
    else:
        frames_needing_bwd.discard(frame_id)
        if owned_frames is not None:
            owned_frames.discard(frame_id)
    if len(frames_needing_bwd) == 0:
        unpatch_compiled_func()


def _cleanup_compiled_backward_state_on_error(frame_id, owned_frames=None):

    def decorator(fn):

        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                cleanup_compiled_backward_state(frame_id, owned_frames)
                raise

        return wrapped

    return decorator


def _cleanup_compiled_backward_backend_state_on_error(owned_frames=None):

    def decorator(fn):

        def wrapped(gm, *args, **kwargs):
            frame_id = gm.meta["dynamo_compile_id"].frame_id
            try:
                return fn(gm, *args, **kwargs)
            except Exception:
                cleanup_compiled_backward_state(frame_id, owned_frames)
                raise

        return wrapped

    return decorator


def register_compile_pass(name: str, opt_pass_fn):
    opt_passes[name] = opt_pass_fn


def init_schedule(schedule):

    assert isinstance(schedule, list), f"schedule should be a list, but got {type(schedule)}"

    for step, passes in schedule:
        assert isinstance(step, int), f"Each step in schedule should be an integer, but got {type(step)}"
        assert isinstance(passes, list), f"Passes at a certain step should be a list, but got {type(passes)}"

    global remaining_schedule
    remaining_schedule = deque(schedule)


def launch_compile_passes(global_steps: int, owned_frames=None):
    """Advance the pass schedule and discard state owned by the previous compile cycle."""
    global next_pass_step, next_passes

    if len(remaining_schedule) > 0 and global_steps == remaining_schedule[0][0]:
        _, next_passes = remaining_schedule.popleft()
        log_rank0(f"Launching compile passes: global_steps={global_steps} passes={next_passes}", True)

        torch._dynamo.reset()
        get_deepcompile_handle().reset()
        graph_order_with_frame_id.clear()
        profiling_results.clear()
        param_manager.clear()
        fwd_real_inputs.clear()
        cleanup_compiled_backward_state(owned_frames=owned_frames)
        frames_partitioned.clear()


def set_time_and_tensor_size(graph_id, graph: Graph, mem, bwd, profiling_results, mem_complete=True):
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
        profiling_results[graph_id].bwd_mem_complete = mem_complete
    else:
        profiling_results[graph_id].fwd_graph = graph
        profiling_results[graph_id].fwd_time = node_time
        profiling_results[graph_id].fwd_tensor_sizes = tensor_sizes
        profiling_results[graph_id].fwd_mem = mem
        profiling_results[graph_id].fwd_mem_complete = mem_complete


def _sync_memory_profile_complete(profile_complete: bool, process_group=None) -> bool:
    if not dist.is_initialized():
        return profile_complete

    complete = torch.tensor([1 if profile_complete else 0], device=torch.device(get_accelerator().current_device()))
    if process_group is None:
        dist.all_reduce(complete, dist.ReduceOp.MIN)
    else:
        dist.all_reduce(complete, dist.ReduceOp.MIN, group=process_group)
    return bool(complete.item())


def evaluate_symint_from_shape_env(sym_int_v):
    assert isinstance(sym_int_v, torch.SymInt)
    # shape_env = sym_int_v.node.shape_env
    # v = shape_env.evaluate_sym_node(sym_int_v.node)
    return sym_int_v.node.hint


_ZERO_PARAMETER_COMPILE_METADATA = ("ds_id", "ds_shape", "ds_persist", "ds_status", "ds_target_dtype")


def set_example_values_to_symints(real_inputs, param_indices=None, real_zero_params=None):
    real_inputs_ret = []

    # Create a set of parameter indices for quick lookup
    param_idx_set = set()
    if param_indices is not None:
        param_idx_set = {i for i, _, _ in param_indices}
    param_ds_ids = {i: ds_id for i, ds_id, _ in (param_indices or [])}
    real_zero_params = real_zero_params or {}

    for i, v in enumerate(real_inputs):
        if isinstance(v, torch.Tensor):
            real_zero_param = real_zero_params.get(param_ds_ids.get(i))
            if i in param_idx_set and real_zero_param is not None:
                # Stored and fake profiling inputs both discard instance-bound
                # ZeRO methods, so recover the original before materialization.
                real_inputs_ret.append(real_zero_param)
                continue

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
                        dummy_v = torch.nn.Parameter(dummy_v, requires_grad=v.requires_grad)
                        # Profiling and graph-parameter consumers use these ZeRO
                        # attributes after symbolic fake inputs are materialized.
                        for attr in _ZERO_PARAMETER_COMPILE_METADATA:
                            if hasattr(v, attr):
                                setattr(dummy_v, attr, getattr(v, attr))

                    real_inputs_ret.append(dummy_v)
            else:
                real_inputs_ret.append(v)
        else:
            if isinstance(v, torch.SymInt):
                real_inputs_ret.append(evaluate_symint_from_shape_env(v))
            else:
                real_inputs_ret.append(v)

    return tuple(real_inputs_ret)


def _get_fw_real_inputs(local_real_inputs, input_storage: InputStorage, graph_id: int, debug_log: bool = False):
    """Resolve real inputs from the one-shot queue, persistent storage, then legacy state."""
    if local_real_inputs:
        return local_real_inputs.popleft()

    if input_storage.has_data():
        if debug_log:
            log_rank0(f"Retrieving real inputs from storage for graph_id={graph_id}", enable=True)
        return input_storage.get()

    if fwd_real_inputs:
        if debug_log:
            log_rank0(f"Retrieving real inputs from legacy global queue for graph_id={graph_id}", enable=True)
        return fwd_real_inputs.pop(0)

    raise RuntimeError(f"No real inputs available for graph_id {graph_id}. "
                       f"Local queue size: {len(local_real_inputs)}, "
                       f"global queue size: {len(fwd_real_inputs)}, "
                       f"storage has data: {input_storage.has_data()}")


def run_opt_passes(opt_passes: List[Callable],
                   gm: GraphModule,
                   graph_id: int,
                   graph_order: List[Tuple[int, bool]],
                   profiling_results,
                   create_inputs_fn,
                   mem_budget: float,
                   param_manager,
                   bwd: bool,
                   debug_log=False,
                   process_group=None) -> None:
    """Apply scheduled graph passes and retain only complete post-pass memory profiles."""

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

            # Re-profiling an already incomplete graph would turn synthetic
            # backfilled metadata into a seemingly valid memory profile.
            operator_profile_complete = _sync_memory_profile_complete(not is_profile_incomplete(gm.graph),
                                                                      process_group)
            if not operator_profile_complete:
                profile_complete = False
                mem = []
            else:
                mem_prof = MemoryProfilingInterpreter(gm, debug_log=debug_log, process_group=process_group)
                mem_prof.run(*create_inputs_fn())
                profile_complete = _sync_memory_profile_complete(mem_prof.profile_complete, process_group)
                if profile_complete:
                    mem = [(name, current_alloc, delta, peak)
                           for name, current_alloc, delta, peak in mem_prof.mem_record]
                else:
                    mem = []
                del mem_prof

            set_time_and_tensor_size(graph_id, gm.graph, mem, bwd, profiling_results, profile_complete)

        with unset_fake_temporarily():
            get_accelerator().synchronize()
            gc.collect()
            get_accelerator().empty_cache()


def make_backend(backend, compile_config, compile_kwargs={}, process_group=None, owned_frames=None):

    register_custom_ops()

    # Extract values from compile_config
    debug_log = compile_config.debug_log
    free_activation = compile_config.free_activation and not is_backend_inductor(backend)

    if owned_frames is None:
        owned_frames = set()

    @_cleanup_compiled_backward_backend_state_on_error(owned_frames)
    def backend_fn(gm: GraphModule, real_inputs):
        graph_id = id(gm.graph)

        # Checking the existence of input tensors requiring grad alone is insufficient to determine `need_backward`.
        # AOT autograd also checks the graph data flow and skips the backward pass if no output requires grad and no
        # input requiring grad is mutated.
        #
        # Instead of replicating AOT autograd's backward pass determination (which is too costly), we infer whether
        # backward pass is needed by checking if the joint graph is partitioned (into a forward and a backward module).
        # This check cannot be placed here because autograd creates the fw/bw compiler callables before graph
        # partitioning. It is thus postponed to the point where the fw compiler is called.
        frame_id = gm.meta["dynamo_compile_id"].frame_id
        graph_order_with_frame_id.add_graph(graph_id, frame_id)

        z3_partition = any(hasattr(v, "ds_id") for v in real_inputs)
        if z3_partition:
            param_indices = [(i, input_val.ds_id, input_val.ds_shape) for i, input_val in enumerate(real_inputs)
                             if isinstance(input_val, torch.nn.Parameter)]
            real_zero_params = {
                input_val.ds_id: input_val
                for input_val in real_inputs if isinstance(input_val, torch.nn.Parameter)
                and hasattr(input_val, "all_gather") and hasattr(input_val, "partition")
            }
        else:
            assert all(hasattr(v, "param_id") for v in real_inputs
                       if isinstance(v, torch.nn.Parameter)), "All param inputs should have param_id"
            param_indices = [(i, input_val.param_id, input_val.shape) for i, input_val in enumerate(real_inputs)
                             if isinstance(input_val, torch.nn.Parameter)]
            real_zero_params = {}

        global fwd_real_inputs

        # Create an InputStorage instance for this specific graph
        # It will be captured by the make_fw_graph closure, eliminating the need for graph ID management
        input_storage = InputStorage(keep_int_input_tensors=compile_config.keep_int_input_tensors,
                                     keep_all_input_tensors=compile_config.keep_all_input_tensors)

        # Store in a closure-local queue and storage (for persistence).
        # The input_storage keeps tensor metadata to handle cases where
        # backend_fn is called once but make_fw_graph is called multiple times
        local_fwd_real_inputs = deque([real_inputs])
        input_storage.put(real_inputs)

        global profiling_results
        if graph_id not in profiling_results:
            profiling_results[graph_id] = ProfilingResult(process_group=process_group)
            profiling_results[graph_id].param_indices = param_indices

        @_cleanup_compiled_backward_state_on_error(frame_id, owned_frames)
        def make_fw_graph(gm, sample_inputs):
            """Apply forward passes with graph-local real inputs and return the rewritten FX graph."""
            time_start = time.time()
            graph_index = len(graph_order_with_frame_id) - 1

            needs_backward = frame_id in frames_partitioned
            graph_order_with_frame_id.set_needs_backward(frame_id, needs_backward)
            profiling_results[graph_id].needs_backward = needs_backward

            if needs_backward:
                if len(frames_needing_bwd) == 0:
                    patch_compiled_func()
                frames_needing_bwd.add(frame_id)
                owned_frames.add(frame_id)

            real_inputs = _get_fw_real_inputs(local_fwd_real_inputs, input_storage, graph_id, debug_log=debug_log)
            real_inputs = set_example_values_to_symints(real_inputs, param_indices, real_zero_params=real_zero_params)

            param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)

            real_inputs_with_rng = real_inputs + tuple(sample_inputs[len(real_inputs):])
            run_opt_passes(
                opt_passes=next_passes,
                gm=gm,
                graph_id=graph_id,
                graph_order=graph_order_with_frame_id.get_graph_order(),
                profiling_results=profiling_results,
                create_inputs_fn=lambda: real_inputs_with_rng,
                mem_budget=.0,  # unused
                param_manager=param_manager,
                bwd=False,
                debug_log=debug_log,
                process_group=process_group)

            opt_pass_times.append(("fwd", graph_index, graph_id, time.time() - time_start))

            log_rank0(f"Fwd end {graph_index} graph_id={graph_id} alloc_mem={get_accelerator().memory_allocated()}",
                      enable=debug_log)

            return gm.graph

        @_cleanup_compiled_backward_state_on_error(frame_id, owned_frames)
        def make_bw_graph(gm, sample_inputs):
            time_start = time.time()

            graph_order = graph_order_with_frame_id.get_graph_order()
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
                debug_log=debug_log,
                process_group=process_group)

            # assert graph_id in param_manager, f"Graph {graph_id} not found in param_manager"

            if free_activation:
                param_names = [n.name for n in param_nodes_bw]
                non_param_input_names = [n.name for n in get_input_nodes(gm.graph) if n.name not in param_names]
                add_free_activations(graph_id, gm.graph,
                                     get_activation_node_names(gm.graph, param_nodes_bw, non_param_input_names))

            cleanup_compiled_backward_state(frame_id, owned_frames)

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

            partition_fn = get_wrapped_partitioner(z3_partition, param_indices, min_cut_rematerialization_partition,
                                                   frame_id, frames_partitioned)
            aot_mod = aot_module_simplified(gm,
                                            real_inputs,
                                            fw_compiler=make_compiler_fn(make_fw_graph),
                                            bw_compiler=make_compiler_fn(make_bw_graph),
                                            partition_fn=partition_fn)
            return torch._dynamo.optimize(**compile_kwargs)(aot_mod)
        elif backend == "inductor":
            restore_aotautograd = patch_create_aot_dispatcher_function(graph_id, z3_partition, make_fw_graph,
                                                                       make_bw_graph, real_inputs, param_indices,
                                                                       param_manager, frame_id, frames_partitioned)
            try:
                with deepcompile_z3_inductor_config_patch(z3_partition):
                    return torch._inductor.compile(gm, real_inputs)
            finally:
                # AotAutograd.__init__ is process-global; never leak this
                # graph-specific compiler wiring into a later compilation.
                restore_aotautograd()

        raise ValueError(f"Unsupported backend {backend}")

    return backend_fn
