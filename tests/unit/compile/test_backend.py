# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import deque
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import torch

from deepspeed.compile.backend import _get_fw_real_inputs, set_example_values_to_symints
from deepspeed.compile import backend as backend_mod
from deepspeed.compile.inductor import patch_create_aot_dispatcher_function
from deepspeed.compile.graph_param import DSGraphParamManager
from deepspeed.compile.input_storage import InputStorage
from deepspeed.compile.patch_compiled_func import (clear_backward_inputs, get_backward_inputs, patch_compiled_func,
                                                   unpatch_compiled_func)
from deepspeed.compile.profilers import ProfilingResult
from deepspeed.compile.profilers import graph_profile as graph_profile_mod
from deepspeed.compile.profilers.graph_profile import _mark_profile_incomplete

_DC_LIBRARIES = []


def test_profiling_result_keeps_existing_positional_field_order():
    graph = torch.fx.Graph()

    result = ProfilingResult(graph)

    assert result.fwd_graph is graph
    assert result.process_group is None


def _define_dc_ops():
    try:
        torch.ops.dc.allgather_param.default
        torch.ops.dc.wait_allgather.default
        torch.ops.dc.release_param.default
        torch.ops.dc.reduce_grad.default
        return
    except AttributeError:
        pass

    lib = torch.library.Library("dc", "FRAGMENT")
    for schema in (
            "allgather_param(Tensor a, int graph_id, int id, ScalarType? dtype = None) -> Tensor",
            "wait_allgather(Tensor(a) a, int graph_id, int id) -> Tensor(a)",
            "release_param(Tensor(a) a, int graph_id, int id, int n_users) -> Tensor(a)",
            "reduce_grad(Tensor a, int graph_id, int id) -> Tensor",
            "free_tensors(Tensor[] tensors) -> ()",
            "end_backward(Tensor[] tensors, int graph_id, bool release_reduce_buckets = True) -> ()",
    ):
        try:
            lib.define(schema)
        except RuntimeError as exc:
            if "already been registered" not in str(exc):
                raise
    _DC_LIBRARIES.append(lib)


def test_forward_real_inputs_are_graph_local():
    local_inputs = (torch.nn.Parameter(torch.ones(2, dtype=torch.float32)), )
    storage = InputStorage()
    storage.put((torch.ones(1, dtype=torch.float32), ))

    selected = _get_fw_real_inputs(deque([local_inputs]), storage, graph_id=7)

    assert selected is local_inputs


def test_forward_real_inputs_fall_back_to_storage_when_local_queue_is_empty():
    storage = InputStorage()
    storage.put((torch.ones(3, dtype=torch.float32), ))

    selected = _get_fw_real_inputs(deque(), storage, graph_id=7)

    assert len(selected) == 1
    assert selected[0].shape == torch.Size([3])
    assert selected[0].dtype is torch.float32


def test_stored_zero_parameter_recovers_original_instance_bound_protocol():
    real_param = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    real_param.ds_id = 123
    real_param.ds_shape = torch.Size([3])
    real_param.ds_persist = False
    real_param.all_gather = MethodType(lambda self, param_list: None, real_param)
    real_param.partition = MethodType(lambda self, param_list, has_been_updated: None, real_param)
    storage = InputStorage()
    storage.put((real_param, ))

    stored_inputs = storage.get()
    materialized = set_example_values_to_symints(stored_inputs, [(0, 123, torch.Size([3]))],
                                                 real_zero_params={123: real_param})

    assert stored_inputs[0] is not real_param
    assert not hasattr(stored_inputs[0], "ds_id")
    assert materialized[0] is real_param
    assert materialized[0].all_gather.__self__ is real_param
    assert materialized[0].partition.__self__ is real_param


def test_symint_materialization_preserves_frozen_zero_parameter_for_profiling_consumers(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    _define_dc_ops()
    calls = []
    real_param = torch.nn.Parameter(torch.empty((2, 3), dtype=torch.bfloat16), requires_grad=False)
    real_param.ds_id = 123
    real_param.ds_shape = torch.Size([2, 3])
    real_param.ds_persist = False

    def all_gather(self, param_list):
        calls.append(("all_gather", param_list))

    def partition(self, param_list, has_been_updated):
        calls.append(("partition", param_list, has_been_updated))

    real_param.all_gather = MethodType(all_gather, real_param)
    real_param.partition = MethodType(partition, real_param)

    with FakeTensorMode() as fake_mode:
        fake_param = fake_mode.from_tensor(real_param)
    fake_param.ds_id = 123
    fake_param.ds_shape = torch.Size([2, 3])
    fake_param.ds_persist = False

    materialized = set_example_values_to_symints((fake_param, ), [(0, 123, torch.Size([2, 3]))],
                                                 real_zero_params={123: real_param})
    assert materialized[0] is real_param
    graph = torch.fx.Graph()
    param_node = graph.placeholder("frozen_zero_param")
    neg_node = graph.call_function(torch.neg, (param_node, ))
    graph.output((neg_node, ))
    manager = DSGraphParamManager(graph, materialized, [(0, 123, torch.Size([2, 3]))])
    managed_param = manager.params[param_node.name].param
    persistent_ds_ids = {
        manager.ds_ids[name]
        for name, graph_param in manager.params.items() if graph_param.param.ds_persist
    }

    assert isinstance(managed_param, torch.nn.Parameter)
    assert managed_param.shape == torch.Size([2, 3])
    assert managed_param.dtype is torch.bfloat16
    assert not managed_param.requires_grad
    assert managed_param.ds_id == 123
    assert managed_param.ds_shape == torch.Size([2, 3])
    assert managed_param.ds_persist is False
    assert persistent_ds_ids == set()

    class FakeEvent:

        def record(self):
            pass

        def elapsed_time(self, other):
            return 0.0

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def random(self):
            return SimpleNamespace(fork_rng=lambda devices: nullcontext())

        def Event(self, enable_timing):
            return FakeEvent()

        def reset_peak_memory_stats(self):
            pass

        def memory_allocated(self):
            return 100

        def max_memory_allocated(self):
            return 100

        def synchronize(self):
            pass

    class FakeDeepCompileHandle:

        def enable_profiling(self, enabled):
            pass

        def clear_all_gathered_params(self):
            pass

    monkeypatch.setattr(graph_profile_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(graph_profile_mod, "get_deepcompile_handle", lambda: FakeDeepCompileHandle())
    monkeypatch.setattr(graph_profile_mod, "_get_mem_usage_out_of_torch", lambda: 0)
    monkeypatch.setattr(graph_profile_mod, "is_comm_op", lambda node: False)
    monkeypatch.setattr(graph_profile_mod, "is_release_node", lambda node: False)
    monkeypatch.setattr(graph_profile_mod.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(graph_profile_mod.dist, "get_rank", lambda: 0)

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    profiler = graph_profile_mod.ProfilingInterpreter(gm, iteration=1, warmup=0)
    profiler.run(*materialized)

    assert not graph_profile_mod.is_profile_incomplete(graph)
    assert [call[0] for call in calls] == ["all_gather", "partition"]


def test_launch_compile_passes_clears_owned_compiled_backward_state(monkeypatch):

    class DummyDeepCompileHandle:

        def reset(self):
            pass

    clear_backward_inputs()
    backend_mod.frames_needing_bwd.clear()
    unpatch_compiled_func()
    original_autograd_function = torch.autograd.Function
    owner = object()
    owned_frames = {(owner, 17)}
    backend_mod.frames_needing_bwd.update(owned_frames)
    patch_compiled_func()
    get_backward_inputs().append((torch.ones(1), ))
    monkeypatch.setattr(backend_mod, "log_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_mod, "get_deepcompile_handle", lambda: DummyDeepCompileHandle())

    backend_mod.init_schedule([(0, [])])
    try:
        backend_mod.launch_compile_passes(0, owned_frames=owned_frames)

        assert owned_frames == set()
        assert backend_mod.frames_needing_bwd == set()
        assert get_backward_inputs() == []
        assert torch.autograd.Function is original_autograd_function
    finally:
        backend_mod.frames_needing_bwd.clear()
        unpatch_compiled_func()


def test_unpatch_compiled_func_clears_backward_inputs():
    clear_backward_inputs()
    patch_compiled_func()
    try:
        get_backward_inputs().append((torch.ones(1), ))
        unpatch_compiled_func()
        assert get_backward_inputs() == []
    finally:
        unpatch_compiled_func()


def _patch_aot_constructor():
    return patch_create_aot_dispatcher_function(graph_id=7,
                                                z3_partition=False,
                                                make_fw_graph=lambda gm, sample_inputs: gm.graph,
                                                make_bw_graph=lambda gm, sample_inputs: gm.graph,
                                                real_inputs=(torch.ones(1), ),
                                                param_indices=[],
                                                param_manager={},
                                                frame_id=0,
                                                frames_partitioned=set())


def test_inductor_aot_constructor_patch_is_restorable():
    from torch._dynamo.backends.common import AotAutograd

    original_init = AotAutograd.__init__
    restore = patch_create_aot_dispatcher_function(graph_id=7,
                                                   z3_partition=False,
                                                   make_fw_graph=lambda gm, sample_inputs: gm.graph,
                                                   make_bw_graph=lambda gm, sample_inputs: gm.graph,
                                                   real_inputs=(torch.ones(1), ),
                                                   param_indices=[],
                                                   param_manager={},
                                                   frame_id=0,
                                                   frames_partitioned=set())
    try:
        assert AotAutograd.__init__ is not original_init
    finally:
        restore()

    assert AotAutograd.__init__ is original_init
    assert not hasattr(AotAutograd, "__original_init")


def test_older_aot_restore_does_not_clobber_newer_patch():
    from torch._dynamo.backends.common import AotAutograd

    original_init = AotAutograd.__init__
    restore_first = _patch_aot_constructor()
    restore_second = _patch_aot_constructor()
    newer_init = AotAutograd.__init__
    try:
        restore_first()
        assert AotAutograd.__init__ is newer_init
        assert hasattr(AotAutograd, "__original_init")
    finally:
        restore_second()

    assert AotAutograd.__init__ is original_init
    assert not hasattr(AotAutograd, "__original_init")


def test_run_opt_passes_skips_memory_profile_for_incomplete_graph(monkeypatch):
    gm = torch.fx.symbolic_trace(lambda x: x + 1)
    profiling_results = {7: ProfilingResult()}

    class UnexpectedMemoryProfiler:

        def __init__(self, *args, **kwargs):
            raise AssertionError("memory profiling should be skipped for incomplete operator profiles")

    def incomplete_profile_pass(gm, *args, **kwargs):
        _mark_profile_incomplete(gm.graph)
        return gm

    monkeypatch.setattr(backend_mod, "MemoryProfilingInterpreter", UnexpectedMemoryProfiler)
    monkeypatch.setattr(backend_mod, "log_rank0", lambda *args, **kwargs: None)

    backend_mod.run_opt_passes(opt_passes=[incomplete_profile_pass],
                               gm=gm,
                               graph_id=7,
                               graph_order=[],
                               profiling_results=profiling_results,
                               create_inputs_fn=lambda: (torch.ones(1), ),
                               mem_budget=0.0,
                               param_manager={},
                               bwd=False)

    assert profiling_results[7].fwd_mem == []
    assert profiling_results[7].fwd_mem_complete is False


def test_run_opt_passes_skips_memory_profile_when_another_rank_is_incomplete(monkeypatch):
    gm = torch.fx.symbolic_trace(lambda x: x + 1)
    profiling_results = {7: ProfilingResult()}

    class UnexpectedMemoryProfiler:

        def __init__(self, *args, **kwargs):
            raise AssertionError("all ranks must skip profiling when any rank has an incomplete operator profile")

    def complete_profile_pass(gm, *args, **kwargs):
        return gm

    monkeypatch.setattr(backend_mod, "MemoryProfilingInterpreter", UnexpectedMemoryProfiler)
    monkeypatch.setattr(backend_mod, "_sync_memory_profile_complete", lambda complete, process_group=None: False)
    monkeypatch.setattr(backend_mod, "log_rank0", lambda *args, **kwargs: None)

    backend_mod.run_opt_passes(opt_passes=[complete_profile_pass],
                               gm=gm,
                               graph_id=7,
                               graph_order=[],
                               profiling_results=profiling_results,
                               create_inputs_fn=lambda: (torch.ones(1), ),
                               mem_budget=0.0,
                               param_manager={},
                               bwd=False)

    assert profiling_results[7].fwd_mem == []
    assert profiling_results[7].fwd_mem_complete is False
