# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
from types import SimpleNamespace

import pytest
import torch
from torch.fx import Graph, GraphModule

import deepspeed.compile.util as compile_util
from deepspeed.compile import backend as backend_mod
from deepspeed.compile import inductor as inductor_mod
from deepspeed.compile import list_schedule as schedule_mod
from deepspeed.compile.passes import prefetch as prefetch_mod
from deepspeed.compile.passes import selective_gather as selective_gather_mod
from deepspeed.compile.profilers import ProfilingResult
from deepspeed.compile.profilers.graph_profile import _backfill_missing_profile_metadata, is_profile_incomplete

_DC_LIBRARIES = []


def _define_dc_ops():
    try:
        torch.ops.dc.allgather_param.default
        torch.ops.dc.wait_allgather.default
        torch.ops.dc.release_param.default
        torch.ops.dc.reduce_grad.default
        return
    except AttributeError:
        pass

    lib = torch.library.Library("dc", "DEF")
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


@pytest.fixture(autouse=True)
def stub_deepcompile_ops(monkeypatch):
    _define_dc_ops()
    no_copy_ops = {torch.ops.dc.wait_allgather.default}
    monkeypatch.setattr(compile_util, "get_no_copy_ops", lambda: no_copy_ops)


def _with_meta(node, tensor_size=0, device_time=0):
    node.meta["tensor_size"] = tensor_size
    if device_time is not None:
        node.meta["device_time"] = device_time
    return node


def _placeholder(graph, name):
    return _with_meta(graph.placeholder(name))


def test_sync_memory_profile_complete_noops_without_distributed(monkeypatch):
    monkeypatch.setattr(backend_mod.dist, "is_initialized", lambda: False)

    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("all_reduce should not run without distributed init")

    monkeypatch.setattr(backend_mod.dist, "all_reduce", fail_all_reduce)

    assert backend_mod._sync_memory_profile_complete(True)
    assert not backend_mod._sync_memory_profile_complete(False)


def test_sync_memory_profile_complete_reduces_asymmetric_failure(monkeypatch):
    monkeypatch.setattr(backend_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend_mod, "get_accelerator", lambda: SimpleNamespace(current_device=lambda: "cpu"))

    def mark_any_rank_failed(tensor, op):
        assert op == backend_mod.dist.ReduceOp.MIN
        tensor[0] = 0

    monkeypatch.setattr(backend_mod.dist, "all_reduce", mark_any_rank_failed)

    assert not backend_mod._sync_memory_profile_complete(True)


def _allgather(graph, arg, ds_id, name, tensor_size=1, device_time=1):
    return _with_meta(
        graph.call_function(torch.ops.dc.allgather_param.default, (arg, 0, ds_id), {"dtype": torch.float16},
                            name=f"allgather_ds_param_{name}_{ds_id}"),
        tensor_size=tensor_size,
        device_time=device_time,
    )


def _wait(graph, arg, ds_id, name):
    return _with_meta(
        graph.call_function(torch.ops.dc.wait_allgather.default, (arg, 0, ds_id),
                            name=f"wait_allgather_ds_param_{name}_{ds_id}"))


def _neg(graph, arg, name, device_time=0):
    return _with_meta(graph.call_function(operator.neg, (arg, ), name=name), device_time=device_time)


def _add(graph, lhs, rhs, name, device_time=0):
    return _with_meta(graph.call_function(operator.add, (lhs, rhs), name=name), device_time=device_time)


def _release(graph, arg, ds_id, name):
    return _with_meta(
        graph.call_function(torch.ops.dc.release_param.default, (arg, 0, ds_id, 1),
                            name=f"release_ds_param_{name}_{ds_id}"))


def _scheduled_names(graph):
    return [node.name for node in schedule_mod.fast_free_schedule(graph, 0, 0, debug_log=True).nodes]


def test_fast_free_schedule_keeps_zero_free_acc_filter():
    graph = Graph()

    safe_param = _placeholder(graph, "safe_param")
    safe_pre_param = _placeholder(graph, "safe_pre_param")
    unsafe_param = _placeholder(graph, "unsafe_param")
    unsafe_extra_param = _placeholder(graph, "unsafe_extra_param")

    safe_pre_ag = _allgather(graph, safe_pre_param, 10, "safe_pre")
    safe_pre_wait = _wait(graph, safe_pre_ag, 10, "safe_pre")
    safe_pre_use = _neg(graph, safe_pre_wait, "safe_pre_use")
    safe_ag = _allgather(graph, _add(graph, safe_param, safe_pre_use, "safe_param_dep"), 11, "safe")
    safe_wait = _wait(graph, safe_ag, 11, "safe")
    safe_use = _neg(graph, safe_wait, "safe_use", device_time=100)
    safe_release = _release(graph, safe_use, 11, "safe")

    unsafe_ag = _allgather(graph, unsafe_param, 20, "unsafe")
    unsafe_wait = _wait(graph, unsafe_ag, 20, "unsafe")
    unsafe_extra_ag = _allgather(graph, unsafe_extra_param, 21, "unsafe_extra")
    unsafe_extra_wait = _wait(graph, unsafe_extra_ag, 21, "unsafe_extra")
    unsafe_use = _add(graph, unsafe_wait, unsafe_extra_wait, "unsafe_use", device_time=1)
    unsafe_release = _release(graph, unsafe_use, 20, "unsafe")

    graph.output((safe_release, unsafe_release))
    graph.lint()

    names = _scheduled_names(graph)

    assert names.index(safe_release.name) < names.index(unsafe_ag.name)
    assert names.index(safe_release.name) < names.index(unsafe_extra_ag.name)


def test_fast_free_schedule_prefers_lower_allgather_pressure_in_zero_free_acc_bucket():
    graph = Graph()

    high_param = _placeholder(graph, "high_param")
    high_pre_param = _placeholder(graph, "high_pre_param")
    low_param = _placeholder(graph, "low_param")
    low_pre_param = _placeholder(graph, "low_pre_param")

    high_pre_ag = _allgather(graph, high_pre_param, 30, "high_pre", tensor_size=100)
    high_pre_wait = _wait(graph, high_pre_ag, 30, "high_pre")
    high_ag = _allgather(graph, _add(graph, high_param, high_pre_wait, "high_param_dep"), 31, "high")
    high_wait = _wait(graph, high_ag, 31, "high")
    high_use = _neg(graph, high_wait, "high_use", device_time=1)
    high_release = _release(graph, high_use, 31, "high")

    low_pre_ag = _allgather(graph, low_pre_param, 40, "low_pre", tensor_size=1)
    low_pre_wait = _wait(graph, low_pre_ag, 40, "low_pre")
    low_ag = _allgather(graph, _add(graph, low_param, low_pre_wait, "low_param_dep"), 41, "low")
    low_wait = _wait(graph, low_ag, 41, "low")
    low_use = _neg(graph, low_wait, "low_use", device_time=100)
    low_release = _release(graph, low_use, 41, "low")

    graph.output((high_release, low_release))
    graph.lint()

    names = _scheduled_names(graph)

    assert names.index(low_release.name) < names.index(high_ag.name)


def test_fast_free_schedule_uses_pressure_tiebreaker_in_fallback_bucket():
    graph = Graph()

    high_param = _placeholder(graph, "fallback_high_param")
    high_extra_param = _placeholder(graph, "fallback_high_extra_param")
    low_param = _placeholder(graph, "fallback_low_param")
    low_extra_param = _placeholder(graph, "fallback_low_extra_param")

    high_ag = _allgather(graph, high_param, 50, "fallback_high", tensor_size=100)
    high_wait = _wait(graph, high_ag, 50, "fallback_high")
    high_extra_ag = _allgather(graph, high_extra_param, 51, "fallback_high_extra", tensor_size=10)
    high_extra_wait = _wait(graph, high_extra_ag, 51, "fallback_high_extra")
    high_use = _add(graph, high_wait, high_extra_wait, "fallback_high_use", device_time=1)
    high_release = _release(graph, high_use, 50, "fallback_high")

    low_ag = _allgather(graph, low_param, 60, "fallback_low", tensor_size=1)
    low_wait = _wait(graph, low_ag, 60, "fallback_low")
    low_extra_ag = _allgather(graph, low_extra_param, 61, "fallback_low_extra", tensor_size=10)
    low_extra_wait = _wait(graph, low_extra_ag, 61, "fallback_low_extra")
    low_use = _add(graph, low_wait, low_extra_wait, "fallback_low_use", device_time=100)
    low_release = _release(graph, low_use, 60, "fallback_low")

    graph.output((high_release, low_release))
    graph.lint()

    names = _scheduled_names(graph)

    assert names.index(low_ag.name) < names.index(high_ag.name)


def test_fast_free_schedule_keeps_single_allgather_release_order():
    graph = Graph()

    param = _placeholder(graph, "param")
    ag = _allgather(graph, param, 70, "single")
    wait = _wait(graph, ag, 70, "single")
    use = _neg(graph, wait, "single_use")
    release = _release(graph, use, 70, "single")

    graph.output((release, ))
    graph.lint()

    names = _scheduled_names(graph)

    assert names.index(ag.name) < names.index(wait.name)
    assert names.index(wait.name) < names.index(use.name)
    assert names.index(use.name) < names.index(release.name)


def test_profile_backfill_makes_partial_profile_safe_for_profile_dependent_passes(monkeypatch):
    graph = Graph()

    param = _placeholder(graph, "partial_profile_param")
    ag = _allgather(graph, param, 90, "partial_profile", device_time=None)
    wait = _wait(graph, ag, 90, "partial_profile")
    use = _neg(graph, wait, "partial_profile_use", device_time=None)
    release = _release(graph, use, 90, "partial_profile")

    ag.meta.pop("tensor_size", None)
    for node in (ag, use):
        node.meta.pop("wall_time", None)
        node.meta.pop("alloc_mem", None)
        node.meta.pop("max_mem", None)

    graph.output((release, ))
    graph.lint()

    _backfill_missing_profile_metadata(graph)
    assert is_profile_incomplete(graph)

    for node in graph.nodes:
        if node in (ag, use):
            assert node.meta["device_time"] == 0.0
        else:
            assert "device_time" in node.meta
        assert "wall_time" in node.meta
        assert "tensor_size" in node.meta
        assert "alloc_mem" in node.meta
        assert "max_mem" in node.meta
    assert ag.meta["tensor_size"] == 0

    names = _scheduled_names(graph)
    assert names.index(ag.name) < names.index(wait.name)
    assert names.index(wait.name) < names.index(use.name)
    assert names.index(use.name) < names.index(release.name)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1024

        def available_memory(self):
            return 1024

    fake_ds_param = SimpleNamespace(numel=7,
                                    dtype=torch.float16,
                                    param=SimpleNamespace(ds_persist=False, ds_shape=(1, )))
    fake_param_manager = {
        0: SimpleNamespace(params={"partial_profile_param": fake_ds_param}, ds_ids={"partial_profile_param": 90})
    }
    profiling_results = {
        0: ProfilingResult(fwd_graph=graph, bwd_graph=None, fwd_mem=[("profiled_before_abort", 0, 0, 0)])
    }
    gm = GraphModule(torch.nn.Module(), graph)
    logs = []
    prefetch_logs = []
    persisted = []

    monkeypatch.setattr(prefetch_mod, "print_rank_0", lambda message: prefetch_logs.append(message))
    assert prefetch_mod.schedule_prefetch(gm,
                                          graph_id=0,
                                          graph_order=[(0, True)],
                                          profiling_results=profiling_results,
                                          create_inputs_fn=lambda: (),
                                          mem_budget=0,
                                          param_manager=fake_param_manager,
                                          bwd=False) is gm
    assert any("incomplete profiling data" in message for message in prefetch_logs)

    monkeypatch.setattr(selective_gather_mod, "print_rank_0", lambda message: logs.append(message))
    monkeypatch.setattr(selective_gather_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(selective_gather_mod, "get_deepcompile_handle",
                        lambda: SimpleNamespace(set_persistent=persisted.append))
    monkeypatch.setattr(selective_gather_mod.dist, "all_reduce", lambda *args, **kwargs: None)

    selective_gather_mod.selective_gather(gm,
                                          graph_id=0,
                                          graph_order=[(0, True)],
                                          profiling_results=profiling_results,
                                          create_inputs_fn=lambda: (),
                                          mem_budget=0,
                                          param_manager=fake_param_manager,
                                          bwd=True)
    assert persisted == []
    assert any("incomplete profiling data" in message for message in logs)


def test_schedule_prefetch_skips_when_memory_profile_incomplete(monkeypatch):
    graph = Graph()

    param = _placeholder(graph, "mem_incomplete_param")
    ag = _allgather(graph, param, 91, "mem_incomplete")
    wait = _wait(graph, ag, 91, "mem_incomplete")
    use = _neg(graph, wait, "mem_incomplete_use")
    release = _release(graph, use, 91, "mem_incomplete")

    graph.output((release, ))
    graph.lint()

    profiling_results = {
        0:
        ProfilingResult(fwd_graph=graph,
                        bwd_graph=None,
                        fwd_mem=[("profiled_before_abort", 0, 0, 0)],
                        fwd_mem_complete=False)
    }
    gm = GraphModule(torch.nn.Module(), graph)
    logs = []

    monkeypatch.setattr(prefetch_mod, "print_rank_0", lambda message: logs.append(message))

    assert prefetch_mod.schedule_prefetch(gm,
                                          graph_id=0,
                                          graph_order=[(0, False)],
                                          profiling_results=profiling_results,
                                          create_inputs_fn=lambda: (),
                                          mem_budget=0,
                                          param_manager={},
                                          bwd=False) is gm
    assert gm.graph is graph
    assert any("incomplete profiling data" in message for message in logs)


def test_graphsafe_rng_state_outputs_are_registered_no_reuse():
    graphsafe_run_with_rng_state = inductor_mod._get_graphsafe_run_with_rng_state()
    if graphsafe_run_with_rng_state is None:
        pytest.skip("graphsafe_run_with_rng_state is unavailable in this torch build")

    calls = []

    def fake_register(op_overload, **kwargs):
        calls.append((op_overload, kwargs))

    assert inductor_mod._register_graphsafe_rng_state_no_reuse(fake_register)
    assert calls == [(graphsafe_run_with_rng_state, {"never_reuse_output": True})]


def test_register_custom_ops_includes_graphsafe_rng_state_no_reuse(monkeypatch):
    graphsafe_run_with_rng_state = inductor_mod._get_graphsafe_run_with_rng_state()
    if graphsafe_run_with_rng_state is None:
        pytest.skip("graphsafe_run_with_rng_state is unavailable in this torch build")

    _define_dc_ops()
    registered_ops = []

    def fake_add_needs_realized_inputs(_op_overload):
        return None

    def fake_register_lowering(op_overload, **_kwargs):

        def record_handler(handler):
            registered_ops.append(op_overload)
            return handler

        return record_handler

    monkeypatch.setattr(inductor_mod, "add_needs_realized_inputs", fake_add_needs_realized_inputs)
    monkeypatch.setattr(inductor_mod, "register_lowering", fake_register_lowering)
    monkeypatch.setattr(inductor_mod, "fallbacks", set())
    monkeypatch.setattr(inductor_mod.Scheduler, "is_dc_patched", True, raising=False)

    inductor_mod.register_custom_ops()

    assert graphsafe_run_with_rng_state in registered_ops
