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
from deepspeed.compile import executor_arena as executor_arena_mod
from deepspeed.compile import inductor as inductor_mod
from deepspeed.compile import list_schedule as schedule_mod
from deepspeed.compile.passes import prefetch as prefetch_mod
from deepspeed.compile.passes import selective_gather as selective_gather_mod
from deepspeed.compile.passes import zero3_compile as zero3_compile_mod
from deepspeed.compile.profilers import ProfilingResult
from deepspeed.compile.profilers.graph_profile import _backfill_missing_profile_metadata, is_profile_incomplete

_TEST_DC_NAMESPACE = "dc_list_schedule_test"
_DC_LIBRARIES = []


def _define_dc_ops():
    test_dc_ops = getattr(torch.ops, _TEST_DC_NAMESPACE)
    try:
        test_dc_ops.allgather_param.default
        test_dc_ops.prefetch_params_fused.default
        test_dc_ops.wait_allgather.default
        test_dc_ops.release_param.default
        test_dc_ops.reduce_grad.default
        test_dc_ops.reload_parameter.default
        test_dc_ops.offload_tensor.default
        return test_dc_ops
    except AttributeError:
        pass

    lib = torch.library.Library(_TEST_DC_NAMESPACE, "DEF")
    for schema in (
            "allgather_param(Tensor a, int graph_id, int id, ScalarType? dtype = None) -> Tensor",
            "prefetch_params_fused(int graph_id, Tensor[] params, int[] ids, "
            "ScalarType[]? dtypes = None) -> Tensor[]",
            "wait_allgather(Tensor(a) a, int graph_id, int id) -> Tensor(a)",
            "release_param(Tensor(a) a, int graph_id, int id, int n_users) -> Tensor(a)",
            "reduce_grad(Tensor a, int graph_id, int id) -> Tensor",
            "reload_parameter(Tensor a, int graph_id, int id) -> ()",
            "free_tensors(Tensor[] tensors) -> ()",
            "end_backward(Tensor[] tensors, int graph_id, bool release_reduce_buckets = True) -> ()",
            "offload_tensor(Tensor a, int id, int id) -> Tensor",
            "reload_tensor(Tensor a, int id, int id) -> Tensor",
            "wait_offload(Tensor a, int id, int id) -> Tensor",
            "wait_reload(Tensor a, int id, int id) -> Tensor",
    ):
        try:
            lib.define(schema)
        except RuntimeError as exc:
            if "already been registered" not in str(exc):
                raise
    _DC_LIBRARIES.append(lib)
    return test_dc_ops


@pytest.fixture(autouse=True)
def stub_deepcompile_ops(monkeypatch):
    test_dc_ops = _define_dc_ops()
    original_dc_ops = torch.ops.dc
    with monkeypatch.context() as fixture_patch:
        fixture_patch.setattr(torch.ops, "dc", test_dc_ops)
        no_copy_ops = {torch.ops.dc.wait_allgather.default}
        fixture_patch.setattr(compile_util, "get_no_copy_ops", lambda: no_copy_ops)
        yield
    assert torch.ops.dc is original_dc_ops


def _with_meta(node, tensor_size=0, device_time=0):
    node.meta["tensor_size"] = tensor_size
    node.meta["alloc_mem"] = 0
    node.meta["profile_mem_start"] = 0
    node.meta["profile_mem_peak"] = 0
    if device_time is not None:
        node.meta["device_time"] = device_time
    return node


def test_stub_deepcompile_ops_uses_isolated_namespace():
    assert torch.ops.dc is getattr(torch.ops, _TEST_DC_NAMESPACE)


def _placeholder(graph, name):
    return _with_meta(graph.placeholder(name))


def test_end_backward_is_not_added_without_reduce_nodes():
    graph = Graph()
    grad = graph.placeholder("grad")
    graph.output((grad, ))

    before = list(graph.nodes)
    zero3_compile_mod.add_end_backward(graph, 7)

    assert list(graph.nodes) == before


def test_sync_memory_profile_complete_noops_without_distributed(monkeypatch):
    monkeypatch.setattr(backend_mod.dist, "is_initialized", lambda: False)

    def fail_all_reduce(*args, **kwargs):
        raise AssertionError("all_reduce should not run without distributed init")

    monkeypatch.setattr(backend_mod.dist, "all_reduce", fail_all_reduce)

    assert backend_mod._sync_memory_profile_complete(True)
    assert not backend_mod._sync_memory_profile_complete(False)


def test_sync_memory_profile_complete_reduces_asymmetric_failure(monkeypatch):
    monkeypatch.setattr(backend_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend_mod, "get_accelerator", lambda: SimpleNamespace(current_device_name=lambda: "cpu"))

    def mark_any_rank_failed(tensor, op):
        assert op == backend_mod.dist.ReduceOp.MIN
        tensor[0] = 0

    monkeypatch.setattr(backend_mod.dist, "all_reduce", mark_any_rank_failed)

    assert not backend_mod._sync_memory_profile_complete(True)


def test_scheduler_debug_env_flag_values(monkeypatch):
    monkeypatch.delenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, raising=False)
    assert not zero3_compile_mod._scheduler_debug_enabled()

    for value in ("", "0", "false", "FALSE", "no", "No"):
        monkeypatch.setenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, value)
        assert not zero3_compile_mod._scheduler_debug_enabled()

    for value in ("1", "true", "TRUE", "yes", "debug"):
        monkeypatch.setenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, value)
        assert zero3_compile_mod._scheduler_debug_enabled()


def test_unreleased_scheduler_debug_alias_is_not_supported(monkeypatch):
    monkeypatch.delenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, raising=False)
    monkeypatch.setenv("DEEPSPEED_DEEPCOMPILE_SCHEDULER_DEBUG", "1")

    assert not zero3_compile_mod._scheduler_debug_enabled()


def test_zero3_scheduler_budget_uses_rank_reduced_non_gathered_peak(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def current_device_name(self):
            return "cpu"

        def total_memory(self):
            return 2000

    max_reductions = iter((850, 200))

    def reduce_budget_inputs(tensor, op, group=None):
        assert group is None
        if op == zero3_compile_mod.dist.ReduceOp.MIN:
            tensor[0] = 1000
        elif op == zero3_compile_mod.dist.ReduceOp.MAX:
            tensor[0] = next(max_reductions)
        else:
            raise AssertionError(f"unexpected reduce op {op}")

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", reduce_budget_inputs)

    graph = Graph()
    param = _placeholder(graph, "budget_builder_param")
    ag = _allgather(graph, param, 1, "budget_builder", tensor_size=200)
    wait = _wait(graph, ag, 1, "budget_builder")
    op = _neg(graph, wait, "budget_builder_op")
    op.meta.update(max_mem=800, profile_mem_start=250, profile_mem_peak=1050)
    release = _release(graph, op, 1, "budget_builder")
    graph.output((release, ))
    graph.lint()
    for node in graph.nodes:
        node.meta.setdefault("alloc_mem", 0)
        node.meta.setdefault("max_mem", 0)
        node.meta.setdefault("profile_mem_start", 0)
        node.meta.setdefault("profile_mem_peak", 0)

    budget = zero3_compile_mod._build_scheduler_budget_from_operator_profile(graph)

    assert budget.source == "profiled_non_gathered_peak_memory_clamped_to_minimum_gather_residency"
    assert budget.total_mem == 1000
    assert budget.profiled_non_gathered_peak_mem == 850
    assert budget.safety_margin == 100
    assert budget.available_mem == 50
    assert budget.max_gathered_bytes == 200


# The helpers below build nodes through Graph.create_node instead of Graph.call_function
# because call_function only accepts an explicit name= on newer torch releases, while
# create_node has supported it on every version this suite runs against.
def _allgather(graph, arg, ds_id, name, tensor_size=1, device_time=1):
    return _with_meta(
        graph.create_node('call_function',
                          torch.ops.dc.allgather_param.default, (arg, 0, ds_id), {"dtype": torch.float16},
                          name=f"allgather_ds_param_{name}_{ds_id}"),
        tensor_size=tensor_size,
        device_time=device_time,
    )


def _wait(graph, arg, ds_id, name):
    return _with_meta(
        graph.create_node('call_function',
                          torch.ops.dc.wait_allgather.default, (arg, 0, ds_id), {},
                          name=f"wait_allgather_ds_param_{name}_{ds_id}"))


def _neg(graph, arg, name, device_time=0):
    return _with_meta(graph.create_node('call_function', operator.neg, (arg, ), {}, name=name),
                      device_time=device_time)


def _add(graph, lhs, rhs, name, device_time=0):
    return _with_meta(graph.create_node('call_function', operator.add, (lhs, rhs), {}, name=name),
                      device_time=device_time)


def _release(graph, arg, ds_id, name, n_users=1):
    return _with_meta(
        graph.create_node('call_function',
                          torch.ops.dc.release_param.default, (arg, 0, ds_id, n_users), {},
                          name=f"release_ds_param_{name}_{ds_id}"))


def test_fused_prefetch_outputs_replace_ordinary_allgather_edges():
    graph = Graph()
    first = _placeholder(graph, "first")
    second = _placeholder(graph, "second")
    first_ag = _allgather(graph, first, 1, "first", tensor_size=256)
    second_ag = _allgather(graph, second, 2, "second", tensor_size=512)
    first_wait = _wait(graph, first_ag, 1, "first")
    second_wait = _wait(graph, second_ag, 2, "second")
    use = _add(graph, first_wait, second_wait, "use")
    first_release = _release(graph, use, 1, "first")
    second_release = _release(graph, first_release, 2, "second")
    output = graph.output((second_release, ))
    graph.lint()

    ordered_nodes = (first, second, [first_ag, second_ag], first_ag, second_ag, first_wait, second_wait, use,
                     first_release, second_release, output)
    rewritten = prefetch_mod._rewrite_fused_prefetch(ordered_nodes, graph_id=0)
    fused = [node for node in rewritten.nodes if node.target == torch.ops.dc.prefetch_params_fused.default]
    waits = [node for node in rewritten.nodes if node.target == torch.ops.dc.wait_allgather.default]

    assert len(fused) == 1
    assert fused[0].args[2] == [1, 2]
    assert fused[0].args[3] == [torch.float16, torch.float16]
    assert not any(node.target == torch.ops.dc.allgather_param.default for node in rewritten.nodes)
    assert len(waits) == 2
    assert all(wait.args[0].target == operator.getitem for wait in waits)
    assert all(wait.args[0].args[0] is fused[0] for wait in waits)
    assert [wait.args[0].args[1] for wait in waits] == [0, 1]
    assert all(len(wait.args) == 3 for wait in waits)
    arena_plan = executor_arena_mod.plan_graph_executor_arena(rewritten)
    assert [(entry.ds_id, entry.occurrence) for entry in arena_plan.packed.entries] == [(1, 0), (2, 0)]
    assert arena_plan.packed.fallbacks == ()


def test_fused_prefetch_mixed_dtype_group_preserves_independent_allgathers():
    graph = Graph()
    first = _placeholder(graph, "mixed_first")
    second = _placeholder(graph, "mixed_second")
    first_ag = _allgather(graph, first, 1, "mixed_first")
    second_ag = _allgather(graph, second, 2, "mixed_second")
    second_ag.kwargs = {}
    first_wait = _wait(graph, first_ag, 1, "mixed_first")
    second_wait = _wait(graph, second_ag, 2, "mixed_second")
    use = _add(graph, first_wait, second_wait, "mixed_use")
    release = _release(graph, use, 1, "mixed")
    output = graph.output((release, ))
    graph.lint()

    rewritten = prefetch_mod._rewrite_fused_prefetch(
        (first, second, [first_ag, second_ag], first_ag, second_ag, first_wait, second_wait, use, release, output),
        graph_id=0)

    assert not any(node.target == torch.ops.dc.prefetch_params_fused.default for node in rewritten.nodes)
    assert len([node for node in rewritten.nodes if node.target == torch.ops.dc.allgather_param.default]) == 2


def test_fused_prefetch_duplicate_ds_id_preserves_single_lease_semantics():
    graph = Graph()
    param = _placeholder(graph, "duplicate_param")
    first_ag = _allgather(graph, param, 3, "duplicate_first")
    second_ag = _allgather(graph, param, 3, "duplicate_second")
    first_wait = _wait(graph, first_ag, 3, "duplicate_first")
    second_wait = _wait(graph, second_ag, 3, "duplicate_second")
    use = _add(graph, first_wait, second_wait, "duplicate_use")
    release = _release(graph, use, 3, "duplicate")
    output = graph.output((release, ))
    graph.lint()

    rewritten = prefetch_mod._rewrite_fused_prefetch(
        (param, [first_ag, second_ag], first_ag, second_ag, first_wait, second_wait, use, release, output), graph_id=0)

    assert not any(node.target == torch.ops.dc.prefetch_params_fused.default for node in rewritten.nodes)
    assert len([node for node in rewritten.nodes if node.target == torch.ops.dc.allgather_param.default]) == 2


def test_graph_executor_arena_excludes_wait_tensor_that_escapes_as_output():
    graph = Graph()
    param = _placeholder(graph, "escaping_param")
    ag = _allgather(graph, param, 11, "escaping", tensor_size=256)
    wait = _wait(graph, ag, 11, "escaping")
    _release(graph, wait, 11, "escaping")
    graph.output((wait, ))
    graph.lint()

    plan = executor_arena_mod.plan_graph_executor_arena(graph)

    assert plan.packed.entries == ()
    assert len(plan.packed.fallbacks) == 1
    assert plan.packed.fallbacks[0].fallback_reason == "graph_output_escape"


def test_graph_executor_arena_lifetime_ends_at_final_multi_user_release():
    graph = Graph()
    param = _placeholder(graph, "multi_user_param")
    ag = _allgather(graph, param, 12, "multi_user", tensor_size=256)
    wait = _wait(graph, ag, 12, "multi_user")
    first_use = _neg(graph, wait, "multi_user_first")
    second_use = _neg(graph, wait, "multi_user_second")
    first_release = _release(graph, first_use, 12, "multi_user_first", n_users=2)
    second_release = _release(graph, second_use, 12, "multi_user_second", n_users=2)
    graph.output((first_release, second_release))
    graph.lint()

    plan = executor_arena_mod.plan_graph_executor_arena(graph)
    positions = {node: index for index, node in enumerate(graph.nodes)}

    assert len(plan.packed.entries) == 1
    assert plan.packed.entries[0].release == positions[second_release]


def test_graph_executor_arena_reuses_slice_for_repeated_ds_id_after_release():
    graph = Graph()
    param = _placeholder(graph, "repeated_param")
    first_ag = _allgather(graph, param, 14, "repeated_first", tensor_size=256)
    first_wait = _wait(graph, first_ag, 14, "repeated_first")
    first_use = _neg(graph, first_wait, "repeated_first_use")
    first_release = _release(graph, first_use, 14, "repeated_first")
    second_ag = _allgather(graph, param, 14, "repeated_second", tensor_size=256)
    second_wait = _wait(graph, second_ag, 14, "repeated_second")
    second_use = _add(graph, first_release, second_wait, "repeated_second_use")
    second_release = _release(graph, second_use, 14, "repeated_second")
    graph.output((second_release, ))
    graph.lint()

    plan = executor_arena_mod.plan_graph_executor_arena(graph)

    assert [(entry.ds_id, entry.occurrence, entry.offset) for entry in plan.packed.entries] == [(14, 0, 0), (14, 1, 0)]


def test_graph_executor_arena_excludes_alias_view_saved_as_output():
    graph = Graph()
    param = _placeholder(graph, "alias_escape_param")
    ag = _allgather(graph, param, 13, "alias_escape", tensor_size=256)
    wait = _wait(graph, ag, 13, "alias_escape")
    view = graph.call_method("view", args=(wait, -1))
    _release(graph, view, 13, "alias_escape")
    graph.output((view, ))
    graph.lint()

    plan = executor_arena_mod.plan_graph_executor_arena(graph)

    assert plan.packed.entries == ()
    assert plan.packed.fallbacks[0].fallback_reason == "graph_output_alias_escape"


def test_graph_executor_arena_excludes_getitem_alias_saved_as_output():
    graph = Graph()
    param = _placeholder(graph, "getitem_escape_param")
    ag = _allgather(graph, param, 15, "getitem_escape", tensor_size=256)
    wait = _wait(graph, ag, 15, "getitem_escape")
    view = graph.call_function(operator.getitem, args=(wait, slice(None)))
    _release(graph, view, 15, "getitem_escape")
    graph.output((view, ))
    graph.lint()

    plan = executor_arena_mod.plan_graph_executor_arena(graph)

    assert plan.packed.entries == ()
    assert plan.packed.fallbacks[0].fallback_reason == "graph_output_alias_escape"


def test_final_arena_planning_excludes_frozen_persistent_param():
    graph = Graph()
    param = graph.placeholder("frozen_param")
    param.meta["val"] = torch.empty(4, dtype=torch.float16)
    use = graph.call_function(operator.neg, (param, ))
    graph.output((use, ))

    registered_param = SimpleNamespace(numel=4, dtype=torch.float16, param=SimpleNamespace(ds_persist=True))
    param_manager = SimpleNamespace(params={"frozen_param": registered_param}, ds_ids={"frozen_param": 77})

    rewritten = zero3_compile_mod.add_gather_and_release(0, graph, param_manager, [param])
    plan = executor_arena_mod.plan_graph_executor_arena(rewritten)

    assert plan.packed.entries == ()
    assert len(plan.packed.fallbacks) == 1
    assert plan.packed.fallbacks[0].ds_id == 77
    assert plan.packed.fallbacks[0].fallback_reason == "persistent_param"


def test_prefetch_size_bounds_exclude_oversized_single_gather(monkeypatch):
    monkeypatch.setattr(prefetch_mod, "MAX_BUFFERED_SIZE", 1024)
    monkeypatch.setattr(prefetch_mod, "MAX_FUSE_SIZE", 512)

    assert prefetch_mod._prefetch_size_admissible(512)
    assert not prefetch_mod._prefetch_size_admissible(513)
    assert not prefetch_mod._prefetch_size_admissible(0)


def _scheduled_graph(graph, scheduler_budget=None):
    return schedule_mod.fast_free_schedule(graph, 0, 0, debug_log=True, scheduler_budget=scheduler_budget)


def _scheduled_names(graph, scheduler_budget=None):
    return [node.name for node in _scheduled_graph(graph, scheduler_budget=scheduler_budget).nodes]


def _scheduler_diagnostics(graph):
    return getattr(graph, schedule_mod.DS_SCHEDULER_BUDGET_DIAGNOSTICS_ATTR)


@pytest.mark.parametrize("case", ["only_consumer", "later_real_use"])
def test_get_last_uses_unconsumed_no_copy_wait(case):
    graph = Graph()
    param = _placeholder(graph, f"{case}_param")
    allgather = _allgather(graph, param, 1, case)
    wait = _wait(graph, allgather, 1, case)

    later_use = _neg(graph, allgather, f"{case}_later_use") if case == "later_real_use" else None
    graph.output((later_use, ) if later_use is not None else ())
    graph.lint()

    node_to_last_use, user_to_last_uses = compile_util.get_last_uses(graph)
    node_to_uses = compile_util.get_real_uses(graph)

    expected_last_use = later_use if later_use is not None else wait
    assert node_to_last_use[allgather] is expected_last_use
    assert user_to_last_uses[expected_last_use] == [allgather]
    assert user_to_last_uses.get(wait, []) == ([] if later_use is not None else [allgather])
    assert node_to_uses[allgather] == ([] if later_use is None else [later_use])


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


def test_fast_free_schedule_counts_live_gathered_bytes_when_filtering_candidates():
    graph = Graph()

    first_param = _placeholder(graph, "budget_first_param")
    high_param = _placeholder(graph, "budget_high_param")
    low_param = _placeholder(graph, "budget_low_param")

    first_ag = _allgather(graph, first_param, 80, "budget_first", tensor_size=70)
    first_wait = _wait(graph, first_ag, 80, "budget_first")

    high_dep = _add(graph, high_param, first_wait, "budget_high_dep")
    high_ag = _allgather(graph, high_dep, 81, "budget_high", tensor_size=40)
    high_wait = _wait(graph, high_ag, 81, "budget_high")
    high_use = _neg(graph, high_wait, "budget_high_use", device_time=1)
    high_release = _release(graph, high_use, 81, "budget_high")

    low_dep = _add(graph, low_param, first_wait, "budget_low_dep")
    low_ag = _allgather(graph, low_dep, 82, "budget_low", tensor_size=20)
    low_wait = _wait(graph, low_ag, 82, "budget_low")
    low_use = _neg(graph, low_wait, "budget_low_use", device_time=100)
    low_release = _release(graph, low_use, 82, "budget_low")

    high_low_pair = _add(graph, high_wait, low_wait, "budget_high_low_pair")
    first_use = _add(graph, first_wait, high_low_pair, "budget_first_use")
    first_release = _release(graph, first_use, 80, "budget_first")

    graph.output((first_release, high_release, low_release))
    graph.lint()

    no_budget_names = _scheduled_names(graph)
    assert no_budget_names.index(high_ag.name) < no_budget_names.index(low_ag.name)

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=100, source="test")
    scheduled_graph = _scheduled_graph(graph, scheduler_budget=budget)
    names = [node.name for node in scheduled_graph.nodes]
    diagnostics = _scheduler_diagnostics(scheduled_graph)

    assert names.index(first_ag.name) < names.index(low_ag.name)
    assert names.index(low_ag.name) < names.index(high_ag.name)
    assert diagnostics["budget_rejections"] > 0
    assert any(record["node"] == high_ag.name for record in diagnostics["budget_rejected_candidates"])


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

        def current_device_name(self):
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


def test_schedule_prefetch_rejected_fused_admission_restores_enabled_backward_demand_arena(monkeypatch):
    graph = Graph()
    param = _placeholder(graph, "admission_param")
    output = graph.output((param, ))
    _with_meta(output)
    graph.lint()

    profile_records = [(node.name, 0, 0, 0) for node in graph.nodes]
    time_records = [(node.name, 0, 0) for node in graph.nodes]
    size_records = [(node.name, 0) for node in graph.nodes]
    profiling_results = {
        0:
        ProfilingResult(bwd_graph=graph,
                        bwd_mem=profile_records,
                        bwd_time=time_records,
                        bwd_tensor_sizes=size_records,
                        bwd_mem_complete=True)
    }
    gm = GraphModule(torch.nn.Module(), graph)

    demand_occurrences = (executor_arena_mod.ArenaOccurrence(ds_id=101,
                                                             occurrence=0,
                                                             first_use=0,
                                                             release=1,
                                                             nbytes=256,
                                                             dtype=torch.float16), )
    final_occurrences = (executor_arena_mod.ArenaOccurrence(ds_id=101,
                                                            occurrence=0,
                                                            first_use=0,
                                                            release=1,
                                                            nbytes=512,
                                                            dtype=torch.float16), )
    plans = iter(
        (
            executor_arena_mod.GraphArenaPlan(demand_occurrences,
                                              executor_arena_mod.pack_executor_arena(demand_occurrences)),
            executor_arena_mod.GraphArenaPlan(final_occurrences,
                                              executor_arena_mod.pack_executor_arena(final_occurrences)),
        ))

    class FakeAccelerator:

        def current_device_name(self):
            return "cpu"

        def total_memory(self):
            return 1024

        def available_memory(self):
            return 1024

        def memory_allocated(self):
            return 0

        def max_memory_allocated(self):
            return 0

    class FakeNative:

        def configure_z3_gather_arena(self, *args):
            self.arena_config = args

    native = FakeNative()
    admissions = []

    def record_admission(plan, demand_profile_bytes, live_budget):
        admission = executor_arena_mod.admit_executor_arena(plan, demand_profile_bytes, live_budget)
        admissions.append(admission)
        return admission

    monkeypatch.setattr(prefetch_mod, "MAX_BUFFERED_SIZE", 255)
    monkeypatch.setattr(prefetch_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(prefetch_mod, "print_rank_0", lambda _message: None)
    monkeypatch.setattr(prefetch_mod, "create_predictor", lambda: lambda _size: 0)
    monkeypatch.setattr(prefetch_mod, "is_profile_incomplete", lambda _graph: False)
    monkeypatch.setattr(prefetch_mod, "plan_graph_executor_arena", lambda _graph: next(plans))
    monkeypatch.setattr(prefetch_mod, "admit_executor_arena", record_admission)
    monkeypatch.setattr(prefetch_mod, "get_deepcompile_handle", lambda: native)
    monkeypatch.setattr(prefetch_mod.dist, "all_reduce", lambda tensor, op, group=None: tensor)

    returned = prefetch_mod.schedule_prefetch(gm,
                                              graph_id=0,
                                              graph_order=[(0, True)],
                                              profiling_results=profiling_results,
                                              create_inputs_fn=lambda: (),
                                              mem_budget=0,
                                              param_manager={},
                                              bwd=True)

    assert returned is gm
    assert gm.graph is graph
    assert [(admission.accepted, admission.capacity, admission.demand_profile_bytes, admission.incremental_bytes)
            for admission in admissions] == [(False, 512, 256, 256), (True, 256, 256, 0)]
    assert gm._deepcompile_executor_arena_plan.packed.capacity == 256
    assert gm._deepcompile_executor_arena_admission.accepted
    assert gm._deepcompile_executor_arena_admission.incremental_bytes == 0
    assert gm._deepcompile_executor_arena_registration.enabled
    assert gm._deepcompile_executor_arena_registration.reason == "accepted"
    assert native.arena_config[0:5] == (0, True, 256, 256, [101])

    admissions.clear()
    plans = iter(
        (
            executor_arena_mod.GraphArenaPlan(demand_occurrences,
                                              executor_arena_mod.pack_executor_arena(demand_occurrences)),
            executor_arena_mod.GraphArenaPlan(final_occurrences,
                                              executor_arena_mod.pack_executor_arena(final_occurrences)),
        ))
    gm = GraphModule(torch.nn.Module(), graph)
    monkeypatch.setattr(prefetch_mod, "MAX_BUFFERED_SIZE", 256)

    prefetch_mod.schedule_prefetch(gm,
                                   graph_id=0,
                                   graph_order=[(0, True)],
                                   profiling_results=profiling_results,
                                   create_inputs_fn=lambda: (),
                                   mem_budget=0,
                                   param_manager={},
                                   bwd=True)

    assert gm.graph is not graph
    assert len(admissions) == 1
    assert gm._deepcompile_executor_arena_plan.packed.capacity == 512
    assert gm._deepcompile_executor_arena_admission.accepted
    assert gm._deepcompile_executor_arena_registration.enabled
    assert native.arena_config[0:4] == (0, True, 512, 256)


def test_graphsafe_rng_state_outputs_are_registered_no_reuse():
    graphsafe_run_with_rng_state = inductor_mod._get_graphsafe_run_with_rng_state()
    if graphsafe_run_with_rng_state is None:
        pytest.skip("graphsafe_run_with_rng_state is unavailable in this torch build")

    calls = []

    def fake_register(op_overload, **kwargs):
        calls.append((op_overload, kwargs))

    assert inductor_mod._register_graphsafe_rng_state_no_reuse(fake_register)
    assert calls == [(graphsafe_run_with_rng_state, {"never_reuse_output": True})]


def test_mark_output_never_reuse_mixed_pytree():

    class FakeIRNode(inductor_mod.IRNode):

        def get_name(self):
            return "tensor_buffer"

    class NonIRLeaf:

        def get_name(self):
            raise AssertionError("get_name must not be called for non-IR outputs")

    graph = SimpleNamespace(never_reuse_buffers=set())
    outputs = (FakeIRNode(), 1, None, NonIRLeaf())

    with inductor_mod.V.set_graph_handler(graph):
        wrapped = torch.utils._pytree.tree_map(lambda out: inductor_mod._mark_output_never_reuse(out, enabled=True),
                                               outputs)

    assert wrapped == outputs
    assert graph.never_reuse_buffers == {"tensor_buffer"}


def test_register_custom_ops_defines_every_op_it_registers(monkeypatch):
    """register_custom_ops must not raise when only _define_dc_ops has run.

    It registers each dc op in sequence, so an op the helper does not define raises AttributeError
    partway through and everything after it -- including the graphsafe rng lowering -- is silently
    never registered. That only shows up where the C++ extension is absent, which is CI.
    """
    _define_dc_ops()
    registered_ops = []

    monkeypatch.setattr(inductor_mod, "add_needs_realized_inputs", lambda _op: None)
    monkeypatch.setattr(inductor_mod, "register_lowering",
                        lambda op_overload, **_kwargs: lambda handler: registered_ops.append(op_overload) or handler)
    monkeypatch.setattr(inductor_mod, "fallbacks", set())
    monkeypatch.setattr(inductor_mod.Scheduler, "is_dc_patched", True, raising=False)

    inductor_mod.register_custom_ops()

    # Every dc op named in register_custom_ops should have reached register_lowering.
    names = {getattr(op, "__name__", str(op)) for op in registered_ops}
    for expected in ("offload_tensor", "wait_offload", "reload_tensor", "wait_reload"):
        assert any(expected in str(op) for op in registered_ops), \
            f"{expected} was never registered; _define_dc_ops is missing its schema. Registered: {sorted(names)}"


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
