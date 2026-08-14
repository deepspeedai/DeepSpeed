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
from deepspeed.compile.passes import zero3_compile as zero3_compile_mod
from deepspeed.compile.profilers import ProfilingResult
from deepspeed.compile.profilers.graph_profile import _backfill_missing_profile_metadata, is_profile_incomplete

_DC_LIBRARIES = []


def _define_dc_ops():
    try:
        torch.ops.dc.allgather_param.default
        torch.ops.dc.prefetch_params_fused.default
        torch.ops.dc.wait_allgather.default
        torch.ops.dc.release_param.default
        torch.ops.dc.reduce_grad.default
        torch.ops.dc.reload_parameter.default
        return
    except AttributeError:
        pass

    lib = torch.library.Library("dc", "DEF")
    for schema in (
            "allgather_param(Tensor a, int graph_id, int id, ScalarType? dtype = None) -> Tensor",
            "prefetch_params_fused(int graph_id, Tensor[] params, int[] ids, ScalarType[]? dtypes = None, int arena_plan_id = -1) -> ()",
            "wait_allgather(Tensor(a) a, int graph_id, int id) -> Tensor(a)",
            "release_param(Tensor(a) a, int graph_id, int id, int n_users) -> Tensor(a)",
            "reduce_grad(Tensor a, int graph_id, int id) -> Tensor",
            "reload_parameter(Tensor a, int graph_id, int id) -> ()",
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
    node.meta["alloc_mem"] = 0
    node.meta["profile_mem_start"] = 0
    node.meta["profile_mem_peak"] = 0
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


def _release(graph, arg, ds_id, name):
    return _with_meta(
        graph.create_node('call_function',
                          torch.ops.dc.release_param.default, (arg, 0, ds_id, 1), {},
                          name=f"release_ds_param_{name}_{ds_id}"))


def _arena_prefetch(graph, params, ds_ids, plan_id, sizes):
    node = graph.create_node('call_function',
                             torch.ops.dc.prefetch_params_fused.default, (0, params, ds_ids, None, plan_id), {},
                             name=f"prefetch_arena_{plan_id}")
    node.meta["prefetch_arena_eligible_ds_ids"] = tuple(dict.fromkeys(ds_ids))
    node.meta["prefetch_arena_bytes_by_ds_id"] = tuple(zip(ds_ids, sizes))
    return node


def _fake_arena_param_manager(params):
    return SimpleNamespace(
        params={
            name: SimpleNamespace(shape=torch.Size(shape), dtype=dtype, param=SimpleNamespace(ds_persist=False))
            for name, _, shape, dtype in params
        },
        ds_ids={
            name: ds_id
            for name, ds_id, _, _ in params
        },
    )


def test_prefetch_arena_interval_packing_alignment_and_reuse():
    packed, reason = prefetch_mod._pack_prefetch_intervals([
        {
            "ds_id": 1,
            "start": 0,
            "end": 2,
            "bytes": 257,
            "requests": [(10, 1)]
        },
        {
            "ds_id": 2,
            "start": 1,
            "end": 3,
            "bytes": 255,
            "requests": [(11, 2)]
        },
        {
            "ds_id": 3,
            "start": 4,
            "end": 5,
            "bytes": 256,
            "requests": [(12, 3)]
        },
    ])

    assert reason is None
    by_ds_id = {interval["ds_id"]: interval for interval in packed["intervals"]}
    assert by_ds_id[1]["size"] == 512
    assert by_ds_id[2]["size"] == 256
    assert by_ds_id[1]["offset"] != by_ds_id[2]["offset"]
    assert by_ds_id[3]["offset"] == by_ds_id[1]["offset"]
    assert packed["max_live_bytes"] == 768
    assert packed["capacity"] == 768


def test_prefetch_arena_interval_packing_uses_best_fit_free_block():
    packed, reason = prefetch_mod._pack_prefetch_intervals([
        {
            "ds_id": 1,
            "start": 0,
            "end": 1,
            "bytes": 10
        },
        {
            "ds_id": 2,
            "start": 0,
            "end": 10,
            "bytes": 4
        },
        {
            "ds_id": 3,
            "start": 1,
            "end": 2,
            "bytes": 6
        },
        {
            "ds_id": 4,
            "start": 3,
            "end": 4,
            "bytes": 6
        },
        {
            "ds_id": 5,
            "start": 3,
            "end": 4,
            "bytes": 10
        },
    ],
                                                           alignment=1)

    assert reason is None
    assert packed["capacity"] == packed["max_live_bytes"] == 20
    offsets = {interval["ds_id"]: interval["offset"] for interval in packed["intervals"]}
    assert offsets[4] == 14
    assert offsets[5] == 0


def test_prefetch_arena_plan_uses_final_padded_bytes_and_coalesces_repeated_ids():
    graph = Graph()
    first = graph.placeholder("first")
    second = graph.placeholder("second")
    _arena_prefetch(graph, [first, first, second], [1, 1, 2], 10, [12, 12, 16])
    release_first = _release(graph, first, 1, "arena_first")
    release_second = _release(graph, second, 2, "arena_second")
    graph.output((release_first, release_second))
    manager = _fake_arena_param_manager([
        ("first", 1, (5, ), torch.float16),
        ("second", 2, (8, ), torch.float16),
    ])

    plan, reason = prefetch_mod._build_prefetch_arena_plan(graph, manager, world_size=2, bwd=False)

    assert reason is None
    assert [(entry["plan_id"], entry["ds_id"], entry["bytes"]) for entry in plan["entries"]] == [(10, 1, 12),
                                                                                                 (10, 2, 16)]
    assert plan["capacity"] == 512
    assert plan["max_live_bytes"] == 512
    assert all(entry["offset"] % prefetch_mod.PREFETCH_ARENA_ALIGNMENT == 0 for entry in plan["entries"])


def test_prefetch_arena_reuse_adds_release_ordering_dependencies():
    graph = Graph()
    first = graph.placeholder("first")
    second = graph.placeholder("second")
    demand_only = graph.placeholder("demand_only")
    first_prefetch = _arena_prefetch(graph, [first], [1], 10, [16])
    first_release = _release(graph, first, 1, "arena_first")
    _release(graph, demand_only, 3, "demand_only")
    second_release = _release(graph, second, 2, "arena_second")
    second_prefetch = _arena_prefetch(graph, [second], [2], 11, [16])
    graph.output((first_prefetch, second_prefetch))

    prefetch_mod._add_prefetch_arena_release_dependencies(graph)
    graph.lint()

    assert first_prefetch.args[1] == [first]
    assert second_prefetch.args[1] == [second, first_release, second_release]
    assert second_prefetch.args[2] == [2]
    assert second_prefetch.meta["prefetch_arena_ordering_dependencies"] == (
        first_release.name,
        second_release.name,
    )


def test_prefetch_arena_plan_reuses_nonoverlapping_slice_but_not_overlapping_slice():
    graph = Graph()
    first = graph.placeholder("first")
    second = graph.placeholder("second")
    third = graph.placeholder("third")
    _arena_prefetch(graph, [first], [1], 10, [256])
    release_first = _release(graph, first, 1, "arena_first")
    _arena_prefetch(graph, [second, third], [2, 3], 11, [256, 256])
    release_second = _release(graph, second, 2, "arena_second")
    release_third = _release(graph, third, 3, "arena_third")
    graph.output((release_first, release_second, release_third))
    manager = _fake_arena_param_manager([
        ("first", 1, (128, ), torch.float16),
        ("second", 2, (128, ), torch.float16),
        ("third", 3, (128, ), torch.float16),
    ])

    plan, reason = prefetch_mod._build_prefetch_arena_plan(graph, manager, world_size=2, bwd=True)

    assert reason is None
    offsets = {entry["ds_id"]: entry["offset"] for entry in plan["entries"]}
    assert offsets[1] == offsets[2]
    assert offsets[2] != offsets[3]
    assert plan["phase"] == 1
    assert plan["capacity"] == plan["max_live_bytes"] == 512


def test_prefetch_arena_plan_reuses_slice_for_sequential_repeated_id():
    graph = Graph()
    param = graph.placeholder("param")
    _arena_prefetch(graph, [param], [1], 10, [256])
    first_release = _release(graph, param, 1, "arena_first")
    _arena_prefetch(graph, [param], [1], 11, [256])
    second_release = _release(graph, param, 1, "arena_second")
    graph.output((first_release, second_release))
    manager = _fake_arena_param_manager([("param", 1, (128, ), torch.float16)])

    plan, reason = prefetch_mod._build_prefetch_arena_plan(graph, manager, world_size=2, bwd=False)

    assert reason is None
    assert [(entry["plan_id"], entry["offset"]) for entry in plan["entries"]] == [(10, 0), (11, 0)]
    assert plan["capacity"] == plan["max_live_bytes"] == 256


@pytest.mark.parametrize("missing", ["release", "scheduled_bytes"])
def test_prefetch_arena_plan_falls_back_on_incomplete_metadata(missing):
    graph = Graph()
    param = graph.placeholder("param")
    prefetch = _arena_prefetch(graph, [param], [1], 10, [12])
    if missing == "scheduled_bytes":
        prefetch.meta["prefetch_arena_bytes_by_ds_id"] = ()
    if missing != "release":
        release = _release(graph, param, 1, "arena_param")
        graph.output((release, ))
    else:
        graph.output((param, ))
    manager = _fake_arena_param_manager([("param", 1, (5, ), torch.float16)])

    plan, reason = prefetch_mod._build_prefetch_arena_plan(graph, manager, world_size=2, bwd=False)

    assert plan is None
    assert reason == ("incomplete_release" if missing == "release" else "missing_scheduled_bytes")


def test_prefetch_arena_plan_falls_back_on_dynamic_shape():
    graph = Graph()
    param = graph.placeholder("param")
    _arena_prefetch(graph, [param], [1], 10, [12])
    release = _release(graph, param, 1, "arena_param")
    graph.output((release, ))
    manager = SimpleNamespace(
        params={"param": SimpleNamespace(shape=(object(), ), dtype=torch.float16)},
        ds_ids={"param": 1},
    )

    plan, reason = prefetch_mod._build_prefetch_arena_plan(graph, manager, world_size=2, bwd=False)

    assert plan is None
    assert reason == "dynamic_shape"


def test_prefetch_arena_budget_admission_replaces_live_gather_charge_with_fixed_capacity():
    graph = Graph()
    param = graph.placeholder("param")
    _arena_prefetch(graph, [param], [1], 10, [12])
    release = _release(graph, param, 1, "arena_param")
    graph.output((release, ))
    manager = _fake_arena_param_manager([("param", 1, (5, ), torch.float16)])
    plan, reason = prefetch_mod._build_prefetch_arena_plan(graph, manager, world_size=2, bwd=False)
    assert reason is None
    mem_dict = {node.name: (500, 500) for node in graph.nodes}

    reserved_mem_dict = {node.name: 500 for node in graph.nodes}
    rejected = prefetch_mod._prefetch_arena_budget_admission(graph, plan, mem_dict, reserved_mem_dict, max_mem=700)
    accepted = prefetch_mod._prefetch_arena_budget_admission(graph, plan, mem_dict, reserved_mem_dict, max_mem=800)

    assert rejected == {
        "accepted": False,
        "max_mem": 700,
        "capacity": 256,
        "original_allocated_peak": 512,
        "arena_allocated_peak": 756,
        "original_modeled_peak": 512,
        "arena_modeled_peak": 756,
        "original_reserved_peak": 500,
        "arena_reserved_peak": 756,
        "incremental_peak": 244,
        "limiting_node": "param",
        "limiting_metric": "allocated_after_pool_drain_and_empty_cache",
        "limiting_pool_reclaimable_bytes": 0,
    }
    assert accepted["accepted"] is True
    # Capacity replaces the already-counted 12 live bytes rather than being
    # added on top of them a second time.
    assert accepted["incremental_peak"] == plan["capacity"] - 12


def test_prefetch_arena_budget_does_not_charge_cache_flushed_before_backing():
    graph = Graph()
    param = graph.placeholder("param")
    _arena_prefetch(graph, [param], [1], 10, [12])
    release = _release(graph, param, 1, "arena_param")
    graph.output((release, ))
    manager = _fake_arena_param_manager([("param", 1, (5, ), torch.float16)])
    plan, reason = prefetch_mod._build_prefetch_arena_plan(graph, manager, world_size=2, bwd=False)
    assert reason is None

    mem_dict = {node.name: (500, 500) for node in graph.nodes}
    reserved_mem_dict = {node.name: 900 for node in graph.nodes}
    admission = prefetch_mod._prefetch_arena_budget_admission(graph, plan, mem_dict, reserved_mem_dict, max_mem=800)

    assert admission["accepted"] is True
    assert admission["arena_allocated_peak"] == 756
    assert admission["arena_modeled_peak"] == 756
    assert admission["arena_reserved_peak"] == 1156
    assert admission["limiting_metric"] == "allocated_after_pool_drain_and_empty_cache"
    assert admission["limiting_pool_reclaimable_bytes"] == 0


def test_training_session_budget_uses_shared_forward_backward_capacity_bound():
    fwd = Graph()
    fwd_param = _placeholder(fwd, "fwd_param")
    _arena_prefetch(fwd, [fwd_param], [1], 10, [12])
    fwd_release = _release(fwd, fwd_param, 1, "fwd_param")
    fwd.output((fwd_release, ))
    manager = _fake_arena_param_manager([("fwd_param", 1, (5, ), torch.float16)])
    fwd_plan, reason = prefetch_mod._build_prefetch_arena_plan(fwd, manager, world_size=2, bwd=False)
    assert reason is None

    bwd = Graph()
    bwd_param = _placeholder(bwd, "bwd_param")
    _arena_prefetch(bwd, [bwd_param], [1], 1_000_010, [300])
    bwd_release = _release(bwd, bwd_param, 1, "bwd_param")
    bwd.output((bwd_release, ))
    bwd_manager = _fake_arena_param_manager([("bwd_param", 1, (150, ), torch.float16)])
    bwd_plan, reason = prefetch_mod._build_prefetch_arena_plan(bwd, bwd_manager, world_size=2, bwd=True)
    assert reason is None
    fwd_mem = {node.name: (100, 100) for node in fwd.nodes}
    fwd_reserved = {node.name: 100 for node in fwd.nodes}
    bwd_mem = {node.name: (100, 100) for node in bwd.nodes}
    bwd_reserved = {node.name: 100 for node in bwd.nodes}

    admission = prefetch_mod._training_session_budget_admission(fwd,
                                                                fwd_plan,
                                                                fwd_mem,
                                                                fwd_reserved,
                                                                bwd,
                                                                bwd_plan,
                                                                bwd_mem,
                                                                bwd_reserved,
                                                                max_mem=700)

    assert admission == {
        "accepted": True,
        "max_mem": 700,
        "capacity_bound": 512,
        "forward_arena_peak": 612,
        "backward_arena_peak": 612,
        "forward_original_peak": 112,
        "backward_original_peak": 400,
        "forward_pool_reclaimable_bytes": 0,
        "backward_pool_reclaimable_bytes": 0,
    }


def test_training_session_budget_subtracts_only_profiled_pool_charge():
    fwd = Graph()
    fwd_param = _placeholder(fwd, "fwd_param")
    _arena_prefetch(fwd, [fwd_param], [1], 10, [12])
    fwd.output((_release(fwd, fwd_param, 1, "fwd_param"), ))
    fwd_plan, reason = prefetch_mod._build_prefetch_arena_plan(fwd,
                                                               _fake_arena_param_manager([("fwd_param", 1, (5, ),
                                                                                           torch.float16)]),
                                                               world_size=2,
                                                               bwd=False)
    assert reason is None

    bwd = Graph()
    bwd_param = _placeholder(bwd, "bwd_param")
    _arena_prefetch(bwd, [bwd_param], [1], 1_000_010, [300])
    bwd.output((_release(bwd, bwd_param, 1, "bwd_param"), ))
    bwd_plan, reason = prefetch_mod._build_prefetch_arena_plan(bwd,
                                                               _fake_arena_param_manager([("bwd_param", 1, (150, ),
                                                                                           torch.float16)]),
                                                               world_size=2,
                                                               bwd=True)
    assert reason is None

    fwd_mem = {node.name: (600, 600) for node in fwd.nodes}
    bwd_mem = {node.name: (600, 600) for node in bwd.nodes}
    reserved = {node.name: 900 for node in (*list(fwd.nodes), *list(bwd.nodes))}
    no_credit = prefetch_mod._training_session_budget_admission(fwd,
                                                                fwd_plan,
                                                                fwd_mem,
                                                                reserved,
                                                                bwd,
                                                                bwd_plan,
                                                                bwd_mem,
                                                                reserved,
                                                                max_mem=700)
    pool_credit = {node.name: 450 for node in (*list(fwd.nodes), *list(bwd.nodes))}
    credited = prefetch_mod._training_session_budget_admission(fwd,
                                                               fwd_plan,
                                                               fwd_mem,
                                                               reserved,
                                                               bwd,
                                                               bwd_plan,
                                                               bwd_mem,
                                                               reserved,
                                                               max_mem=700,
                                                               forward_pool_reclaimable_dict=pool_credit,
                                                               backward_pool_reclaimable_dict=pool_credit)

    assert no_credit["accepted"] is False
    assert credited["accepted"] is True
    assert credited["forward_arena_peak"] == 662
    assert credited["backward_arena_peak"] == 662
    assert credited["forward_pool_reclaimable_bytes"] == 450
    assert credited["backward_pool_reclaimable_bytes"] == 450


def test_schedule_prefetch_configures_arena_by_default_from_final_graph(monkeypatch):
    graph = Graph()
    param = _placeholder(graph, "param")
    ag = _allgather(graph, param, 1, "arena", tensor_size=12)
    ag.meta["allgather_allocation_bytes"] = 12
    wait = _wait(graph, ag, 1, "arena")
    use = _neg(graph, wait, "arena_use")
    release = _release(graph, use, 1, "arena")
    graph.output((release, ))
    graph.lint()

    records = [(node.name, 0, 0, 0) for node in graph.nodes]
    times = [(node.name, 1, 1) for node in graph.nodes]
    sizes = [(node.name, int(node.meta.get("tensor_size", 0))) for node in graph.nodes]
    profile = ProfilingResult(fwd_graph=graph,
                              fwd_mem=records,
                              fwd_time=times,
                              fwd_tensor_sizes=sizes,
                              process_group="test-group")
    for node in graph.nodes:
        node.meta["profile_reserved_peak"] = 0
    manager = _fake_arena_param_manager([("param", 1, (5, ), torch.float16)])
    configured = []
    reductions = []

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1 << 20

        def available_memory(self):
            return 1 << 20

        def memory_allocated(self):
            return 0

        def max_memory_allocated(self):
            return 0

    monkeypatch.delenv("DEEPSPEED_COMPILE_PREFETCH_ARENA", raising=False)
    monkeypatch.setattr(prefetch_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(prefetch_mod, "create_predictor", lambda: lambda _: 1)
    monkeypatch.setattr(prefetch_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(prefetch_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(prefetch_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(prefetch_mod.dist, "all_reduce", lambda *args, **kwargs: reductions.append(args))
    monkeypatch.setattr(prefetch_mod, "get_deepcompile_handle",
                        lambda: SimpleNamespace(configure_z3_prefetch_arena=lambda *args: configured.append(args)))
    gm = GraphModule(torch.nn.Module(), graph)

    result = prefetch_mod.schedule_prefetch(gm,
                                            graph_id=0,
                                            graph_order=[(0, False)],
                                            profiling_results={0: profile},
                                            create_inputs_fn=lambda: (),
                                            mem_budget=0,
                                            param_manager={0: manager},
                                            bwd=False)

    prefetch_nodes = [node for node in result.graph.nodes if node.target == torch.ops.dc.prefetch_params_fused.default]
    # The scheduler's existing memory-budget reduction remains, but arena-plan
    # consensus is deferred to native execution where graph order is stable.
    assert len(reductions) == 1
    assert len(prefetch_nodes) == 1
    assert prefetch_nodes[0].args[4] == 0
    assert prefetch_nodes[0].meta["prefetch_arena_eligible_ds_ids"] == (1, )
    assert len(configured) == 1
    assert configured[0][:5] == (0, 0, False, 256, 256)
    assert configured[0][5] > 0
    assert configured[0][6:] == ([0], [1], [0], [12])


def test_schedule_prefetch_shared_budget_rejection_restores_original_call_shape(monkeypatch):
    graph = Graph()
    param = _placeholder(graph, "param")
    ag = _allgather(graph, param, 1, "arena", tensor_size=12)
    ag.meta["allgather_allocation_bytes"] = 12
    wait = _wait(graph, ag, 1, "arena")
    use = _neg(graph, wait, "arena_use")
    release = _release(graph, use, 1, "arena")
    graph.output((release, ))
    graph.lint()

    # 90% of 1 MiB is 943,718 bytes. The fixed 256-byte backing takes the
    # modeled peak above that common budget even though the original 12-byte
    # live gather remains admissible.
    records = [(node.name, 943_500, 0, 943_500) for node in graph.nodes]
    times = [(node.name, 1, 1) for node in graph.nodes]
    sizes = [(node.name, int(node.meta.get("tensor_size", 0))) for node in graph.nodes]
    profile = ProfilingResult(fwd_graph=graph,
                              fwd_mem=records,
                              fwd_time=times,
                              fwd_tensor_sizes=sizes,
                              process_group="test-group")
    for node in graph.nodes:
        node.meta["profile_reserved_peak"] = 943_500
    manager = _fake_arena_param_manager([("param", 1, (5, ), torch.float16)])
    configured = []
    logs = []

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1 << 20

        def available_memory(self):
            return 1 << 20

        def memory_allocated(self):
            return 0

        def max_memory_allocated(self):
            return 0

    monkeypatch.setattr(prefetch_mod, "print_rank_0", logs.append)
    monkeypatch.setattr(prefetch_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(prefetch_mod, "create_predictor", lambda: lambda _: 1)
    monkeypatch.setattr(prefetch_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(prefetch_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(prefetch_mod.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(prefetch_mod, "get_deepcompile_handle",
                        lambda: SimpleNamespace(configure_z3_prefetch_arena=lambda *args: configured.append(args)))
    gm = GraphModule(torch.nn.Module(), graph)

    result = prefetch_mod.schedule_prefetch(gm,
                                            graph_id=0,
                                            graph_order=[(0, False)],
                                            profiling_results={0: profile},
                                            create_inputs_fn=lambda: (),
                                            mem_budget=0,
                                            param_manager={0: manager},
                                            bwd=False)

    prefetch_nodes = [node for node in result.graph.nodes if node.target == torch.ops.dc.prefetch_params_fused.default]
    assert len(prefetch_nodes) == 1
    assert len(prefetch_nodes[0].args) == 3
    assert "prefetch_arena_eligible_ds_ids" not in prefetch_nodes[0].meta
    assert configured == []
    assert any("accepted=0" in message for message in logs)
    assert any("fallback=shared_budget" in message for message in logs)


def test_schedule_prefetch_training_forward_registers_pending_session_plan(monkeypatch):
    graph = Graph()
    param = _placeholder(graph, "param")
    ag = _allgather(graph, param, 1, "arena", tensor_size=12)
    ag.meta["allgather_allocation_bytes"] = 12
    wait = _wait(graph, ag, 1, "arena")
    use = _neg(graph, wait, "arena_use")
    release = _release(graph, use, 1, "arena")
    graph.output((release, ))
    records = [(node.name, 0, 0, 0) for node in graph.nodes]
    profile = ProfilingResult(fwd_graph=graph,
                              fwd_mem=records,
                              fwd_time=[(node.name, 1, 1) for node in graph.nodes],
                              fwd_tensor_sizes=[(node.name, int(node.meta.get("tensor_size", 0)))
                                                for node in graph.nodes],
                              needs_backward=True,
                              process_group="test-group")
    for node in graph.nodes:
        node.meta["profile_reserved_peak"] = 0
    manager = _fake_arena_param_manager([("param", 1, (5, ), torch.float16)])
    configured = []
    logs = []

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1 << 20

        def available_memory(self):
            return 1 << 20

        def memory_allocated(self):
            return 0

        def max_memory_allocated(self):
            return 0

    monkeypatch.setattr(prefetch_mod, "print_rank_0", logs.append)
    monkeypatch.setattr(prefetch_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(prefetch_mod, "create_predictor", lambda: lambda _: 1)
    monkeypatch.setattr(prefetch_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(prefetch_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(prefetch_mod.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        prefetch_mod, "get_deepcompile_handle",
        lambda: SimpleNamespace(configure_z3_prefetch_arena=lambda *args: configured.append(args),
                                disable_z3_prefetch_arena=lambda *args: None))

    result = prefetch_mod.schedule_prefetch(GraphModule(torch.nn.Module(), graph),
                                            graph_id=0,
                                            graph_order=[(0, True)],
                                            profiling_results={0: profile},
                                            create_inputs_fn=lambda: (),
                                            mem_budget=0,
                                            param_manager={0: manager},
                                            bwd=False)

    prefetch_nodes = [node for node in result.graph.nodes if node.target == torch.ops.dc.prefetch_params_fused.default]
    assert len(prefetch_nodes) == 1
    assert prefetch_nodes[0].args[4] == 0
    assert len(configured) == 1
    assert configured[0][:5] == (0, 0, True, 256, 256)
    assert profile.prefetch_arena_forward_plan is not None
    assert profile.prefetch_arena_session_accepted is None
    assert any("prefetch_arena_session_pending" in message for message in logs)


def test_schedule_prefetch_training_backward_decides_pending_session(monkeypatch):
    fwd = Graph()
    fwd_param = _placeholder(fwd, "fwd_param")
    _arena_prefetch(fwd, [fwd_param], [1], 0, [12])
    fwd_release = _release(fwd, fwd_param, 1, "fwd_param")
    fwd.output((fwd_release, ))
    fwd_plan, reason = prefetch_mod._build_prefetch_arena_plan(fwd,
                                                               _fake_arena_param_manager([("fwd_param", 1, (5, ),
                                                                                           torch.float16)]),
                                                               world_size=2,
                                                               bwd=False)
    assert reason is None

    bwd = Graph()
    bwd_param = _placeholder(bwd, "bwd_param")
    bwd_ag = _allgather(bwd, bwd_param, 1, "bwd_param", tensor_size=12)
    bwd_ag.meta["allgather_allocation_bytes"] = 12
    bwd_wait = _wait(bwd, bwd_ag, 1, "bwd_param")
    bwd_release = _release(bwd, bwd_wait, 1, "bwd_param")
    bwd.output((bwd_release, ))
    records = [(node.name, 0, 0, 0) for node in bwd.nodes]
    profile = ProfilingResult(bwd_graph=bwd,
                              bwd_mem=records,
                              bwd_time=[(node.name, 1, 1) for node in bwd.nodes],
                              bwd_tensor_sizes=[(node.name, int(node.meta.get("tensor_size", 0)))
                                                for node in bwd.nodes],
                              needs_backward=True,
                              process_group="test-group",
                              prefetch_arena_forward_graph=fwd,
                              prefetch_arena_forward_plan=fwd_plan,
                              prefetch_arena_forward_mem={node.name: (0, 0)
                                                          for node in fwd.nodes},
                              prefetch_arena_forward_reserved_mem={node.name: 0
                                                                   for node in fwd.nodes},
                              prefetch_arena_forward_pool_reclaimable={node.name: 0
                                                                       for node in fwd.nodes})
    for node in bwd.nodes:
        node.meta["profile_reserved_peak"] = 0
    manager = _fake_arena_param_manager([("bwd_param", 1, (5, ), torch.float16)])
    configured = []
    logs = []

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1 << 20

        def available_memory(self):
            return 1 << 20

        def memory_allocated(self):
            return 0

        def max_memory_allocated(self):
            return 0

    monkeypatch.setattr(prefetch_mod, "print_rank_0", logs.append)
    monkeypatch.setattr(prefetch_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(prefetch_mod, "create_predictor", lambda: lambda _: 1)
    monkeypatch.setattr(prefetch_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(prefetch_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(prefetch_mod.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        prefetch_mod, "get_deepcompile_handle",
        lambda: SimpleNamespace(configure_z3_prefetch_arena=lambda *args: configured.append(args),
                                disable_z3_prefetch_arena=lambda *args: None))

    result = prefetch_mod.schedule_prefetch(GraphModule(torch.nn.Module(), bwd),
                                            graph_id=0,
                                            graph_order=[(0, True)],
                                            profiling_results={0: profile},
                                            create_inputs_fn=lambda: (),
                                            mem_budget=0,
                                            param_manager={0: manager},
                                            bwd=True)

    prefetch_nodes = [node for node in result.graph.nodes if node.target == torch.ops.dc.prefetch_params_fused.default]
    assert len(prefetch_nodes) == 1
    assert prefetch_nodes[0].args[4] == 1_000_000
    assert len(configured) == 1
    assert configured[0][:5] == (0, 1, True, 256, 256)
    assert profile.prefetch_arena_session_accepted is True
    assert profile.prefetch_arena_session_capacity_bound == 256
    assert any("prefetch_arena_session_admission" in message and "accepted=1" in message for message in logs)


def test_schedule_prefetch_missing_reserved_profile_restores_original_call_shape(monkeypatch):
    graph = Graph()
    param = _placeholder(graph, "param")
    ag = _allgather(graph, param, 1, "arena", tensor_size=12)
    ag.meta["allgather_allocation_bytes"] = 12
    wait = _wait(graph, ag, 1, "arena")
    use = _neg(graph, wait, "arena_use")
    release = _release(graph, use, 1, "arena")
    graph.output((release, ))
    graph.lint()

    records = [(node.name, 0, 0, 0) for node in graph.nodes]
    times = [(node.name, 1, 1) for node in graph.nodes]
    sizes = [(node.name, int(node.meta.get("tensor_size", 0))) for node in graph.nodes]
    profile = ProfilingResult(fwd_graph=graph,
                              fwd_mem=records,
                              fwd_time=times,
                              fwd_tensor_sizes=sizes,
                              process_group="test-group")
    manager = _fake_arena_param_manager([("param", 1, (5, ), torch.float16)])
    configured = []
    logs = []

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1 << 20

        def available_memory(self):
            return 1 << 20

        def memory_allocated(self):
            return 0

        def max_memory_allocated(self):
            return 0

    monkeypatch.setattr(prefetch_mod, "print_rank_0", logs.append)
    monkeypatch.setattr(prefetch_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(prefetch_mod, "create_predictor", lambda: lambda _: 1)
    monkeypatch.setattr(prefetch_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(prefetch_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(prefetch_mod.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(prefetch_mod, "get_deepcompile_handle",
                        lambda: SimpleNamespace(configure_z3_prefetch_arena=lambda *args: configured.append(args)))
    gm = GraphModule(torch.nn.Module(), graph)

    result = prefetch_mod.schedule_prefetch(gm,
                                            graph_id=0,
                                            graph_order=[(0, False)],
                                            profiling_results={0: profile},
                                            create_inputs_fn=lambda: (),
                                            mem_budget=0,
                                            param_manager={0: manager},
                                            bwd=False)

    prefetch_nodes = [node for node in result.graph.nodes if node.target == torch.ops.dc.prefetch_params_fused.default]
    assert len(prefetch_nodes) == 1
    assert len(prefetch_nodes[0].args) == 3
    assert configured == []
    assert any("fallback=incomplete_reserved_profile" in message for message in logs)


def _scheduled_graph(graph, scheduler_budget=None):
    return schedule_mod.fast_free_schedule(graph, 0, 0, debug_log=True, scheduler_budget=scheduler_budget)


def _scheduled_names(graph, scheduler_budget=None):
    return [node.name for node in _scheduled_graph(graph, scheduler_budget=scheduler_budget).nodes]


def _scheduler_diagnostics(graph):
    return getattr(graph, schedule_mod.DS_SCHEDULER_BUDGET_DIAGNOSTICS_ATTR)


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
