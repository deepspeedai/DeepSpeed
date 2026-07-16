# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import operator
import sys
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
from deepspeed.compile.profilers import graph_profile as graph_profile_mod
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
    process_group = object()

    def mark_any_rank_failed(tensor, op, group=None):
        assert op == backend_mod.dist.ReduceOp.MIN
        assert group is process_group
        tensor[0] = 0

    monkeypatch.setattr(backend_mod.dist, "all_reduce", mark_any_rank_failed)

    assert not backend_mod._sync_memory_profile_complete(True, process_group)


def test_get_last_uses_handles_dead_no_copy_node():
    graph = Graph()
    param = _placeholder(graph, "dead_wait_param")
    wait = _wait(graph, param, 1, "dead_wait")
    graph.output(())
    graph.lint()

    node_to_last_use, user_to_last_uses = compile_util.get_last_uses(graph)
    node_to_uses = compile_util.get_real_uses(graph)

    assert node_to_last_use[param] is wait
    assert user_to_last_uses[wait] == [param]
    assert node_to_uses[param] == []


def test_zero3_scheduler_budget_uses_rank_reduced_non_gathered_peak(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 2000

        def memory_allocated(self):
            return 50

    def reduce_budget_inputs(tensor, op):
        if op == zero3_compile_mod.dist.ReduceOp.MIN:
            tensor[0] = 1000
        elif op == zero3_compile_mod.dist.ReduceOp.MAX:
            tensor[0] = 850
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

    assert budget.source == "profiled_non_gathered_peak_memory"
    assert budget.total_mem == 1000
    assert budget.profiled_non_gathered_peak_mem == 850
    assert budget.safety_margin == 100
    assert budget.max_gathered_bytes == 50


def test_zero3_scheduler_budget_reconstructs_live_activations_before_transient_peak(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: SimpleNamespace(memory_allocated=lambda: 100))

    graph = Graph()
    value = _placeholder(graph, "absolute_peak_input")
    activation = _neg(graph, value, "absolute_peak_activation")
    activation.meta.update(alloc_mem=300, max_mem=300, profile_mem_start=100, profile_mem_peak=400)
    transient = _neg(graph, activation, "absolute_peak_transient")
    transient.meta.update(alloc_mem=50, max_mem=200, profile_mem_start=400, profile_mem_peak=600)
    graph.output((transient, ))
    graph.lint()

    peak = zero3_compile_mod._rank_max_operator_profiled_non_gathered_peak(graph)

    # The absolute record retains the first node's live activation while the
    # second node reaches a further transient peak.
    assert peak == 600


def test_zero3_scheduler_budget_uses_absolute_peaks_after_inter_node_reclamation(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: False)

    graph = Graph()
    value = _placeholder(graph, "reclaimed_peak_input")
    first = _neg(graph, value, "reclaimed_first")
    first.meta.update(alloc_mem=400, max_mem=400, profile_mem_start=100, profile_mem_peak=500)
    second = _neg(graph, first, "reclaimed_second")
    second.meta.update(alloc_mem=-300, max_mem=250, profile_mem_start=200, profile_mem_peak=450)
    graph.output((second, ))
    graph.lint()

    peak = zero3_compile_mod._rank_max_operator_profiled_non_gathered_peak(graph)

    assert peak == 500


def test_absolute_profile_memory_adds_external_once_and_max_reduces_asymmetric_ranks(monkeypatch):

    class FakeAccelerator:

        def memory_allocated(self):
            return 100

        def max_memory_allocated(self):
            return 140

    monkeypatch.setattr(graph_profile_mod, "get_accelerator", lambda: FakeAccelerator())

    assert graph_profile_mod._absolute_profile_memory(50) == (150, 190)

    def reduce_to_worst_rank(values, op):
        assert op == graph_profile_mod.dist.ReduceOp.MAX
        values[0] = 175
        values[1] = 410

    monkeypatch.setattr(graph_profile_mod.dist, "all_reduce", reduce_to_worst_rank)
    assert graph_profile_mod._rank_max_profile_memory(150, 190, torch.device("cpu"), distributed=True) == (175, 410)


def test_external_memory_profile_maps_nvml_device_and_excludes_reserved_allocator_bytes(monkeypatch):
    nvml_device_ids = []
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda device_id: nvml_device_ids.append(device_id) or "handle",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=1000))
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    class FakeAccelerator:

        def current_device(self):
            return 0

        def _get_nvml_gpu_id(self, device_id):
            assert device_id == 0
            return 5

        def memory_allocated(self):
            return 400

        def memory_reserved(self):
            return 600

    monkeypatch.setattr(graph_profile_mod, "get_accelerator", lambda: FakeAccelerator())

    assert graph_profile_mod._get_mem_usage_out_of_torch() == 400
    assert nvml_device_ids == [5]


def test_external_memory_profile_does_not_count_reserved_to_allocated_reuse_twice(monkeypatch):
    fake_nvml = SimpleNamespace(nvmlInit=lambda: None,
                                nvmlDeviceGetHandleByIndex=lambda device_id: "handle",
                                nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=1000))
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    class FakeAccelerator:
        allocated = 400

        def current_device(self):
            return 0

        def memory_allocated(self):
            return self.allocated

        def max_memory_allocated(self):
            return self.allocated

        def memory_reserved(self):
            return 800

    accelerator = FakeAccelerator()
    monkeypatch.setattr(graph_profile_mod, "get_accelerator", lambda: accelerator)

    external_mem = graph_profile_mod._get_mem_usage_out_of_torch()
    first_absolute, _ = graph_profile_mod._absolute_profile_memory(external_mem)
    accelerator.allocated = 700
    second_absolute, _ = graph_profile_mod._absolute_profile_memory(external_mem)

    assert external_mem == 200
    assert (first_absolute, second_absolute) == (600, 900)


def test_zero3_scheduler_incomplete_profile_uses_identical_collective_on_asymmetric_ranks(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 2)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 2000

        def memory_allocated(self):
            return 50

        def available_memory(self):
            return 500

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())

    def make_graph(observed):
        graph = Graph()
        param = _placeholder(graph, f"asymmetric_param_{observed}")
        ag = _allgather(graph, param, 1, f"asymmetric_{observed}", tensor_size=800)
        wait = _wait(graph, ag, 1, f"asymmetric_{observed}")
        op = _neg(graph, wait, f"asymmetric_op_{observed}")
        if observed:
            op.meta.update(alloc_mem=300, max_mem=500, profile_mem_start=850, profile_mem_peak=1300)
        release = _release(graph, op, 1, f"asymmetric_{observed}")
        graph.output((release, ))
        graph.lint()
        _backfill_missing_profile_metadata(graph, profile_complete=False)
        return GraphModule(torch.nn.Module(), graph)

    sequences = []
    budgets = []
    for observed in (True, False):
        calls = []

        def reduce_asymmetric_rank(tensor, op):
            calls.append((op, tensor.dtype))
            tensor[0] = 0

        monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", reduce_asymmetric_rank)
        budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(make_graph(observed))
        assert budget is None
        assert disabled_reason == "incomplete_operator_profile"
        sequences.append(calls)
        budgets.append(budget)

    assert sequences[0] == sequences[1]
    assert sequences[0] == [(zero3_compile_mod.dist.ReduceOp.MIN, torch.int32)]
    assert budgets == [None, None]


def test_zero3_scheduler_budget_skips_incomplete_operator_profile_metadata():
    graph = Graph()
    graph.output(())
    graph.lint()
    _backfill_missing_profile_metadata(graph, profile_complete=False)

    budget = zero3_compile_mod._build_scheduler_budget_from_operator_profile(graph)

    assert budget is None


def test_zero3_scheduler_budget_skips_incomplete_operator_profile(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod, "get_accelerator",
                        lambda: SimpleNamespace(current_device=lambda: "cpu", available_memory=lambda: 0))
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", lambda *args, **kwargs: None)
    graph = Graph()
    graph.output(())
    graph.lint()
    _backfill_missing_profile_metadata(graph, profile_complete=False)
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert budget is None
    assert disabled_reason == "incomplete_operator_profile"


def test_zero3_scheduler_budget_disables_incomplete_profile_with_unprofiled_high_memory_suffix(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 1)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 1000

        def memory_allocated(self):
            return 50

        def available_memory(self):
            return 500

    def reduce_budget_inputs(tensor, op):
        if tensor.dtype == torch.int32:
            tensor[0] = 0

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", reduce_budget_inputs)

    graph = Graph()
    param = _placeholder(graph, "partial_budget_param")
    ag = _allgather(graph, param, 1, "partial_budget", tensor_size=800)
    wait = _wait(graph, ag, 1, "partial_budget")
    op = _neg(graph, wait, "partial_budget_observed_op")
    op.meta.update(max_mem=800, profile_mem_start=850, profile_mem_peak=1650)
    unprofiled_suffix = _neg(graph, op, "partial_budget_unprofiled_high_memory_suffix")
    release = _release(graph, op, 1, "partial_budget")
    graph.output((release, unprofiled_suffix))
    graph.lint()
    _backfill_missing_profile_metadata(graph, profile_complete=False)
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert budget is None
    assert disabled_reason == "incomplete_operator_profile"


def test_zero3_scheduler_budget_skips_non_distributed_memory_profile(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: False)
    graph = Graph()
    graph.output(())
    graph.lint()
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert budget is None
    assert disabled_reason == "non_distributed"


def test_zero3_scheduler_budget_skips_when_budget_cannot_constrain(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", lambda *args, **kwargs: None)

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 2000

        def memory_allocated(self):
            return 50

    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: FakeAccelerator())

    graph = Graph()
    param = _placeholder(graph, "nonbinding_budget_param")
    ag = _allgather(graph, param, 1, "nonbinding_budget", tensor_size=200)
    wait = _wait(graph, ag, 1, "nonbinding_budget")
    op = _neg(graph, wait, "nonbinding_budget_op")
    op.meta.update(max_mem=800, profile_mem_start=250, profile_mem_peak=1050)
    release = _release(graph, op, 1, "nonbinding_budget")
    graph.output((release, ))
    graph.lint()
    for node in graph.nodes:
        node.meta.setdefault("alloc_mem", 0)
        node.meta.setdefault("max_mem", 0)
        node.meta.setdefault("profile_mem_start", 0)
        node.meta.setdefault("profile_mem_peak", 0)
    gm = GraphModule(torch.nn.Module(), graph)

    budget, disabled_reason = zero3_compile_mod._scheduler_budget_from_operator_profile(gm)

    assert budget is None
    assert disabled_reason == "budget_not_constraining"


def test_profiled_non_gathered_peak_conservatively_keeps_observed_peak():
    graph = Graph()
    param = _placeholder(graph, "nongathered_peak_param")
    ag = _allgather(graph, param, 2, "nongathered_peak", tensor_size=200)
    wait = _wait(graph, ag, 2, "nongathered_peak")
    release = _release(graph, wait, 2, "nongathered_peak")
    graph.output((release, ))
    graph.lint()

    assert schedule_mod.profiled_non_gathered_peak(graph, [(ag.name, 900, 0, 900), (wait.name, 950, 0, 950),
                                                           (release.name, 920, 0, 920)]) == 950


def test_profiled_non_gathered_peak_does_not_subtract_nonresident_upfront_gathers():
    graph = Graph()
    first_param = _placeholder(graph, "first_upfront_param")
    second_param = _placeholder(graph, "second_upfront_param")
    first_ag = _allgather(graph, first_param, 20, "first_upfront", tensor_size=200)
    second_ag = _allgather(graph, second_param, 21, "second_upfront", tensor_size=300)
    activation = _neg(graph, second_ag, "activation_heavy_node")
    first_release = _release(graph, activation, 20, "first_upfront")
    second_release = _release(graph, first_release, 21, "second_upfront")
    graph.output((second_release, ))
    graph.lint()

    # Operator profiling invalidates gathered buffers between nodes, so the
    # activation peak is not guaranteed to include either upfront gather.  The
    # scheduler budget must not subtract hypothetical source-order residency.
    mem_records = [(first_ag.name, 200, 0, 200), (second_ag.name, 300, 0, 300), (activation.name, 1000, 0, 1000)]
    assert schedule_mod.profiled_non_gathered_peak(graph, mem_records) == 1000


def test_zero3_stamps_padded_allgather_allocation_metadata(monkeypatch):
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", lambda: 8)
    graph = Graph()

    param = _placeholder(graph, "metadata_padded_param")
    ag = _allgather(graph, param, 3, "metadata_padded", tensor_size=102)
    wait = _wait(graph, ag, 3, "metadata_padded")
    release = _release(graph, wait, 3, "metadata_padded")
    graph.output((release, ))
    graph.lint()

    zero3_compile_mod._set_allgather_allocation_metadata(graph)

    assert ag.meta["allgather_allocation_bytes"] == 112


def test_zero3_stamps_replicated_param_allgather_allocation_metadata():
    graph = Graph()

    param = _placeholder(graph, "replicated_metadata_param")
    param.meta["val"] = torch.empty((8, ), dtype=torch.float16)
    use = _neg(graph, param, "replicated_metadata_use")
    graph.output((use, ))
    graph.lint()

    param_manager = SimpleNamespace(params={param.name: SimpleNamespace(dtype=torch.bfloat16, numel=777)},
                                    ds_ids={param.name: 3})

    zero3_compile_mod.add_gather_and_release(0, graph, param_manager, [param])

    ag_nodes = [node for node in graph.nodes if node.target == torch.ops.dc.allgather_param.default]
    assert len(ag_nodes) == 1
    assert ag_nodes[0].meta["allgather_allocation_bytes"] == 1554


def test_zero3_gathers_output_only_param_for_backward_passthrough():
    graph = Graph()

    param = _placeholder(graph, "output_only_param")
    param.meta["val"] = torch.empty((8, ), dtype=torch.float16)
    graph.output((param, ))
    graph.lint()

    param_manager = SimpleNamespace(params={param.name: SimpleNamespace(dtype=torch.bfloat16, numel=777)},
                                    ds_ids={param.name: 3})

    new_graph = zero3_compile_mod.add_gather_and_release(0, graph, param_manager, [param])
    new_graph.lint()

    ag_nodes = [node for node in new_graph.nodes if node.target == torch.ops.dc.allgather_param.default]
    wait_nodes = [node for node in new_graph.nodes if node.target == torch.ops.dc.wait_allgather.default]
    release_nodes = [node for node in new_graph.nodes if node.target == torch.ops.dc.release_param.default]
    output_node = next(node for node in new_graph.nodes if node.op == "output")

    assert len(ag_nodes) == 1
    assert len(wait_nodes) == 1
    assert release_nodes == []
    assert ag_nodes[0].args[0].name == param.name
    assert wait_nodes[0].args[0] is ag_nodes[0]
    assert output_node.args == ((wait_nodes[0], ), )


def test_zero3_scheduler_debug_logs_disabled_budget(monkeypatch, capsys):
    monkeypatch.setenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, "1")
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: False)
    graph = Graph()
    graph.output(())
    graph.lint()

    zero3_compile_mod._log_scheduler_result(7,
                                            bwd=True,
                                            scheduler_budget=None,
                                            disabled_reason="missing_or_incomplete_memory_profile",
                                            graph=graph)

    captured = capsys.readouterr()
    assert "budget_enabled=False" in captured.out
    assert "disabled_reason=missing_or_incomplete_memory_profile" in captured.out


def test_zero3_final_schedule_fingerprint_detects_rank_mismatch(monkeypatch):
    monkeypatch.setenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, "1")
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod, "get_accelerator", lambda: SimpleNamespace(current_device=lambda: "cpu"))

    def reduce_mismatched_fingerprints(value, op):
        value[0] = 1 if op == zero3_compile_mod.dist.ReduceOp.MIN else 2

    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", reduce_mismatched_fingerprints)
    graph = Graph()
    graph.output(())

    with pytest.raises(RuntimeError, match="final schedule fingerprint mismatch"):
        zero3_compile_mod._validate_final_schedule_fingerprint(graph, graph_id=7, bwd=False)


def test_zero3_final_schedule_fingerprint_is_safe_without_distributed(monkeypatch, capsys):
    monkeypatch.setenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, "1")
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce",
                        lambda *args, **kwargs: pytest.fail("non-distributed debug must not use a collective"))
    graph = Graph()
    graph.output(())

    fingerprint = zero3_compile_mod._validate_final_schedule_fingerprint(graph, graph_id=8, bwd=True)

    assert fingerprint == zero3_compile_mod._final_schedule_fingerprint(graph)
    assert "final_schedule_fingerprint" in capsys.readouterr().out


def test_zero3_final_schedule_fingerprint_is_absent_when_debug_is_disabled(monkeypatch):
    monkeypatch.delenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, raising=False)
    monkeypatch.delenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV_LEGACY, raising=False)
    monkeypatch.setattr(zero3_compile_mod, "_final_schedule_fingerprint",
                        lambda graph: pytest.fail("non-debug scheduling must not compute a fingerprint"))
    graph = Graph()
    graph.output(())

    assert zero3_compile_mod._validate_final_schedule_fingerprint(graph, graph_id=9, bwd=False) is None


def test_zero3_scheduler_collectives_stay_with_each_data_parallel_group(monkeypatch):
    monkeypatch.setenv(zero3_compile_mod.SCHEDULER_DEBUG_ENV, "1")
    monkeypatch.setattr(zero3_compile_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(zero3_compile_mod, "get_accelerator",
                        lambda: SimpleNamespace(current_device=lambda: "cpu", total_memory=lambda: 2000))
    group_a = object()
    group_b = object()
    group_sizes = {group_a: 2, group_b: 8}
    collective_groups = []

    def get_world_size(group=None):
        assert group in group_sizes
        return group_sizes[group]

    def all_reduce(tensor, op, group=None):
        assert group in group_sizes
        collective_groups.append(group)

    monkeypatch.setattr(zero3_compile_mod.dist, "get_world_size", get_world_size)
    monkeypatch.setattr(zero3_compile_mod.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(zero3_compile_mod.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(graph_profile_mod.dist, "all_reduce", all_reduce)

    def make_group_graph(name):
        graph = Graph()
        param = _placeholder(graph, f"{name}_param")
        ag = _allgather(graph, param, 1, name, tensor_size=102)
        wait = _wait(graph, ag, 1, name)
        op = _neg(graph, wait, f"{name}_op")
        release = _release(graph, op, 1, name)
        graph.output((release, ))
        graph.lint()
        for node in graph.nodes:
            node.meta.setdefault("alloc_mem", 0)
            node.meta.setdefault("max_mem", 0)
            node.meta.setdefault("profile_mem_start", 100)
            node.meta.setdefault("profile_mem_peak", 1000)
        return GraphModule(torch.nn.Module(), graph), ag

    gm_a, ag_a = make_group_graph("group_a")
    gm_b, ag_b = make_group_graph("group_b_with_different_graph")

    zero3_compile_mod._scheduler_budget_from_operator_profile(gm_a, process_group=group_a)
    zero3_compile_mod._validate_final_schedule_fingerprint(gm_a.graph, graph_id=10, bwd=False, process_group=group_a)
    graph_profile_mod._rank_max_profile_memory(100, 200, torch.device("cpu"), distributed=True, process_group=group_a)
    zero3_compile_mod._scheduler_budget_from_operator_profile(gm_b, process_group=group_b)
    zero3_compile_mod._validate_final_schedule_fingerprint(gm_b.graph, graph_id=11, bwd=True, process_group=group_b)
    graph_profile_mod._rank_max_profile_memory(300, 400, torch.device("cpu"), distributed=True, process_group=group_b)

    assert ag_a.meta["allgather_allocation_bytes"] == 104
    assert ag_b.meta["allgather_allocation_bytes"] == 112
    assert collective_groups
    first_group_b = collective_groups.index(group_b)
    assert all(group is group_a for group in collective_groups[:first_group_b])
    assert all(group is group_b for group in collective_groups[first_group_b:])


def test_prefetch_and_selective_gather_collectives_stay_with_data_parallel_group(monkeypatch):
    try:
        torch.ops.dc.reload_parameter.default
    except AttributeError:
        library = torch.library.Library("dc", "FRAGMENT")
        library.define("reload_parameter(Tensor a, int graph_id, int id) -> ()")
        _DC_LIBRARIES.append(library)

    group_a = object()
    group_b = object()
    collective_groups = []

    class FakeAccelerator:

        def current_device(self):
            return "cpu"

        def total_memory(self):
            return 2000

        def available_memory(self):
            return 1800

        def memory_allocated(self):
            return 0

        def max_memory_allocated(self):
            return 0

    def all_reduce(tensor, op, group=None):
        assert group in (group_a, group_b)
        collective_groups.append(group)

    monkeypatch.setattr(prefetch_mod.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(prefetch_mod, "get_accelerator", FakeAccelerator)
    monkeypatch.setattr(prefetch_mod, "create_predictor", lambda: lambda _: 0.0)
    monkeypatch.setattr(prefetch_mod, "print_rank_0", lambda _: None)
    monkeypatch.setattr(selective_gather_mod, "get_accelerator", FakeAccelerator)
    monkeypatch.setattr(selective_gather_mod, "get_deepcompile_handle",
                        lambda: SimpleNamespace(set_persistent=lambda _: None))
    monkeypatch.setattr(selective_gather_mod, "print_rank_0", lambda _: None)

    def run_passes(group, graph_id):
        graph = Graph()
        value = _placeholder(graph, f"group_{graph_id}_input")
        result = _neg(graph, value, f"group_{graph_id}_result")
        graph.output((result, ))
        graph.lint()
        mem = [(node.name, 0, 0, 0) for node in graph.nodes]
        timing = [(node.name, 0.0, 0.0) for node in graph.nodes]
        tensor_sizes = [(node.name, 0) for node in graph.nodes]
        profile = ProfilingResult(fwd_graph=graph,
                                  bwd_graph=graph,
                                  needs_backward=True,
                                  fwd_mem=mem,
                                  bwd_mem=mem,
                                  fwd_time=timing,
                                  bwd_time=timing,
                                  fwd_tensor_sizes=tensor_sizes,
                                  bwd_tensor_sizes=tensor_sizes,
                                  process_group=group)
        profiling_results = {graph_id: profile}
        gm = GraphModule(torch.nn.Module(), graph)
        prefetch_mod.schedule_prefetch(gm,
                                       graph_id=graph_id,
                                       graph_order=[(graph_id, True)],
                                       profiling_results=profiling_results,
                                       create_inputs_fn=lambda: (),
                                       mem_budget=0,
                                       param_manager={},
                                       bwd=False)
        selective_gather_mod.selective_gather(gm,
                                              graph_id=graph_id,
                                              graph_order=[(graph_id, True)],
                                              profiling_results=profiling_results,
                                              create_inputs_fn=lambda: (),
                                              mem_budget=0,
                                              param_manager={},
                                              bwd=True)

    run_passes(group_a, 0)
    run_passes(group_b, 1)

    assert collective_groups == [group_a, group_a, group_b, group_b]


def _allgather(graph, arg, ds_id, name, tensor_size=1, device_time=1, allocation_size=None):
    node = _with_meta(
        graph.call_function(torch.ops.dc.allgather_param.default, (arg, 0, ds_id), {"dtype": torch.float16},
                            name=f"allgather_ds_param_{name}_{ds_id}"),
        tensor_size=tensor_size,
        device_time=device_time,
    )
    if allocation_size is not None:
        node.meta["allgather_allocation_bytes"] = allocation_size
    return node


def _wait(graph, arg, ds_id, name):
    return _with_meta(
        graph.call_function(torch.ops.dc.wait_allgather.default, (arg, 0, ds_id),
                            name=f"wait_allgather_ds_param_{name}_{ds_id}"))


def _neg(graph, arg, name, device_time=0):
    return _with_meta(graph.call_function(operator.neg, (arg, ), name=name), device_time=device_time)


def _add(graph, lhs, rhs, name, device_time=0):
    return _with_meta(graph.call_function(operator.add, (lhs, rhs), name=name), device_time=device_time)


def _release(graph, arg, ds_id, name, n_users=1):
    return _with_meta(
        graph.call_function(torch.ops.dc.release_param.default, (arg, 0, ds_id, n_users),
                            name=f"release_ds_param_{name}_{ds_id}"))


def _scheduled_graph(graph, scheduler_budget=None):
    return schedule_mod.fast_free_schedule(
        graph,
        0,
        0,
        debug_log=True,
        scheduler_budget=scheduler_budget,
    )


def _scheduled_names(graph, scheduler_budget=None):
    return [node.name for node in _scheduled_graph(graph, scheduler_budget=scheduler_budget).nodes]


def _scheduler_diagnostics(graph):
    return getattr(graph, schedule_mod.SCHEDULER_BUDGET_DIAGNOSTICS_ATTR)


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


def test_fast_free_schedule_continues_to_higher_count_candidates_when_lowest_count_exceeds_budget():
    graph = Graph()

    first_param = _placeholder(graph, "budget_count_first_param")
    high_param = _placeholder(graph, "budget_count_high_param")
    low_param = _placeholder(graph, "budget_count_low_param")
    extra_param = _placeholder(graph, "budget_count_extra_param")

    first_ag = _allgather(graph, first_param, 100, "budget_count_first", tensor_size=70)
    first_wait = _wait(graph, first_ag, 100, "budget_count_first")

    high_dep = _add(graph, high_param, first_wait, "budget_count_high_dep")
    high_ag = _allgather(graph, high_dep, 101, "budget_count_high", tensor_size=60)
    high_wait = _wait(graph, high_ag, 101, "budget_count_high")
    high_use = _neg(graph, high_wait, "budget_count_high_use")
    high_release = _release(graph, high_use, 101, "budget_count_high")

    low_dep = _add(graph, low_param, first_wait, "budget_count_low_dep")
    low_ag = _allgather(graph, low_dep, 102, "budget_count_low", tensor_size=20)
    low_wait = _wait(graph, low_ag, 102, "budget_count_low")
    extra_dep = _add(graph, extra_param, low_wait, "budget_count_extra_dep")
    extra_ag = _allgather(graph, extra_dep, 103, "budget_count_extra", tensor_size=20)
    extra_wait = _wait(graph, extra_ag, 103, "budget_count_extra")
    low_use = _add(graph, low_wait, extra_wait, "budget_count_low_use")
    low_release = _release(graph, low_use, 102, "budget_count_low")
    extra_release = _release(graph, low_use, 103, "budget_count_extra")

    first_use = _add(graph, first_wait, low_wait, "budget_count_first_use")
    first_release = _release(graph, first_use, 100, "budget_count_first")

    graph.output((first_release, high_release, low_release, extra_release))
    graph.lint()

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=120, source="test")
    names = _scheduled_names(graph, scheduler_budget=budget)

    assert names.index(low_ag.name) < names.index(high_ag.name)


def test_fast_free_schedule_counts_padded_allgather_allocation_bytes():
    graph = Graph()

    param = _placeholder(graph, "budget_padded_param")
    ag = _allgather(graph, param, 110, "budget_padded", tensor_size=102, allocation_size=112)
    wait = _wait(graph, ag, 110, "budget_padded")
    use = _neg(graph, wait, "budget_padded_use")
    release = _release(graph, use, 110, "budget_padded")

    graph.output((release, ))
    graph.lint()

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=105, source="test")
    scheduled_graph = _scheduled_graph(graph, scheduler_budget=budget)
    diagnostics = _scheduler_diagnostics(scheduled_graph)

    assert diagnostics["budget_overflows"][0]["candidate_allgather_bytes"] == 112
    assert diagnostics["budget_overflows"][0]["over_budget_bytes"] == 7
    assert diagnostics["budget_overflows"][0]["path"] == "until_free"


def test_fast_free_schedule_records_diagnostic_when_no_candidate_fits_budget():
    graph = Graph()

    first_param = _placeholder(graph, "budget_fail_first_param")
    high_param = _placeholder(graph, "budget_fail_high_param")
    low_param = _placeholder(graph, "budget_fail_low_param")

    first_ag = _allgather(graph, first_param, 90, "budget_fail_first", tensor_size=80)
    first_wait = _wait(graph, first_ag, 90, "budget_fail_first")

    high_dep = _add(graph, high_param, first_wait, "budget_fail_high_dep")
    high_ag = _allgather(graph, high_dep, 91, "budget_fail_high", tensor_size=40)
    high_wait = _wait(graph, high_ag, 91, "budget_fail_high")
    high_use = _neg(graph, high_wait, "budget_fail_high_use", device_time=1)
    high_release = _release(graph, high_use, 91, "budget_fail_high")

    low_dep = _add(graph, low_param, first_wait, "budget_fail_low_dep")
    low_ag = _allgather(graph, low_dep, 92, "budget_fail_low", tensor_size=30)
    low_wait = _wait(graph, low_ag, 92, "budget_fail_low")
    low_use = _neg(graph, low_wait, "budget_fail_low_use", device_time=100)
    low_release = _release(graph, low_use, 92, "budget_fail_low", n_users=2)

    first_use = _add(graph, first_wait, low_wait, "budget_fail_first_use")
    first_release = _release(graph, first_use, 90, "budget_fail_first")
    low_last_release = _release(graph, first_use, 92, "budget_fail_low_last", n_users=2)

    graph.output((first_release, high_release, low_release, low_last_release))
    graph.lint()

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=100, source="test")
    scheduled_graph = _scheduled_graph(graph, scheduler_budget=budget)
    names = [node.name for node in scheduled_graph.nodes]
    diagnostics = _scheduler_diagnostics(scheduled_graph)

    assert names.index(first_ag.name) < names.index(low_ag.name)
    assert diagnostics["budget_rejections"] > 0
    assert diagnostics["budget_overflows"][0]["source"] == "test"
    assert diagnostics["budget_overflows"][0]["max_gathered_bytes"] == 100
    assert diagnostics["budget_overflows"][0]["over_budget_bytes"] > 0


def test_fast_free_schedule_over_budget_fallback_prefers_lower_peak_before_live_memory():
    graph = Graph()

    first_param = _placeholder(graph, "budget_debt_first_param")
    helper_param = _placeholder(graph, "budget_debt_helper_param")

    first_ag = _allgather(graph, first_param, 120, "budget_debt_first", tensor_size=80)
    first_wait = _wait(graph, first_ag, 120, "budget_debt_first")
    helper_dep = _add(graph, helper_param, first_wait, "budget_debt_helper_dep")
    helper_ag = _allgather(graph, helper_dep, 121, "budget_debt_helper", tensor_size=30)
    helper_wait = _wait(graph, helper_ag, 121, "budget_debt_helper")
    use = _add(graph, first_wait, helper_wait, "budget_debt_use")
    first_release = _release(graph, use, 120, "budget_debt_first")
    helper_release = _release(graph, use, 121, "budget_debt_helper")

    graph.output((first_release, helper_release))
    graph.lint()

    budget = schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=50, source="test")
    scheduled_graph = _scheduled_graph(graph, scheduler_budget=budget)
    diagnostics = _scheduler_diagnostics(scheduled_graph)
    first_selection = diagnostics["selected"][0]

    assert first_selection["path"] == "until_ag"
    assert first_selection["schedule_until_ag_peak_mem"] < first_selection["schedule_until_free_peak_mem"]
    assert first_selection["schedule_until_ag_live_mem"] > first_selection["schedule_until_free_live_mem"]
    assert diagnostics["budget_overflows"][0]["path"] == "until_ag"


def test_over_budget_fallback_prefers_lower_peak_before_ending_residency():
    graph = Graph()
    high_peak_node = _placeholder(graph, "high_peak_zero_residency")
    low_peak_node = _placeholder(graph, "low_peak_nonzero_residency")

    high_peak_task = schedule_mod.AllgatherTask(node=high_peak_node,
                                                allgather_cost=0,
                                                free_cost=0,
                                                allgathered_mem=1000,
                                                allgather_acc_mem=1000,
                                                free_acc_mem=0,
                                                last_use=high_peak_node,
                                                n_scheduled_ags=1,
                                                schedule_until_ag=[high_peak_node],
                                                schedule_until_free=[high_peak_node],
                                                schedule_until_ag_peak_mem=1000,
                                                schedule_until_free_peak_mem=1000,
                                                schedule_until_ag_live_mem=1000,
                                                schedule_until_free_live_mem=0)
    low_peak_task = schedule_mod.AllgatherTask(node=low_peak_node,
                                               allgather_cost=0,
                                               free_cost=0,
                                               allgathered_mem=51,
                                               allgather_acc_mem=51,
                                               free_acc_mem=0,
                                               last_use=low_peak_node,
                                               n_scheduled_ags=1,
                                               schedule_until_ag=[low_peak_node],
                                               schedule_until_free=[low_peak_node],
                                               schedule_until_ag_peak_mem=51,
                                               schedule_until_free_peak_mem=51,
                                               schedule_until_ag_live_mem=51,
                                               schedule_until_free_live_mem=51)

    selected, _ = schedule_mod._select_over_budget_allgather_task([high_peak_task, low_peak_task],
                                                                  schedule_mod.SchedulerMemoryBudget(
                                                                      max_gathered_bytes=50, source="test"))

    assert selected is low_peak_task


def test_candidate_peak_resets_after_overflow_is_released():
    graph = Graph()
    first_param = _placeholder(graph, "historical_overflow_param")
    next_param = _placeholder(graph, "later_fitting_param")
    first_ag = _allgather(graph, first_param, 130, "historical_overflow", allocation_size=100)
    first_release = _release(graph, first_ag, 130, "historical_overflow")
    next_ag = _allgather(graph, next_param, 131, "later_fitting", allocation_size=40)

    tracker = schedule_mod._GatheredParamTracker({130: 1, 131: 1})
    tracker.apply(first_ag)
    tracker.apply(first_release)
    assert tracker.live_bytes == 0
    assert tracker.peak_bytes == 100

    candidate_peak, candidate_live = schedule_mod._simulate_path_stats(tracker, [next_ag])

    assert candidate_peak == 40
    assert candidate_live == 40
    assert schedule_mod._fits_budget(schedule_mod.SchedulerMemoryBudget(max_gathered_bytes=50, source="test"),
                                     candidate_peak)
    assert tracker.live_bytes == 0
    assert tracker.peak_bytes == 100


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
