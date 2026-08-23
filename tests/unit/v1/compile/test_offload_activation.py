# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import deepspeed.compile.passes.offload_activation as offload_pass
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.torch import required_torch_version

from unit.common import DistributedTest
from unit.util import bf16_required_version_check, skip_on_arch
from unit.v1.compile.util import compare_loss

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.6),
                                reason="DeepCompile requires Pytorch version 2.6 or above")

# 8M floats = 32MB, comfortably above the pass's own 10MB minimum.
LARGE_NUMEL = 8 * 1024 * 1024
LARGE_SIZE = LARGE_NUMEL * 4


@pytest.fixture(autouse=True)
def _reset_offload_pass_globals():
    # The plan lives in module globals; reset it so tests pass in any order.
    yield
    offload_pass._offload_plans.clear()
    offload_pass.reset_offload_activation_stats()
    offload_pass._h2d_bytes_per_sec = None


def _ensure_dc_ops():
    # The ops come from the compiled extension; only their Meta kernels are added from Python.
    from deepspeed.compile.util import is_deepcompile_supported

    if not is_deepcompile_supported():
        pytest.skip("DeepCompile is not supported in this environment")
    try:
        from deepspeed.compile.util import get_deepcompile_handle
        get_deepcompile_handle()
    except Exception as e:
        pytest.skip(f"DeepCompile extension is not loadable here: {e}")
    offload_pass.register_activation_offload_ops()


def _meta_tensor(numel):
    return torch.empty(numel, device="meta")


def _add_node(graph, target, args, name, numel):
    node = graph.create_node('call_function', target, args, {}, name=name)
    node.meta["val"] = _meta_tensor(numel)
    return node


def _make_gm(graph):
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _make_fwd_graph(saved_numels=(LARGE_NUMEL, LARGE_NUMEL)):
    """A forward graph shaped like a partitioned one: the caller's output first, then saved values.

    x -> act_0 -> act_1 -> out. The activations are also returned so the backward pass can read
    them; `out` stands for the value the caller receives and must never be offloaded.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = _meta_tensor(16)

    saved = []
    current = x
    for i, numel in enumerate(saved_numels):
        current = _add_node(graph, torch.relu, (current, ), f"act_{i}", numel)
        saved.append(current)

    out = _add_node(graph, torch.sum, (current, ), "out", 1)
    graph.output((out, *saved))
    return graph


def _make_profile(graph, peak=10**9, num_fwd_outputs=1):
    mem = [(node.name, peak, 0, peak) for node in graph.nodes]
    return SimpleNamespace(num_fwd_outputs=num_fwd_outputs, fwd_mem=mem, bwd_mem=[], bwd_time=[])


def _call(pass_fn, gm, profile, bwd, param_manager=None, graph_id=0):
    return pass_fn(gm,
                   graph_id, [(graph_id, True)], {graph_id: profile},
                   lambda: (),
                   0.0,
                   param_manager if param_manager is not None else {},
                   bwd=bwd)


def _pin_observed_floor(monkeypatch, observed=0):
    """Hold the observed-memory clamp still.

    The planner takes the larger of its profiled floor and what the allocator has already handed
    out. That second term is live state, so without pinning it a test asserting an exact plan size
    passes or fails depending on what else has run in the process.
    """
    monkeypatch.setattr(offload_pass, "_get_mem_usage_out_of_torch", lambda: 0)
    real = offload_pass.get_accelerator()
    monkeypatch.setattr(
        offload_pass, "get_accelerator", lambda: SimpleNamespace(max_memory_allocated=lambda: observed,
                                                                 current_device=real.current_device,
                                                                 total_memory=real.total_memory,
                                                                 memory_reserved=lambda: 0,
                                                                 memory_allocated=lambda: 0))


def _run_pass(gm, profile, bwd, param_manager=None, graph_id=0):
    """Drive both halves the way the schedule does: move everything, then keep what fits.

    The floor pass is profiled by the caller in production; here the profile handed to the planner
    stands in for that measurement.
    """
    if bwd:
        # The floor half owns the backward now: it inserts a copy back for everything it moved out,
        # so the steps before the planner runs at WARMUP execute a complete configuration.
        return _call(offload_pass.offload_activation_floor, gm, profile, True, param_manager, graph_id)

    _call(offload_pass.offload_activation_floor, gm, profile, False, param_manager, graph_id)
    return _call(offload_pass.offload_activation, gm, profile, False, param_manager, graph_id)


def _node_names(graph):
    return [node.name for node in graph.nodes]


def _node_by_name(graph, name):
    return next(node for node in graph.nodes if node.name == name)


@pytest.fixture
def forced_budget(monkeypatch):
    # A budget far below the profiled peak forces every candidate out.
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_BUDGET_GB", "1e-6")


@pytest.fixture
def ample_budget(monkeypatch):
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_BUDGET_GB", "100")
    # A budget alone does not leave the planner any room: it takes the larger of its profiled floor
    # and what the device has already handed out, so on a shared accelerator the observed term can
    # exceed the budget and force everything out. Pin it so "ample" means ample.
    _pin_observed_floor(monkeypatch)


def test_margin_is_measured_not_guessed(monkeypatch):
    # The margin stands for memory the allocator holds without using. A flat tenth of an H200 is
    # 14GiB, which at the sequence lengths that need this pass is most of the room the planner has
    # to give activations back with, so it is read from the allocator instead.
    _ensure_dc_ops()
    total = 100 * 1024**3
    accelerator = SimpleNamespace(total_memory=lambda: total,
                                  memory_reserved=lambda: 42 * 1024**3,
                                  memory_allocated=lambda: 40 * 1024**3)

    # A reading below the calibrated floor does not lower the margin. This is the whole safety
    # property: the quantity measured here is allocator slack at pass time, which is near zero
    # because the memory-heavy phase has not run, while the margin has to cover peak-time
    # fragmentation and error in the floor profile. Reading 2% and reserving 2% is what put a
    # seq4096 plan at 99% of the card.
    assert offload_pass._measured_margin(accelerator) == offload_pass.MIN_MEASURED_MARGIN

    # An unreadable or suspiciously perfect allocator lands on the same floor.
    idle = SimpleNamespace(total_memory=lambda: total, memory_reserved=lambda: 0, memory_allocated=lambda: 0)
    assert offload_pass._measured_margin(idle) == offload_pass.MIN_MEASURED_MARGIN

    # An allocator genuinely holding more than the floor raises the margin -- the measurement is
    # allowed to act, but only in the direction that reserves more.
    fat = SimpleNamespace(total_memory=lambda: total,
                          memory_reserved=lambda: 55 * 1024**3,
                          memory_allocated=lambda: 40 * 1024**3)
    assert offload_pass._measured_margin(fat) == pytest.approx(0.15)
    assert offload_pass._measured_margin(fat) > offload_pass.MIN_MEASURED_MARGIN

    # And a badly fragmented one cannot reserve the whole card away from the planner.
    fragmented = SimpleNamespace(total_memory=lambda: total,
                                 memory_reserved=lambda: 90 * 1024**3,
                                 memory_allocated=lambda: 0)
    assert offload_pass._measured_margin(fragmented) == offload_pass.MAX_MEASURED_MARGIN

    # The clamp must not collapse to a point, or _measured_margin becomes a constant function
    # wearing the appearance of a measurement.
    assert offload_pass.MAX_MEASURED_MARGIN > offload_pass.MIN_MEASURED_MARGIN


def test_floor_half_is_runnable_on_its_own(monkeypatch):
    """Everything the floor moves out in the forward is copied back in the backward.

    The planner runs at WARMUP, so the steps before it execute what the floor produced and nothing
    else. If the floor offloaded without reloading, those steps would hand a host buffer to a CUDA
    matmul -- which is the failure the split schedule would otherwise have introduced.
    """
    _ensure_dc_ops()
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_MIN_SIZE_MB", "0")
    fwd_graph = _make_fwd_graph()
    fwd_gm = _make_gm(fwd_graph)
    _call(offload_pass.offload_activation_floor, fwd_gm, _make_profile(fwd_graph), False, None, 0)

    offloaded = [n for n in fwd_gm.graph.nodes if n.name.startswith("wait_offload_")]
    assert offloaded, "the floor half moved nothing out"

    bwd_graph = _make_bwd_graph()
    bwd_gm = _make_gm(bwd_graph)
    _call(offload_pass.offload_activation_floor, bwd_gm, _make_bwd_profile(bwd_graph), True, None, 0)

    reloaded = {n.name[len("wait_reload_"):] for n in bwd_gm.graph.nodes if n.name.startswith("wait_reload_")}
    moved = {n.name[len("wait_offload_"):] for n in offloaded}
    assert moved <= reloaded, f"offloaded without a copy back: {sorted(moved - reloaded)}"


def test_planner_drops_the_reload_for_what_it_keeps_resident(monkeypatch):
    """Bringing an activation back in the forward must remove its copy back in the backward.

    The backward placeholder now carries the tensor itself rather than a host buffer, so a reload
    left behind would copy from a buffer nothing wrote.
    """
    _ensure_dc_ops()
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_MIN_SIZE_MB", "0")
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_BUDGET_GB", "1000")

    fwd_graph = _make_fwd_graph()
    fwd_gm = _make_gm(fwd_graph)
    _call(offload_pass.offload_activation_floor, fwd_gm, _make_profile(fwd_graph), False, None, 0)

    bwd_graph = _make_bwd_graph()
    bwd_gm = _make_gm(bwd_graph)
    _call(offload_pass.offload_activation_floor, bwd_gm, _make_bwd_profile(bwd_graph), True, None, 0)
    before = sum(1 for n in bwd_gm.graph.nodes if n.name.startswith("wait_reload_"))
    assert before > 0

    # A budget with room to spare: the planner brings everything back, so every reload must go.
    _call(offload_pass.offload_activation, fwd_gm, _make_profile(fwd_graph, peak=10**6), False, None, 0)
    _call(offload_pass.offload_activation, bwd_gm, _make_bwd_profile(bwd_graph), True, None, 0)

    after = sum(1 for n in bwd_gm.graph.nodes if n.name.startswith("wait_reload_"))
    assert after == 0, f"{after} reloads left for activations that were kept resident"


def test_floor_never_reads_below_what_was_already_observed(monkeypatch):
    """The profile covers the compiled forward graph; work outside it is invisible to the floor.

    A tiled loss drives a nested backward the profile never sees, so the floor can read below the
    run's real peak. What the allocator has already handed out cannot.
    """
    _ensure_dc_ops()
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_MIN_SIZE_MB", "0")
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_BUDGET_GB", "1000")
    monkeypatch.setattr(offload_pass, "_get_mem_usage_out_of_torch", lambda: 0)

    # The observed high-water mark reaches the budget, so a floor clamped to it leaves no headroom.
    # The profiled floor is tiny, so without the clamp the planner would think it had the whole
    # budget to spend -- which is the seq4096 failure in miniature.
    observed = 1000 * 10**9
    monkeypatch.setattr(
        offload_pass, "get_accelerator", lambda: SimpleNamespace(max_memory_allocated=lambda: observed,
                                                                 current_device=lambda: "cpu",
                                                                 total_memory=lambda: 1000 * 10**9,
                                                                 memory_reserved=lambda: 0,
                                                                 memory_allocated=lambda: 0))
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    assert _run_pass(gm, _make_profile(graph, peak=10**6), bwd=False) is None
    starved = offload_pass.get_offload_activation_stats()["offload_nodes"]

    # Same graph and budget, but nothing observed: the profile is all there is, so the planner is
    # free to bring activations back.
    offload_pass.reset_offload_activation_stats()
    offload_pass._offload_plans.clear()
    monkeypatch.setattr(
        offload_pass, "get_accelerator", lambda: SimpleNamespace(max_memory_allocated=lambda: 0,
                                                                 current_device=lambda: "cpu",
                                                                 total_memory=lambda: 1000 * 10**9,
                                                                 memory_reserved=lambda: 0,
                                                                 memory_allocated=lambda: 0))
    graph2 = _make_fwd_graph()
    gm2 = _make_gm(graph2)
    assert _run_pass(gm2, _make_profile(graph2, peak=10**6), bwd=False) is None
    generous = offload_pass.get_offload_activation_stats()["offload_nodes"]

    assert starved > generous, (
        "a floor below the observed high-water mark must not license bringing activations back: "
        f"observed-clamped left {starved} offloaded, unclamped left {generous}")


def test_floor_peak_agrees_across_ranks(monkeypatch):
    # Every rank profiles its own device, and the out-of-torch term behind that profile reads NVML
    # for that one device, so no two ranks measure the same floor. The planner spends the gap
    # between the floor and the budget, so a rank that read a higher floor hands back more
    # activations -- on the device that could least afford them. The busiest rank decides, the same
    # way the smallest device decides the budget.
    _ensure_dc_ops()
    reduce_ops = []

    class _FakeDist:
        ReduceOp = SimpleNamespace(MAX="max")

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(tensor, op):
            reduce_ops.append(op)
            # Stand in for the rank that measured the highest floor.
            tensor.fill_(900.0)

    monkeypatch.setattr(offload_pass, "dist", _FakeDist)
    monkeypatch.setattr(offload_pass, "get_accelerator", lambda: SimpleNamespace(current_device=lambda: "cpu"))

    assert offload_pass._agree_on_floor_peak(100) == pytest.approx(900.0)
    assert reduce_ops == ["max"], "the floor must be reduced with MAX, not averaged or left local"


def test_floor_peak_without_distributed_is_local(monkeypatch):
    _ensure_dc_ops()
    monkeypatch.setattr(offload_pass, "dist", SimpleNamespace(is_initialized=lambda: False))

    assert offload_pass._agree_on_floor_peak(1234) == pytest.approx(1234.0)


def test_margin_override_wins(monkeypatch):
    _ensure_dc_ops()
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_MARGIN", "0.05")
    accelerator = SimpleNamespace(total_memory=lambda: 100 * 1024**3,
                                  memory_reserved=lambda: 90 * 1024**3,
                                  memory_allocated=lambda: 0)

    assert offload_pass._measured_margin(accelerator) == pytest.approx(0.05)


def test_register_ops_adds_meta_kernels():
    _ensure_dc_ops()

    for name in offload_pass._ACTIVATION_OFFLOAD_OPS:
        assert torch._C._dispatch_has_kernel_for_dispatch_key(f"dc::{name}", "Meta"), \
            f"dc::{name} has no Meta kernel, so a graph containing it cannot be traced"

    lib_first = offload_pass._activation_ops_lib
    offload_pass.register_activation_offload_ops()
    assert offload_pass._activation_ops_lib is lib_first


def test_fwd_offloads_saved_activations(forced_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)

    # The pass rewrites the graph in place and returns None, which tells the caller not to replay
    # the graph for a fresh memory profile. Nothing in this schedule reads that profile, and the
    # replay is what ran out of memory on a run that was already at the wall.
    assert _run_pass(gm, _make_profile(graph), bwd=False) is None

    names = _node_names(graph)
    assert [n for n in names if n.startswith("offload_act_")] == ["offload_act_0", "offload_act_1"]

    # The backward pass receives the host buffers; the value the caller receives is untouched.
    output_args = _node_by_name(graph, "output").args[0]
    assert output_args[0].name == "out"
    assert [node.name for node in output_args[1:]] == ["wait_offload_act_0", "wait_offload_act_1"]

    # Each copy starts right after the tensor's last use...
    assert names.index("offload_act_0") == names.index("act_1") + 1
    assert names.index("offload_act_1") == names.index("out") + 1
    # ...and each wait immediately follows its own copy. That placement is load-bearing: it ends the
    # copy before anything else runs, so the compiler is free to recycle the buffer afterwards.
    # Letting the copy run on while other work proceeds would need the buffer marked never-reused,
    # and under inductor that keeps it alive for the whole forward -- no memory is released at all.
    for activation in ("act_0", "act_1"):
        assert names.index(f"wait_offload_{activation}") == names.index(f"offload_{activation}") + 1

    assert offload_pass.get_offload_activation_stats()["offload_nodes"] == 2
    graph.lint()


def test_pass_does_not_ask_for_a_reprofiling_replay(forced_budget):
    # DeepCompile replays a graph after any pass that returns it, to measure the result. That replay
    # runs at the memory the pass was called in to relieve: on a run near the wall it ran out of
    # memory on one rank, whose profiler then dropped out of the per-node collectives while the
    # other ranks waited, and the job died half an hour later on a collective timeout. This pass
    # runs last and nothing reads its profile, so it declines the replay and recompiles itself.
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)

    assert _run_pass(gm, _make_profile(graph), bwd=False) is None
    assert [n for n in _node_names(graph) if n.startswith("offload_")], "the graph was not rewritten"
    assert "offload_tensor" in gm.code, "the rewritten graph was not recompiled into the module"


def test_fwd_moves_everything_when_the_profile_is_missing(forced_budget):
    # Even the floor could not be profiled, which means memory ran out with every activation already
    # on the host. Nothing can be brought back on that evidence, so they all stay moved.
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    profile = _make_profile(graph)
    profile.fwd_mem = []

    _run_pass(gm, profile, bwd=False)

    assert [n for n in _node_names(graph) if n.startswith("offload_")] == ["offload_act_0", "offload_act_1"]


def test_fwd_skips_parameters(forced_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    # ZeRO already manages parameters; moving one here would fight it.
    param_manager = {0: SimpleNamespace(param_names=["act_0"])}

    _run_pass(gm, _make_profile(graph), bwd=False, param_manager=param_manager)

    assert [n for n in _node_names(graph) if n.startswith("offload_")] == ["offload_act_1"]


def test_fwd_skips_values_the_caller_also_receives(forced_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    # act_1 is returned to the caller as well as saved: offloading it would hand the caller a host
    # tensor instead of the device tensor it expects.
    output_node = _node_by_name(graph, "output")
    saved = output_node.args[0][1:]
    output_node.args = ((_node_by_name(graph, "act_1"), *saved), )

    _run_pass(gm, _make_profile(graph), bwd=False)

    assert [n for n in _node_names(graph) if n.startswith("offload_")] == ["offload_act_0"]


def test_fwd_skips_values_saved_twice(forced_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    # The backward graph would receive such a value as two placeholders under two names, and only
    # the one named after this node would be reloaded; the other would keep the host buffer.
    output_node = _node_by_name(graph, "output")
    out, act_0, act_1 = output_node.args[0]
    output_node.args = ((out, act_0, act_1, act_1), )

    _run_pass(gm, _make_profile(graph), bwd=False)

    assert [n for n in _node_names(graph) if n.startswith("offload_")] == ["offload_act_0"]


def test_fwd_skips_values_that_alias_another_tensor(forced_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    # A view shares the storage of the tensor it came from, so copying it out frees nothing while
    # that tensor is still live, and releasing the storage would break its other readers.
    _node_by_name(graph, "act_0").target = torch.ops.aten.view.default

    _run_pass(gm, _make_profile(graph), bwd=False)

    assert [n for n in _node_names(graph) if n.startswith("offload_")] == ["offload_act_1"]


def test_fwd_skips_values_already_on_the_host(forced_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    # Nothing to move, and the copy op holds its input through record_stream, which a host tensor
    # does not support. Graphs do carry such values: small index and mask tensors stay on the host.
    _node_by_name(graph, "act_0").meta["val"] = torch.empty(LARGE_NUMEL, device="cpu")

    _run_pass(gm, _make_profile(graph), bwd=False)

    assert [n for n in _node_names(graph) if n.startswith("offload_")] == ["offload_act_1"]


def test_fwd_skips_values_that_are_not_activations(forced_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    # Attention saves its random-number state for the backward pass as an integer tensor that lives
    # on the host, while its traced metadata claims the accelerator (an op's outputs take their
    # device from its inputs). Copying it out crashes, so dtype, not device, is the test.
    _node_by_name(graph, "act_0").meta["val"] = torch.empty(LARGE_NUMEL, dtype=torch.int64, device="meta")

    _run_pass(gm, _make_profile(graph), bwd=False)

    assert [n for n in _node_names(graph) if n.startswith("offload_")] == ["offload_act_1"]


def test_fwd_skips_small_tensors(forced_budget):
    _ensure_dc_ops()
    # 1024 floats is far below the 10MB threshold.
    graph = _make_fwd_graph(saved_numels=(1024, 1024))
    gm = _make_gm(graph)

    assert _run_pass(gm, _make_profile(graph), bwd=False) is None
    assert not [n for n in _node_names(graph) if n.startswith("offload_")]


def test_fwd_skips_symbolic_shapes(forced_budget):
    _ensure_dc_ops()
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    try:
        with FakeTensorMode(shape_env=shape_env):
            symbolic_val = torch.empty(shape_env.create_unbacked_symint(), 1024)
    except Exception as e:
        pytest.skip(f"cannot build a symbolic-shape fake tensor here: {e}")

    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    # A host buffer is allocated once per value and reused every step, so a shape that can change
    # between steps would be copied into a buffer of the wrong size.
    _node_by_name(graph, "act_0").meta["val"] = symbolic_val

    _run_pass(gm, _make_profile(graph), bwd=False)

    assert [n for n in _node_names(graph) if n.startswith("offload_")] == ["offload_act_1"]


def test_fwd_no_op_when_memory_fits(ample_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)

    assert _run_pass(gm, _make_profile(graph, peak=10**9), bwd=False) is None
    assert not [n for n in _node_names(graph) if n.startswith("offload_")]
    assert offload_pass.get_offload_activation_stats()["offload_nodes"] == 0


def test_fwd_keeps_resident_only_what_the_budget_has_room_for(monkeypatch):
    _pin_observed_floor(monkeypatch)
    # The profile the planner reads is the floor: taken with everything already moved out. Headroom
    # between that floor and the budget is what can come back. 40MB of room fits one 32MB
    # activation, so exactly one returns to the device and the other stays on the host.
    _ensure_dc_ops()
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_BUDGET_GB", "1")
    graph = _make_fwd_graph()
    gm = _make_gm(graph)
    floor_peak = int(1e9) - 40 * 1024 * 1024

    _run_pass(gm, _make_profile(graph, peak=floor_peak), bwd=False)

    assert len([n for n in _node_names(graph) if n.startswith("offload_")]) == 1


def test_fwd_without_partition_info_is_skipped(forced_budget):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    gm = _make_gm(graph)

    assert _run_pass(gm, _make_profile(graph, num_fwd_outputs=None), bwd=False) is None
    assert not [n for n in _node_names(graph) if n.startswith("offload_")]


def test_fwd_rerun_replans_instead_of_stacking(forced_budget):
    _ensure_dc_ops()
    first_graph = _make_fwd_graph()
    _run_pass(_make_gm(first_graph), _make_profile(first_graph), bwd=False)
    assert len(offload_pass._offload_plans[0]) == 2

    # A later compile phase runs the pass again from the original graph.
    second_graph = _make_fwd_graph()
    _run_pass(_make_gm(second_graph), _make_profile(second_graph), bwd=False)

    assert len(offload_pass._offload_plans[0]) == 2
    assert len([n for n in _node_names(second_graph) if n.startswith("offload_")]) == 2


def _make_bwd_graph(activation_names=("act_0", "act_1"), num_compute_nodes=6):
    """A backward graph that reads the saved activations late, leaving room to start copies early."""
    graph = torch.fx.Graph()
    grad = graph.placeholder("grad")
    grad.meta["val"] = _meta_tensor(16)

    placeholders = {}
    for name in activation_names:
        node = graph.placeholder(name)
        node.meta["val"] = _meta_tensor(LARGE_NUMEL)
        placeholders[name] = node

    current = grad
    for i in range(num_compute_nodes):
        current = _add_node(graph, torch.relu, (current, ), f"bwd_{i}", 16)

    # The activations are read at the end, in reverse order of the forward pass.
    for name in reversed(activation_names):
        current = _add_node(graph, torch.add, (current, placeholders[name]), f"use_{name}", 16)

    graph.output((current, ))
    return graph


def _make_bwd_profile(graph, peak=0, node_time_ms=1.0):
    return SimpleNamespace(num_fwd_outputs=1,
                           fwd_mem=[],
                           bwd_mem=[(node.name, peak, 0, peak) for node in graph.nodes],
                           bwd_time=[(node.name, node_time_ms, node_time_ms) for node in graph.nodes])


def _plan_for(names, graph_id=0, size=LARGE_SIZE):
    offload_pass._offload_plans[graph_id] = OrderedDict((name, (index + 1, size)) for index, name in enumerate(names))


def test_bwd_reloads_before_first_use_and_rewrites_readers(forced_budget, monkeypatch):
    _ensure_dc_ops()
    monkeypatch.setattr(offload_pass, "_h2d_bandwidth", lambda: 10e9)
    graph = _make_bwd_graph()
    gm = _make_gm(graph)
    _plan_for(["act_0", "act_1"])

    assert _run_pass(gm, _make_bwd_profile(graph), bwd=True) is None

    names = _node_names(graph)
    for activation in ("act_0", "act_1"):
        assert f"reload_{activation}" in names
        assert names.index(f"wait_reload_{activation}") == names.index(f"use_{activation}") - 1
        # The reader takes the reloaded tensor, not the host buffer it replaced.
        assert _node_by_name(graph, f"use_{activation}").args[1].name == f"wait_reload_{activation}"

    assert offload_pass.get_offload_activation_stats()["reload_nodes"] == 2
    graph.lint()


def test_bwd_starts_the_copy_early_enough_to_hide_it(ample_budget, monkeypatch):
    _ensure_dc_ops()
    monkeypatch.setattr(offload_pass, "_h2d_bandwidth", lambda: 10e9)
    graph = _make_bwd_graph()
    gm = _make_gm(graph)
    _plan_for(["act_1"])

    _run_pass(gm, _make_bwd_profile(graph, node_time_ms=1.0), bwd=True)

    names = _node_names(graph)
    between = names[names.index("reload_act_1") + 1:names.index("use_act_1")]
    # 32MB at 10GB/s takes 3.2ms, so the copy starts four 1ms nodes ahead of its reader.
    assert [n for n in between if n.startswith("bwd_")] == ["bwd_2", "bwd_3", "bwd_4", "bwd_5"]


def test_bwd_keeps_the_copy_late_when_memory_is_tight(monkeypatch):
    _ensure_dc_ops()
    monkeypatch.setattr(offload_pass, "_h2d_bandwidth", lambda: 10e9)
    monkeypatch.setenv("DS_DC_OFFLOAD_ACT_BUDGET_GB", "0.05")
    graph = _make_bwd_graph()
    gm = _make_gm(graph)
    _plan_for(["act_1"])

    # Every node already sits at 40MB of the 50MB budget, so a 32MB reload only fits immediately
    # before its reader.
    _run_pass(gm, _make_bwd_profile(graph, peak=40 * 1024 * 1024, node_time_ms=1.0), bwd=True)

    names = _node_names(graph)
    assert names.index("use_act_1") - names.index("reload_act_1") == 2


def test_bwd_reloads_just_in_time_when_the_profile_is_missing(forced_budget, monkeypatch):
    # Without a backward profile every node reads as using no memory, so the headroom check cannot
    # refuse anything and every copy back is hoisted as early as it will go. The backward pass then
    # holds everything the forward pass moved out. Measured: the same plan that completes with the
    # copies coming back one at a time dies when they are all hoisted.
    _ensure_dc_ops()
    monkeypatch.setattr(offload_pass, "_h2d_bandwidth", lambda: 10e9)
    graph = _make_bwd_graph()
    gm = _make_gm(graph)
    _plan_for(["act_1"])
    profile = _make_bwd_profile(graph)
    profile.bwd_mem = []

    _run_pass(gm, profile, bwd=True)

    names = _node_names(graph)
    # Immediately before its reader, not hoisted above the compute in between.
    assert names.index("use_act_1") - names.index("reload_act_1") == 2


def test_bwd_without_a_plan_is_skipped(forced_budget):
    _ensure_dc_ops()
    graph = _make_bwd_graph()
    gm = _make_gm(graph)

    assert _run_pass(gm, _make_bwd_profile(graph), bwd=True) is None
    assert not [n for n in _node_names(graph) if n.startswith("reload_")]


def test_rejects_other_offload_targets():
    # Each offload pass plans against the whole memory budget on its own, so two of them together
    # move far more data than the run needs. The check runs before init_z3 removes any hooks, which
    # is why a bare stub engine reaches it.
    from deepspeed.compile.init_z3 import init_z3

    engine = SimpleNamespace(zero_use_cpu_optimizer=lambda: False)
    for conflicting in ("offload_parameters", "offload_opt_states"):
        compile_config = SimpleNamespace(offload_activation=True, offload_parameters=False, offload_opt_states=False)
        setattr(compile_config, conflicting, True)

        with pytest.raises(ValueError, match="offload_activation"):
            init_z3(engine, "inductor", compile_config, {})


class TestOffloadActivation(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    @pytest.mark.parametrize('gradient_accumulation_steps', [1, 2])
    def test_offload_activation_correctness(self, dtype, gradient_accumulation_steps):
        from deepspeed.compile.util import is_deepcompile_supported

        skip_on_arch(min_arch=8)
        if not bf16_required_version_check():
            pytest.skip(
                "DeepSpeed BFloat16 tests need NCCL >= 2.10.3, CUDA >=11.0, and HW support for BFloat16 to run correctly"
            )
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU does not support this test yet")
        if not is_deepcompile_supported():
            pytest.skip("DeepCompile is not supported in this environment")

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": 3
            },
            "compile": {
                "deepcompile": True,
                "offload_activation": True
            },
            "bf16": {
                "enabled": True
            },
        }

        # Same configuration with offloading off: isolates the pass, so a missed stream ordering
        # shows up as loss drift far below compare_loss's cross-stage tolerance.
        config_no_offload = deepcopy(config)
        config_no_offload["compile"]["offload_activation"] = False
        # A wider model gives inductor buffers worth reusing. Reuse of a buffer whose copy out is
        # still in flight corrupts the activation silently, so the arithmetic check below is only
        # as good as the reuse pressure the model creates.
        iteration = 8 * gradient_accumulation_steps
        losses_no_offload = compare_loss(self, config_no_offload, dtype, iteration=iteration, hidden_dim_override=512)

        # This model's activations are far below the real size threshold and its memory nowhere
        # near any real budget, so force both.
        os.environ["DS_DC_OFFLOAD_ACT_BUDGET_GB"] = "0.000001"
        os.environ["DS_DC_OFFLOAD_ACT_MIN_SIZE_MB"] = "0"
        try:
            offload_pass.reset_offload_activation_stats()
            # The pass engages at the WARMUP phase, so 8 optimizer steps give several offloaded ones.
            losses_offload = compare_loss(self, config, dtype, iteration=iteration, hidden_dim_override=512)
        finally:
            del os.environ["DS_DC_OFFLOAD_ACT_BUDGET_GB"]
            del os.environ["DS_DC_OFFLOAD_ACT_MIN_SIZE_MB"]

        stats = offload_pass.get_offload_activation_stats()
        assert stats["offload_nodes"] > 0, "no activation was offloaded"
        # offload_nodes counts what was ever built, so it stays positive even if a later rebuild
        # dropped every offload -- exactly what a broken WARMUP schedule did once, leaving this test
        # green while the pass had stopped offloading anything. planned_offloads is the end state.
        assert stats["planned_offloads"] > 0, \
            "the last plan left nothing offloaded; the pass built offload nodes and then lost them"
        # Activations that never come back would be a silent failure of the backward half.
        assert stats["reload_nodes"] > 0, "offloaded activations were never reloaded"

        # Both runs are identically seeded, so moving activations must not change the arithmetic.
        for step, (ref, got) in enumerate(zip(losses_no_offload, losses_offload)):
            assert got == pytest.approx(ref, rel=1e-4, abs=1e-5), \
                f"offloading changed the loss at step {step}: {ref} vs {got}"
