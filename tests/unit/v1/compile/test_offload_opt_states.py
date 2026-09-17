# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import deepspeed.compile.passes.offload_adam_states as offload_pass
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_states import _make_offload_state_key
from deepspeed.utils.torch import required_torch_version

from unit.common import DistributedTest
from unit.util import bf16_required_version_check, skip_on_arch
from unit.v1.compile.util import compare_loss

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=2.6),
                                reason="DeepCompile requires Pytorch version 2.6 or above")


@pytest.fixture(autouse=True)
def _reset_offload_pass_globals():
    # Planning state lives in module globals; reset it so tests pass in any order.
    yield
    offload_pass.offload_tasks.clear()
    offload_pass.offload_tasks_scheduled.clear()
    offload_pass.offload_tasks_inserted = 0
    offload_pass.reload_tasks_remaining = []
    offload_pass.total_reload_mem = 0
    offload_pass._op_task_registry.clear()
    offload_pass._empty_cache_pending = False
    offload_pass.reset_offload_op_stats()


def _ensure_dc_ops():
    # Registration only needs torch.library, so these tests run the same way with or without the
    # compiled extension present.
    offload_pass.register_offload_ops()


def _make_fake_optimizer():
    param_key = torch.zeros(4)
    state = {
        _make_offload_state_key("exp_avg"): torch.zeros(8),
        _make_offload_state_key("exp_avg_sq"): torch.zeros(8),
    }
    return SimpleNamespace(state={param_key: state},
                           fp32_partitioned_groups_flat=[torch.zeros(8)],
                           hp_params_pin_buffers=[torch.zeros(8)])


def _make_fwd_graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(4)
    a = graph.call_function(torch.relu, (x, ))
    b = graph.call_function(torch.relu, (a, ))
    graph.output(b)
    return graph


def _mem_rows(graph):
    return [(node.name, 100, 0, 100) for node in graph.nodes]


def _run_fwd_pass(monkeypatch, graph, budget_gb="0"):
    monkeypatch.setattr(offload_pass.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(offload_pass, "offload_adam_states_sync", lambda: None)
    monkeypatch.setattr(offload_pass, "reload_adam_states_sync", lambda: None)
    monkeypatch.setattr(offload_pass, "sync_reload_states", lambda: None)
    monkeypatch.setattr(offload_pass, "optimizer", _make_fake_optimizer())
    # The default zero budget forces every task to be scheduled at the first compute node.
    monkeypatch.setenv("DS_DC_OFFLOAD_OPT_BUDGET_GB", budget_gb)
    prof = SimpleNamespace(fwd_mem=_mem_rows(graph), bwd_mem=[])
    return offload_pass.offload_opt_states_inc(graph, 0, [(0, True)], {0: prof}, 0.0, None, bwd=False)


@pytest.mark.parametrize("offload_device,rejected", [("cpu", True), ("nvme", True), ("none", False), ("unset", False)])
def test_rejects_zero_optimizer_offload(offload_device, rejected):
    # ZeRO's own optimizer offload keeps the state off the accelerator for the whole step and runs
    # the optimizer there; this pass keeps it resident when memory allows. Both owning the same
    # state silently produces wrong behaviour, so the combination is refused up front -- before
    # init_z3 removes any hooks, which is why a bare stub engine reaches the check. An offload
    # section that is present but inert must still be accepted: `offload_optimizer: {}` takes the
    # default device "none", so only cpu and nvme actually move the state.
    from deepspeed.compile.init_z3 import init_z3

    engine = SimpleNamespace(zero_use_cpu_optimizer=lambda: offload_device in ("cpu", "nvme"))
    compile_config = SimpleNamespace(offload_opt_states=True)

    if rejected:
        with pytest.raises(ValueError, match="offload_optimizer"):
            init_z3(engine, "inductor", compile_config, {})
    else:
        # The stub has nothing past the check, so reaching anything else proves it was accepted.
        with pytest.raises(AttributeError) as excinfo:
            init_z3(engine, "inductor", compile_config, {})
        assert "offload_optimizer" not in str(excinfo.value)


def test_register_offload_ops_idempotent():
    _ensure_dc_ops()
    lib_first = offload_pass._offload_ops_lib
    offload_pass.register_offload_ops()
    assert offload_pass._offload_ops_lib is lib_first

    for name, _, _ in offload_pass._OFFLOAD_OP_SPECS:
        overload = getattr(torch.ops.dc, name).default
        assert overload is not None
        assert overload in torch.fx.node._side_effectful_functions


def _registered_effect(overload):
    # torch moved this registry: up to 2.7 it is a SIDE_EFFECTS dict, and later versions expose a
    # _get_effect accessor backed by the library registry instead.
    from torch._higher_order_ops import effects

    if hasattr(effects, "_get_effect"):
        return effects._get_effect(overload)
    return effects.SIDE_EFFECTS.get(overload)


def test_offload_ops_registered_with_ordered_effects():
    if offload_pass._register_effectful_op is None:
        pytest.skip("torch without the effects registry; the pass falls back to the inductor DCE patch")
    _ensure_dc_ops()
    from torch._higher_order_ops.effects import _EffectType

    for name, _, _ in offload_pass._OFFLOAD_OP_SPECS:
        overload = getattr(torch.ops.dc, name).default
        # Only the ORDERED effect protects these ops from inductor's scheduler DCE;
        # _side_effectful_functions covers FX's DCE alone. The next test proves the mechanism.
        assert _registered_effect(overload) == _EffectType.ORDERED


def test_side_effect_ops_survive_stock_inductor():
    # Stock inductor drops an anchor-only `-> ()` op unless it carries an ORDERED effect, and
    # keeps its program order with one. A throwaway namespace keeps this runnable on CPU.
    if offload_pass._register_effectful_op is None:
        pytest.skip("torch without the effects registry; the pass falls back to the inductor DCE patch")

    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
    from torch._inductor.scheduler import Scheduler

    if getattr(Scheduler, "is_dc_patched", False):
        pytest.skip("inductor DCE already patched out in this process; mechanism not observable")

    if not hasattr(test_side_effect_ops_survive_stock_inductor, "_probe"):
        lib = torch.library.Library("dctest_probe", "DEF")
        lib.define("side_op(Tensor anchor, int idx) -> ()")
        calls = []
        lib.impl("side_op", lambda anchor, idx: calls.append(idx), "CompositeExplicitAutograd")
        lib.impl("side_op", lambda anchor, idx: None, "Meta")
        op = torch.ops.dctest_probe.side_op.default
        torch.fx.node._side_effectful_functions.add(op)
        _register_effectful_op(op, _EffectType.ORDERED)
        # The ops deregister if the library object is collected.
        test_side_effect_ops_survive_stock_inductor._probe = (lib, op, calls)
    _, op, calls = test_side_effect_ops_survive_stock_inductor._probe

    def f(x):
        return ((x * 2).relu().sin().sum(), )

    gm = make_fx(f)(torch.randn(8, 8))
    nodes = list(gm.graph.nodes)
    placeholder = nodes[0]
    compute = [n for n in nodes if n.op == "call_function"]
    out = next(n for n in nodes if n.op == "output")
    with gm.graph.inserting_before(compute[0]):
        gm.graph.create_node("call_function", op, (placeholder, 0), {})
    with gm.graph.inserting_before(compute[-1]):
        gm.graph.create_node("call_function", op, (placeholder, 1), {})
    with gm.graph.inserting_before(out):
        gm.graph.create_node("call_function", op, (placeholder, 2), {})
    gm.recompile()

    compiled = torch._inductor.compile(gm, [torch.randn(8, 8)])
    calls.clear()
    compiled(torch.randn(8, 8))
    assert calls == [0, 1, 2], f"side-effect ops dropped or reordered by stock inductor: {calls}"


def test_fwd_insertion_schedules_all_tasks_under_forced_budget(monkeypatch):
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, graph)

    names = [n.name for n in graph.nodes]
    launch_names = [n for n in names if n.startswith("offload_opt_")]

    assert len(launch_names) == 3, f"expected exp_avg/exp_avg_sq/hp_param launches, got {launch_names}"
    assert any("hp_param" in n for n in launch_names)
    # The launch op frees via record_stream, so no sync nodes are inserted.
    assert not any("sync" in n for n in launch_names)

    # All copies launch at the top of the graph, before all compute.
    first_compute = names.index("relu")
    assert max(names.index(n) for n in launch_names) < first_compute


def test_partial_offload_when_budget_allows_residency(monkeypatch):
    # Peaks are profiled with states offloaded, so residency adds on top: a 150B budget against
    # a 100B peak admits one 32B task, so two must stay scheduled.
    _ensure_dc_ops()
    graph = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, graph, budget_gb="1.5e-7")

    launch_names = [n.name for n in graph.nodes if n.name.startswith("offload_opt_")]
    assert len(launch_names) == 2
    assert len(offload_pass.offload_tasks_scheduled) == 2


def test_for_init_offloads_before_profiling(monkeypatch):
    calls = []
    monkeypatch.setattr(offload_pass, "offload_adam_states_sync", lambda: calls.append(1))

    ret = offload_pass.offload_adam_states_for_init(None, 0, [(0, True)], None, None, 0.0, None, bwd=False)
    assert ret is None
    assert calls == [1]

    ret = offload_pass.offload_adam_states_for_init(None, 0, [(0, True)], None, None, 0.0, None, bwd=True)
    assert ret is None
    assert calls == [1]


def test_pass_reruns_do_not_double_append(monkeypatch):
    _ensure_dc_ops()
    graph_first = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, graph_first)
    assert len(offload_pass.offload_tasks_scheduled) == 3

    graph_second = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, graph_second)
    assert len(offload_pass.offload_tasks_scheduled) == 3

    launch_names = [n.name for n in graph_second.nodes if n.name.startswith("offload_opt_")]
    assert len(launch_names) == 3


def test_bwd_insertion_reloads_at_graph_end(monkeypatch):
    _ensure_dc_ops()
    fwd_graph = _make_fwd_graph()
    _run_fwd_pass(monkeypatch, fwd_graph)

    bwd_graph = torch.fx.Graph()
    tangent = bwd_graph.placeholder("tangent")
    tangent.meta["val"] = torch.empty(4)
    a = bwd_graph.call_function(torch.relu, (tangent, ))
    bwd_graph.output(a)
    prof = SimpleNamespace(fwd_mem=[], bwd_mem=_mem_rows(bwd_graph))

    offload_pass.offload_opt_states_inc(bwd_graph, 0, [(0, True)], {0: prof}, 0.0, None, bwd=True)

    names = [n.name for n in bwd_graph.nodes]
    reload_names = [n for n in names if n.startswith("reload_opt_")]

    assert "empty_cache" in names
    assert len(reload_names) == 3
    assert "reload_copy_stream_sync" in names
    # A zero budget leaves no mid-graph headroom, so all reloads land at the end.
    assert names.index("reload_copy_stream_sync") > max(names.index(n) for n in reload_names)
    # Running the backward pass re-arms the once-per-phase empty_cache.
    assert offload_pass._empty_cache_pending is True


def test_empty_cache_runs_once_per_phase(monkeypatch):
    calls = []
    monkeypatch.setattr(offload_pass, "get_accelerator", lambda: SimpleNamespace(empty_cache=lambda: calls.append(1)))

    offload_pass._empty_cache_pending = True
    offload_pass._opt_empty_cache_impl(None)
    offload_pass._opt_empty_cache_impl(None)

    assert len(calls) == 1


class TestOffloadOptStates(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    @pytest.mark.parametrize('dtype', [torch.bfloat16])
    def test_offload_opt_states_correctness(self, dtype):
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
                "offload_opt_states": True
            },
            "bf16": {
                "enabled": True
            },
        }

        # Same configuration with offloading off: isolates the pass, so a missed stream
        # ordering shows up as loss drift far below compare_loss's cross-stage tolerance.
        config_no_offload = deepcopy(config)
        config_no_offload["compile"]["offload_opt_states"] = False
        losses_no_offload = compare_loss(self, config_no_offload, dtype, iteration=8)

        # Force every state out, hp_param included, whatever the device actually has.
        os.environ["DS_DC_OFFLOAD_OPT_BUDGET_GB"] = "0.000001"
        try:
            offload_pass.reset_offload_op_stats()
            # The offload schedule engages at step 1 (not WARMUP); 8 iterations give several
            # steady offloaded steps.
            losses_offload = compare_loss(self, config, dtype, iteration=8)
        finally:
            del os.environ["DS_DC_OFFLOAD_OPT_BUDGET_GB"]

        stats = offload_pass.get_offload_op_stats()
        assert stats["launches"] > 0, "offload launch ops never executed"
        # The load-bearing assertion: reloads are skipped while profiling, so a nonzero count
        # proves the ops ran in the compiled graph rather than only in the profilers.
        assert stats["reloads"] > 0, "reload ops never executed outside profiling"
        assert stats["reloads"] <= stats["launches"]

        # Both runs are identically seeded, so moving states must not change the arithmetic.
        for step, (ref, got) in enumerate(zip(losses_no_offload, losses_offload)):
            assert got == pytest.approx(ref, rel=1e-4, abs=1e-5), \
                f"offloading changed the loss at step {step}: {ref} vs {got}"
