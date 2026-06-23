# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Gradient and optimizer policy checks for AutoEP + AutoTP folding."""

import glob
import json
import os

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.checkpoint.autoep_universal import validate_folding_metadata
from deepspeed.checkpoint.constants import FOLDING_FAMILY, FOLDING_METADATA_KEY, FOLDING_PARAM_FAMILIES
from deepspeed.module_inject.auto_ep_folding import (
    AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE,
    AUTOEP_FOLDING_GRAD_REDUCE_EXPERT_TP_CANCEL,
    AUTOEP_FOLDING_GRAD_REDUCE_SKIP,
    AUTOEP_FOLDING_GRAD_REDUCE_SUM,
    autoep_folding_gradient_reduction_strategy,
    mark_autoep_folding_partial_router_parameter,
    mark_autoep_folding_router_parameter,
    mark_autoep_folding_sp_sharded_layernorm_parameter,
)
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.utils import safe_get_full_grad
from deepspeed.utils import groups
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoEOnlyTransformer,
    engine_input_dtype,
    make_autoep_config,
    run_cpu_gloo_test,
    seed_everything,
    skip_unless_h100_tests_enabled,
)

from deepspeed.module_inject.auto_ep_config import AutoEPConfig, validate_autoep_config


def _folding_spec(mp_mode="tp", tp_size=2):
    return type("Spec", (), {"tp_size": tp_size, "mp_mode": mp_mode})()


def test_zero_offload_paths_fail_fast_until_per_family_replica_groups_are_proven():
    for kwargs in ({"zero_offload_optimizer": True}, {"zero_offload_param": True}):
        with pytest.raises(ValueError, match="offload"):
            validate_autoep_config(AutoEPConfig(enabled=True, autoep_size=2),
                                   world_size=4,
                                   pp_size=1,
                                   tp_size=2,
                                   sp_size=1,
                                   zero_stage=2,
                                   **kwargs)


def test_zero3_composition_remains_separate_lane():
    with pytest.raises(ValueError, match="separate ZeRO-3 composition lane"):
        validate_autoep_config(AutoEPConfig(enabled=True, autoep_size=2),
                               world_size=4,
                               pp_size=1,
                               tp_size=2,
                               sp_size=1,
                               zero_stage=3)


def _folded_zero2_config(*, mixed_precision=True):
    config = make_autoep_config(zero_stage=2, ep_size=2, mixed_precision=mixed_precision)
    config["gradient_accumulation_steps"] = 2
    config["gradient_clipping"] = 0.0
    if not mixed_precision:
        config["optimizer"]["params"]["torch_adam"] = True
    config["tensor_parallel"] = {
        "autotp_size": 2,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        },
    }
    return config


def test_autoep_folding_gradient_strategy_uses_parameter_family():
    router = torch.nn.Parameter(torch.ones(2))
    mark_autoep_folding_router_parameter(router)
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("tp"),
                                                       router) == AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE)

    partial_router = torch.nn.Parameter(torch.ones(2))
    mark_autoep_folding_partial_router_parameter(partial_router)
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("tp"),
                                                       partial_router) == AUTOEP_FOLDING_GRAD_REDUCE_SUM)
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("sp"),
                                                       partial_router) == AUTOEP_FOLDING_GRAD_REDUCE_SUM)
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("replicated"),
                                                       router) == AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE)

    dense_or_layernorm = torch.nn.Parameter(torch.ones(2))
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("tp"),
                                                       dense_or_layernorm) == AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE)

    sp_layernorm = torch.nn.Parameter(torch.ones(2))
    mark_autoep_folding_sp_sharded_layernorm_parameter(sp_layernorm)
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("sp"),
                                                       sp_layernorm) == AUTOEP_FOLDING_GRAD_REDUCE_SUM)
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("tp"),
                                                       sp_layernorm) == AUTOEP_FOLDING_GRAD_REDUCE_AVERAGE)

    # Routed experts cancel the restore all-gather tp_size factor (divide-by-tp, no
    # TP all_reduce); their data-parallel reduction stays on the EP/EDP path.
    expert = torch.nn.Parameter(torch.ones(2))
    expert.allreduce = False
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("tp"),
                                                       expert) == AUTOEP_FOLDING_GRAD_REDUCE_EXPERT_TP_CANCEL)
    # With folding disabled (tp_size == 1) experts still SKIP.
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("tp", tp_size=1),
                                                       expert) == AUTOEP_FOLDING_GRAD_REDUCE_SKIP)

    model_parallel = torch.nn.Parameter(torch.ones(2))
    model_parallel.tensor_model_parallel = True
    assert (autoep_folding_gradient_reduction_strategy(_folding_spec("tp"),
                                                       model_parallel) == AUTOEP_FOLDING_GRAD_REDUCE_SKIP)


@pytest.mark.parametrize(
    ("param_name", "mark_router", "mp_mode", "expected_grad"),
    (
        ("model.layers.0.mlp.router.gate.weight", True, "tp", 1.0),
        ("model.layers.0.mlp.router.gate.weight", True, "sp", 1.0),
        ("model.layers.0.mlp.router.gate.weight", True, "replicated", 1.0),
        ("model.layers.0.input_layernorm.weight", False, "tp", 1.0),
    ),
)
def test_tp_replicated_gradient_reducer_respects_param_family(monkeypatch, param_name, mark_router, mp_mode,
                                                              expected_grad):
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param)
    if mark_router:
        mark_autoep_folding_router_parameter(param)
    engine = object.__new__(DeepSpeedEngine)
    engine._autoep_folding_spec = _folding_spec(mp_mode)
    engine.__dict__["optimizer"] = None
    engine.__dict__["module"] = type("ModuleStub", (),
                                     {"named_parameters": lambda self: iter([(param_name, param)])})()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(groups, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    def fake_all_reduce(tensor, group=None):
        tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    engine._reduce_autoep_folding_tp_replicated_gradients()

    assert torch.equal(param.grad, torch.full_like(param.grad, expected_grad))


def test_zero2_tp_gradient_reducer_skips_incomplete_ds_grad(monkeypatch):
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param)
    param.ds_grad_is_ready = False
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.partition_gradients = True
    optimizer.autoep_folding_tp_group = object()
    optimizer.autoep_folding_spec = _folding_spec("tp")
    calls = []

    def fake_all_reduce(tensor, group=None):
        calls.append(tensor.clone())
        tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    optimizer._maybe_reduce_autoep_folding_tp_gradient(param, param.grad)

    assert calls == []
    torch.testing.assert_close(param.grad, torch.ones_like(param.grad))


@pytest.mark.parametrize(("mark_router", "expected_grad"), ((True, 1.0), (False, 1.0)))
def test_zero2_tp_gradient_reducer_uses_shared_param_family_strategy(monkeypatch, mark_router, expected_grad):
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param)
    if mark_router:
        mark_autoep_folding_router_parameter(param)
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.partition_gradients = True
    optimizer.autoep_folding_tp_group = object()
    optimizer.autoep_folding_spec = _folding_spec("tp")

    def fake_all_reduce(tensor, group=None):
        tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    optimizer._maybe_reduce_autoep_folding_tp_gradient(param, param.grad)

    torch.testing.assert_close(param.grad, torch.full_like(param.grad, expected_grad))


def _folded_zero2_tp2_ep4_config():
    config = _folded_zero2_config(mixed_precision=False)
    config["expert_parallel"]["autoep_size"] = 4
    config["communication_data_type"] = "fp32"
    return config


def _folded_zero0_tp2_ep4_config():
    config = make_autoep_config(zero_stage=0, ep_size=4, mixed_precision=False)
    config["gradient_accumulation_steps"] = 2
    config["gradient_clipping"] = 0.0
    config["communication_data_type"] = "fp32"
    config["optimizer"]["params"]["torch_adam"] = True
    config["expert_parallel"]["autoep_size"] = 4
    config["tensor_parallel"] = {
        "autotp_size": 2,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        },
    }
    return config


def _zero2_baseline_config():
    config = {
        **{
            key: value
            for key, value in make_autoep_config(zero_stage=2, ep_size=1, mixed_precision=False).items() if key != "expert_parallel"
        },
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 0.0,
    }
    config["communication_data_type"] = "fp32"
    config["optimizer"]["params"]["torch_adam"] = True
    return config


def _zero0_baseline_config():
    config = {
        **{
            key: value
            for key, value in make_autoep_config(zero_stage=0, ep_size=1, mixed_precision=False).items() if key != "expert_parallel"
        },
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 0.0,
    }
    config["communication_data_type"] = "fp32"
    config["optimizer"]["params"]["torch_adam"] = True
    return config


GATE_BASELINE = "model.layers.0.mlp.gate.weight"
GATE_FOLDED = "model.layers.0.mlp.router.gate.weight"
INPUT_LAYERNORM = "model.layers.0.input_layernorm.weight"


def _router_grad_model():
    return MockMoEOnlyTransformer(num_layers=1, num_experts=4, hidden_size=64, intermediate_size=128)


def _make_logical_batches(engine, *, logical_dp_world_size, logical_dp_rank, grad_accum, seed):
    batches = []
    for accum_idx in range(grad_accum):
        batch_idx = accum_idx * logical_dp_world_size + logical_dp_rank
        generator = torch.Generator().manual_seed(seed + batch_idx)
        batch = torch.randn((1, 4, 64), generator=generator, dtype=engine_input_dtype(engine))
        batches.append(batch.to(engine.device))
    return batches


def _run_router_grad_boundary(engine, *, logical_dp_world_size, logical_dp_rank, seed):
    batches = _make_logical_batches(engine,
                                    logical_dp_world_size=logical_dp_world_size,
                                    logical_dp_rank=logical_dp_rank,
                                    grad_accum=2,
                                    seed=seed)
    for batch_idx, batch in enumerate(batches):
        loss = engine(batch).float().mean()
        engine.backward(loss)
        if batch_idx + 1 < len(batches):
            engine.step()


def _full_grad_by_suffix(engine, suffix):
    for name, param in engine.module.named_parameters():
        if name.endswith(suffix):
            grad = safe_get_full_grad(param)
            assert grad is not None, f"Expected full grad for {name}"
            return grad.detach().float().cpu().clone()
    raise AssertionError(f"Missing parameter ending with {suffix}")


def _cpu_folded_zero0_router_gate_and_layernorm_worker(rank, world_size, _shared_tmpdir):
    seed = 1234
    tp_size = 2
    logical_dp_world_size = world_size // tp_size
    logical_dp_rank = rank // tp_size

    seed_everything(seed)
    reference_state = _router_grad_model().state_dict()

    baseline_model = _router_grad_model()
    baseline_model.load_state_dict(reference_state)
    baseline_engine, _, _, _ = deepspeed.initialize(model=baseline_model, config=_zero0_baseline_config())
    _run_router_grad_boundary(baseline_engine,
                              logical_dp_world_size=logical_dp_world_size,
                              logical_dp_rank=logical_dp_rank,
                              seed=seed)
    baseline_gate = _full_grad_by_suffix(baseline_engine, GATE_BASELINE)
    baseline_layernorm = _full_grad_by_suffix(baseline_engine, INPUT_LAYERNORM)

    folded_model = _router_grad_model()
    folded_model.load_state_dict(reference_state)
    folded_engine, _, _, _ = deepspeed.initialize(model=folded_model, config=_folded_zero0_tp2_ep4_config())
    _run_router_grad_boundary(folded_engine,
                              logical_dp_world_size=logical_dp_world_size,
                              logical_dp_rank=logical_dp_rank,
                              seed=seed)
    folded_gate = _full_grad_by_suffix(folded_engine, GATE_FOLDED)
    folded_layernorm = _full_grad_by_suffix(folded_engine, INPUT_LAYERNORM)

    metrics = {
        "rank": rank,
        "gate": _grad_parity_metrics(folded_gate, baseline_gate),
        "layernorm": _grad_parity_metrics(folded_layernorm, baseline_layernorm),
    }
    if rank == 0:
        print("FOLDED_ENGINE_ZERO0_ROUTER_GATE_LAYERNORM_GRAD_PARITY " + json.dumps(metrics, sort_keys=True))
    torch.testing.assert_close(folded_gate,
                               baseline_gate,
                               atol=1e-1,
                               rtol=5e-3,
                               msg=f"Folded zero_stage=0 router/gate grad must match baseline; metrics={metrics}")
    torch.testing.assert_close(folded_layernorm,
                               baseline_layernorm,
                               atol=1e-1,
                               rtol=5e-3,
                               msg=f"Folded zero_stage=0 LayerNorm grad must match baseline; metrics={metrics}")


def _cpu_folded_zero2_router_gate_and_layernorm_worker(rank, world_size, _shared_tmpdir):
    seed = 1234
    tp_size = 2
    logical_dp_world_size = world_size // tp_size
    logical_dp_rank = rank // tp_size

    seed_everything(seed)
    reference_state = _router_grad_model().state_dict()

    baseline_model = _router_grad_model()
    baseline_model.load_state_dict(reference_state)
    baseline_engine, _, _, _ = deepspeed.initialize(model=baseline_model, config=_zero2_baseline_config())
    _run_router_grad_boundary(baseline_engine,
                              logical_dp_world_size=logical_dp_world_size,
                              logical_dp_rank=logical_dp_rank,
                              seed=seed)
    baseline_gate = _full_grad_by_suffix(baseline_engine, GATE_BASELINE)
    baseline_layernorm = _full_grad_by_suffix(baseline_engine, INPUT_LAYERNORM)

    folded_model = _router_grad_model()
    folded_model.load_state_dict(reference_state)
    folded_engine, _, _, _ = deepspeed.initialize(model=folded_model, config=_folded_zero2_tp2_ep4_config())
    _run_router_grad_boundary(folded_engine,
                              logical_dp_world_size=logical_dp_world_size,
                              logical_dp_rank=logical_dp_rank,
                              seed=seed)
    folded_gate = _full_grad_by_suffix(folded_engine, GATE_FOLDED)
    folded_layernorm = _full_grad_by_suffix(folded_engine, INPUT_LAYERNORM)

    metrics = {
        "rank": rank,
        "gate": _grad_parity_metrics(folded_gate, baseline_gate),
        "layernorm": _grad_parity_metrics(folded_layernorm, baseline_layernorm),
    }
    if rank == 0:
        print("FOLDED_ZERO2_ROUTER_GATE_LAYERNORM_GRAD_PARITY " + json.dumps(metrics, sort_keys=True))
    torch.testing.assert_close(folded_gate,
                               baseline_gate,
                               atol=1e-1,
                               rtol=5e-3,
                               msg=f"Folded ZeRO-2 router/gate grad must match baseline; metrics={metrics}")
    torch.testing.assert_close(folded_layernorm,
                               baseline_layernorm,
                               atol=1e-1,
                               rtol=5e-3,
                               msg=f"Folded ZeRO-2 LayerNorm grad must match baseline; metrics={metrics}")


def _grad_parity_metrics(actual, expected):
    diff = actual - expected
    expected_norm_sq = expected.square().sum().item()
    actual_norm = actual.norm().item()
    expected_norm = expected.norm().item()
    scale = actual.mul(expected).sum().item() / expected_norm_sq if expected_norm_sq else 0.0
    return {
        "scale_vs_expected": scale,
        "scale_vs_baseline": scale,
        "max_abs": diff.abs().max().item(),
        "rel_norm": diff.norm().item() / expected_norm,
        "actual_norm": actual_norm,
        "expected_norm": expected_norm,
        "folded_norm": actual_norm,
        "baseline_norm": expected_norm,
    }


def test_cpu_gloo_folded_zero0_router_gate_and_layernorm_grad_parity(tmpdir):
    run_cpu_gloo_test(_cpu_folded_zero0_router_gate_and_layernorm_worker, tmpdir, world_size=8)


def test_cpu_gloo_folded_zero2_router_gate_and_layernorm_grad_parity(tmpdir):
    run_cpu_gloo_test(_cpu_folded_zero2_router_gate_and_layernorm_worker, tmpdir, world_size=8)


# ---------------------------------------------------------------------------
# Cross-lane EP (expert parallel spanning TP lanes; expert_width = ep need NOT be a
# subset of dp). These reuse the tp2/ep4 workers at world_size=4 (where ep=4 > dp=2)
# and add a tp4/ep4/dp1 worker (EP group == TP group). The router/gate and LayerNorm
# AVERAGE-over-TP convention is unchanged because the dedup/restore key on the
# token-replication (TP) group, independent of the EP layout.
# ---------------------------------------------------------------------------


def _folded_zero0_tp4_ep4_config():
    config = make_autoep_config(zero_stage=0, ep_size=4, mixed_precision=False)
    config["gradient_accumulation_steps"] = 2
    config["gradient_clipping"] = 0.0
    config["communication_data_type"] = "fp32"
    config["optimizer"]["params"]["torch_adam"] = True
    config["expert_parallel"]["autoep_size"] = 4
    config["tensor_parallel"] = {
        "autotp_size": 4,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        },
    }
    return config


def _cpu_folded_tp4_ep4_router_gate_and_layernorm_worker(rank, world_size, _shared_tmpdir):
    seed = 1234
    tp_size = 4
    logical_dp_world_size = world_size // tp_size
    logical_dp_rank = rank // tp_size

    seed_everything(seed)
    reference_state = _router_grad_model().state_dict()

    baseline_model = _router_grad_model()
    baseline_model.load_state_dict(reference_state)
    baseline_engine, _, _, _ = deepspeed.initialize(model=baseline_model, config=_zero0_baseline_config())
    _run_router_grad_boundary(baseline_engine,
                              logical_dp_world_size=logical_dp_world_size,
                              logical_dp_rank=logical_dp_rank,
                              seed=seed)
    baseline_gate = _full_grad_by_suffix(baseline_engine, GATE_BASELINE)
    baseline_layernorm = _full_grad_by_suffix(baseline_engine, INPUT_LAYERNORM)

    folded_model = _router_grad_model()
    folded_model.load_state_dict(reference_state)
    folded_engine, _, _, _ = deepspeed.initialize(model=folded_model, config=_folded_zero0_tp4_ep4_config())
    _run_router_grad_boundary(folded_engine,
                              logical_dp_world_size=logical_dp_world_size,
                              logical_dp_rank=logical_dp_rank,
                              seed=seed)
    folded_gate = _full_grad_by_suffix(folded_engine, GATE_FOLDED)
    folded_layernorm = _full_grad_by_suffix(folded_engine, INPUT_LAYERNORM)

    metrics = {
        "rank": rank,
        "gate": _grad_parity_metrics(folded_gate, baseline_gate),
        "layernorm": _grad_parity_metrics(folded_layernorm, baseline_layernorm)
    }
    if rank == 0:
        print("FOLDED_CROSSLANE_TP4_EP4_GRAD_PARITY " + json.dumps(metrics, sort_keys=True))
    torch.testing.assert_close(folded_gate,
                               baseline_gate,
                               atol=1e-1,
                               rtol=5e-3,
                               msg=f"Cross-lane tp4/ep4 router/gate grad must match baseline; metrics={metrics}")
    torch.testing.assert_close(folded_layernorm,
                               baseline_layernorm,
                               atol=1e-1,
                               rtol=5e-3,
                               msg=f"Cross-lane tp4/ep4 LayerNorm grad must match baseline; metrics={metrics}")


def test_cpu_gloo_crosslane_tp2_ep4_zero0_router_gate_and_layernorm_grad_parity(tmpdir):
    # world=4: tp2/ep4 => ep=4 > dp=2 (cross-lane; EP group spans both TP lanes and DP ranks).
    run_cpu_gloo_test(_cpu_folded_zero0_router_gate_and_layernorm_worker, tmpdir, world_size=4)


def test_cpu_gloo_crosslane_tp2_ep4_zero2_router_gate_and_layernorm_grad_parity(tmpdir):
    run_cpu_gloo_test(_cpu_folded_zero2_router_gate_and_layernorm_worker, tmpdir, world_size=4)


def test_cpu_gloo_crosslane_tp4_ep4_dp1_zero0_router_gate_and_layernorm_grad_parity(tmpdir):
    # world=4: tp4/ep4/dp1 => EP group == TP group == {0,1,2,3}.
    run_cpu_gloo_test(_cpu_folded_tp4_ep4_router_gate_and_layernorm_worker, tmpdir, world_size=4)


# ---------------------------------------------------------------------------
# Expert-weight gradient parity. The folded forward all-gathers expert outputs into
# a replicated full view in ``restore_combined`` (backward injects a ``tp_size``
# factor); routed experts must cancel it (EXPERT_TP_CANCEL ``/tp_size``) or their
# gradients reach the optimizer ``tp_size`` times too large. This is invisible to
# scale-invariant Adam, so this test uses SGD and compares the post-step expert
# weights against a non-folded baseline. It covers the MVP shape and cross-lane
# shapes including edp>1.
# ---------------------------------------------------------------------------

EXPERTS_W1 = "experts.w1"  # folded GroupedExperts gate half (num_local, ffn, hidden)
EXPERTS_W3 = "experts.w3"  # folded GroupedExperts up half  (num_local, ffn, hidden)
EXPERTS_W2 = "experts.w2"  # folded GroupedExperts down     (num_local, hidden, ffn)
GATE_UP_PROJ = "mlp.experts.gate_up_proj"  # baseline fused gate||up (num_experts, 2*ffn, hidden)
DOWN_PROJ = "mlp.experts.down_proj"  # baseline down (num_experts, hidden, ffn)


def _sgd_baseline_config(zero_stage=0):
    config = {
        key: value
        for key, value in make_autoep_config(zero_stage=zero_stage, ep_size=1, mixed_precision=False).items()
        if key != "expert_parallel"
    }
    config["gradient_accumulation_steps"] = 1
    config["gradient_clipping"] = 0.0
    config["communication_data_type"] = "fp32"
    config["optimizer"] = {"type": "SGD", "params": {"lr": 1.0}}
    config["zero_allow_untested_optimizer"] = True  # SGD under ZeRO is "untested"; this is a grad-parity probe
    return config


def _folded_sgd_config(tp_size, ep_size, zero_stage=0):
    config = make_autoep_config(zero_stage=zero_stage, ep_size=ep_size, mixed_precision=False)
    config["gradient_accumulation_steps"] = 1
    config["gradient_clipping"] = 0.0
    config["communication_data_type"] = "fp32"
    config["optimizer"] = {"type": "SGD", "params": {"lr": 1.0}}
    config["zero_allow_untested_optimizer"] = True  # SGD under ZeRO is "untested"; this is a grad-parity probe
    config["expert_parallel"]["autoep_size"] = ep_size
    config["tensor_parallel"] = {
        "autotp_size": tp_size,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        },
    }
    return config


def _one_sgd_step(engine, *, tp_size, seed):
    rank = dist.get_rank()
    generator = torch.Generator().manual_seed(seed + (rank // tp_size))
    x = torch.randn((1, 4, 64), generator=generator, dtype=engine_input_dtype(engine)).to(engine.device)
    loss = engine(x).float().mean()
    engine.backward(loss)
    engine.step()
    return loss


def _local_param_by_suffix(module, suffix):
    for name, param in module.named_parameters():
        if name.endswith(suffix):
            return param.detach().float().cpu()
    raise AssertionError(f"Missing parameter ending with {suffix}")


def _gather_full_experts(layer, suffix):
    """All-gather a local routed-expert tensor over the EP group into the full
    (num_experts, ...) tensor, ordered by ep_rank (expert_start = ep_rank * num_local)."""
    local = None
    for name, param in layer.named_parameters():
        if name.endswith(suffix):
            local = param.detach().contiguous()
            break
    assert local is not None, f"Missing parameter ending with {suffix}"
    ep_group = layer.ep_group
    ep_world = dist.get_world_size(group=ep_group)
    gathered = [torch.empty_like(local) for _ in range(ep_world)]
    dist.all_gather(gathered, local, group=ep_group)
    return torch.cat([chunk.float().cpu() for chunk in gathered], dim=0)


def _expert_weight_parity_worker(rank, world_size, tp_size, ep_size, zero_stage=0):
    seed = 1234
    seed_everything(seed)
    reference_state = _router_grad_model().state_dict()

    baseline_model = _router_grad_model()
    baseline_model.load_state_dict(reference_state)
    baseline_engine, _, _, _ = deepspeed.initialize(model=baseline_model, config=_sgd_baseline_config(zero_stage))
    _one_sgd_step(baseline_engine, tp_size=tp_size, seed=seed)
    base_gate_up = _local_param_by_suffix(baseline_engine.module, GATE_UP_PROJ)  # (E, 2*ffn, h)
    base_down = _local_param_by_suffix(baseline_engine.module, DOWN_PROJ)  # (E, h, ffn)

    folded_model = _router_grad_model()
    folded_model.load_state_dict(reference_state)
    folded_engine, _, _, _ = deepspeed.initialize(model=folded_model,
                                                  config=_folded_sgd_config(tp_size, ep_size, zero_stage))
    _one_sgd_step(folded_engine, tp_size=tp_size, seed=seed)
    layer = folded_engine.module.model.layers[0].mlp
    full_w1 = _gather_full_experts(layer, EXPERTS_W1)
    full_w3 = _gather_full_experts(layer, EXPERTS_W3)
    full_w2 = _gather_full_experts(layer, EXPERTS_W2)
    folded_gate_up = torch.cat([full_w1, full_w3], dim=1)  # (E, 2*ffn, h)

    if rank == 0:
        gu_scale = folded_gate_up.mul(base_gate_up).sum().item() / base_gate_up.square().sum().item()
        dn_scale = full_w2.mul(base_down).sum().item() / base_down.square().sum().item()
        print(f"EXPERT_WEIGHT_PARITY tp{tp_size}_ep{ep_size}_w{world_size}_z{zero_stage} "
              f"gate_up_post_step_scale={gu_scale:.6f} down_post_step_scale={dn_scale:.6f}")
    # Post-step weight equality proves the *applied* expert update matches the baseline.
    # A tp_size over-scaling would diverge these by ~lr*(tp-1)*grad (lr=1), far above tol.
    torch.testing.assert_close(
        folded_gate_up,
        base_gate_up,
        atol=1e-4,
        rtol=1e-4,
        msg="Folded routed-expert gate/up weights must match non-folded baseline after one SGD step")
    torch.testing.assert_close(
        full_w2,
        base_down,
        atol=1e-4,
        rtol=1e-4,
        msg="Folded routed-expert down weights must match non-folded baseline after one SGD step")


def _expert_weight_parity_mvp_tp2_ep4_z0(rank, world_size, _tmp):
    _expert_weight_parity_worker(rank, world_size, tp_size=2, ep_size=4, zero_stage=0)


def _expert_weight_parity_tp4_ep4_z0(rank, world_size, _tmp):
    _expert_weight_parity_worker(rank, world_size, tp_size=4, ep_size=4, zero_stage=0)


def _expert_weight_parity_tp4_ep4_z2(rank, world_size, _tmp):
    _expert_weight_parity_worker(rank, world_size, tp_size=4, ep_size=4, zero_stage=2)


def test_cpu_gloo_expert_weight_parity_mvp_tp2_ep4(tmpdir):
    # MVP shape (ep=4 <= dp=4, edp=2). Guards the pre-existing expert over-scaling fix.
    run_cpu_gloo_test(_expert_weight_parity_mvp_tp2_ep4_z0, tmpdir, world_size=8)


def test_cpu_gloo_expert_weight_parity_crosslane_tp4_ep4_dp1(tmpdir):
    # Cross-lane, edp=1: EP group == TP group == {0,1,2,3}.
    run_cpu_gloo_test(_expert_weight_parity_tp4_ep4_z0, tmpdir, world_size=4)


def test_cpu_gloo_expert_weight_parity_crosslane_tp4_ep4_edp2(tmpdir):
    # Cross-lane, edp=2: EP groups span TP lanes and DP ranks.
    run_cpu_gloo_test(_expert_weight_parity_tp4_ep4_z0, tmpdir, world_size=8)


def test_cpu_gloo_expert_weight_parity_crosslane_tp4_ep4_dp1_zero2(tmpdir):
    # Cross-lane expert fix must also apply on the ZeRO-2 reducer path (different hook).
    run_cpu_gloo_test(_expert_weight_parity_tp4_ep4_z2, tmpdir, world_size=4)


def _assert_zero_optimizer_folding_metadata(checkpoint_dir):
    optim_paths = sorted(glob.glob(os.path.join(str(checkpoint_dir), "folded-zero2", "*_optim_states.pt")))
    assert optim_paths
    saw_metadata = False
    for path in optim_paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if FOLDING_METADATA_KEY not in state:
            continue
        if state[FOLDING_METADATA_KEY][FOLDING_FAMILY] != "zero_optimizer_state":
            continue
        saw_metadata = True
        folding = validate_folding_metadata(state,
                                            tp_size=2,
                                            ep_size=2,
                                            family="zero_optimizer_state",
                                            zero_partition_group="per_family",
                                            zero_partition_count=2)
        assert folding[FOLDING_FAMILY] == "zero_optimizer_state"
        param_families = folding[FOLDING_PARAM_FAMILIES]
        routed_entries = {name: meta for name, meta in param_families.items() if ".experts." in name}
        assert routed_entries
        assert all(meta["family"] == "routed_expert" for meta in routed_entries.values())
        assert all(meta["zero_partition_group"] == "edp" for meta in routed_entries.values())
        dense_entries = {name: meta for name, meta in param_families.items() if ".dense." in name}
        assert dense_entries
        assert all(meta["family"] == "dense" for meta in dense_entries.values())
        assert all(meta["zero_partition_group"] == "dense_dp" for meta in dense_entries.values())
    assert saw_metadata


def _cpu_folded_zero2_worker(_rank, _world_size, _shared_tmpdir):
    seed_everything(1234)
    engine, _, _, _ = deepspeed.initialize(model=MockMoEOnlyTransformer(),
                                           config=_folded_zero2_config(mixed_precision=False))
    folded_layers = [module for module in engine.module.modules() if isinstance(module, AutoEPMoELayer)]
    assert folded_layers
    assert all(layer.folding_group_handles is not None for layer in folded_layers)
    torch.manual_seed(1234)
    x = torch.randn(1, 4, 64, device=engine.device, dtype=engine_input_dtype(engine))
    dist.broadcast(x, groups.get_tensor_model_parallel_src_rank(), group=groups.get_tensor_model_parallel_group())
    loss = engine(x).float().mean()
    engine.backward(loss)
    engine.step()
    engine.save_checkpoint(str(_shared_tmpdir), tag="folded-zero2")
    dist.barrier()
    _assert_zero_optimizer_folding_metadata(_shared_tmpdir)
    assert torch.isfinite(loss.detach()).item()


def test_cpu_gloo_folded_zero2_optimizer_state_smoke(tmpdir):
    run_cpu_gloo_test(_cpu_folded_zero2_worker, tmpdir, world_size=4)


class TestH100FoldedZero12(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def test_h100_zero12_per_family_optimizer_state(self):
        skip_unless_h100_tests_enabled("H100 optimizer-state node")

        seed_everything(1234)
        engine, _, _, _ = deepspeed.initialize(model=MockMoEOnlyTransformer(), config=_folded_zero2_config())
        folded_layers = [module for module in engine.module.modules() if isinstance(module, AutoEPMoELayer)]
        assert folded_layers
        assert all(layer.folding_group_handles is not None for layer in folded_layers)
        torch.manual_seed(1234)
        x = torch.randn(1, 4, 64, device=engine.device, dtype=engine_input_dtype(engine))
        dist.broadcast(x, groups.get_tensor_model_parallel_src_rank(), group=groups.get_tensor_model_parallel_group())
        loss = engine(x).float().mean()
        engine.backward(loss)
        engine.step()
        assert torch.isfinite(loss.detach()).item()


class TestH100FoldedRouterGateGradParityTP2EP4(DistributedTest):
    world_size = 8
    reuse_dist_env = False

    def test_folded_router_gate_and_layernorm_grad_match_nonfolded_zero2_baseline(self):
        skip_unless_h100_tests_enabled("H100 folded router/gate and LayerNorm gradient parity node")

        seed = 1234
        tp_size = 2
        logical_dp_world_size = self.world_size // tp_size
        logical_dp_rank = dist.get_rank() // tp_size

        seed_everything(seed)
        reference_state = _router_grad_model().state_dict()
        baseline_model = _router_grad_model()
        baseline_model.load_state_dict(reference_state)
        baseline_engine, _, _, _ = deepspeed.initialize(model=baseline_model, config=_zero2_baseline_config())
        _run_router_grad_boundary(baseline_engine,
                                  logical_dp_world_size=logical_dp_world_size,
                                  logical_dp_rank=logical_dp_rank,
                                  seed=seed)
        baseline_gate_grad = _full_grad_by_suffix(baseline_engine, GATE_BASELINE)
        baseline_layernorm_grad = _full_grad_by_suffix(baseline_engine, INPUT_LAYERNORM)

        folded_model = _router_grad_model()
        folded_model.load_state_dict(reference_state)
        folded_engine, _, _, _ = deepspeed.initialize(model=folded_model, config=_folded_zero2_tp2_ep4_config())
        _run_router_grad_boundary(folded_engine,
                                  logical_dp_world_size=logical_dp_world_size,
                                  logical_dp_rank=logical_dp_rank,
                                  seed=seed)

        folded_gate_grad = _full_grad_by_suffix(folded_engine, GATE_FOLDED)
        folded_layernorm_grad = _full_grad_by_suffix(folded_engine, INPUT_LAYERNORM)
        metrics = {
            "nodeid": "tests/unit/v1/moe/test_autoep_autotp_grad_parity.py::"
            "TestH100FoldedRouterGateGradParityTP2EP4::"
            "test_folded_router_gate_and_layernorm_grad_match_nonfolded_zero2_baseline",
            "rank": dist.get_rank(),
            "gate": _grad_parity_metrics(folded_gate_grad, baseline_gate_grad),
            "layernorm": _grad_parity_metrics(folded_layernorm_grad, baseline_layernorm_grad),
        }
        if dist.get_rank() == 0:
            print("FOLDED_ROUTER_GATE_LAYERNORM_GRAD_PARITY " + json.dumps(metrics, sort_keys=True))

        torch.testing.assert_close(folded_gate_grad,
                                   baseline_gate_grad,
                                   atol=1e-1,
                                   rtol=5e-3,
                                   msg=("Folded TP2-EP4 router/gate grad must match the non-folded ZeRO-2 "
                                        f"baseline; metrics={metrics}"))
        torch.testing.assert_close(folded_layernorm_grad,
                                   baseline_layernorm_grad,
                                   atol=1e-1,
                                   rtol=5e-3,
                                   msg=("Folded TP2-EP4 LayerNorm grad must match the non-folded ZeRO-2 "
                                        f"baseline; metrics={metrics}"))
