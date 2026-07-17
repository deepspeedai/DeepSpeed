# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Runtime wiring checks for AutoEP + AutoTP folding."""

import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
from deepspeed.module_inject.auto_ep_config import AutoEPConfig, MoELayerSpec
from deepspeed.module_inject.auto_ep_folding import FoldingGroupHandles, build_folding_spec, local_folding_ranks
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.module_inject.auto_tp import AutoTP
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


def _make_spec(**overrides):
    defaults = dict(
        moe_module_name="model.layers.0.mlp",
        model_family="mixtral",
        router_name="gate",
        experts_name="experts",
        expert_storage="fused_3d",
        expert_w1_name="gate_up_proj",
        expert_w2_name="down_proj",
        expert_w3_name=None,
        num_experts=4,
        top_k=2,
        hidden_size=8,
        ffn_hidden_size=16,
        score_func="softmax",
        score_apply="post",
        route_norm=True,
        gate_bias=False,
        return_router_logits=False,
        router_logits_capture_target="none",
        router_logits_capture_index=None,
        router_logits_capture_layer_name=None,
        has_shared_experts=False,
        shared_experts_name="",
        shared_experts_gate_name="",
    )
    defaults.update(overrides)
    return MoELayerSpec(**defaults)


class TinySourceMoE(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.randn(4, 32, 8))
        self.experts.down_proj = nn.Parameter(torch.randn(4, 8, 16))


def test_folded_layer_binds_explicit_group_handles(monkeypatch):
    layer = AutoEPMoELayer(_make_spec(), TinySourceMoE(), ep_size=2, ep_rank=0, config=AutoEPConfig(enabled=True))
    spec = build_folding_spec(world_size=4, pp_size=1, tp_size=2, ep_size=2, etp_size=1)
    local = local_folding_ranks(0, spec)
    handles = FoldingGroupHandles(
        spec=spec,
        tp_group=object(),
        dense_dp_group=object(),
        ep_group=object(),
        edp_group=object(),
        ep_group_name="ep_size_2",
        tp_ranks=local["tp"],
        dense_dp_ranks=local["dense_dp"],
        ep_ranks=local["ep"],
        edp_ranks=local["edp"],
    )
    monkeypatch.setattr("deepspeed.module_inject.auto_ep_layer.dist.get_rank", lambda group=None: 0)

    layer.set_deepspeed_parallelism(folding_group_handles=handles)

    assert layer.folding_group_handles is handles
    assert layer.tp_group is handles.tp_group
    assert layer.ep_group is handles.ep_group
    assert layer.ep_group_name == "ep_size_2"


def test_autotp_reaches_autoep_shared_experts(monkeypatch):

    class AutoEPLike(nn.Module):

        def __init__(self):
            super().__init__()
            self._is_autoep_layer = True
            self.shared_experts = nn.Linear(8, 8, bias=False)
            self.shared_experts_gate = nn.Linear(8, 8, bias=False)

    model = nn.Module()
    model.moe = AutoEPLike()
    autotp = AutoTP.__new__(AutoTP)
    calls = []
    monkeypatch.setattr(autotp, "_replace_autoep_shared_experts", lambda child, name: calls.append((child, name)))

    AutoTP._replace_module(autotp, model)

    assert calls == [(model.moe, "moe")]


def _folded_config(zero_stage=0, *, ep_size=2, mixed_precision=True):
    config = make_autoep_config(zero_stage=zero_stage, ep_size=ep_size, mixed_precision=mixed_precision)
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


def _tp_consistent_input(engine, *, seed=1234):
    torch.manual_seed(seed)
    x = torch.randn(1, 4, 64, device=engine.device, dtype=engine_input_dtype(engine))
    dist.broadcast(x, groups.get_tensor_model_parallel_src_rank(), group=groups.get_tensor_model_parallel_group())
    return x


def _initialize_folded_engine(*, zero_stage=0, ep_size=2, mixed_precision=True):
    seed_everything(1234)
    return deepspeed.initialize(model=MockMoEOnlyTransformer(),
                                config=_folded_config(zero_stage=zero_stage,
                                                      ep_size=ep_size,
                                                      mixed_precision=mixed_precision))


def _assert_nonzero_named_grad(engine, *name_fragments):
    grad_total = 0.0
    matched = False
    for name, param in engine.module.named_parameters():
        if not any(fragment in name for fragment in name_fragments):
            continue
        if param.grad is None:
            continue
        matched = True
        grad_total += param.grad.detach().float().abs().sum().item()
    assert matched, f"no gradients found for parameters matching {name_fragments}"
    assert grad_total > 0.0


def _cpu_folded_runtime_worker(_rank, _world_size, _shared_tmpdir):
    engine, _, _, _ = _initialize_folded_engine(zero_stage=0, mixed_precision=False)
    assert engine.autotp_size() == 2
    assert groups.get_tensor_model_parallel_world_size() == 2
    folded_layers = [module for module in engine.module.modules() if isinstance(module, AutoEPMoELayer)]
    assert folded_layers
    assert all(layer.folding_group_handles is not None for layer in folded_layers)

    x = _tp_consistent_input(engine)
    loss = engine(x).float().mean()
    engine.backward(loss)
    _assert_nonzero_named_grad(engine, "experts.")
    _assert_nonzero_named_grad(engine, "router", "gate")
    engine.step()
    assert torch.isfinite(loss.detach()).item()


def test_cpu_gloo_folded_runtime_smoke(tmpdir):
    run_cpu_gloo_test(_cpu_folded_runtime_worker, tmpdir, world_size=4)


class TestH100FoldedRuntime(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def test_h100_folded_tp2_ep2_runtime(self):
        skip_unless_h100_tests_enabled("H100 runtime node")

        engine, _, _, _ = _initialize_folded_engine(zero_stage=0)
        assert engine.autotp_size() == 2
        assert groups.get_tensor_model_parallel_world_size() == 2
        folded_layers = [module for module in engine.module.modules() if isinstance(module, AutoEPMoELayer)]
        assert folded_layers
        assert all(layer.folding_group_handles is not None for layer in folded_layers)

        x = _tp_consistent_input(engine)
        loss = engine(x).float().mean()
        engine.backward(loss)
        _assert_nonzero_named_grad(engine, "experts.")
        _assert_nonzero_named_grad(engine, "router", "gate")
        engine.step()
        assert torch.isfinite(loss.detach()).item()


class TestH100FoldedRuntimeReference(DistributedTest):
    world_size = 4
    reuse_dist_env = False

    def test_h100_folded_tp2_ep2_finite_loss_smoke(self):
        skip_unless_h100_tests_enabled("H100 benchmark node")

        engine, _, _, _ = _initialize_folded_engine(zero_stage=0)
        x = _tp_consistent_input(engine)
        losses = []
        for _ in range(2):
            loss = engine(x).float().mean()
            engine.backward(loss)
            engine.step()
            losses.append(float(loss.detach().cpu()))
        assert all(torch.isfinite(torch.tensor(value)) for value in losses)


class TestH100FoldedRuntimeTP2EP4(DistributedTest):
    world_size = 8
    reuse_dist_env = False

    def test_h100_folded_tp2_ep4_runtime(self):
        skip_unless_h100_tests_enabled("H100 TP2-EP4 runtime node")

        engine, _, _, _ = _initialize_folded_engine(zero_stage=0, ep_size=4)
        assert engine.autotp_size() == 2
        assert groups.get_tensor_model_parallel_world_size() == 2
        folded_layers = [module for module in engine.module.modules() if isinstance(module, AutoEPMoELayer)]
        assert folded_layers
        assert all(layer.folding_group_handles is not None for layer in folded_layers)

        x = _tp_consistent_input(engine)
        loss = engine(x).float().mean()
        engine.backward(loss)
        _assert_nonzero_named_grad(engine, "experts.")
        _assert_nonzero_named_grad(engine, "router", "gate")
        engine.step()
        assert torch.isfinite(loss.detach()).item()
