# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Compact critical-path tests for AutoEP."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import deepspeed.runtime.engine as ds_engine
from deepspeed.module_inject.auto_ep import AutoEP, _resolve_route_scale
from deepspeed.module_inject.auto_ep_config import (
    AutoEPConfig,
    MoELayerSpec,
    PRESET_MODELS,
    fill_autoep_config_from_hf,
    parse_autoep_config,
    validate_autoep_config,
    validate_autoep_post_detection,
)
from deepspeed.module_inject.auto_ep_layer import (
    AutoEPMoELayer,
    apply_scores_before_experts_if_enabled,
    combine_from_routed,
    resolve_score_apply_mode,
)
from deepspeed.module_inject.auto_ep_preset_adapters import get_preset_adapter
from deepspeed.module_inject.auto_ep_presets.registry import (
    preset_name_for_hf_model_type,
    unsupported_preset_for_hf_model_type,
)
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.moe.ep_kernels import TokenReorderer
from deepspeed.moe.ep_router import TokenChoiceTopKRouter
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils import groups
from unit.v1.moe.autoep_test_utils import (
    MockMoEBlock,
    MockMoETransformer,
    UNSUPPORTED_LOAD_BALANCE_VALUES,
    assert_causal_lm_outputs_close,
    assert_load_balance_coeff_rejection_message,
    replace_autoep_layers,
    skip_unless_transformers_has,
    state_matched_models,
    tiny_mixtral_config,
)


def _runtime_config(**kwargs):
    kwargs.setdefault("use_grouped_mm", False)
    return AutoEPConfig(**kwargs)


def _make_spec(**kwargs):
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
        hidden_size=64,
        ffn_hidden_size=128,
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
    defaults.update(kwargs)
    return MoELayerSpec(**defaults)


def _assert_same_dtype_device(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


class MockSharedExpert(nn.Module):

    def __init__(self, hidden_size=64):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class MockDeepSeekV3Config:
    model_type = "deepseek_v3"
    n_routed_experts = 8
    num_experts_per_tok = 2
    hidden_size = 64
    moe_intermediate_size = 128
    n_group = 4
    topk_group = 2
    routed_scaling_factor = 2.5


class MockDeepSeekV3Expert(nn.Module):

    def __init__(self, hidden_size=64, ffn_hidden=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, hidden_size, bias=False)


class MockDeepSeekV3MoEBlock(nn.Module):

    def __init__(self, num_experts=8, ffn_hidden=128, hidden_size=64):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([MockDeepSeekV3Expert(hidden_size, ffn_hidden) for _ in range(num_experts)])
        self.shared_experts = MockSharedExpert(hidden_size)


class MockDeepSeekV3Transformer(nn.Module):

    def __init__(self, num_layers=2, num_experts=8):
        super().__init__()
        self.config = MockDeepSeekV3Config()
        self.config.n_routed_experts = num_experts
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([self._make_layer(num_experts) for _ in range(num_layers)])

    @staticmethod
    def _make_layer(num_experts):
        layer = nn.Module()
        layer.mlp = MockDeepSeekV3MoEBlock(num_experts)
        return layer


class TestAutoEPConfig:

    def test_parse_and_validate_enabled_size_contract(self):
        disabled = parse_autoep_config({})
        assert disabled.enabled is False
        assert disabled.autoep_size == 1
        assert disabled.load_balance_coeff is None
        assert disabled._load_balance_coeff_explicit is False

        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 4,
            "preset_model": "mixtral",
            "load_balance_coeff": None,
            "score_apply": "pre",
            "route_scale": 2.0,
        })

        assert config.enabled is True
        assert config.autoep_size == 4
        assert config.preset_model == "mixtral"
        assert config.load_balance_coeff is None
        assert config._load_balance_coeff_explicit is True
        assert config.score_apply == "pre"
        assert config.route_scale == 2.0
        validate_autoep_config(config, world_size=4, pp_size=1, tp_size=1, sp_size=1)

    @pytest.mark.parametrize("value", UNSUPPORTED_LOAD_BALANCE_VALUES)
    def test_load_balance_coeff_rejected_at_parse(self, value):
        with pytest.raises(ValueError) as exc_info:
            parse_autoep_config({"enabled": True, "load_balance_coeff": value})
        assert_load_balance_coeff_rejection_message(exc_info.value, value)

    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("value", [0.01, False, "0.01"])
    def test_load_balance_coeff_rejected_by_validate(self, enabled, value):
        config = AutoEPConfig(enabled=enabled, load_balance_coeff=value)

        with pytest.raises(ValueError) as exc_info:
            validate_autoep_config(config, world_size=1, pp_size=1, tp_size=1, sp_size=1)
        assert_load_balance_coeff_rejection_message(exc_info.value, value)

    def test_ep_size_validation_rejects_invalid_topology(self):
        with pytest.raises(ValueError, match="AutoTP"):
            validate_autoep_config(AutoEPConfig(enabled=True, autoep_size=2),
                                   world_size=8,
                                   pp_size=1,
                                   tp_size=2,
                                   sp_size=1)
        with pytest.raises(ValueError, match="must divide the stage size"):
            validate_autoep_config(AutoEPConfig(enabled=True, autoep_size=3),
                                   world_size=8,
                                   pp_size=1,
                                   tp_size=1,
                                   sp_size=1)
        with pytest.raises(ValueError, match="exceeds num_experts"):
            validate_autoep_post_detection(AutoEPConfig(enabled=True, autoep_size=16), [_make_spec(num_experts=8)])

    def test_configure_expert_parallel_uses_engine_mpu_sequence_parallel_size(self, monkeypatch):

        class SequenceParallelMPU:

            def get_model_parallel_world_size(self):
                return 1

            def get_sequence_parallel_world_size(self):
                return 2

        class EmptyAutoEP:

            def __init__(self, model, config):
                pass

            def ep_parser(self):
                return []

        observed = {}

        def record_validate(config, world_size, pp_size, tp_size, sp_size):
            observed["validate"] = {
                "world_size": world_size,
                "pp_size": pp_size,
                "tp_size": tp_size,
                "sp_size": sp_size,
            }

        def record_create(**kwargs):
            observed["create"] = kwargs

        monkeypatch.setattr(groups, "mpu", None)
        monkeypatch.setattr(groups, "_get_sequence_parallel_world_size", lambda: 1)
        monkeypatch.setattr(groups, "_create_expert_and_data_parallel", record_create)
        monkeypatch.setattr(groups, "_get_expert_parallel_group", lambda name: object())
        monkeypatch.setattr(ds_engine.dist, "get_world_size", lambda: 4)
        monkeypatch.setattr(ds_engine.dist, "get_rank", lambda group=None: 0)
        monkeypatch.setattr("deepspeed.module_inject.auto_ep.AutoEP", EmptyAutoEP)
        monkeypatch.setattr("deepspeed.module_inject.auto_ep_config.validate_autoep_config", record_validate)

        engine = object.__new__(DeepSpeedEngine)
        engine.mpu = SequenceParallelMPU()
        engine._config = SimpleNamespace(
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=2),
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            use_data_before_expert_parallel_=False,
        )

        engine._configure_expert_parallel(model=nn.Module())

        assert groups.mpu is None
        assert observed["validate"]["sp_size"] == 2
        assert observed["create"]["mp_size"] == 2
        assert observed["create"]["mp_mode"] == "sp"

    def test_configure_expert_parallel_rejects_bwc_tensor_model_parallel_mpu(self, monkeypatch):

        class TensorParallelMPU:

            def get_tensor_model_parallel_world_size(self):
                return 2

        monkeypatch.setattr(groups, "_get_sequence_parallel_world_size", lambda: 1)

        engine = object.__new__(DeepSpeedEngine)
        engine.mpu = TensorParallelMPU()
        engine._config = SimpleNamespace(
            expert_parallel_config=AutoEPConfig(enabled=True, autoep_size=2),
            tensor_parallel_config=SimpleNamespace(autotp_size=1),
            use_data_before_expert_parallel_=False,
        )

        with pytest.raises(ValueError, match="bwc_tensor_model_parallel_world_size=2"):
            engine._configure_expert_parallel(model=nn.Module())

    def test_autoep_sequence_parallel_size_falls_back_to_groups_helper(self, monkeypatch):
        monkeypatch.setattr(groups, "_get_sequence_parallel_world_size", lambda: 3)

        engine = object.__new__(DeepSpeedEngine)
        engine.mpu = object()

        assert engine._autoep_sequence_parallel_world_size() == 3

    def test_preset_registry_core_contracts(self):
        assert set(PRESET_MODELS) == {"mixtral", "qwen3_moe", "qwen3_5_moe", "deepseek_v2", "deepseek_v3"}
        assert preset_name_for_hf_model_type("mixtral") == "mixtral"
        assert preset_name_for_hf_model_type("qwen2_moe") == "qwen3_moe"
        assert preset_name_for_hf_model_type("llama4_text") is None

        qwen35 = unsupported_preset_for_hf_model_type("qwen3_5_moe")
        assert qwen35 is not None
        assert "qwen3_5_moe_text" in qwen35[1].unsupported_hf_model_type_notes["qwen3_5_moe"]
        assert PRESET_MODELS["deepseek_v2"].supports_expert_bias is False
        assert PRESET_MODELS["deepseek_v3"].unsupported_router_bias_names == ()

    def test_fill_autoep_config_from_hf_defaults(self):
        config = AutoEPConfig(enabled=True, autoep_size=2)

        fill_autoep_config_from_hf(config, MockDeepSeekV3Config())

        assert config.num_expert_groups == 4
        assert config.num_limited_groups == 2
        assert config.route_scale == pytest.approx(2.5)

    def test_fill_autoep_config_from_hf_preserves_explicit_values(self):
        config = AutoEPConfig(enabled=True,
                              autoep_size=2,
                              num_expert_groups=8,
                              num_limited_groups=1,
                              routed_scaling_factor=3.0,
                              route_scale=3.0)

        fill_autoep_config_from_hf(config, MockDeepSeekV3Config())

        assert config.num_expert_groups == 8
        assert config.num_limited_groups == 1
        assert config.route_scale == pytest.approx(3.0)

    @pytest.mark.parametrize("value", ["2.5", True, float("nan"), float("inf")])
    def test_invalid_routed_scaling_factor_rejected(self, value):
        with pytest.raises(ValueError, match="routed_scaling_factor"):
            _resolve_route_scale(AutoEPConfig(enabled=True, routed_scaling_factor=value), None)


class TestRoutingAndLayerSemantics:

    def test_router_route_scale_and_group_limited_routing(self):
        base = TokenChoiceTopKRouter(64, 8, 4, 2, 2, "softmax", False, 1.0, False)
        scaled = TokenChoiceTopKRouter(64, 8, 4, 2, 2, "softmax", False, 2.5, False)
        scaled.load_state_dict(base.state_dict())
        x = torch.randn(50, 64)

        base_scores, base_experts, base_counts = base(x)
        scaled_scores, scaled_experts, scaled_counts = scaled(x)

        assert torch.equal(scaled_experts, base_experts)
        assert torch.allclose(scaled_scores, base_scores * 2.5, atol=1e-5)
        assert torch.equal(scaled_counts, base_counts)
        assert base_counts.shape == (8, )

    def test_grouped_experts_and_token_reorderer(self):
        experts = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=False)
        nn.init.normal_(experts.w1, std=0.02)
        nn.init.normal_(experts.w2, std=0.02)
        nn.init.normal_(experts.w3, std=0.02)
        out = experts(torch.randn(8, 64), torch.tensor([2, 2, 2, 2]))
        assert out.shape == (8, 64)
        assert not torch.isnan(out).any()

        top_scores = torch.randn(20, 2)
        selected_experts = torch.randint(0, 4, (20, 2))
        scores_sorted, indices_sorted, counts = TokenReorderer(num_experts=4, top_k=2)(top_scores, selected_experts)
        assert scores_sorted.shape == (40, )
        assert set(indices_sorted.tolist()) == set(range(40))
        assert torch.equal(counts, torch.bincount(selected_experts.reshape(-1), minlength=4).to(counts.dtype))

    def test_score_application_and_combine(self):
        x = torch.randn(4, 8)
        scores = torch.tensor([0.25, 0.5, 0.75, 1.0])
        expected = x.float() * scores.reshape(-1, 1)
        torch.testing.assert_close(apply_scores_before_experts_if_enabled(x, scores, "pre"), expected.to(x.dtype))

        spec = _make_spec(score_apply="post")
        assert resolve_score_apply_mode(spec, "auto") == "post"
        expert_output = torch.ones(4, 8)
        top_scores = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        out = combine_from_routed(expert_output, top_scores, torch.arange(4), 2, "post", "weighted_sum", (1, 2, 8))
        torch.testing.assert_close(out[0, 0], torch.ones(8))

    def test_autoep_layer_forward_and_expert_bias_rejection(self):
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        layer = AutoEPMoELayer(_make_spec(route_scale=2.5),
                               source,
                               ep_size=1,
                               ep_rank=0,
                               config=_runtime_config(enabled=True, autoep_size=1))
        out = layer(torch.randn(2, 8, 64))
        assert layer._is_autoep_layer is True
        assert layer.num_experts == 4
        assert layer.router.route_scale == pytest.approx(2.5)
        assert out.shape == (2, 8, 64)
        assert not torch.isnan(out).any()

        with pytest.raises(ValueError, match="load_balance_coeff/expert_bias"):
            AutoEPMoELayer(_make_spec(model_family="no_bias_family", supports_expert_bias=False),
                           source,
                           ep_size=1,
                           ep_rank=0,
                           config=AutoEPConfig(enabled=True, autoep_size=1, load_balance_coeff=0.02))


class TestModelDetectionAndReplacement:

    def test_mixtral_detect_replace_and_mock_forward(self):
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=1, preset_model="mixtral"))
        specs = auto_ep.ep_parser()

        assert len(specs) == 2
        assert specs[0].model_family == "mixtral"
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)
        assert isinstance(model.model.layers[0].mlp, AutoEPMoELayer)
        assert model(torch.randn(1, 4, 64)).shape == (1, 4, 100)

    def test_fused_replacement_preserves_frozen_experts_and_trainable_router(self):
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1).to(dtype=torch.bfloat16)
        source = model.model.layers[0].mlp
        source.experts.gate_up_proj.requires_grad_(False)
        source.experts.down_proj.requires_grad_(False)
        source.gate.weight.requires_grad_(True)

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=1, preset_model="mixtral"))
        spec = auto_ep.ep_parser()[0]
        auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)

        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, AutoEPMoELayer)
        assert replaced.experts.w1.requires_grad is False
        assert replaced.experts.w2.requires_grad is False
        assert replaced.experts.w3.requires_grad is False
        assert replaced.router.gate.weight.requires_grad is True
        _assert_same_dtype_device(replaced.router.gate.weight, source.gate.weight)
        _assert_same_dtype_device(replaced.experts.w1, source.experts.gate_up_proj)
        _assert_same_dtype_device(replaced.experts.w2, source.experts.down_proj)
        _assert_same_dtype_device(replaced.experts.w3, source.experts.gate_up_proj)

    def test_module_list_replacement_preserves_frozen_experts_and_trainable_router(self, monkeypatch):
        monkeypatch.setattr(get_preset_adapter("deepseek_v3"), "_installed_transformers_version", lambda: "5.0.0")
        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=4).to(dtype=torch.bfloat16)
        source = model.model.layers[0].mlp
        for expert in source.experts:
            for param in expert.parameters():
                param.requires_grad_(False)
        source.gate.weight.requires_grad_(True)
        source.gate.e_score_correction_bias = nn.Parameter(torch.zeros(4,
                                                                       dtype=source.gate.weight.dtype,
                                                                       device=source.gate.weight.device),
                                                           requires_grad=True)

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        spec = auto_ep.ep_parser()[0]
        auto_ep.replace_moe_layer(spec, ep_size=2, ep_rank=0)

        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, AutoEPMoELayer)
        assert replaced.experts.w1.requires_grad is False
        assert replaced.experts.w2.requires_grad is False
        assert replaced.experts.w3.requires_grad is False
        assert replaced.router.gate.weight.requires_grad is True
        assert replaced.router.e_score_correction_bias.requires_grad is True
        _assert_same_dtype_device(replaced.router.gate.weight, source.gate.weight)
        _assert_same_dtype_device(replaced.router.e_score_correction_bias, source.gate.e_score_correction_bias)
        _assert_same_dtype_device(replaced.experts.w1, source.experts[0].gate_proj.weight)
        _assert_same_dtype_device(replaced.experts.w2, source.experts[0].down_proj.weight)
        _assert_same_dtype_device(replaced.experts.w3, source.experts[0].up_proj.weight)

    def test_module_list_mixed_expert_requires_grad_flags_are_rejected(self, monkeypatch):
        monkeypatch.setattr(get_preset_adapter("deepseek_v3"), "_installed_transformers_version", lambda: "5.0.0")
        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=4)
        source = model.model.layers[0].mlp
        source.experts[0].gate_proj.weight.requires_grad_(False)
        source.experts[1].gate_proj.weight.requires_grad_(True)

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        spec = auto_ep.ep_parser()[0]
        with pytest.raises(ValueError, match="mixed requires_grad flags"):
            auto_ep.replace_moe_layer(spec, ep_size=2, ep_rank=0)

        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=4)
        source = model.model.layers[0].mlp
        source.experts[1].gate_proj.to(dtype=torch.float64)

        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        spec = auto_ep.ep_parser()[0]
        with pytest.raises(ValueError, match="mixed dtype/device"):
            auto_ep.replace_moe_layer(spec, ep_size=2, ep_rank=0)

    def test_hf_mixtral_causal_lm_matches_autoep_with_router_logits(self):
        transformers = pytest.importorskip("transformers")
        skip_unless_transformers_has(transformers,
                                     "MixtralConfig",
                                     "MixtralForCausalLM",
                                     min_version="5.0.0",
                                     reason="Mixtral AutoEP router-logit capture")

        torch.manual_seed(1234)
        config = tiny_mixtral_config(transformers)
        native_model, autoep_model = state_matched_models(transformers.MixtralForCausalLM, config)
        replace_autoep_layers(autoep_model, "mixtral")
        assert_causal_lm_outputs_close(native_model,
                                       autoep_model,
                                       output_router_logits=True,
                                       compare_router_logits=True,
                                       compare_aux_loss=True,
                                       compare_logits=False)

    def test_qwen_adapter_guards(self, monkeypatch):
        monkeypatch.setattr(get_preset_adapter("qwen3_moe"), "_installed_transformers_version", lambda: "5.0.0")
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
        model.config.model_type = "qwen2_moe"
        model.config.num_experts = model.config.num_local_experts

        specs = AutoEP(model, _runtime_config(enabled=True, autoep_size=1)).ep_parser()

        assert len(specs) == 1
        assert specs[0].model_family == "qwen3_moe"

        model.config.model_type = "qwen3_5_moe"
        with pytest.raises(ValueError, match="qwen3_5_moe_text"):
            AutoEP(model, _runtime_config(enabled=True, autoep_size=1))._resolve_presets()

    def test_deepseek_v3_detection_and_score_correction_bias_copy(self, monkeypatch):
        monkeypatch.setattr(get_preset_adapter("deepseek_v3"), "_installed_transformers_version", lambda: "5.0.0")
        model = MockDeepSeekV3Transformer(num_layers=1, num_experts=8)
        auto_ep = AutoEP(model, _runtime_config(enabled=True, autoep_size=2))
        specs = auto_ep.ep_parser()

        assert len(specs) == 1
        assert specs[0].model_family == "deepseek_v3"
        assert specs[0].expert_storage == "module_list"
        assert specs[0].expert_w1_name == "gate_proj"
        assert specs[0].has_shared_experts is True

        source_bias = torch.arange(8, dtype=torch.float32)
        model.model.layers[0].mlp.gate.e_score_correction_bias = nn.Parameter(source_bias.clone())

        auto_ep.replace_moe_layer(specs[0], ep_size=2, ep_rank=0)

        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, AutoEPMoELayer)
        assert replaced.router.e_score_correction_bias is not None
        torch.testing.assert_close(replaced.router.e_score_correction_bias, source_bias)
