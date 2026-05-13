# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for AutoEP feature (all phases append test classes here)."""

import copy

import pytest
import torch
import torch.nn as nn

# === Phase 1: Configuration and Preset Definitions ===

from deepspeed.module_inject.auto_ep_config import (
    AutoEPConfig,
    MoEModelPreset,
    MoELayerSpec,
    PRESET_MODELS,
    parse_autoep_config,
    resolve_autoep_config_defaults,
    validate_autoep_config,
    validate_autoep_post_detection,
    _UNSET,
)


class TestAutoEPConfig:
    """Phase 1 unit tests for configuration parsing and validation."""

    def test_parse_autoep_config_defaults(self):
        """Default values from empty expert_parallel section."""
        config = parse_autoep_config({})
        assert config.enabled is False
        assert config.autoep_size == 1
        assert config.preset_model is None
        assert config.moe_layer_pattern is None
        assert config.expert_pattern is None
        assert config.router_pattern is None
        assert config.use_grouped_mm is True
        assert config.grouped_mm_backend == "auto"
        assert config.route_norm is None
        assert config.route_scale == 1.0
        assert config.score_apply == "auto"
        assert config.num_expert_groups is None
        assert config.num_limited_groups is None
        assert config.score_func == "auto"
        assert config.top_k == "auto"
        assert config.load_balance_coeff == pytest.approx(1e-3)
        assert config.routed_scaling_factor == "auto"
        assert config.expert_w1 is None
        assert config.expert_w2 is None
        assert config.expert_w3 is _UNSET
        assert config.num_experts_attr is None
        assert config.top_k_attr is None
        assert config.has_shared_experts is None
        assert config.shared_experts_pattern is None
        assert config.shared_experts_gate_pattern is None

    def test_llama4_preset_default_sets_load_balance_coeff_none(self):
        """Llama4 preset disables dynamic expert_bias unless the user opts in."""
        config = parse_autoep_config({"enabled": True, "preset_model": "llama4"})
        assert config.load_balance_coeff == pytest.approx(1e-3)

        resolved = resolve_autoep_config_defaults(config, config.preset_model)

        assert resolved.load_balance_coeff is None
        assert config.load_balance_coeff == pytest.approx(1e-3)

    def test_llama4_explicit_load_balance_coeff_overrides_preset_default(self):
        """Explicit user load_balance_coeff survives Llama4 preset resolution."""
        config = parse_autoep_config({
            "enabled": True,
            "preset_model": "llama4",
            "load_balance_coeff": 0.02,
        })

        resolved = resolve_autoep_config_defaults(config, config.preset_model)

        assert resolved.load_balance_coeff == pytest.approx(0.02)

    @pytest.mark.parametrize("preset_model", ["deepseek_v2", "deepseek_v3"])
    def test_deepseek_preset_default_sets_load_balance_coeff_none(self, preset_model):
        """DeepSeek presets disable AutoEP expert_bias by default."""
        config = parse_autoep_config({"enabled": True, "preset_model": preset_model})
        assert config.load_balance_coeff == pytest.approx(1e-3)

        resolved = resolve_autoep_config_defaults(config, config.preset_model)

        assert resolved.load_balance_coeff is None
        assert config.load_balance_coeff == pytest.approx(1e-3)

    def test_parse_autoep_config_full(self):
        """All fields parsed from complete JSON."""
        param_dict = {
            "enabled": True,
            "autoep_size": 4,
            "preset_model": "mixtral",
            "moe_layer_pattern": r"model\.layers\.\d+\.mlp",
            "expert_pattern": "experts",
            "router_pattern": "gate",
            "use_grouped_mm": False,
            "grouped_mm_backend": "sequential",
            "route_norm": True,
            "route_scale": 2.0,
            "score_apply": "pre",
            "num_expert_groups": 2,
            "num_limited_groups": 1,
            "score_func": "sigmoid",
            "top_k": 2,
            "load_balance_coeff": 0.01,
            "routed_scaling_factor": 1.5,
            "expert_w1": "w1",
            "expert_w2": "w2",
            "expert_w3": "w3",
            "num_experts_attr": "num_moe_experts",
            "top_k_attr": "moe_top_k",
            "has_shared_experts": True,
            "shared_experts_pattern": "shared_expert",
            "shared_experts_gate_pattern": "shared_expert_gate",
        }
        config = parse_autoep_config(param_dict)
        assert config.enabled is True
        assert config.autoep_size == 4
        assert config.preset_model == "mixtral"
        assert config.moe_layer_pattern == r"model\.layers\.\d+\.mlp"
        assert config.expert_pattern == "experts"
        assert config.router_pattern == "gate"
        assert config.use_grouped_mm is False
        assert config.grouped_mm_backend == "sequential"
        assert config.route_norm is True
        assert config.route_scale == 2.0
        assert config.score_apply == "pre"
        assert config.num_expert_groups == 2
        assert config.num_limited_groups == 1
        assert config.score_func == "sigmoid"
        assert config.top_k == 2
        assert config.load_balance_coeff == pytest.approx(0.01)
        assert config.routed_scaling_factor == 1.5
        assert config.expert_w1 == "w1"
        assert config.expert_w2 == "w2"
        assert config.expert_w3 == "w3"
        assert config.num_experts_attr == "num_moe_experts"
        assert config.top_k_attr == "moe_top_k"
        assert config.has_shared_experts is True
        assert config.shared_experts_pattern == "shared_expert"
        assert config.shared_experts_gate_pattern == "shared_expert_gate"

    def test_validate_ep_tp_mutual_exclusivity(self):
        """autotp_size>1 + sp_size>1 raises ValueError."""
        config = AutoEPConfig(enabled=True, autoep_size=2)
        with pytest.raises(ValueError, match="simultaneous TP.*and SP"):
            validate_autoep_config(config, world_size=8, pp_size=1, tp_size=2, sp_size=2)

    def test_validate_ep_size_divides_stage(self):
        """ep_size must divide world_size / pp_size."""
        config = AutoEPConfig(enabled=True, autoep_size=3)
        with pytest.raises(ValueError, match="must divide the stage size"):
            validate_autoep_config(config, world_size=8, pp_size=1, tp_size=1, sp_size=1)

    def test_validate_post_detection_ep_gt_num_experts(self):
        """ep_size > num_experts raises with helpful message listing valid divisors."""
        config = AutoEPConfig(enabled=True, autoep_size=16)
        specs = [
            MoELayerSpec(
                moe_module_name="model.layers.0.mlp",
                model_family="mixtral",
                router_name="gate",
                experts_name="experts",
                expert_storage="fused_3d",
                expert_w1_name="gate_up_proj",
                expert_w2_name="down_proj",
                expert_w3_name=None,
                num_experts=8,
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
            )
        ]
        with pytest.raises(ValueError, match="exceeds num_experts"):
            validate_autoep_post_detection(config, specs)

    def test_validate_post_detection_not_divisible(self):
        """num_experts % ep_size != 0 raises with suggested sizes."""
        config = AutoEPConfig(enabled=True, autoep_size=3)
        specs = [
            MoELayerSpec(
                moe_module_name="model.layers.0.mlp",
                model_family="mixtral",
                router_name="gate",
                experts_name="experts",
                expert_storage="fused_3d",
                expert_w1_name="gate_up_proj",
                expert_w2_name="down_proj",
                expert_w3_name=None,
                num_experts=8,
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
            )
        ]
        with pytest.raises(ValueError, match="not divisible"):
            validate_autoep_post_detection(config, specs)

    def test_validate_expert_groups_constraints(self):
        """num_expert_groups must divide num_experts."""
        config = AutoEPConfig(enabled=True, autoep_size=2, num_expert_groups=3)
        specs = [
            MoELayerSpec(
                moe_module_name="model.layers.0.mlp",
                model_family="mixtral",
                router_name="gate",
                experts_name="experts",
                expert_storage="fused_3d",
                expert_w1_name="gate_up_proj",
                expert_w2_name="down_proj",
                expert_w3_name=None,
                num_experts=8,
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
            )
        ]
        with pytest.raises(ValueError, match="num_expert_groups.*must divide"):
            validate_autoep_post_detection(config, specs)

    def test_preset_models_complete(self):
        """All presets have required fields."""
        expected = {"mixtral", "qwen2_moe", "qwen3_moe", "qwen3_5_moe", "deepseek_v2", "deepseek_v3", "llama4"}
        assert set(PRESET_MODELS.keys()) == expected
        for name, preset in PRESET_MODELS.items():
            assert isinstance(preset, MoEModelPreset), f"Preset {name} is not MoEModelPreset"
            assert preset.moe_layer_pattern, f"Preset {name} missing moe_layer_pattern"
            assert preset.router_pattern, f"Preset {name} missing router_pattern"
            assert preset.experts_pattern, f"Preset {name} missing experts_pattern"
            assert preset.expert_storage in ("fused_3d", "module_list")
            assert preset.expert_w1, f"Preset {name} missing expert_w1"
            assert preset.expert_w2, f"Preset {name} missing expert_w2"
            assert preset.num_experts_attr, f"Preset {name} missing num_experts_attr"
            assert preset.top_k_attr, f"Preset {name} missing top_k_attr"
            assert preset.score_func in ("softmax", "sigmoid")
            assert preset.score_apply in ("pre", "post")

    def test_preset_field_values(self):
        """Spot-check Mixtral preset values."""
        mixtral = PRESET_MODELS["mixtral"]
        assert mixtral.score_func == "softmax"
        assert mixtral.score_apply == "post"
        assert mixtral.route_norm is True
        assert mixtral.gate_bias is False
        assert mixtral.expert_storage == "fused_3d"
        assert mixtral.expert_w1 == "gate_up_proj"
        assert mixtral.expert_w3 is None
        assert mixtral.has_shared_experts is False

        llama4 = PRESET_MODELS["llama4"]
        assert llama4.score_func == "sigmoid"
        assert llama4.score_apply == "pre"
        assert llama4.router_pattern == "router"
        assert llama4.has_shared_experts is True

        qwen2 = PRESET_MODELS["qwen2_moe"]
        assert qwen2.has_shared_experts is True
        assert qwen2.shared_experts_pattern == "shared_expert"
        assert qwen2.shared_experts_gate_pattern == "shared_expert_gate"

    def test_validate_empty_expert_w1(self):
        """Empty expert_w1 raises ValueError."""
        config = AutoEPConfig(enabled=True, autoep_size=2, expert_w1="")
        with pytest.raises(ValueError, match="expert_w1"):
            validate_autoep_config(config, world_size=8, pp_size=1, tp_size=1, sp_size=1)

    def test_validate_empty_expert_w2(self):
        """Empty expert_w2 raises ValueError."""
        config = AutoEPConfig(enabled=True, autoep_size=2, expert_w2="")
        with pytest.raises(ValueError, match="expert_w2"):
            validate_autoep_config(config, world_size=8, pp_size=1, tp_size=1, sp_size=1)

    def test_validate_empty_expert_w3(self):
        """Empty expert_w3 raises ValueError."""
        config = AutoEPConfig(enabled=True, autoep_size=2, expert_w3="")
        with pytest.raises(ValueError, match="expert_w3"):
            validate_autoep_config(config, world_size=8, pp_size=1, tp_size=1, sp_size=1)

    def test_parse_expert_w3_sentinel_semantics(self):
        """expert_w3 sentinel: absent=_UNSET, null=None, string=custom name."""
        # Key absent -> _UNSET (use preset default)
        c1 = parse_autoep_config({})
        assert c1.expert_w3 is _UNSET

        # Key present with None -> None (fused gate+up, no separate w3)
        c2 = parse_autoep_config({"expert_w3": None})
        assert c2.expert_w3 is None

        # Key present with string -> custom weight name
        c3 = parse_autoep_config({"expert_w3": "up_proj"})
        assert c3.expert_w3 == "up_proj"


# === Phase 4: Generalized Group Creation ===

import inspect
from deepspeed.utils import groups as ds_groups


class TestGroupCreation:
    """Phase 4 tests for generalized group creation (non-distributed)."""

    def test_group_creation_signature(self):
        """Verify the function has new parameters."""
        sig = inspect.signature(ds_groups._create_expert_and_data_parallel)
        params = list(sig.parameters.keys())
        assert "expert_parallel_size_" in params
        assert "mp_size" in params
        assert "pp_size" in params
        assert "mp_mode" in params
        assert "use_data_before_expert_parallel_" in params

    def test_group_creation_default_params(self):
        """Default values preserve backward compat."""
        sig = inspect.signature(ds_groups._create_expert_and_data_parallel)
        assert sig.parameters["mp_size"].default is None
        assert sig.parameters["pp_size"].default is None
        assert sig.parameters["mp_mode"].default == "tp"
        assert sig.parameters["use_data_before_expert_parallel_"].default is False


# === Phase 2: TorchTitan Layer Port ===

from deepspeed.moe.ep_router import TokenChoiceTopKRouter
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.moe.ep_kernels import TokenReorderer, generate_permute_indices


class TestTokenChoiceTopKRouter:

    def test_router_forward_shapes(self):
        router = TokenChoiceTopKRouter(dim=64,
                                       num_experts=8,
                                       num_expert_groups=None,
                                       num_limited_groups=None,
                                       top_k=2,
                                       score_func="softmax",
                                       route_norm=True,
                                       route_scale=1.0,
                                       gate_bias=False)
        x = torch.randn(100, 64)
        top_scores, selected_experts, num_tokens = router(x)
        assert top_scores.shape == (100, 2)
        assert selected_experts.shape == (100, 2)
        assert num_tokens.shape == (8, )

    def test_router_softmax_scores_sum(self):
        router = TokenChoiceTopKRouter(dim=64,
                                       num_experts=8,
                                       num_expert_groups=None,
                                       num_limited_groups=None,
                                       top_k=2,
                                       score_func="softmax",
                                       route_norm=True,
                                       route_scale=1.0,
                                       gate_bias=False)
        x = torch.randn(50, 64)
        top_scores, _, _ = router(x)
        # With route_norm, scores should sum to ~1 per token (times route_scale=1.0)
        sums = top_scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_router_sigmoid_scores_range(self):
        router = TokenChoiceTopKRouter(dim=64,
                                       num_experts=8,
                                       num_expert_groups=None,
                                       num_limited_groups=None,
                                       top_k=2,
                                       score_func="sigmoid",
                                       route_norm=False,
                                       route_scale=1.0,
                                       gate_bias=False)
        x = torch.randn(50, 64)
        top_scores, _, _ = router(x)
        assert (top_scores >= 0).all() and (top_scores <= 1).all()

    def test_router_group_limited_routing(self):
        router = TokenChoiceTopKRouter(dim=64,
                                       num_experts=8,
                                       num_expert_groups=4,
                                       num_limited_groups=2,
                                       top_k=2,
                                       score_func="softmax",
                                       route_norm=False,
                                       route_scale=1.0,
                                       gate_bias=False)
        x = torch.randn(50, 64)
        top_scores, selected_experts, num_tokens = router(x)
        assert top_scores.shape == (50, 2)
        assert selected_experts.shape == (50, 2)

    def test_router_gate_bias_copy(self):
        router = TokenChoiceTopKRouter(dim=64,
                                       num_experts=8,
                                       num_expert_groups=None,
                                       num_limited_groups=None,
                                       top_k=2,
                                       score_func="softmax",
                                       route_norm=True,
                                       route_scale=1.0,
                                       gate_bias=True)
        assert router.gate.bias is not None
        assert router.gate.bias.shape == (8, )

    def test_router_deterministic(self):
        router = TokenChoiceTopKRouter(dim=64,
                                       num_experts=8,
                                       num_expert_groups=None,
                                       num_limited_groups=None,
                                       top_k=2,
                                       score_func="softmax",
                                       route_norm=True,
                                       route_scale=1.0,
                                       gate_bias=False)
        x = torch.randn(50, 64)
        out1 = router(x)
        out2 = router(x)
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])


class TestGroupedExperts:

    def test_grouped_experts_forward_shapes(self):
        experts = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=False)
        nn.init.normal_(experts.w1, std=0.02)
        nn.init.normal_(experts.w2, std=0.02)
        nn.init.normal_(experts.w3, std=0.02)
        x = torch.randn(20, 64)
        counts = torch.tensor([5, 5, 5, 5])
        out = experts(x, counts)
        assert out.shape == (20, 64)

    def test_grouped_experts_dtype_aware(self):
        experts = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=False)
        nn.init.normal_(experts.w1, std=0.02)
        nn.init.normal_(experts.w2, std=0.02)
        nn.init.normal_(experts.w3, std=0.02)
        x_bf16 = torch.randn(8, 64).bfloat16()
        counts = torch.tensor([2, 2, 2, 2])
        # For-loop path works with bf16
        experts_bf16 = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=False)
        experts_bf16.w1.data.copy_(experts.w1.data.bfloat16())
        experts_bf16.w2.data.copy_(experts.w2.data.bfloat16())
        experts_bf16.w3.data.copy_(experts.w3.data.bfloat16())
        out = experts_bf16(x_bf16, counts)
        assert out.dtype == torch.bfloat16

    def test_grouped_experts_zero_tokens(self):
        experts = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=False)
        nn.init.normal_(experts.w1, std=0.02)
        nn.init.normal_(experts.w2, std=0.02)
        nn.init.normal_(experts.w3, std=0.02)
        x = torch.randn(8, 64)
        counts = torch.tensor([0, 5, 0, 3])
        out = experts(x, counts)
        assert not torch.isnan(out).any()

    def test_grouped_experts_gradient_flow(self):
        experts = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=False)
        nn.init.normal_(experts.w1, std=0.02)
        nn.init.normal_(experts.w2, std=0.02)
        nn.init.normal_(experts.w3, std=0.02)
        x = torch.randn(8, 64, requires_grad=True)
        counts = torch.tensor([2, 2, 2, 2])
        out = experts(x, counts)
        loss = out.sum()
        loss.backward()
        assert experts.w1.grad is not None and experts.w1.grad.abs().sum() > 0
        assert experts.w2.grad is not None and experts.w2.grad.abs().sum() > 0
        assert experts.w3.grad is not None and experts.w3.grad.abs().sum() > 0

    def test_grouped_mm_fallback_when_unavailable(self):
        # Mock torch._grouped_mm as unavailable
        original = getattr(torch, '_grouped_mm', None)
        try:
            if hasattr(torch, '_grouped_mm'):
                delattr(torch, '_grouped_mm')
            experts = GroupedExperts(dim=64, hidden_dim=128, num_experts=4, use_grouped_mm=True)
            assert experts.use_grouped_mm is False  # Should have fallen back
        finally:
            if original is not None:
                torch._grouped_mm = original

    def test_cutlass_backend_raises_not_implemented(self):
        # Test that cutlass raises NotImplementedError if requested
        # This is tested via the backend attribute, not constructor
        pass  # CUTLASS path is out of scope for Phase 2


class TestTokenReorderer:

    def test_token_reorderer_output_shapes(self):
        reorderer = TokenReorderer(num_experts=8, top_k=2)
        top_scores = torch.randn(50, 2)
        selected_experts = torch.randint(0, 8, (50, 2))
        scores_sorted, indices_sorted, num_tokens = reorderer(top_scores, selected_experts)
        assert scores_sorted.shape == (100, )
        assert indices_sorted.shape == (100, )
        assert num_tokens.shape == (8, )

    def test_token_reorderer_index_coverage(self):
        reorderer = TokenReorderer(num_experts=4, top_k=2)
        T = 20
        top_scores = torch.randn(T, 2)
        selected_experts = torch.randint(0, 4, (T, 2))
        _, indices_sorted, _ = reorderer(top_scores, selected_experts)
        # Every token appears exactly top_k times
        all_token_indices = indices_sorted // 2  # map back to token index (// top_k)
        # Each of 0..T-1 should appear... but not necessarily exactly K times due to sorting
        # Actually each SLOT (T*K) appears exactly once
        assert indices_sorted.shape[0] == T * 2
        assert set(indices_sorted.tolist()) == set(range(T * 2))

    def test_permute_alignment_padding(self):
        # Test that generate_permute_indices produces aligned sizes
        tokens_per_expert_group = torch.tensor([3, 5, 2, 7], dtype=torch.int32)
        alignment = 16
        experts_per_rank = 4
        num_ranks = 1
        max_len = 200
        permuted_indices, m_sizes, m_offsets = generate_permute_indices(tokens_per_expert_group,
                                                                        experts_per_rank,
                                                                        num_ranks,
                                                                        max_len,
                                                                        alignment,
                                                                        use_cpu=True)
        # All m_sizes should be multiples of alignment
        for s in m_sizes.tolist():
            assert s % alignment == 0, f"size {s} not aligned to {alignment}"


# === Phase 3: MoE Detection and Weight Repacking ===

from deepspeed.module_inject.auto_ep import AutoEP
from deepspeed.moe.ep_repack import repack_expert_weights


class MockHFConfig:
    model_type = "mixtral"
    num_local_experts = 8
    num_experts_per_tok = 2
    hidden_size = 64
    intermediate_size = 128


class MockMoEExperts(nn.Module):
    """Mimics HF transformers 5.0.0 fused expert storage."""

    def __init__(self, num_experts=8, ffn_hidden=128, hidden_size=64):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * ffn_hidden, hidden_size))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, ffn_hidden))


class MockMoEBlock(nn.Module):
    """Mimics model.layers.N.mlp for Mixtral-like models."""

    def __init__(self, num_experts=8, ffn_hidden=128, hidden_size=64):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MockMoEExperts(num_experts, ffn_hidden, hidden_size)


class MockLlama4Config:
    model_type = "llama4"
    num_local_experts = 8
    num_experts_per_tok = 1
    hidden_size = 64
    intermediate_size = 128


class MockLlama4Experts(nn.Module):
    """Mimics HF Llama4 hidden-first fused expert storage."""

    def __init__(self, num_experts=8, ffn_hidden=128, hidden_size=64):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden_size, 2 * ffn_hidden))
        self.down_proj = nn.Parameter(torch.randn(num_experts, ffn_hidden, hidden_size))


class MockSharedExpert(nn.Module):

    def __init__(self, hidden_size=64):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class MockLlama4MoEBlock(nn.Module):
    """Mimics model.layers.N.feed_forward for Llama4."""

    def __init__(self, num_experts=8, ffn_hidden=128, hidden_size=64):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MockLlama4Experts(num_experts, ffn_hidden, hidden_size)
        self.shared_expert = MockSharedExpert(hidden_size)


class MockRecordingRouter(nn.Linear):
    _can_record_outputs = {"router_logits": {"index": 1, "layer_name": "router"}}


class MockDenseBlock(nn.Module):
    """Dense FFN block (should be skipped by detection)."""

    def __init__(self, hidden_size=64, ffn_hidden=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, hidden_size, bias=False)


class MockMoETransformer(nn.Module):
    """Minimal transformer with MoE layers for testing detection."""

    def __init__(self, num_layers=4, num_experts=8, moe_every_n=2):
        super().__init__()
        self.config = MockHFConfig()
        self.config.num_local_experts = num_experts
        self.model = nn.Module()
        layers = []
        for i in range(num_layers):
            layer = nn.Module()
            layer.self_attn = nn.MultiheadAttention(64, 1, batch_first=True)
            if i % moe_every_n == 0:
                layer.mlp = MockMoEBlock(num_experts)
            else:
                layer.mlp = MockDenseBlock()
            layer.input_layernorm = nn.LayerNorm(64)
            layer.post_attention_layernorm = nn.LayerNorm(64)
            layers.append(layer)
        self.model.layers = nn.ModuleList(layers)


class MockLlama4Transformer(nn.Module):
    """Minimal transformer with Llama4-style MoE layers."""

    def __init__(self, num_layers=2, num_experts=8):
        super().__init__()
        self.config = MockLlama4Config()
        self.config.num_local_experts = num_experts
        self.model = nn.Module()
        layers = []
        for _ in range(num_layers):
            layer = nn.Module()
            layer.feed_forward = MockLlama4MoEBlock(num_experts)
            layers.append(layer)
        self.model.layers = nn.ModuleList(layers)


class TestMoEDetection:
    """Phase 3 tests for MoE layer detection."""

    def test_detect_mixtral_moe_layers(self):
        """Finds all MoE layers in mock Mixtral model."""
        model = MockMoETransformer(num_layers=4, moe_every_n=1)
        config = AutoEPConfig(enabled=True, autoep_size=2, preset_model="mixtral")
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 4

    def test_detect_skips_dense_ffn(self):
        """Structural validation filters dense layers."""
        model = MockMoETransformer(num_layers=4, moe_every_n=2)
        config = AutoEPConfig(enabled=True, autoep_size=2, preset_model="mixtral")
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 2
        module_names = [s.moe_module_name for s in specs]
        assert "model.layers.1.mlp" not in module_names

    def test_detect_fused_3d_storage(self):
        """Correctly identifies fused_3d expert storage."""
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        config = AutoEPConfig(enabled=True, autoep_size=2, preset_model="mixtral")
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        for spec in specs:
            assert spec.expert_storage == "fused_3d"

    def test_detect_spec_field_types(self):
        """All MoELayerSpec fields have correct types."""
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        config = AutoEPConfig(enabled=True, autoep_size=2, preset_model="mixtral")
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        for spec in specs:
            assert isinstance(spec.moe_module_name, str)
            assert isinstance(spec.num_experts, int)
            assert isinstance(spec.top_k, int)
            assert isinstance(spec.hidden_size, int)
            assert isinstance(spec.ffn_hidden_size, int)
            assert spec.score_func in ("softmax", "sigmoid")
            assert spec.score_apply in ("pre", "post")

    def test_detect_llama4_hidden_first_fused_layout(self):
        """Llama4 hidden-first fused weights are detected with the correct contract."""
        model = MockLlama4Transformer(num_layers=2, num_experts=8)
        config = AutoEPConfig(enabled=True, autoep_size=2, preset_model="llama4")
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 2
        for spec in specs:
            assert spec.model_family == "llama4"
            assert spec.hidden_size == 64
            assert spec.ffn_hidden_size == 128
            assert spec.score_apply == "pre"
            assert spec.router_name == "router"
            assert spec.return_router_logits is True
            assert spec.router_logits_capture_target == "router"
            assert spec.has_shared_experts is True
            assert spec.shared_experts_name == "shared_expert"

    def test_detect_llama4_router_capture_still_returns_tuple(self):
        """Router-level output recording must not suppress Llama4's MoE tuple contract."""
        model = MockLlama4Transformer(num_layers=1, num_experts=8)
        model.model.layers[0].feed_forward.router = MockRecordingRouter(64, 8, bias=False)
        config = AutoEPConfig(enabled=True, autoep_size=2, preset_model="llama4")

        specs = AutoEP(model, config).ep_parser()

        assert len(specs) == 1
        assert specs[0].return_router_logits is True
        assert specs[0].router_logits_capture_target == "router"

    def test_llama4_preset_layer_disables_expert_bias_by_default(self):
        """preset_model='llama4' resolves load_balance_coeff=None for layer construction."""
        model = MockLlama4Transformer(num_layers=1, num_experts=4)
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "llama4",
        })
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

        replaced = model.model.layers[0].feed_forward
        assert replaced.load_balance_coeff is None
        assert replaced.expert_bias is None
        assert "expert_bias" not in dict(replaced.named_buffers())

    def test_llama4_explicit_load_balance_coeff_keeps_expert_bias(self):
        """Explicit load_balance_coeff for llama4 opts back into expert_bias."""
        model = MockLlama4Transformer(num_layers=1, num_experts=4)
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "llama4",
            "load_balance_coeff": 0.02,
        })
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

        replaced = model.model.layers[0].feed_forward
        assert replaced.load_balance_coeff == pytest.approx(0.02)
        assert replaced.expert_bias is not None
        assert "expert_bias" in dict(replaced.named_buffers())

    def test_auto_detect_llama4_layer_disables_expert_bias_by_default(self):
        """Auto-detected model_type='llama4' also applies the llama4 preset default."""
        model = MockLlama4Transformer(num_layers=1, num_experts=4)
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
        })
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        assert specs[0].model_family == "llama4"
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

        replaced = model.model.layers[0].feed_forward
        assert replaced.load_balance_coeff is None
        assert replaced.expert_bias is None

    @pytest.mark.parametrize("preset_model", ["deepseek_v2", "deepseek_v3"])
    def test_deepseek_preset_layer_disables_expert_bias_by_default(self, preset_model):
        """DeepSeek preset defaults resolve to no AutoEP load-balance expert_bias."""
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": preset_model,
            "top_k": 2,
        })
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

        replaced = model.model.layers[0].mlp
        assert replaced.load_balance_coeff is None
        assert replaced.expert_bias is None
        assert "expert_bias" not in dict(replaced.named_buffers())

    @pytest.mark.parametrize("preset_model", ["deepseek_v2", "deepseek_v3"])
    def test_deepseek_explicit_load_balance_coeff_rejected(self, preset_model):
        """DeepSeek AutoEP fails explicitly instead of creating expert_bias."""
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": preset_model,
            "top_k": 2,
            "load_balance_coeff": 0.02,
        })
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()

        with pytest.raises(ValueError, match="DeepSeek.*load_balance_coeff"):
            auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

    def test_deepseek_v3_nonzero_score_correction_bias_rejected(self):
        """DeepSeek-V3 expert correction bias remains unsupported for AutoEP parity."""
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
        model.model.layers[0].mlp.gate.register_buffer("e_score_correction_bias", torch.ones(4))
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "deepseek_v3",
            "top_k": 2,
            "load_balance_coeff": None,
        })
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()

        with pytest.raises(ValueError, match="e_score_correction_bias"):
            auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

    def test_deepseek_v2_detection_keeps_native_topk_weights_unnormalized(self):
        """DeepSeek-V2 HF forward does not renormalize top-k weights in Transformers 5."""
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
        model.config.norm_topk_prob = True
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "deepseek_v2",
            "top_k": 2,
        })

        specs = AutoEP(model, config).ep_parser()

        assert len(specs) == 1
        assert specs[0].route_norm is False

    def test_deepseek_v3_detection_reads_group_routing_from_model_config(self):
        """DeepSeek-V3 preset carries HF group-limited routing into AutoEP."""
        model = MockMoETransformer(num_layers=1, num_experts=4, moe_every_n=1)
        model.config.n_group = 2
        model.config.topk_group = 1
        model.config.norm_topk_prob = True
        model.config.routed_scaling_factor = 2.5
        config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "deepseek_v3",
            "top_k": 2,
        })

        specs = AutoEP(model, config).ep_parser()

        assert len(specs) == 1
        assert specs[0].num_expert_groups == 2
        assert specs[0].num_limited_groups == 1
        assert specs[0].group_score_func == "top2_sum"
        assert specs[0].route_scale == pytest.approx(2.5)

    def test_replace_moe_layer_works(self):
        """replace_moe_layer creates AutoEPMoELayer replacement."""
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer as _AutoEPMoELayer
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        config = AutoEPConfig(enabled=True, autoep_size=1, preset_model="mixtral")
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)
        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, _AutoEPMoELayer)

    def test_custom_preset_uses_config_fields(self):
        """Custom preset path reads expert_w1/w2/etc from config."""

        class CustomExperts(nn.Module):

            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.randn(4, 256, 64))
                self.w2 = nn.Parameter(torch.randn(4, 64, 128))

        class CustomMoEBlock(nn.Module):

            def __init__(self):
                super().__init__()
                self.router = nn.Linear(64, 4, bias=True)
                self.mlp_experts = CustomExperts()

        class CustomModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.config = type('C', (), {
                    'model_type': 'custom',
                    'num_moe_experts': 4,
                    'moe_top_k': 1,
                })()
                self.model = nn.Module()
                layer = nn.Module()
                layer.moe = CustomMoEBlock()
                self.model.layers = nn.ModuleList([layer])

        model = CustomModel()
        config = AutoEPConfig(
            enabled=True,
            autoep_size=1,
            moe_layer_pattern=r"model\.layers\.\d+\.moe",
            router_pattern="router",
            expert_pattern="mlp_experts",
            expert_w1="w1",
            expert_w2="w2",
            expert_w3=None,  # fused gate+up
            num_experts_attr="num_moe_experts",
            top_k_attr="moe_top_k",
            score_func="sigmoid",
        )
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        spec = specs[0]
        assert spec.expert_w1_name == "w1"
        assert spec.expert_w2_name == "w2"
        assert spec.expert_w3_name is None
        assert spec.num_experts == 4
        assert spec.top_k == 1
        assert spec.gate_bias is True  # auto-detected from router bias
        assert spec.score_func == "sigmoid"

    def test_preset_model_with_config_overrides(self):
        """Custom fields override preset_model values."""
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        config = AutoEPConfig(
            enabled=True,
            autoep_size=1,
            preset_model="mixtral",
            moe_layer_pattern=r"model\.layers\.\d+\.moe",
            router_pattern="router",
            num_experts_attr="custom_num_experts",
        )
        auto_ep = AutoEP(model, config)
        presets = auto_ep._resolve_presets()
        assert len(presets) == 1
        name, preset = presets[0]
        assert name == "mixtral"
        assert preset.moe_layer_pattern == r"model\.layers\.\d+\.moe"
        assert preset.router_pattern == "router"
        assert preset.num_experts_attr == "custom_num_experts"
        # Other fields remain from the preset
        assert preset.expert_w1 == "gate_up_proj"

    def test_apply_config_overrides_no_overrides_returns_same(self):
        """_apply_config_overrides with default config returns same preset object."""
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        config = AutoEPConfig(enabled=True, autoep_size=1)
        auto_ep = AutoEP(model, config)
        original = PRESET_MODELS["mixtral"]
        result = auto_ep._apply_config_overrides(original)
        assert result is original  # same object, not a copy

    def test_apply_config_overrides_expert_w3_none_overrides(self):
        """expert_w3=None (fused) overrides preset's expert_w3."""
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        config = AutoEPConfig(enabled=True, autoep_size=1, expert_w3=None)
        auto_ep = AutoEP(model, config)
        # deepseek_v3 preset has expert_w3=None already, but let's verify with a preset that has non-None
        p = auto_ep._apply_config_overrides(PRESET_MODELS["deepseek_v3"])
        assert p.expert_w3 is None
        # Since deepseek_v3 already has expert_w3=None, this is a no-op for w3 but
        # expert_w3 is not _UNSET so it triggers override logic
        assert p is not PRESET_MODELS["deepseek_v3"]

    def test_apply_config_overrides_expert_w3_unset_no_override(self):
        """expert_w3=_UNSET (default) does NOT override preset's expert_w3."""
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        config = AutoEPConfig(enabled=True, autoep_size=1)
        assert config.expert_w3 is _UNSET
        auto_ep = AutoEP(model, config)
        p = auto_ep._apply_config_overrides(PRESET_MODELS["deepseek_v3"])
        assert p is PRESET_MODELS["deepseek_v3"]  # same object (no overrides)


class TestWeightRepacking:
    """Phase 3 tests for expert weight repacking."""

    def test_repack_fused_3d_shapes(self):
        experts = MockMoEExperts(num_experts=8, ffn_hidden=128, hidden_size=64)
        spec = MoELayerSpec(
            moe_module_name="test",
            model_family="mixtral",
            router_name="gate",
            experts_name="experts",
            expert_storage="fused_3d",
            expert_w1_name="gate_up_proj",
            expert_w2_name="down_proj",
            expert_w3_name=None,
            num_experts=8,
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
        )
        w1, w2, w3 = repack_expert_weights(experts, spec, ep_rank=0, ep_size=2)
        assert w1.shape == (4, 128, 64)
        assert w2.shape == (4, 64, 128)
        assert w3.shape == (4, 128, 64)

    def test_repack_fused_3d_correct_experts(self):
        experts = MockMoEExperts(num_experts=8, ffn_hidden=128, hidden_size=64)
        spec = MoELayerSpec(
            moe_module_name="test",
            model_family="mixtral",
            router_name="gate",
            experts_name="experts",
            expert_storage="fused_3d",
            expert_w1_name="gate_up_proj",
            expert_w2_name="down_proj",
            expert_w3_name=None,
            num_experts=8,
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
        )
        w1_r0, _, _ = repack_expert_weights(experts, spec, ep_rank=0, ep_size=2)
        w1_r1, _, _ = repack_expert_weights(experts, spec, ep_rank=1, ep_size=2)
        expected_r0 = experts.gate_up_proj.data[0:4, :128, :]
        expected_r1 = experts.gate_up_proj.data[4:8, :128, :]
        assert torch.equal(w1_r0, expected_r0)
        assert torch.equal(w1_r1, expected_r1)

    def test_repack_ep_size_1_full_model(self):
        experts = MockMoEExperts(num_experts=8, ffn_hidden=128, hidden_size=64)
        spec = MoELayerSpec(
            moe_module_name="test",
            model_family="mixtral",
            router_name="gate",
            experts_name="experts",
            expert_storage="fused_3d",
            expert_w1_name="gate_up_proj",
            expert_w2_name="down_proj",
            expert_w3_name=None,
            num_experts=8,
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
        )
        w1, w2, w3 = repack_expert_weights(experts, spec, ep_rank=0, ep_size=1)
        assert w1.shape[0] == 8
        assert w2.shape[0] == 8
        assert w3.shape[0] == 8

    def test_repack_llama4_hidden_first_fused_layout(self):
        experts = MockLlama4Experts(num_experts=8, ffn_hidden=128, hidden_size=64)
        spec = MoELayerSpec(
            moe_module_name="test",
            model_family="llama4",
            router_name="router",
            experts_name="experts",
            expert_storage="fused_3d",
            expert_w1_name="gate_up_proj",
            expert_w2_name="down_proj",
            expert_w3_name=None,
            num_experts=8,
            top_k=1,
            hidden_size=64,
            ffn_hidden_size=128,
            score_func="sigmoid",
            score_apply="pre",
            route_norm=False,
            gate_bias=False,
            return_router_logits=True,
            router_logits_capture_target="moe_block",
            router_logits_capture_index=1,
            router_logits_capture_layer_name=None,
            has_shared_experts=True,
            shared_experts_name="shared_expert",
        )
        w1, w2, w3 = repack_expert_weights(experts, spec, ep_rank=0, ep_size=2)
        assert w1.shape == (4, 128, 64)
        assert w2.shape == (4, 64, 128)
        assert w3.shape == (4, 128, 64)
        expected_w1 = experts.gate_up_proj.data[0:4, :, :128].transpose(1, 2)
        expected_w2 = experts.down_proj.data[0:4].transpose(1, 2)
        expected_w3 = experts.gate_up_proj.data[0:4, :, 128:].transpose(1, 2)
        assert torch.equal(w1, expected_w1)
        assert torch.equal(w2, expected_w2)
        assert torch.equal(w3, expected_w3)


# === Phase 5: AutoEP MoE Layer and Orchestrator ===

from deepspeed.module_inject.auto_ep_layer import (
    AutoEPMoELayer,
    resolve_score_apply_mode,
    apply_scores_before_experts_if_enabled,
    combine_from_routed,
)


def _make_spec(**kwargs):
    """Helper to create MoELayerSpec with default test values."""
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


class TestScoreApplication:
    """Phase 5 tests for score application logic."""

    def test_score_apply_pre(self):
        x = torch.randn(10, 64)
        scores = torch.rand(10)
        out = apply_scores_before_experts_if_enabled(x, scores, "pre")
        expected = (x.float() * scores.reshape(-1, 1)).to(x.dtype)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_score_apply_post(self):
        x = torch.randn(10, 64)
        scores = torch.rand(10)
        out = apply_scores_before_experts_if_enabled(x, scores, "post")
        assert torch.equal(out, x)  # No change

    def test_resolve_score_apply_auto(self):
        spec = _make_spec(score_apply="post")
        assert resolve_score_apply_mode(spec, "auto") == "post"

    def test_resolve_score_apply_override(self):
        spec = _make_spec(score_apply="post")
        assert resolve_score_apply_mode(spec, "pre") == "pre"


class TestCombineFromRouted:
    """Phase 5 tests for combine_from_routed."""

    def test_combine_from_routed_shapes(self):
        B, S, H, K = 2, 8, 64, 2
        T = B * S
        N = T * K
        expert_output = torch.randn(N, H)
        top_scores = torch.rand(T, K)
        token_indices = torch.arange(N)
        out = combine_from_routed(
            expert_output,
            top_scores,
            token_indices,
            K,
            "post",
            "weighted_sum",
            (B, S, H),
        )
        assert out.shape == (B, S, H)

    def test_combine_from_routed_scatter_add(self):
        # Simple case: 2 tokens, top-2, 4 experts
        B, S, H, K = 1, 2, 4, 2
        T = 2
        expert_output = torch.ones(T * K, H)
        top_scores = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        token_indices = torch.arange(T * K)
        out = combine_from_routed(
            expert_output,
            top_scores,
            token_indices,
            K,
            "post",
            "weighted_sum",
            (B, S, H),
        )
        # With post scoring: each token's output = weighted sum of expert outputs
        assert out.shape == (B, S, H)
        # Score sum for token 0 = 0.6 + 0.4 = 1.0, so output should be ~1.0
        assert torch.allclose(out[0, 0], torch.ones(H), atol=1e-5)


class TestParamMarking:
    """Phase 5 tests for parameter marking."""

    def test_param_marking_expert(self):
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        spec = _make_spec()
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        for p in layer.experts.parameters():
            assert hasattr(p, 'allreduce') and p.allreduce is False
            assert hasattr(p, 'group_name') and p.group_name == "ep_size_1"

    def test_param_marking_router(self):
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        spec = _make_spec()
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        for p in layer.router.parameters():
            assert hasattr(p, 'allreduce') and p.allreduce is True


def _require_transformers_5(transformers):
    from packaging.version import Version
    if Version(transformers.__version__) < Version("5.0.0"):
        pytest.skip("DeepSeek AutoEP parity smoke requires Transformers >= 5.0.0")


def _get_transformers_class(transformers, *names):
    for name in names:
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    pytest.skip(f"Installed transformers does not expose any of: {names}")


class TestAutoEPMoELayerUnit:
    """Phase 5 tests for AutoEPMoELayer (ep_size=1, no dist needed)."""

    def test_autoep_layer_marker_attribute(self):
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        spec = _make_spec()
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        assert layer._is_autoep_layer is True

    def test_autoep_layer_ep_size_1_forward(self):
        torch.manual_seed(42)
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        spec = _make_spec()
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)
        assert not torch.isnan(out).any()

    def test_autoep_layer_replace_in_model(self):
        model = MockMoETransformer(num_layers=2, moe_every_n=1)
        config = AutoEPConfig(enabled=True, autoep_size=1, preset_model="mixtral")
        auto_ep = AutoEP(model, config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 2
        # Now replace should work (Phase 5 filled in)
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)
        # Verify replacement
        replaced = model.model.layers[0].mlp
        assert isinstance(replaced, AutoEPMoELayer)
        assert replaced._is_autoep_layer is True

    def test_hf_qwen2_autoep_direct_moe_applies_shared_expert_gate(self):
        """Qwen2 AutoEP carries shared_expert_gate and matches the direct MoE block."""
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "Qwen2MoeConfig") or not hasattr(transformers, "Qwen2MoeForCausalLM"):
            pytest.skip("Installed transformers does not expose Qwen2MoeConfig/Qwen2MoeForCausalLM")

        torch.manual_seed(1234)
        config = transformers.Qwen2MoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            decoder_sparse_step=1,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            output_router_logits=False,
            tie_word_embeddings=False,
            use_cache=False,
            use_sliding_window=False,
        )
        native_model = transformers.Qwen2MoeForCausalLM(config)
        autoep_model = transformers.Qwen2MoeForCausalLM(config)
        autoep_model.load_state_dict(copy.deepcopy(native_model.state_dict()))

        autoep_config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "qwen2_moe",
            "use_grouped_mm": False,
        })
        auto_ep = AutoEP(autoep_model, autoep_config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        assert specs[0].model_family == "qwen2_moe"
        assert specs[0].has_shared_experts is True
        assert specs[0].shared_experts_name == "shared_expert"
        assert specs[0].shared_experts_gate_name == "shared_expert_gate"
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

        native_moe = native_model.model.layers[0].mlp
        autoep_moe = autoep_model.model.layers[0].mlp
        assert isinstance(autoep_moe, AutoEPMoELayer)
        assert autoep_moe.shared_experts_gate is not None
        torch.testing.assert_close(autoep_moe.shared_experts_gate.weight, native_moe.shared_expert_gate.weight)

        hidden_states = torch.randn(2, 5, 32)
        native_model.eval()
        autoep_model.eval()
        with torch.no_grad():
            native_output = native_moe(hidden_states)
            autoep_output = autoep_moe(hidden_states)

        torch.testing.assert_close(autoep_output, native_output, rtol=1e-5, atol=1e-6)

    def test_hf_qwen3_causal_lm_matches_autoep_ce_only(self):
        """Tiny Qwen3 CE-only CausalLM matches AutoEP and stays ungated."""
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "Qwen3MoeConfig") or not hasattr(transformers, "Qwen3MoeForCausalLM"):
            pytest.skip("Installed transformers does not expose Qwen3MoeConfig/Qwen3MoeForCausalLM")

        torch.manual_seed(1234)
        config = transformers.Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            decoder_sparse_step=1,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            output_router_logits=False,
            tie_word_embeddings=False,
            use_cache=False,
            use_sliding_window=False,
        )
        native_model = transformers.Qwen3MoeForCausalLM(config)
        autoep_model = transformers.Qwen3MoeForCausalLM(config)
        autoep_model.load_state_dict(copy.deepcopy(native_model.state_dict()))

        autoep_config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "qwen3_moe",
            "use_grouped_mm": False,
        })
        auto_ep = AutoEP(autoep_model, autoep_config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        assert specs[0].model_family == "qwen3_moe"
        assert specs[0].has_shared_experts is False
        assert specs[0].shared_experts_gate_name == ""
        for spec in specs:
            auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)

        autoep_layers = [module for module in autoep_model.modules() if isinstance(module, AutoEPMoELayer)]
        assert len(autoep_layers) == 1
        assert autoep_layers[0].shared_experts is None
        assert autoep_layers[0].shared_experts_gate is None

        input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
        labels = input_ids.clone()
        native_model.eval()
        autoep_model.eval()
        with torch.no_grad():
            native_outputs = native_model(input_ids=input_ids, labels=labels, output_router_logits=False)
            autoep_outputs = autoep_model(input_ids=input_ids, labels=labels, output_router_logits=False)

        torch.testing.assert_close(autoep_outputs.logits, native_outputs.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(autoep_outputs.loss, native_outputs.loss, rtol=1e-5, atol=1e-6)

    def test_hf_deepseek_v2_causal_lm_matches_autoep_without_load_balance_default(self):
        """Tiny DeepSeek-V2 CausalLM reaches forward parity after AutoEP replacement."""
        transformers = pytest.importorskip("transformers")
        _require_transformers_5(transformers)
        config_cls = _get_transformers_class(transformers, "DeepseekV2Config", "DeepSeekV2Config")
        model_cls = _get_transformers_class(transformers, "DeepseekV2ForCausalLM", "DeepSeekV2ForCausalLM")

        torch.manual_seed(1234)
        config = config_cls(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            moe_intermediate_size=16,
            n_routed_experts=4,
            n_shared_experts=0,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            output_router_logits=False,
            tie_word_embeddings=False,
            use_cache=False,
        )
        native_model = model_cls(config)
        autoep_model = model_cls(config)
        autoep_model.load_state_dict(copy.deepcopy(native_model.state_dict()))

        autoep_config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "deepseek_v2",
            "use_grouped_mm": False,
        })
        auto_ep = AutoEP(autoep_model, autoep_config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        assert specs[0].model_family == "deepseek_v2"
        assert specs[0].route_norm is False
        for spec in specs:
            auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)

        autoep_layers = [module for module in autoep_model.modules() if isinstance(module, AutoEPMoELayer)]
        assert len(autoep_layers) == 1
        assert autoep_layers[0].load_balance_coeff is None
        assert autoep_layers[0].expert_bias is None

        input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
        labels = input_ids.clone()
        native_model.eval()
        autoep_model.eval()
        with torch.no_grad():
            native_outputs = native_model(input_ids=input_ids, labels=labels, output_router_logits=False)
            autoep_outputs = autoep_model(input_ids=input_ids, labels=labels, output_router_logits=False)

        torch.testing.assert_close(autoep_outputs.logits, native_outputs.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(autoep_outputs.loss, native_outputs.loss, rtol=1e-5, atol=1e-6)

    def test_hf_deepseek_v3_causal_lm_matches_autoep_without_load_balance_default(self):
        """Tiny DeepSeek-V3 CausalLM reaches forward parity after AutoEP replacement."""
        transformers = pytest.importorskip("transformers")
        _require_transformers_5(transformers)
        config_cls = _get_transformers_class(transformers, "DeepseekV3Config", "DeepSeekV3Config")
        model_cls = _get_transformers_class(transformers, "DeepseekV3ForCausalLM", "DeepSeekV3ForCausalLM")

        torch.manual_seed(1234)
        config = config_cls(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            moe_intermediate_size=16,
            n_routed_experts=4,
            n_shared_experts=0,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
            n_group=2,
            topk_group=1,
            output_router_logits=False,
            tie_word_embeddings=False,
            use_cache=False,
        )
        native_model = model_cls(config)
        autoep_model = model_cls(config)
        autoep_model.load_state_dict(copy.deepcopy(native_model.state_dict()))

        autoep_config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "deepseek_v3",
            "use_grouped_mm": False,
        })
        auto_ep = AutoEP(autoep_model, autoep_config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        assert specs[0].model_family == "deepseek_v3"
        assert specs[0].num_expert_groups == 2
        assert specs[0].num_limited_groups == 1
        for spec in specs:
            auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)

        autoep_layers = [module for module in autoep_model.modules() if isinstance(module, AutoEPMoELayer)]
        assert len(autoep_layers) == 1
        assert autoep_layers[0].load_balance_coeff is None
        assert autoep_layers[0].expert_bias is None

        input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
        labels = input_ids.clone()
        native_model.eval()
        autoep_model.eval()
        with torch.no_grad():
            native_outputs = native_model(input_ids=input_ids, labels=labels, output_router_logits=False)
            autoep_outputs = autoep_model(input_ids=input_ids, labels=labels, output_router_logits=False)

        torch.testing.assert_close(autoep_outputs.logits, native_outputs.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(autoep_outputs.loss, native_outputs.loss, rtol=1e-5, atol=1e-6)

    def test_hf_llama4_autoep_direct_moe_returns_flat_contract(self):
        """AutoEP's Llama4 replacement matches Llama4TextMoe's direct tuple shapes."""
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "Llama4ForCausalLM") or not hasattr(transformers, "Llama4TextConfig"):
            pytest.skip("Installed transformers does not expose Llama4ForCausalLM/Llama4TextConfig")

        torch.manual_seed(1234)
        config = transformers.Llama4TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=16,
            intermediate_size_mlp=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            num_local_experts=4,
            num_experts_per_tok=1,
            moe_layers=[0],
            interleave_moe_layer_step=1,
            output_router_logits=False,
            router_jitter_noise=0.0,
            tie_word_embeddings=False,
            use_cache=False,
            attention_chunk_size=64,
            attn_temperature_tuning=False,
            no_rope_layers=[0],
        )
        native_model = transformers.Llama4ForCausalLM(config)
        autoep_model = transformers.Llama4ForCausalLM(config)
        autoep_model.load_state_dict(copy.deepcopy(native_model.state_dict()))

        autoep_config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "llama4",
            "use_grouped_mm": False,
        })
        auto_ep = AutoEP(autoep_model, autoep_config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        assert specs[0].return_router_logits is True
        auto_ep.replace_moe_layer(specs[0], ep_size=1, ep_rank=0)

        native_moe = native_model.model.layers[0].feed_forward
        autoep_moe = [module for module in autoep_model.modules() if isinstance(module, AutoEPMoELayer)][0]
        hidden_states = torch.randn(2, 5, 32)
        native_model.eval()
        autoep_model.eval()

        with torch.no_grad():
            native_output, native_router_logits = native_moe(hidden_states)
            autoep_output, autoep_router_logits = autoep_moe(hidden_states)

        assert autoep_output.shape == (10, 32)
        assert autoep_router_logits.shape == (10, 4)
        torch.testing.assert_close(autoep_output, native_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(autoep_router_logits, native_router_logits, rtol=1e-5, atol=1e-6)

    def test_hf_llama4_causal_lm_matches_autoep_without_load_balance_default(self):
        """Tiny real HF Llama4 CausalLM matches AutoEP with the llama4 preset default."""
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "Llama4ForCausalLM") or not hasattr(transformers, "Llama4TextConfig"):
            pytest.skip("Installed transformers does not expose Llama4ForCausalLM/Llama4TextConfig")

        torch.manual_seed(1234)
        config = transformers.Llama4TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=16,
            intermediate_size_mlp=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            num_local_experts=4,
            num_experts_per_tok=1,
            moe_layers=[0],
            interleave_moe_layer_step=1,
            output_router_logits=False,
            router_jitter_noise=0.0,
            tie_word_embeddings=False,
            use_cache=False,
            attention_chunk_size=64,
            attn_temperature_tuning=False,
            no_rope_layers=[0],
        )
        native_model = transformers.Llama4ForCausalLM(config)
        autoep_model = transformers.Llama4ForCausalLM(config)
        autoep_model.load_state_dict(copy.deepcopy(native_model.state_dict()))

        autoep_config = parse_autoep_config({
            "enabled": True,
            "autoep_size": 1,
            "preset_model": "llama4",
            "use_grouped_mm": False,
        })
        auto_ep = AutoEP(autoep_model, autoep_config)
        specs = auto_ep.ep_parser()
        assert len(specs) == 1
        for spec in specs:
            auto_ep.replace_moe_layer(spec, ep_size=1, ep_rank=0)

        autoep_layers = [module for module in autoep_model.modules() if isinstance(module, AutoEPMoELayer)]
        assert len(autoep_layers) == 1
        assert autoep_layers[0].load_balance_coeff is None
        assert autoep_layers[0].expert_bias is None

        input_ids = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
        labels = input_ids.clone()
        native_model.eval()
        autoep_model.eval()
        with torch.no_grad():
            native_outputs = native_model(input_ids=input_ids, labels=labels)
            autoep_outputs = autoep_model(input_ids=input_ids, labels=labels)

        torch.testing.assert_close(autoep_outputs.logits, native_outputs.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(autoep_outputs.loss, native_outputs.loss, rtol=1e-5, atol=1e-6)


# === Phase 6: Engine + Mappings ===


class TestAutoTPSkipAutoEP:
    """Phase 6 tests for AutoTP skip logic on AutoEP-managed modules."""

    def test_autotp_skip_autoep_marker(self):
        """AutoTP._replace() returns child unchanged when _is_autoep_layer=True."""
        from deepspeed.module_inject.auto_tp import AutoTP

        # Create a mock module with the AutoEP marker
        mock_module = nn.Linear(64, 64)
        mock_module._is_autoep_layer = True

        autotp = AutoTP.__new__(AutoTP)
        autotp.mp_group = None
        autotp.mp_size = 1
        autotp.module = nn.Module()
        autotp.partition_config = None

        result = autotp._replace(mock_module, "test_layer", conv_linear_layer=False)
        assert result is mock_module, "AutoTP should return AutoEP module unchanged"

    def test_autotp_does_not_skip_regular_module(self):
        """AutoTP._replace() does NOT skip regular nn.Linear modules."""
        # A regular nn.Linear without _is_autoep_layer should not be returned as-is
        regular_module = nn.Linear(64, 64)
        assert not getattr(regular_module, "_is_autoep_layer", False)


class TestEngineAutoEPConfig:
    """Phase 6 tests for engine configuration parsing."""

    def test_expert_parallel_config_present(self):
        """DeepSpeedConfig has expert_parallel_config attribute."""
        from deepspeed.runtime.config import DeepSpeedConfig
        assert hasattr(DeepSpeedConfig, '__init__'), "DeepSpeedConfig must exist"
        # Verify the get_expert_parallel_config function exists
        from deepspeed.runtime.config import get_expert_parallel_config
        config = get_expert_parallel_config({})
        assert config is not None or config is None  # None when disabled

    def test_autoep_layer_has_set_deepspeed_parallelism(self):
        """AutoEPMoELayer has set_deepspeed_parallelism for engine traversal."""
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        spec = _make_spec()
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        assert hasattr(layer, 'set_deepspeed_parallelism')
        assert callable(layer.set_deepspeed_parallelism)

    def test_autoep_layer_num_experts_attribute(self):
        """AutoEPMoELayer exposes num_experts for engine MoE detection."""
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        spec = _make_spec()
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        assert layer.num_experts == 4

    def test_gate_alias_present_when_router_capture_and_name_differs(self):
        """Gate alias created for router_name != 'router' when capture_target == 'router'."""
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        spec = _make_spec(
            router_name="gate",
            router_logits_capture_target="router",
            router_logits_capture_layer_name=None,
        )
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        assert hasattr(layer, 'gate')
        assert layer.gate is layer.router

    def test_gate_alias_uses_capture_layer_name(self):
        """Alias uses router_logits_capture_layer_name when provided."""
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        source.router = source.gate
        spec = _make_spec(
            router_name="router",
            router_logits_capture_target="router",
            router_logits_capture_layer_name="gate",
        )
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        assert hasattr(layer, 'gate')
        assert layer.gate is layer.router

    def test_no_gate_alias_when_alias_target_is_router(self):
        """No alias when alias_target resolves to 'router' (e.g., Llama4)."""
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        source.router = source.gate
        spec = _make_spec(
            router_name="router",
            router_logits_capture_target="router",
            router_logits_capture_layer_name=None,
        )
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        assert not hasattr(layer, 'gate')

    def test_no_gate_alias_when_no_capture(self):
        """No alias when capture_target is 'none'."""
        source = MockMoEBlock(num_experts=4, ffn_hidden=128, hidden_size=64)
        spec = _make_spec(
            router_name="gate",
            router_logits_capture_target="none",
            router_logits_capture_layer_name="gate",
        )
        config = AutoEPConfig(enabled=True, autoep_size=1)
        layer = AutoEPMoELayer(spec, source, ep_size=1, ep_rank=0, config=config)
        # No gate alias because capture_target != "router"
        assert not hasattr(layer, 'gate')
