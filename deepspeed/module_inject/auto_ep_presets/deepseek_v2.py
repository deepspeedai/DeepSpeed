# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DeepSeek-V2 AutoEP preset and parser adapter."""

from __future__ import annotations

from deepspeed.module_inject.auto_ep_presets.base import (
    AutoEPConfig,
    AutoEPPresetAdapter,
    GroupRoutingConfig,
    MoEModelPreset,
)

PRESET_NAME = "deepseek_v2"

PRESET = MoEModelPreset(
    moe_layer_pattern=r"model\.layers\.\d+\.mlp",
    router_pattern="gate",
    experts_pattern="experts",
    expert_storage="fused_3d",
    expert_w1="gate_up_proj",
    expert_w2="down_proj",
    expert_w3=None,
    num_experts_attr="n_routed_experts",
    top_k_attr="num_experts_per_tok",
    score_func="softmax",
    score_apply="post",
    route_norm=False,
    gate_bias=False,
    has_shared_experts=True,
    shared_experts_pattern="shared_experts",
    autoep_config_defaults={"load_balance_coeff": None},
    supports_expert_bias=False,
    preset_adapter="deepseek_v2",
    hf_model_types=("deepseek_v2", ),
    min_transformers_version="5.0.0",
    docs_support_notes=("load_balance_coeff / expert-bias auxiliary-loss-free load balancing "
                        "is not currently supported; non-null values are rejected."),
)


class DeepSeekV2PresetAdapter(AutoEPPresetAdapter):
    """DeepSeek-V2 keeps native top-k normalization and optional group-limited routing."""

    def _requires_transformers_version_validation(self) -> bool:
        return True

    def resolve_route_norm(
        self,
        config: AutoEPConfig,
        preset: MoEModelPreset,
        model_config,
    ) -> bool:
        if config.route_norm is not None:
            return config.route_norm
        return preset.route_norm

    def resolve_group_routing(
        self,
        config: AutoEPConfig,
        model_config,
    ) -> GroupRoutingConfig:
        group_routing = super().resolve_group_routing(config, model_config)
        if getattr(model_config, 'topk_method', None) != "group_limited_greedy":
            return group_routing

        return GroupRoutingConfig(
            num_expert_groups=group_routing.num_expert_groups or getattr(model_config, 'n_group', None),
            num_limited_groups=group_routing.num_limited_groups or getattr(model_config, 'topk_group', None),
            group_score_func="max",
        )


PRESET_ADAPTERS = {
    "deepseek_v2": DeepSeekV2PresetAdapter(),
}
