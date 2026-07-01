# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DeepSeek-V3 AutoEP preset and parser adapter."""

from __future__ import annotations

from dataclasses import replace

import torch.nn as nn

from deepspeed.module_inject.auto_ep_presets.base import AutoEPConfig, AutoEPPresetAdapter, GroupRoutingConfig, MoEModelPreset

PRESET_NAME = "deepseek_v3"

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
    score_func="sigmoid",
    score_apply="post",
    route_norm=False,
    gate_bias=False,
    has_shared_experts=True,
    shared_experts_pattern="shared_experts",
    autoep_config_defaults={"load_balance_coeff": None},
    supports_expert_bias=False,
    unsupported_router_bias_names=(),
    preset_adapter="deepseek_v3",
    hf_model_types=("deepseek_v3", ),
    min_transformers_version="5.0.0",
    docs_support_notes=("load_balance_coeff / expert-bias auxiliary-loss-free load balancing "
                        "is not currently supported; non-null values are rejected."),
)


class DeepSeekV3PresetAdapter(AutoEPPresetAdapter):
    """DeepSeek-V3 always carries group-limited routing fields when present."""

    def _requires_transformers_version_validation(self) -> bool:
        return True

    def resolve_group_routing(
        self,
        config: AutoEPConfig,
        model_config,
    ) -> GroupRoutingConfig:
        group_routing = super().resolve_group_routing(config, model_config)
        return GroupRoutingConfig(
            num_expert_groups=group_routing.num_expert_groups or getattr(model_config, 'n_group', None),
            num_limited_groups=group_routing.num_limited_groups or getattr(model_config, 'topk_group', None),
            group_score_func=group_routing.group_score_func,
        )

    def resolve_expert_layout(
        self,
        experts_module: nn.Module,
        preset: MoEModelPreset,
    ) -> MoEModelPreset:
        if not isinstance(experts_module, nn.ModuleList) or len(experts_module) == 0:
            return preset

        default_fused_layout = (preset.expert_storage == "fused_3d" and preset.expert_w1 == "gate_up_proj"
                                and preset.expert_w2 == "down_proj" and preset.expert_w3 is None)
        if not default_fused_layout:
            return preset

        expert0 = experts_module[0]
        if not all(_has_expert_projection(expert0, name) for name in ("gate_proj", "up_proj", "down_proj")):
            return preset

        return replace(
            preset,
            expert_storage="module_list",
            expert_w1="gate_proj",
            expert_w2="down_proj",
            expert_w3="up_proj",
        )


def _has_expert_projection(expert_module: nn.Module, name: str) -> bool:
    projection = getattr(expert_module, name, None)
    if projection is None:
        return False
    if isinstance(projection, (nn.Linear, nn.Parameter)):
        return True
    return hasattr(projection, "weight")


PRESET_ADAPTERS = {
    "deepseek_v3": DeepSeekV3PresetAdapter(),
}
