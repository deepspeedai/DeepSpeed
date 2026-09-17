# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Mixtral AutoEP preset."""

from __future__ import annotations

from deepspeed.module_inject.auto_ep_presets.base import MoEModelPreset, TransformersTopLevelRouterLogitsAdapter

PRESET_NAME = "mixtral"

PRESET = MoEModelPreset(
    moe_layer_pattern=r"model\.layers\.\d+\.mlp",
    router_pattern="gate",
    experts_pattern="experts",
    expert_storage="fused_3d",
    expert_w1="gate_up_proj",
    expert_w2="down_proj",
    expert_w3=None,
    num_experts_attr="num_local_experts",
    top_k_attr="num_experts_per_tok",
    score_func="softmax",
    score_apply="post",
    route_norm=True,
    gate_bias=False,
    preset_adapter="mixtral",
    hf_model_types=("mixtral", ),
    min_transformers_version="5.0.0",
)

PRESET_ADAPTERS = {
    "mixtral":
    TransformersTopLevelRouterLogitsAdapter(
        display_name="Mixtral",
        hf_model_types=("mixtral", ),
        class_name_fragments=("Mixtral", ),
    ),
}
