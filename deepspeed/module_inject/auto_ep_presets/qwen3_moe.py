# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Qwen3-MoE AutoEP preset."""

from __future__ import annotations

from deepspeed.module_inject.auto_ep_presets.base import MoEModelPreset

PRESET_NAME = "qwen3_moe"

PRESET = MoEModelPreset(
    moe_layer_pattern=r"model\.layers\.\d+\.mlp",
    router_pattern="gate",
    experts_pattern="experts",
    expert_storage="fused_3d",
    expert_w1="gate_up_proj",
    expert_w2="down_proj",
    expert_w3=None,
    num_experts_attr="num_experts",
    top_k_attr="num_experts_per_tok",
    score_func="softmax",
    score_apply="post",
    route_norm=True,
    gate_bias=False,
    has_shared_experts=True,
    shared_experts_pattern="shared_expert",
    shared_experts_gate_pattern="shared_expert_gate",
    hf_model_types=("qwen3_moe", "qwen2_moe"),
    min_transformers_version="5.0.0",
    docs_support_notes=("Also covers Qwen2-MoE when the installed Transformers build uses the "
                        "validated fused expert layout."),
)
