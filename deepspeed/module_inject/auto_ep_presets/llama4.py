# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Llama4 AutoEP preset and parser adapter."""

from __future__ import annotations

from dataclasses import replace

from deepspeed.module_inject.auto_ep_presets.base import AutoEPPresetAdapter, ForwardContract, MoEModelPreset

PRESET_NAME = "llama4"

PRESET = MoEModelPreset(
    moe_layer_pattern=r"model\.layers\.\d+\.feed_forward",
    router_pattern="router",
    experts_pattern="experts",
    expert_storage="fused_3d",
    expert_w1="gate_up_proj",
    expert_w2="down_proj",
    expert_w3=None,
    num_experts_attr="num_local_experts",
    top_k_attr="num_experts_per_tok",
    score_func="sigmoid",
    score_apply="pre",
    route_norm=False,
    gate_bias=False,
    has_shared_experts=True,
    shared_experts_pattern="shared_expert",
    autoep_config_defaults={"load_balance_coeff": None},
    preset_adapter="llama4",
    hf_model_types=("llama4", "llama4_text"),
    min_transformers_version="5.0.0",
)


class Llama4PresetAdapter(AutoEPPresetAdapter):
    """Llama4 MoE returns a flat hidden-state tensor with raw router logits."""

    def adjust_forward_contract(self, contract: ForwardContract) -> ForwardContract:
        capture_target = contract.capture_target
        if capture_target == "none":
            capture_target = "router"

        return replace(
            contract,
            return_router_logits=True,
            capture_target=capture_target,
            router_logits_capture_mode="raw",
            moe_output_shape="flat",
        )


PRESET_ADAPTERS = {
    "llama4": Llama4PresetAdapter(),
}
