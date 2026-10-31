# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""MiniMax-M3 AutoEP preset and parser adapter.

MiniMax-M3 is 425.9 B parameters, 25.7 B active: 60 layers, the first 3 dense and the remaining
57 mixture-of-experts with 128 experts, top-4, plus one shared expert.

Two things set it apart from the presets before it:

1. The expert MLP activation is the clamped GPT-OSS form, not SwiGLU:
   ``(clamp(up, -7, 7) + 1) * clamp(gate, max=7) * sigmoid(1.702 * clamp(gate, max=7))``.
   Substituting ``silu(gate) * up`` is not an approximation: on standard-normal input the two
   differ by 0.41 mean absolute with correlation 0.74, so AutoEP would silently train a different
   function. ``expert_activation="swiglu_oai"`` selects the right one; the ``swiglu_alpha`` /
   ``swiglu_limit`` values the model states replace the preset's defaults at detection time.

2. The router scores with sigmoid, selects with a correction bias held in a buffer (as in
   DeepSeek-V3), normalises the selected weights, and scales the routed output by
   ``routed_scaling_factor`` (2.0) before adding the shared expert. All of this is already
   supported: the bias path and the scale are read from the model.

The text backbone is what AutoEP replaces, so the model type is ``minimax_m3_vl_text``; the
composite ``minimax_m3_vl`` config carries a vision tower that has no MoE layers. Router logits are
recorded at the model level as ``OutputRecorder(MiniMaxM3VLTopKRouter, index=0)``, the raw logits,
the same pattern as Mixtral, so the same recorder adapter applies.
"""

from __future__ import annotations

from deepspeed.module_inject.auto_ep_presets.base import MoEModelPreset, TransformersTopLevelRouterLogitsAdapter

PRESET_NAME = "minimax_m3"

PRESET = MoEModelPreset(
    moe_layer_pattern=r"model\.layers\.\d+\.mlp",
    router_pattern="gate",
    experts_pattern="experts",
    expert_storage="fused_3d",
    expert_w1="gate_up_proj",
    expert_w2="down_proj",
    expert_w3=None,
    # AutoEP reads these from the MODEL CONFIG, where MiniMax names them num_local_experts /
    # num_experts_per_tok; the router module's own num_experts / top_k attributes are not used.
    num_experts_attr="num_local_experts",
    top_k_attr="num_experts_per_tok",
    score_func="sigmoid",
    score_apply="post",
    route_norm=True,
    gate_bias=False,
    has_shared_experts=True,
    shared_experts_pattern="shared_experts",
    expert_activation="swiglu_oai",
    expert_activation_alpha=1.702,
    expert_activation_limit=7.0,
    autoep_config_defaults={"load_balance_coeff": None},
    supports_expert_bias=False,
    preset_adapter="minimax_m3",
    hf_model_types=("minimax_m3_vl_text", ),
    unsupported_hf_model_type_notes={
        "minimax_m3_vl": ("AutoEP replaces the MiniMax-M3 text backbone; pass the text-backbone "
                          "model/config with model_type='minimax_m3_vl_text'."),
    },
    min_transformers_version="5.15.0",
    docs_support_notes=("Requires the MiniMax-M3 text-backbone minimax_m3_vl_text model type. The expert "
                        "MLP uses the clamped GPT-OSS activation (swiglu_oai), selected by the preset. "
                        "load_balance_coeff / expert-bias auxiliary-loss-free load balancing is not "
                        "currently supported; non-null values are rejected."),
)


class MiniMaxM3PresetAdapter(TransformersTopLevelRouterLogitsAdapter):
    """Mixtral-style router-logit recorder retargeting, plus the Transformers version gate."""

    def _requires_transformers_version_validation(self) -> bool:
        return True


PRESET_ADAPTERS = {
    "minimax_m3":
    MiniMaxM3PresetAdapter(
        display_name="MiniMax-M3",
        hf_model_types=("minimax_m3_vl_text", ),
        class_name_fragments=("MiniMaxM3VL", ),
    ),
}
