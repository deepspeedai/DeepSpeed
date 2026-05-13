# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Qwen3.5-MoE AutoEP preset and parser adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch.nn as nn

from deepspeed.module_inject.auto_ep_presets.base import (
    AutoEPPresetAdapter,
    ForwardContract,
    MoELayerSpec,
    MoEModelPreset,
)
from deepspeed.utils import logger

PRESET_NAME = "qwen3_5_moe"

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
    preset_adapter="qwen3_5_moe",
    hf_model_types=("qwen3_5_moe_text", ),
    unsupported_hf_model_type_notes={
        "qwen3_5_moe": ("AutoEP supports the Qwen3.5 text backbone preset path; pass the "
                        "text-backbone model/config with model_type='qwen3_5_moe_text'.")
    },
    min_transformers_version="5.2.0",
    docs_support_notes="Requires the Qwen3.5 text-backbone qwen3_5_moe_text model type.",
)


class Qwen35MoePresetAdapter(AutoEPPresetAdapter):
    """Qwen3.5 MoE exposes router logits through HF output recording."""

    def _requires_transformers_version_validation(self) -> bool:
        return True

    def adjust_forward_contract(self, contract: ForwardContract) -> ForwardContract:
        # HF records Qwen3.5 router output on Qwen3_5MoeTopKRouter. AutoEP replaces
        # the owning MoE block, so replacement output index 1 is used for recorder retargeting.
        return replace(
            contract,
            return_router_logits=True,
            capture_target="router",
            capture_index=1,
        )

    def retarget_transformers_output_recorders(
        self,
        model: nn.Module,
        spec: MoELayerSpec,
        replacement: nn.Module,
        retargeted_keys: set[str],
        remove_output_capture_hooks: Callable[[nn.Module], int],
    ) -> None:
        recorder_key = f"{spec.model_family}:{replacement.__class__.__module__}.{replacement.__class__.__qualname__}"
        if recorder_key in retargeted_keys:
            return
        retargeted_keys.add(recorder_key)

        try:
            from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder
        except Exception as exc:
            logger.warning(f"AutoEP: could not retarget Qwen3.5 router-logit output capture: {exc}")
            return

        retargeted = 0
        replacement_cls = replacement.__class__
        for module in model.modules():
            module_config = getattr(module, "config", None)
            model_type = getattr(module_config, "model_type", None)
            class_name = module.__class__.__name__
            if model_type != "qwen3_5_moe_text" and "Qwen3_5Moe" not in class_name:
                continue

            registry_key = str(module.__class__)
            record_outputs = getattr(module, "_can_record_outputs", None)
            registry_outputs = _CAN_RECORD_REGISTRY.get(registry_key)
            base_outputs = record_outputs if isinstance(record_outputs, dict) else registry_outputs
            if not isinstance(base_outputs, dict) or "router_logits" not in base_outputs:
                continue

            retargeted_outputs = dict(base_outputs)
            retargeted_outputs["router_logits"] = OutputRecorder(replacement_cls, index=1)
            module._can_record_outputs = retargeted_outputs
            _CAN_RECORD_REGISTRY[registry_key] = retargeted_outputs

            if getattr(module, "_output_capturing_hooks_installed", False):
                removed = remove_output_capture_hooks(module)
                if removed:
                    logger.debug(f"AutoEP: removed {removed} stale HF output-capturing hook(s) "
                                 f"from {class_name}.")
            module._output_capturing_hooks_installed = False
            retargeted += 1

        if retargeted:
            logger.info("AutoEP: retargeted Qwen3.5 HF router-logit output capture to record "
                        f"{replacement_cls.__name__} output index 1 on {retargeted} module(s).")
        else:
            logger.warning("AutoEP: Qwen3.5 AutoEP conversion did not find a HF output-capture registry "
                           "entry for router_logits.")


PRESET_ADAPTERS = {
    "qwen3_5_moe": Qwen35MoePresetAdapter(),
}
