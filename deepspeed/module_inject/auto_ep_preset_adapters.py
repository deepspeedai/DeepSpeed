# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Preset-specific AutoEP parser adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal

import torch.nn as nn

from deepspeed.module_inject.auto_ep_config import AutoEPConfig, MoELayerSpec, MoEModelPreset
from deepspeed.utils import logger


@dataclass(frozen=True)
class GroupRoutingConfig:
    num_expert_groups: int | None
    num_limited_groups: int | None
    group_score_func: Literal["max", "top2_sum"] = "top2_sum"


@dataclass(frozen=True)
class ForwardContract:
    return_router_logits: bool = False
    capture_target: Literal["moe_block", "router", "none"] = "none"
    capture_index: int | None = None
    capture_layer_name: str | None = None
    router_logits_capture_mode: Literal["raw", "post_score"] = "post_score"
    moe_output_shape: Literal["batched", "flat"] = "batched"


class AutoEPPresetAdapter:
    """Default behavior shared by presets without model-specific parser rules."""

    def resolve_route_norm(
        self,
        config: AutoEPConfig,
        preset: MoEModelPreset,
        model_config,
    ) -> bool:
        if config.route_norm is not None:
            return config.route_norm

        cfg_norm = getattr(model_config, 'norm_topk_prob', None)
        if cfg_norm is not None:
            return bool(cfg_norm)
        return preset.route_norm

    def resolve_group_routing(
        self,
        config: AutoEPConfig,
        model_config,
    ) -> GroupRoutingConfig:
        return GroupRoutingConfig(
            num_expert_groups=config.num_expert_groups,
            num_limited_groups=config.num_limited_groups,
        )

    def adjust_forward_contract(self, contract: ForwardContract) -> ForwardContract:
        return contract

    def retarget_transformers_output_recorders(
        self,
        model: nn.Module,
        spec: MoELayerSpec,
        replacement: nn.Module,
        retargeted_keys: set[str],
        remove_output_capture_hooks: Callable[[nn.Module], int],
    ) -> None:
        return


class DeepSeekV2PresetAdapter(AutoEPPresetAdapter):
    """DeepSeek-V2 keeps native top-k normalization and optional group-limited routing."""

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


class DeepSeekV3PresetAdapter(AutoEPPresetAdapter):
    """DeepSeek-V3 always carries group-limited routing fields when present."""

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


class Qwen35MoePresetAdapter(AutoEPPresetAdapter):
    """Qwen3.5 MoE exposes router logits through HF output recording."""

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


_PRESET_ADAPTERS: dict[str, AutoEPPresetAdapter] = {
    "default": AutoEPPresetAdapter(),
    "deepseek_v2": DeepSeekV2PresetAdapter(),
    "deepseek_v3": DeepSeekV3PresetAdapter(),
    "qwen3_5_moe": Qwen35MoePresetAdapter(),
    "llama4": Llama4PresetAdapter(),
}


def get_preset_adapter(adapter_name: str) -> AutoEPPresetAdapter:
    adapter = _PRESET_ADAPTERS.get(adapter_name)
    if adapter is None:
        raise ValueError(f"Unknown AutoEP preset adapter '{adapter_name}'")
    return adapter
