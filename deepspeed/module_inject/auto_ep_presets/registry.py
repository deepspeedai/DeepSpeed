# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""AutoEP preset registry and config override helpers."""

from __future__ import annotations

import copy
from dataclasses import replace

from deepspeed.module_inject.auto_ep_presets.base import (
    _UNSET,
    AutoEPConfig,
    AutoEPPresetAdapter,
    MoEModelPreset,
)
from deepspeed.module_inject.auto_ep_presets import deepseek_v2, deepseek_v3, mixtral, qwen3_5_moe, qwen3_moe
from deepspeed.utils import logger

_PRESET_MODULES = (
    mixtral,
    qwen3_moe,
    qwen3_5_moe,
    deepseek_v2,
    deepseek_v3,
)

PRESET_MODELS: dict[str, MoEModelPreset] = {module.PRESET_NAME: module.PRESET for module in _PRESET_MODULES}

_PRESET_ADAPTERS: dict[str, AutoEPPresetAdapter] = {
    "default": AutoEPPresetAdapter(),
}
for _module in _PRESET_MODULES:
    _PRESET_ADAPTERS.update(getattr(_module, "PRESET_ADAPTERS", {}))


def _validate_registered_preset_adapters(
    preset_models: dict[str, MoEModelPreset] | None = None,
    preset_adapters: dict[str, AutoEPPresetAdapter] | None = None,
) -> None:
    """Fail fast if a registered preset references an adapter that is not registered."""
    preset_models = PRESET_MODELS if preset_models is None else preset_models
    preset_adapters = _PRESET_ADAPTERS if preset_adapters is None else preset_adapters

    missing_presets = []
    for preset_name, preset in preset_models.items():
        if preset.preset_adapter not in preset_adapters:
            missing_presets.append((preset_name, preset.preset_adapter))

    if not missing_presets:
        return

    details = ", ".join(f"{preset_name}:{adapter_name}" for preset_name, adapter_name in missing_presets)
    raise RuntimeError(f"AutoEP preset registry is inconsistent; missing preset_adapter registration(s): {details}")


_validate_registered_preset_adapters()

_PRESET_DEFAULT_EXPLICIT_FLAGS = {
    "load_balance_coeff": "_load_balance_coeff_explicit",
}


def available_preset_names() -> tuple[str, ...]:
    """Return built-in AutoEP preset names in registry order."""
    return tuple(PRESET_MODELS.keys())


def get_preset(preset_name: str) -> MoEModelPreset:
    """Return a registered AutoEP preset by name."""
    preset = PRESET_MODELS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown preset_model '{preset_name}'. Available presets: {list(available_preset_names())}")
    return preset


def get_preset_adapter(adapter_name: str) -> AutoEPPresetAdapter:
    """Return a registered AutoEP preset adapter by name."""
    adapter = _PRESET_ADAPTERS.get(adapter_name)
    if adapter is None:
        raise ValueError(f"Unknown AutoEP preset adapter '{adapter_name}'")
    return adapter


def preset_name_for_hf_model_type(model_type: str) -> str | None:
    """Return the AutoEP preset name for a supported HF model_type."""
    for preset_name, preset in PRESET_MODELS.items():
        if model_type in preset.hf_model_types:
            return preset_name
    return None


def unsupported_preset_for_hf_model_type(model_type: str) -> tuple[str, MoEModelPreset] | None:
    """Return a preset carrying an actionable diagnostic for an unsupported HF model_type."""
    for preset_name, preset in PRESET_MODELS.items():
        if model_type in preset.unsupported_hf_model_type_notes:
            return preset_name, preset
    return None


def resolve_autoep_config_defaults(config: AutoEPConfig, preset_name: str | None) -> AutoEPConfig:
    """Return config with preset-level AutoEP defaults applied where the user did not override."""
    if preset_name is None or preset_name not in PRESET_MODELS:
        return config

    preset_defaults = PRESET_MODELS[preset_name].autoep_config_defaults
    if not preset_defaults:
        return config

    resolved = copy.copy(config)
    for field_name, default_value in preset_defaults.items():
        explicit_flag = _PRESET_DEFAULT_EXPLICIT_FLAGS.get(field_name)
        if explicit_flag is None:
            continue
        if not getattr(config, explicit_flag, False):
            setattr(resolved, field_name, default_value)
    return resolved


def apply_config_overrides(config: AutoEPConfig, preset: MoEModelPreset) -> MoEModelPreset:
    """Apply explicit AutoEP config overrides to a preset.

    Return the original preset object when there are no overrides. When overrides
    are present, return a dataclass copy so the registered preset remains unchanged.
    """
    overrides = {}
    if config.moe_layer_pattern is not None:
        overrides["moe_layer_pattern"] = config.moe_layer_pattern
    if config.router_pattern is not None:
        overrides["router_pattern"] = config.router_pattern
    if config.expert_pattern is not None:
        overrides["experts_pattern"] = config.expert_pattern
    if config.expert_w1 is not None:
        overrides["expert_w1"] = config.expert_w1
    if config.expert_w2 is not None:
        overrides["expert_w2"] = config.expert_w2
    if config.expert_w3 is not _UNSET:
        overrides["expert_w3"] = config.expert_w3
    if config.num_experts_attr is not None:
        overrides["num_experts_attr"] = config.num_experts_attr
    if config.top_k_attr is not None:
        overrides["top_k_attr"] = config.top_k_attr
    if config.has_shared_experts is not None:
        overrides["has_shared_experts"] = config.has_shared_experts
    if config.shared_experts_pattern is not None:
        overrides["shared_experts_pattern"] = config.shared_experts_pattern
    if config.shared_experts_gate_pattern is not None:
        overrides["shared_experts_gate_pattern"] = config.shared_experts_gate_pattern
    if not overrides:
        return preset
    return replace(preset, **overrides)


def resolve_preset_candidates(
    config: AutoEPConfig,
    model_config,
) -> list[tuple[str, MoEModelPreset]]:
    """Resolve ordered preset candidates for AutoEP detection."""
    if config.preset_model is not None:
        preset = apply_config_overrides(config, get_preset(config.preset_model))
        _validate_preset_compatibility(config.preset_model, preset, model_config)
        return [(config.preset_model, preset)]

    if model_config is not None:
        model_type = getattr(model_config, 'model_type', None)
        if model_type:
            preset_name = preset_name_for_hf_model_type(model_type)
            if preset_name is not None:
                logger.info(f"AutoEP: auto-detected model_type='{model_type}', using preset '{preset_name}'")
                preset = apply_config_overrides(config, get_preset(preset_name))
                _validate_preset_compatibility(preset_name, preset, model_config)
                return [(preset_name, preset)]

            unsupported_preset = unsupported_preset_for_hf_model_type(model_type)
            if unsupported_preset is not None:
                preset_name, preset = unsupported_preset
                _validate_preset_compatibility(preset_name, preset, model_config)

    if config.moe_layer_pattern:
        return [("custom", _build_custom_preset(config))]

    return [(name, apply_config_overrides(config, preset)) for name, preset in PRESET_MODELS.items()]


def _validate_preset_compatibility(
    preset_name: str,
    preset: MoEModelPreset,
    model_config,
) -> None:
    adapter = get_preset_adapter(preset.preset_adapter)
    adapter.validate_compatibility(preset_name, preset, model_config)


def _build_custom_preset(config: AutoEPConfig) -> MoEModelPreset:
    return MoEModelPreset(
        moe_layer_pattern=config.moe_layer_pattern,
        router_pattern=config.router_pattern or "gate",
        experts_pattern=config.expert_pattern or "experts",
        expert_storage="fused_3d",
        expert_w1=config.expert_w1 or "gate_up_proj",
        expert_w2=config.expert_w2 or "down_proj",
        expert_w3=(None if config.expert_w3 is _UNSET else config.expert_w3),
        num_experts_attr=config.num_experts_attr or "num_local_experts",
        top_k_attr=config.top_k_attr or "num_experts_per_tok",
        score_func=(config.score_func if config.score_func != "auto" else "softmax"),
        score_apply=(config.score_apply if config.score_apply != "auto" else "post"),
        route_norm=(config.route_norm if config.route_norm is not None else True),
        gate_bias=False,
        has_shared_experts=(config.has_shared_experts if config.has_shared_experts is not None else False),
        shared_experts_pattern=config.shared_experts_pattern or "",
        shared_experts_gate_pattern=config.shared_experts_gate_pattern or "",
    )
