# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP configuration: config parsing, model presets, and validation."""

from __future__ import annotations

from deepspeed.module_inject.auto_ep_presets.base import (
    _UNSET,
    _raise_unsupported_load_balance_coeff,
    AutoEPConfig,
    MoELayerSpec,
    MoEModelPreset,
)
from deepspeed.module_inject.auto_ep_presets.registry import (
    PRESET_MODELS,
    available_preset_names,
    resolve_autoep_config_defaults,
)
from deepspeed.utils import logger

__all__ = [
    "_UNSET",
    "AutoEPConfig",
    "MoELayerSpec",
    "MoEModelPreset",
    "PRESET_MODELS",
    "parse_autoep_config",
    "resolve_autoep_config_defaults",
    "validate_autoep_config",
    "validate_autoep_post_detection",
]

# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def parse_autoep_config(param_dict: dict) -> AutoEPConfig:
    """Parse the 'expert_parallel' section from DS config JSON."""
    if not param_dict:
        return AutoEPConfig()

    config = AutoEPConfig()
    config.enabled = param_dict.get("enabled", False)
    config.autoep_size = param_dict.get("autoep_size", 1)
    config.preset_model = param_dict.get("preset_model", None)
    config.moe_layer_pattern = param_dict.get("moe_layer_pattern", None)
    config.expert_pattern = param_dict.get("expert_pattern", None)
    config.router_pattern = param_dict.get("router_pattern", None)
    config.use_grouped_mm = param_dict.get("use_grouped_mm", True)
    config.route_norm = param_dict.get("route_norm", None)
    config.route_scale = param_dict.get("route_scale", 1.0)
    config.score_apply = param_dict.get("score_apply", "auto")
    config.combine_impl = param_dict.get("combine_impl", "auto")
    config.num_expert_groups = param_dict.get("num_expert_groups", None)
    config.num_limited_groups = param_dict.get("num_limited_groups", None)
    config.score_func = param_dict.get("score_func", "auto")
    config.top_k = param_dict.get("top_k", "auto")
    if "load_balance_coeff" in param_dict:
        value = param_dict["load_balance_coeff"]
        if value is not None:
            _raise_unsupported_load_balance_coeff(value)
        config.load_balance_coeff = None
        config._load_balance_coeff_explicit = True
    else:
        config.load_balance_coeff = None
        config._load_balance_coeff_explicit = False
    config.routed_scaling_factor = param_dict.get("routed_scaling_factor", "auto")
    config.expert_w1 = param_dict.get("expert_w1", None)
    config.expert_w2 = param_dict.get("expert_w2", None)
    # expert_w3: key absent → _UNSET (preset default); key present with null → None (fused); key present with string → custom name
    if "expert_w3" in param_dict:
        config.expert_w3 = param_dict["expert_w3"]  # None or string
    else:
        config.expert_w3 = _UNSET
    config.num_experts_attr = param_dict.get("num_experts_attr", None)
    config.top_k_attr = param_dict.get("top_k_attr", None)
    config.has_shared_experts = param_dict.get("has_shared_experts", None)
    config.shared_experts_pattern = param_dict.get("shared_experts_pattern", None)
    config.shared_experts_gate_pattern = param_dict.get("shared_experts_gate_pattern", None)

    return config


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_autoep_config(
    config: AutoEPConfig,
    world_size: int,
    pp_size: int,
    tp_size: int,
    sp_size: int,
) -> None:
    """Validate config constraints. Raises ValueError on invalid config."""
    if config.load_balance_coeff is not None:
        _raise_unsupported_load_balance_coeff(config.load_balance_coeff)

    if not config.enabled:
        return

    if tp_size > 1:
        raise ValueError("AutoEP does not currently support AutoTP "
                         f"(tensor_parallel.autotp_size={tp_size}). Disable AutoTP for this run; "
                         "AutoEP+AutoTP support is planned as follow-up work.")

    # ep_size must divide the stage size (world_size / pp_size)
    stage_size = world_size // pp_size
    if stage_size % config.autoep_size != 0:
        raise ValueError(f"autoep_size={config.autoep_size} must divide the stage size "
                         f"(world_size={world_size} / pp_size={pp_size} = {stage_size}). "
                         f"Valid autoep_size values: {_divisors(stage_size)}")

    # Validate preset_model if specified
    if config.preset_model is not None and config.preset_model not in PRESET_MODELS:
        raise ValueError(f"Unknown preset_model '{config.preset_model}'. "
                         f"Available presets: {list(available_preset_names())}")

    # Validate score_apply
    valid_score_apply = ("auto", "pre", "post")
    if config.score_apply not in valid_score_apply:
        raise ValueError(f"score_apply must be one of {valid_score_apply}, "
                         f"got '{config.score_apply}'")

    # Validate combine_impl
    valid_combine_impl = ("auto", "weighted_sum", "legacy_bmm")
    if config.combine_impl not in valid_combine_impl:
        raise ValueError(f"combine_impl must be one of {valid_combine_impl}, "
                         f"got '{config.combine_impl}'")

    # Validate score_func
    valid_score_func = ("auto", "softmax", "sigmoid")
    if config.score_func not in valid_score_func:
        raise ValueError(f"score_func must be one of {valid_score_func}, "
                         f"got '{config.score_func}'")

    # Validate group-limited routing constraints
    if config.num_limited_groups is not None:
        if config.num_limited_groups < 1:
            raise ValueError(f"num_limited_groups must be >= 1, got {config.num_limited_groups}")

    if config.num_expert_groups is not None:
        if config.num_expert_groups < 1:
            raise ValueError(f"num_expert_groups must be >= 1, got {config.num_expert_groups}")
        if config.num_limited_groups is not None and config.num_limited_groups > config.num_expert_groups:
            raise ValueError(f"num_limited_groups ({config.num_limited_groups}) must be <= "
                             f"num_expert_groups ({config.num_expert_groups})")
        logger.warning("num_expert_groups is set; interaction with EP topology "
                       "is not yet optimized.")

    # Warn if autoep_size == 1 (no EP needed)
    if config.autoep_size == 1:
        logger.warning("autoep_size=1 means every rank owns all experts with no AllToAll. "
                       "AutoEP replacement remains enabled, but expert-parallel communication "
                       "is bypassed because every rank owns all experts.")

    # Helper validators (local to validate_autoep_config)
    def _validate_attr_name(field_name: str, value, *, allow_dot: bool = False) -> None:
        if value is None:
            return
        if not isinstance(value, str) or value == "":
            raise ValueError(f"{field_name} must be a non-empty string")
        if not allow_dot and "." in value:
            raise ValueError(f"{field_name} must be a direct attribute name (no dots)")

    # Validate expert weight names
    _validate_attr_name("expert_w1", config.expert_w1)
    _validate_attr_name("expert_w2", config.expert_w2)
    if config.expert_w3 is not _UNSET and config.expert_w3 is not None:
        _validate_attr_name("expert_w3", config.expert_w3)

    # Validate model.config attribute names
    _validate_attr_name("num_experts_attr", config.num_experts_attr)
    _validate_attr_name("top_k_attr", config.top_k_attr)

    # Validate child-name fields (direct attribute names, not regex/path)
    _validate_attr_name("router_pattern", config.router_pattern)
    _validate_attr_name("expert_pattern", config.expert_pattern)
    _validate_attr_name("shared_experts_pattern", config.shared_experts_pattern)
    _validate_attr_name("shared_experts_gate_pattern", config.shared_experts_gate_pattern)

    # Validate has_shared_experts type
    if config.has_shared_experts is not None and not isinstance(config.has_shared_experts, bool):
        raise ValueError("has_shared_experts must be a boolean when set")

    # Warn if explicit top_k overrides top_k_attr
    if isinstance(config.top_k, int) and config.top_k_attr is not None:
        logger.warning("top_k is explicitly set; top_k_attr will be ignored.")

    if config.routed_scaling_factor != "auto" and not isinstance(config.routed_scaling_factor, (int, float)):
        raise ValueError("routed_scaling_factor must be a number or 'auto'")

    # Validate shared expert field pairing
    if config.has_shared_experts is True and not config.shared_experts_pattern:
        logger.warning("has_shared_experts=True but shared_experts_pattern is not set. "
                       "Shared expert detection requires both fields.")
    if config.shared_experts_pattern and config.has_shared_experts is not True:
        logger.warning(f"shared_experts_pattern='{config.shared_experts_pattern}' is set "
                       f"but has_shared_experts is not True. Pattern will be ignored.")
    if config.shared_experts_gate_pattern and config.has_shared_experts is not True:
        logger.warning(f"shared_experts_gate_pattern='{config.shared_experts_gate_pattern}' is set "
                       f"but has_shared_experts is not True. Pattern will be ignored.")

    # Warn if custom override fields are set alongside preset_model or auto-detect
    custom_fields_set = []
    if config.moe_layer_pattern is not None:
        custom_fields_set.append("moe_layer_pattern")
    if config.router_pattern is not None:
        custom_fields_set.append("router_pattern")
    if config.expert_pattern is not None:
        custom_fields_set.append("expert_pattern")
    if config.expert_w1 is not None:
        custom_fields_set.append("expert_w1")
    if config.expert_w2 is not None:
        custom_fields_set.append("expert_w2")
    if config.expert_w3 is not _UNSET:
        custom_fields_set.append("expert_w3")
    if config.num_experts_attr is not None:
        custom_fields_set.append("num_experts_attr")
    if config.top_k_attr is not None:
        custom_fields_set.append("top_k_attr")
    if config.has_shared_experts is not None:
        custom_fields_set.append("has_shared_experts")
    if config.shared_experts_pattern is not None:
        custom_fields_set.append("shared_experts_pattern")
    if config.shared_experts_gate_pattern is not None:
        custom_fields_set.append("shared_experts_gate_pattern")
    if custom_fields_set and config.preset_model is not None:
        logger.warning(f"Custom preset fields {custom_fields_set} are set alongside "
                       f"preset_model='{config.preset_model}'. Custom fields will override "
                       f"preset defaults during detection.")
    if custom_fields_set and config.preset_model is None and config.moe_layer_pattern is None:
        logger.warning(f"Custom preset fields {custom_fields_set} are set without preset_model or "
                       f"moe_layer_pattern. Overrides will apply to auto-detected presets or try-all.")


def validate_autoep_post_detection(
    config: AutoEPConfig,
    specs: list[MoELayerSpec],
) -> None:
    """Post-detection validation: ep_size vs num_experts constraints."""
    if not config.enabled or not specs:
        return

    for spec in specs:
        # ep_size must not exceed num_experts
        if config.autoep_size > spec.num_experts:
            valid_divisors = _divisors(spec.num_experts)
            raise ValueError(f"autoep_size={config.autoep_size} exceeds num_experts="
                             f"{spec.num_experts} in layer '{spec.moe_module_name}'. "
                             f"Each rank must own at least one expert. "
                             f"Valid autoep_size values (divisors of {spec.num_experts}): "
                             f"{valid_divisors}")

        # num_experts must be divisible by ep_size
        if spec.num_experts % config.autoep_size != 0:
            valid_sizes = [d for d in _divisors(spec.num_experts) if d <= spec.num_experts]
            raise ValueError(f"num_experts={spec.num_experts} in layer "
                             f"'{spec.moe_module_name}' is not divisible by "
                             f"autoep_size={config.autoep_size}. "
                             f"Suggested autoep_size values: {valid_sizes}")

        num_expert_groups = spec.num_expert_groups if spec.num_expert_groups is not None else config.num_expert_groups
        num_limited_groups = spec.num_limited_groups if spec.num_limited_groups is not None else config.num_limited_groups

        # Validate group-limited routing constraints after layer-specific defaults.
        if num_limited_groups is not None and num_expert_groups is None:
            raise ValueError(f"num_limited_groups requires num_expert_groups to be set "
                             f"in layer '{spec.moe_module_name}'")

        if num_expert_groups is not None:
            if num_expert_groups < 1:
                raise ValueError(f"num_expert_groups must be >= 1 in layer '{spec.moe_module_name}', "
                                 f"got {num_expert_groups}")
            if spec.num_experts % num_expert_groups != 0:
                raise ValueError(f"num_expert_groups ({num_expert_groups}) must divide "
                                 f"num_experts ({spec.num_experts}) in layer "
                                 f"'{spec.moe_module_name}'")
            if num_limited_groups is None:
                raise ValueError(f"num_limited_groups must be set when num_expert_groups is set "
                                 f"in layer '{spec.moe_module_name}'")
            if num_limited_groups < 1:
                raise ValueError(f"num_limited_groups must be >= 1 in layer '{spec.moe_module_name}', "
                                 f"got {num_limited_groups}")
            if num_limited_groups > num_expert_groups:
                raise ValueError(f"num_limited_groups ({num_limited_groups}) must be <= "
                                 f"num_expert_groups ({num_expert_groups}) in layer "
                                 f"'{spec.moe_module_name}'")


def _divisors(n: int) -> list[int]:
    """Return sorted list of positive divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def fill_autoep_config_from_hf(config: AutoEPConfig, model_config) -> None:
    """Back-fill AutoEPConfig fields from HF model config when user hasn't set them.

    HF field names (e.g. n_group, topk_group, routed_scaling_factor) differ from
    AutoEP's internal names, so we map them explicitly rather than relying on the
    user to duplicate these values in the DS config JSON.
    """
    if model_config is None:
        return
    # n_group / topk_group: DeepSeek-style node-limited routing groups
    if config.num_expert_groups is None:
        config.num_expert_groups = getattr(model_config, 'n_group', None)
    if config.num_limited_groups is None:
        config.num_limited_groups = getattr(model_config, 'topk_group', None)
    # routed_scaling_factor: sigmoid score scaling (DeepSeek-V3 / Moonlight)
    if config.routed_scaling_factor == "auto":
        hf_scale = getattr(model_config, 'routed_scaling_factor', None)
        if hf_scale is not None:
            config.route_scale = float(hf_scale)
