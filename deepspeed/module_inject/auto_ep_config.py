# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP configuration: config parsing, model presets, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deepspeed.utils import logger

# Sentinel for "not specified in config, use preset default".
# Unlike None (which means "fused gate+up, no separate w3"), _UNSET means
# the user did not set the field at all.  Compare with `is _UNSET`.
_UNSET = object()

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MoEModelPreset:
    """Preset configuration for a known MoE model family."""

    moe_layer_pattern: str  # Regex matching MoE module names
    router_pattern: str  # Child name for router/gate (e.g., "gate")
    experts_pattern: str  # Child name for experts (e.g., "experts")
    expert_storage: Literal["fused_3d", "module_list"]
    expert_w1: str  # Weight name: "gate_up_proj" (fused) or "gate_proj"/"w1"
    expert_w2: str  # Weight name: "down_proj" or "w2"
    expert_w3: str | None  # None (fused gate+up) or "up_proj"/"w3"
    num_experts_attr: str  # model.config attribute name for num_experts
    top_k_attr: str  # model.config attribute name for top_k
    score_func: Literal["softmax", "sigmoid"]
    score_apply: Literal["pre", "post"]
    route_norm: bool  # Default top-k renormalization
    gate_bias: bool  # Whether router gate has bias
    has_shared_experts: bool = False
    shared_experts_pattern: str = ""


@dataclass
class MoELayerSpec:
    """Detected MoE layer specification for a single module in the model."""

    moe_module_name: str  # e.g., "model.layers.0.mlp"
    model_family: str  # e.g., "mixtral", "qwen3_moe"
    router_name: str  # e.g., "gate"
    experts_name: str  # e.g., "experts"
    expert_storage: Literal["fused_3d", "module_list"]
    expert_w1_name: str
    expert_w2_name: str
    expert_w3_name: str | None
    num_experts: int
    top_k: int
    hidden_size: int
    ffn_hidden_size: int
    score_func: Literal["softmax", "sigmoid"]
    score_apply: Literal["pre", "post"]
    route_norm: bool
    gate_bias: bool
    return_router_logits: bool
    router_logits_capture_target: Literal["moe_block", "router", "none"]
    router_logits_capture_index: int | None
    router_logits_capture_layer_name: str | None
    has_shared_experts: bool
    shared_experts_name: str


@dataclass
class AutoEPConfig:
    """User-facing configuration parsed from DS config JSON."""

    enabled: bool = False
    autoep_size: int = 1
    preset_model: str | None = None
    moe_layer_pattern: str | None = None
    expert_pattern: str | None = None
    router_pattern: str | None = None
    use_grouped_mm: bool = True
    grouped_mm_backend: Literal["auto", "torch", "cutlass", "sequential"] = "auto"
    route_norm: bool | None = None  # None = auto-detect from model config
    route_scale: float = 1.0
    score_apply: Literal["auto", "pre", "post"] = "auto"
    num_expert_groups: int | None = None
    num_limited_groups: int | None = None
    score_func: Literal["auto", "softmax", "sigmoid"] = "auto"
    top_k: int | str = "auto"  # int or "auto"
    load_balance_coeff: float | None = 1e-3
    routed_scaling_factor: float | str = "auto"  # float or "auto"
    # Custom preset fields (override defaults in custom/built-in preset paths)
    expert_w1: str | None = None
    expert_w2: str | None = None
    expert_w3: object = _UNSET  # _UNSET = use preset default; None = fused gate+up; str = custom name
    num_experts_attr: str | None = None
    top_k_attr: str | None = None
    has_shared_experts: bool | None = None
    shared_experts_pattern: str | None = None


# ---------------------------------------------------------------------------
# Preset model definitions
# ---------------------------------------------------------------------------

PRESET_MODELS: dict[str, MoEModelPreset] = {
    "mixtral":
    MoEModelPreset(
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
    ),
    "qwen3_moe":
    MoEModelPreset(
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
    ),
    "deepseek_v2":
    MoEModelPreset(
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
        route_norm=True,
        gate_bias=False,
        has_shared_experts=True,
        shared_experts_pattern="shared_experts",
    ),
    "deepseek_v3":
    MoEModelPreset(
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
    ),
    "llama4":
    MoEModelPreset(
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
        score_apply="post",
        route_norm=False,
        gate_bias=False,
        has_shared_experts=True,
        shared_experts_pattern="shared_expert",
    ),
}

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
    config.grouped_mm_backend = param_dict.get("grouped_mm_backend", "auto")
    config.route_norm = param_dict.get("route_norm", None)
    config.route_scale = param_dict.get("route_scale", 1.0)
    config.score_apply = param_dict.get("score_apply", "auto")
    config.num_expert_groups = param_dict.get("num_expert_groups", None)
    config.num_limited_groups = param_dict.get("num_limited_groups", None)
    config.score_func = param_dict.get("score_func", "auto")
    config.top_k = param_dict.get("top_k", "auto")
    config.load_balance_coeff = param_dict.get("load_balance_coeff", 1e-3)
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
    if not config.enabled:
        return

    # TP + SP mutual exclusivity
    if tp_size > 1 and sp_size > 1:
        raise ValueError(f"AutoEP does not support simultaneous TP (autotp_size={tp_size}) "
                         f"and SP (sequence_parallel_size={sp_size}). Use one or the other.")

    # ep_size must divide the stage size (world_size / pp_size)
    stage_size = world_size // pp_size
    if stage_size % config.autoep_size != 0:
        raise ValueError(f"autoep_size={config.autoep_size} must divide the stage size "
                         f"(world_size={world_size} / pp_size={pp_size} = {stage_size}). "
                         f"Valid autoep_size values: {_divisors(stage_size)}")

    # Validate preset_model if specified
    if config.preset_model is not None and config.preset_model not in PRESET_MODELS:
        raise ValueError(f"Unknown preset_model '{config.preset_model}'. "
                         f"Available presets: {list(PRESET_MODELS.keys())}")

    # Validate grouped_mm_backend
    valid_backends = ("auto", "torch", "cutlass", "sequential")
    if config.grouped_mm_backend not in valid_backends:
        raise ValueError(f"grouped_mm_backend must be one of {valid_backends}, "
                         f"got '{config.grouped_mm_backend}'")

    # Validate score_apply
    valid_score_apply = ("auto", "pre", "post")
    if config.score_apply not in valid_score_apply:
        raise ValueError(f"score_apply must be one of {valid_score_apply}, "
                         f"got '{config.score_apply}'")

    # Validate score_func
    valid_score_func = ("auto", "softmax", "sigmoid")
    if config.score_func not in valid_score_func:
        raise ValueError(f"score_func must be one of {valid_score_func}, "
                         f"got '{config.score_func}'")

    # Validate num_expert_groups constraints
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
                       "AutoEP replacement will be bypassed; the model runs as-is with DP.")

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

    # Validate has_shared_experts type
    if config.has_shared_experts is not None and not isinstance(config.has_shared_experts, bool):
        raise ValueError("has_shared_experts must be a boolean when set")

    # Warn if explicit top_k overrides top_k_attr
    if isinstance(config.top_k, int) and config.top_k_attr is not None:
        logger.warning("top_k is explicitly set; top_k_attr will be ignored.")

    # Validate shared expert field pairing
    if config.has_shared_experts is True and not config.shared_experts_pattern:
        logger.warning("has_shared_experts=True but shared_experts_pattern is not set. "
                       "Shared expert detection requires both fields.")
    if config.shared_experts_pattern and config.has_shared_experts is not True:
        logger.warning(f"shared_experts_pattern='{config.shared_experts_pattern}' is set "
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

        # Validate num_expert_groups divides num_experts
        if config.num_expert_groups is not None:
            if spec.num_experts % config.num_expert_groups != 0:
                raise ValueError(f"num_expert_groups ({config.num_expert_groups}) must divide "
                                 f"num_experts ({spec.num_experts}) in layer "
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
