# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
AutoEP configuration dataclasses and preset model specs.

Ported from the prototype branch (tohtana/add_autoep) with minor
adaptations for DeepSpeed conventions:
  - DeepSpeedConfigModel replaced with plain dataclass (avoids Pydantic dep)
  - parse_autoep_config / validate_* helpers match original API

Usage in ds_config.json::

    {
      "expert_parallel": {
        "enabled": true,
        "autoep_size": 8,
        "preset_model": "mixtral"
      }
    }
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ===================================================================
# MoE layer specification
# ===================================================================


@dataclass
class MoELayerSpec:
    """Specification of a detected MoE layer in the model.

    Attributes:
        parent:      Parent module that contains *child_name*.
        child_name:  Attribute name on *parent* that is the MoE layer.
        layer_idx:   Global layer index (order in which layers were found).
        num_experts: Total number of experts in this layer.
        dim:         Model hidden dimension.
        ffn_dim:     Expert FFN intermediate dimension.
        gate_bias:   Whether the router gate has a bias term.
        top_k:       Number of experts each token is routed to.
    """
    parent: object
    child_name: str
    layer_idx: int
    num_experts: int
    dim: int
    ffn_dim: int
    gate_bias: bool
    top_k: int


# ===================================================================
# Model preset specs
# ===================================================================


@dataclass
class MoEModelPreset:
    """Structural description of a supported MoE architecture.

    Fields map to attribute paths in the model's forward hierarchy.
    """
    # Attribute names to traverse from the root module to reach one MoE block
    # e.g. ["model", "layers"]  means model.model.layers[i]
    layers_path: List[str]

    # Attribute name of the MoE sub-layer inside a single decoder block
    # e.g. "block_sparse_moe" for Mixtral
    moe_layer_attr: str

    # Attribute name of the router/gate inside the MoE sub-layer
    gate_attr: str

    # Attribute names for the expert weights:
    #   experts_attr  → module holding the expert collection
    # For fused_3d format (transformers 5.0+): gate_up_proj / down_proj
    # For module_list format: individual expert modules
    experts_attr: str

    # Number of activated experts per token
    top_k: int

    # Whether the gate linear has a bias
    gate_bias: bool = False

    # Storage format of expert weights
    # "fused_3d"    → experts.gate_up_proj[E, 2*ffn, dim]  (HF ≥5.0)
    # "module_list" → nn.ModuleList of individual expert modules
    expert_storage: str = "fused_3d"

    # Attribute that exposes num_experts from the MoE sub-layer
    num_experts_attr: str = "num_experts"

    # Attribute name for ffn_dim (inside the expert module or moe layer)
    ffn_dim_attr: Optional[str] = None


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

PRESET_MODELS: Dict[str, MoEModelPreset] = {
    "mixtral":
    MoEModelPreset(
        layers_path=["model", "layers"],
        moe_layer_attr="block_sparse_moe",
        gate_attr="gate",
        experts_attr="experts",
        top_k=2,
        gate_bias=False,
        expert_storage="module_list",
        num_experts_attr="num_experts",
        ffn_dim_attr=None,  # auto-inferred from w1.out_features
    ),
    "qwen3_moe":
    MoEModelPreset(
        layers_path=["model", "layers"],
        moe_layer_attr="mlp",
        gate_attr="gate",
        experts_attr="experts",
        top_k=8,
        gate_bias=False,
        expert_storage="module_list",
        num_experts_attr="num_experts",
        ffn_dim_attr=None,
    ),
    "deepseek_v2":
    MoEModelPreset(
        layers_path=["model", "layers"],
        moe_layer_attr="mlp",
        gate_attr="gate",
        experts_attr="experts",
        top_k=6,
        gate_bias=False,
        expert_storage="module_list",
        num_experts_attr="num_experts",
        ffn_dim_attr=None,
    ),
    "deepseek_v3":
    MoEModelPreset(
        layers_path=["model", "layers"],
        moe_layer_attr="mlp",
        gate_attr="gate",
        experts_attr="experts",
        top_k=8,
        gate_bias=False,
        expert_storage="module_list",
        num_experts_attr="num_experts",
        ffn_dim_attr=None,
    ),
    "llama4":
    MoEModelPreset(
        layers_path=["model", "layers"],
        moe_layer_attr="feed_forward",
        gate_attr="router",
        experts_attr="experts",
        top_k=1,
        gate_bias=False,
        expert_storage="fused_3d",
        num_experts_attr="num_experts",
        ffn_dim_attr="intermediate_size",
    ),
}

# ===================================================================
# AutoEP configuration
# ===================================================================


@dataclass
class AutoEPConfig:
    """Runtime configuration for AutoEP.

    Attributes:
        enabled:      Whether AutoEP is active.
        autoep_size:  Expert parallel world size (EP group size).
                      Must evenly divide the total number of experts.
        preset_model: Key into PRESET_MODELS, or None for manual spec.
        layer_specs:  Optional list of per-layer overrides (advanced).
    """
    enabled: bool = False
    autoep_size: int = 1
    preset_model: Optional[str] = None
    layer_specs: List[dict] = field(default_factory=list)


# ===================================================================
# Config parsing helpers
# ===================================================================


def parse_autoep_config(param_dict: dict) -> AutoEPConfig:
    """Parse the ``expert_parallel`` block from a DeepSpeed config dict.

    Args:
        param_dict: The full DeepSpeed config dictionary.

    Returns:
        An :class:`AutoEPConfig` instance (disabled if the block is absent).
    """
    ep_cfg = param_dict.get("expert_parallel", {})
    if not ep_cfg:
        return AutoEPConfig(enabled=False)

    enabled = ep_cfg.get("enabled", False)
    if not enabled:
        return AutoEPConfig(enabled=False)

    autoep_size = ep_cfg.get("autoep_size", 1)
    preset_model = ep_cfg.get("preset_model", None)
    layer_specs = ep_cfg.get("layer_specs", [])

    return AutoEPConfig(
        enabled=True,
        autoep_size=autoep_size,
        preset_model=preset_model,
        layer_specs=layer_specs,
    )


def validate_autoep_config(config: AutoEPConfig, world_size: int) -> None:
    """Validate the AutoEP configuration before model initialisation.

    Args:
        config:     Parsed AutoEP config.
        world_size: Global process-group world size.

    Raises:
        ValueError: If the config is internally inconsistent.
    """
    if not config.enabled:
        return

    if config.autoep_size <= 0:
        raise ValueError(f"autoep_size must be > 0, got {config.autoep_size}")

    if world_size % config.autoep_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by autoep_size ({config.autoep_size}).")

    if config.preset_model is not None and config.preset_model not in PRESET_MODELS:
        raise ValueError(f"Unknown preset_model '{config.preset_model}'. "
                         f"Available presets: {sorted(PRESET_MODELS.keys())}")


def validate_autoep_post_detection(
    config: AutoEPConfig,
    layer_specs: List[MoELayerSpec],
) -> None:
    """Validate EP config after the model has been scanned for MoE layers.

    Args:
        config:      Parsed AutoEP config.
        layer_specs: List of detected :class:`MoELayerSpec` objects.

    Raises:
        ValueError: If num_experts is not divisible by autoep_size.
    """
    if not config.enabled:
        return

    if not layer_specs:
        raise ValueError("AutoEP is enabled but no MoE layers were detected in the model. "
                         "Check preset_model or layer_specs.")

    for spec in layer_specs:
        if spec.num_experts % config.autoep_size != 0:
            raise ValueError(f"num_experts ({spec.num_experts}) for layer {spec.layer_idx} "
                             f"is not divisible by autoep_size ({config.autoep_size}).")
