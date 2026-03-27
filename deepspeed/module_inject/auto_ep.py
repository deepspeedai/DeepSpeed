# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
AutoEP: automatic Expert Parallelism setup for pre-trained MoE models.

Two public entry points:
  - ``AutoEP(model, config).ep_parser()``    – detect MoE layers
  - ``AutoEP.replace_moe_layer(spec, ...)``  – replace a single layer

Ported from the prototype branch (tohtana/add_autoep).
"""

import logging
from typing import List, Optional

import torch.nn as nn

from deepspeed.module_inject.auto_ep_config import (
    AutoEPConfig,
    MoELayerSpec,
    MoEModelPreset,
    PRESET_MODELS,
)

logger = logging.getLogger(__name__)


class AutoEP:
    """Detect and replace MoE layers in a model with AutoEP equivalents.

    Args:
        model:   The model to process (typically a ``PreTrainedModel``).
        config:  Parsed :class:`AutoEPConfig`.
    """

    def __init__(self, model: nn.Module, config: AutoEPConfig):
        self.model = model
        self.config = config

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def ep_parser(self) -> List[MoELayerSpec]:
        """Scan the model and return a list of :class:`MoELayerSpec` objects.

        Raises:
            ValueError: If ``preset_model`` is set but not found, or if no
                MoE layers are detected when AutoEP is enabled.
        """
        preset_name = self.config.preset_model
        if preset_name is not None:
            preset = PRESET_MODELS[preset_name]
            return self._parse_with_preset(preset)

        # Manual layer_specs fallback (not yet implemented; raise clearly)
        raise NotImplementedError("AutoEP without a preset_model requires explicit layer_specs. "
                                  "Set 'preset_model' in the expert_parallel config, or contribute "
                                  "a manual detection path.")

    @staticmethod
    def replace_moe_layer(
        spec: MoELayerSpec,
        ep_size: int,
        ep_rank: int,
        ep_group,
        preset: Optional[MoEModelPreset] = None,
    ) -> None:
        """Replace the MoE sub-layer described by *spec* with an AutoEPMoELayer.

        The replacement is done in-place on ``spec.parent``.

        Args:
            spec:      Specification returned by :meth:`ep_parser`.
            ep_size:   Expert-parallel world size (EP group size).
            ep_rank:   This rank's position in the EP group.
            ep_group:  PyTorch distributed process group for expert comms.
            preset:    Model preset, used to re-derive expert storage format.
        """
        # Import here to avoid circular imports at module level
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

        original_layer = getattr(spec.parent, spec.child_name)

        new_layer = AutoEPMoELayer(
            original_layer=original_layer,
            spec=spec,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
            preset=preset,
        )

        setattr(spec.parent, spec.child_name, new_layer)
        logger.debug(
            "AutoEP: replaced layer %d (%s.%s) with AutoEPMoELayer "
            "(ep_size=%d, ep_rank=%d, num_experts=%d)",
            spec.layer_idx,
            type(spec.parent).__name__,
            spec.child_name,
            ep_size,
            ep_rank,
            spec.num_experts,
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _parse_with_preset(self, preset: MoEModelPreset) -> List[MoELayerSpec]:
        """Walk the model using the preset's path configuration."""
        # Traverse from root to the layer list
        container = self.model
        for attr in preset.layers_path:
            container = getattr(container, attr)

        specs: List[MoELayerSpec] = []
        for layer_idx, block in enumerate(container):
            moe_layer = getattr(block, preset.moe_layer_attr, None)
            if moe_layer is None:
                # Dense layer (e.g. first/last block in some models)
                continue

            num_experts = getattr(moe_layer, preset.num_experts_attr)
            dim, ffn_dim = self._infer_dims(moe_layer, preset)

            spec = MoELayerSpec(
                parent=block,
                child_name=preset.moe_layer_attr,
                layer_idx=layer_idx,
                num_experts=num_experts,
                dim=dim,
                ffn_dim=ffn_dim,
                gate_bias=preset.gate_bias,
                top_k=preset.top_k,
            )
            specs.append(spec)

        logger.info("AutoEP: detected %d MoE layers (preset=%s)", len(specs), self.config.preset_model)
        return specs

    @staticmethod
    def _infer_dims(moe_layer: nn.Module, preset: MoEModelPreset):
        """Infer (hidden_dim, ffn_dim) from the expert weights."""
        experts = getattr(moe_layer, preset.experts_attr)

        if preset.expert_storage == "fused_3d":
            # gate_up_proj: [E, 2*ffn, dim]  /  down_proj: [E, dim, ffn]
            gate_up = getattr(experts, "gate_up_proj")
            # gate_up shape: (E, 2*ffn_dim, hidden_dim)
            ffn_dim = gate_up.shape[1] // 2
            dim = gate_up.shape[2]
        elif preset.expert_storage == "module_list":
            # ModuleList of expert modules; inspect first expert
            first_expert = experts[0]
            # Typical attr names across models: w1 / gate_proj
            for w_attr in ("w1", "gate_proj", "fc1"):
                w = getattr(first_expert, w_attr, None)
                if w is not None:
                    # w: Linear(ffn_dim, hidden_dim)
                    # weight shape: (ffn_dim, hidden_dim)
                    ffn_dim = w.weight.shape[0]
                    dim = w.weight.shape[1]
                    break
            else:
                raise AttributeError(f"Cannot determine dim/ffn_dim from expert module {type(first_expert)}. "
                                     "None of [w1, gate_proj, fc1] found.")
        else:
            raise ValueError(f"Unknown expert_storage format: {preset.expert_storage}")

        return dim, ffn_dim
