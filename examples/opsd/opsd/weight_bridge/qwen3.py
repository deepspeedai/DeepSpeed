# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Weight bridge for Qwen3 dense models.

Qwen3-dense uses the same overall layout as Qwen2 with one addition:
per-head RMSNorm applied to the query and key projections before attention::

    model.layers.{i}.self_attn.q_norm.weight   # shape [head_dim]
    model.layers.{i}.self_attn.k_norm.weight   # shape [head_dim]

These weights are 1-D over ``head_dim`` (not ``num_heads * head_dim``), so they
are **replicated** across TP ranks: every rank owns a subset of heads but each
head normalises with the same per-head-dim scalars.

Qwen3-MoE (the ``Qwen3MoeForCausalLM`` family) is **not** covered here — MoE
introduces gate/expert routing and per-expert MLPs that need their own bridge.
Add a sibling ``qwen3_moe.py`` when that path becomes a priority.
"""

from typing import Optional

from opsd.weight_bridge.base import ParallelKind
from opsd.weight_bridge.qwen2 import Qwen2WeightBridge


class Qwen3WeightBridge(Qwen2WeightBridge):
    arch = "qwen3"

    _Q_NORM = "self_attn.q_norm.weight"
    _K_NORM = "self_attn.k_norm.weight"

    def _extra_layer_kind(self, suffix: str) -> Optional[ParallelKind]:
        if suffix in (self._Q_NORM, self._K_NORM):
            return ParallelKind.REPLICATED
        return None
