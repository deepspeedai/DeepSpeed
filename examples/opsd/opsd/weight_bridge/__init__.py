# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Architecture-specific bridges that slice HuggingFace weights for vLLM TP.

A bridge takes the student's full ``(name, tensor)`` pairs (after we've
gathered them across ZeRO-3 ranks) and emits the per-vLLM-rank slices ready
to push into vLLM's ``model.load_weights(...)``.

vLLM internally fuses Q/K/V into ``qkv_proj`` and gate/up into ``gate_up_proj``.
We do **not** pre-fuse on our side — vLLM's loader already understands the
unfused HuggingFace layout — so the bridge only needs to know each parameter's
parallel kind (column / row / vocab / replicated) and slice on the right dim.
"""

from opsd.weight_bridge.base import ParallelKind, WeightBridge
from opsd.weight_bridge.qwen2 import Qwen2WeightBridge
from opsd.weight_bridge.qwen3 import Qwen3WeightBridge

__all__ = ["WeightBridge", "ParallelKind", "Qwen2WeightBridge", "Qwen3WeightBridge", "get_bridge"]


def get_bridge(arch: str) -> WeightBridge:
    """Look up a bridge by architecture key (matches HF's ``model_type``)."""
    key = arch.lower()
    if key in ("qwen2", "qwen2.5"):
        return Qwen2WeightBridge()
    if key in ("qwen3", ):
        return Qwen3WeightBridge()
    raise ValueError(f"No weight bridge registered for arch {arch!r}; "
                     f"add a sibling of opsd/weight_bridge/qwen2.py and register here")
