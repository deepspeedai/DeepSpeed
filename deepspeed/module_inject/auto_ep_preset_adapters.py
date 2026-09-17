# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compatibility shim for AutoEP preset adapter APIs."""

from deepspeed.module_inject.auto_ep_presets.base import (
    AutoEPPresetAdapter,
    ForwardContract,
    GroupRoutingConfig,
    TransformersTopLevelRouterLogitsAdapter,
)
from deepspeed.module_inject.auto_ep_presets.deepseek_v2 import DeepSeekV2PresetAdapter
from deepspeed.module_inject.auto_ep_presets.deepseek_v3 import DeepSeekV3PresetAdapter
from deepspeed.module_inject.auto_ep_presets.qwen3_5_moe import Qwen35MoePresetAdapter
from deepspeed.module_inject.auto_ep_presets.registry import get_preset_adapter

__all__ = [
    "AutoEPPresetAdapter",
    "DeepSeekV2PresetAdapter",
    "DeepSeekV3PresetAdapter",
    "ForwardContract",
    "GroupRoutingConfig",
    "Qwen35MoePresetAdapter",
    "TransformersTopLevelRouterLogitsAdapter",
    "get_preset_adapter",
]
