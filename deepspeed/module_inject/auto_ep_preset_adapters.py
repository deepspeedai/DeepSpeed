# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Compatibility shim for AutoEP preset adapter APIs."""

from deepspeed.module_inject.auto_ep_presets.base import AutoEPPresetAdapter, ForwardContract, GroupRoutingConfig
from deepspeed.module_inject.auto_ep_presets.registry import get_preset_adapter

__all__ = [
    "AutoEPPresetAdapter",
    "ForwardContract",
    "GroupRoutingConfig",
    "get_preset_adapter",
]
