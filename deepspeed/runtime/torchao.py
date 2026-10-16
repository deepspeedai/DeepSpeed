# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Lazy TorchAO registry integration for DeepSpeed expert containers."""

from dataclasses import dataclass
from functools import partial

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from deepspeed.accelerator import get_accelerator
from deepspeed.moe.ep_experts import GroupedExperts


@dataclass
class GroupedExpertsConfig(AOBaseConfig):
    config: AOBaseConfig


@register_quantize_module_handler(GroupedExpertsConfig)
def _configure_grouped_experts(module, config):
    if not isinstance(module, GroupedExperts):
        raise TypeError("GroupedExpertsConfig requires a DeepSpeed GroupedExperts module")
    provider = get_accelerator().get_hifloat8_provider()
    if provider is None:
        raise RuntimeError("No HiFloat8 provider for GroupedExperts")
    module.grouped_mm = partial(provider.hifloat8_grouped_mm, config=config.config)
    return module
