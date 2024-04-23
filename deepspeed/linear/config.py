# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """
    Configuration settings for LoRAOptimizedLinear.

    Attributes:
        lora_r (int): LoRA attention dimension, also know as the rank. Defaults is 64.
        lora_alpha (float): LoRA scaling factor, default is 16.
        base_weight_sharding (int): The degree to which the base weights are sharded,
            should typically be set to the data-parallel world size to maximize the memory
            reduction benefits. Defaults to 1, which means this feature is disabled.
    """
    lora_r: int = 64
    lora_alpha: float = 16.
    base_weight_sharding: int = 1


@dataclass
class QuantizationConfig:
    """
    Configuration settings for quantization for LoRAOptimizedLinear, QuantizedLinear,
    and QuantizedParameter

    Attributes:
        q_bits (int): The number of bits used for quantization. Default is 8.
        mantissa_bits (int): The number of bits reserved for the mantissa in fixed-point quantization. Default is 3.
        group_size (int): The size of the group used for quantization. Default is 512.
    """
    q_bits: int = 8
    mantissa_bits: int = 3
    group_size: int = 512
