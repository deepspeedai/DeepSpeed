# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
import deepspeed.comm as dist

from deepspeed.accelerator import get_accelerator
from deepspeed.linear import OptimizedLinear, LoRAConfig, QuantizationConfig
from unit.common import DistributedTest

from deepspeed.ops.op_builder import FPQuantizerBuilder

if not deepspeed.ops.__compatible_ops__[FPQuantizerBuilder.NAME]:
    pytest.skip("FPQuantizer op is not available on this system", allow_module_level=True)


class TestBasicLinear(DistributedTest):
    world_size = 2

    def test(self):
        lora_config = None
        quantization_config = None

        input_features = 64  # Number of input features
        output_features = 64  # Number of output features
        batch_size = 1  # Number of samples in a batch

        linear_layer = OptimizedLinear(input_dim=input_features,
                                       output_dim=output_features,
                                       lora_config=lora_config,
                                       quantization_config=quantization_config,
                                       dtype=torch.bfloat16)

        dummy_input = torch.rand(batch_size, input_features, dtype=torch.bfloat16)
        output = linear_layer(dummy_input)
        assert output.shape == (batch_size, output_features)


@pytest.mark.parametrize("base_weight_sharding", [1, 2])
class TestLoRALinear(DistributedTest):
    world_size = 2

    def test(self, base_weight_sharding):
        rank = dist.get_rank()
        lora_config = None
        quantization_config = None

        input_features = 64  # Number of input features
        output_features = 64  # Number of output features
        batch_size = 5  # Number of samples in a batch

        lora_config = LoRAConfig(lora_r=16, lora_alpha=16, base_weight_sharding=base_weight_sharding)

        linear_layer = OptimizedLinear(input_dim=input_features,
                                       output_dim=output_features,
                                       lora_config=lora_config,
                                       quantization_config=quantization_config,
                                       dtype=torch.bfloat16)
        device = get_accelerator().current_device_name()
        linear_layer = linear_layer.to(device)
        if rank == 0:
            for n, p in linear_layer.named_parameters():
                print(f"{n}, {p.shape}")

        dummy_input = torch.rand(batch_size, input_features, device=device, dtype=torch.bfloat16)

        output = linear_layer(dummy_input)
        assert output.shape == (batch_size, output_features)


@pytest.mark.parametrize("q_bits", [8, 6])
class TestQuantLinear(DistributedTest):
    world_size = 2

    def test(self, q_bits):
        rank = dist.get_rank()
        lora_config = None

        input_features = 64  # Number of input features
        output_features = 64  # Number of output features
        batch_size = 5  # Number of samples in a batch

        lora_config = None
        quantization_config = QuantizationConfig(q_bits=q_bits)

        linear_layer = OptimizedLinear(input_dim=input_features,
                                       output_dim=output_features,
                                       lora_config=lora_config,
                                       quantization_config=quantization_config,
                                       dtype=torch.bfloat16)
        device = get_accelerator().current_device_name()
        linear_layer = linear_layer.to(device)
        dummy_input = torch.rand([batch_size, input_features], device=device, dtype=torch.bfloat16)

        output = linear_layer(dummy_input)
        assert output.shape == (batch_size, output_features)


@pytest.mark.parametrize("base_weight_sharding", [1, 2], ids=['bws1', 'bws2'])
@pytest.mark.parametrize("q_bits", [8, 6], ids=['qbit8', 'qbit6'])
class TestOptimizedLinear(DistributedTest):
    world_size = 2

    def test(self, base_weight_sharding, q_bits):
        rank = dist.get_rank()
        lora_config = None

        input_features = 64  # Number of input features
        output_features = 64  # Number of output features
        batch_size = 5  # Number of samples in a batch

        lora_config = LoRAConfig(lora_r=16, lora_alpha=16, base_weight_sharding=base_weight_sharding)
        quantization_config = QuantizationConfig(q_bits=q_bits)

        linear_layer = OptimizedLinear(input_dim=input_features,
                                       output_dim=output_features,
                                       lora_config=lora_config,
                                       quantization_config=quantization_config,
                                       dtype=torch.bfloat16)
        device = get_accelerator().current_device_name()
        linear_layer = linear_layer.to(device)
        dummy_input = torch.rand([batch_size, input_features], device=device, dtype=torch.bfloat16)
        output = linear_layer(dummy_input)
        assert output.shape == (batch_size, output_features)
