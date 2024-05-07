# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import is_dataclass
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist

from .config import LoRAConfig, QuantizationConfig
from .quantization import QuantizedParameter, QuantizedLinear


class OptimizedLinear(nn.Module):
    """
    Optimized version of nn.Linear that adds features such as:
      * LoRA w. base weight sharding
      * FP [6,8,12] quantization

    Arguments:
        input_dim: Required: size of each input sample
        output_dim: Required: size of each output sample
        bias: Optional: If set to False, the layer will not learn an additive bias. Default: False
        lora_config: Optional: LoRAConfig defining lora features and base-weight-sharding degree
        quantization_config: Optional: QuantizationConfig defining quantization features
        dtype: Optional: parameter dtype, only supports bfloat16 currently

    Returns:
        Returns a new nn.Module depending on the input config. Either native
        torch.nn.Linear, QuantizedLinear, or the full-featured DSOptimizedLinear.
    """

    def __new__(self,
                input_dim: int,
                output_dim: int,
                bias: bool = False,
                lora_config: LoRAConfig = None,
                quantization_config: QuantizationConfig = None,
                dtype=torch.bfloat16):

        if quantization_config is not None and not is_dataclass(quantization_config):
            raise ValueError(f"Expecting QuantizationConfig but received {type(quantization_config)}")
        if lora_config is not None and not is_dataclass(lora_config):
            raise ValueError(f"Expecting LoRAConfig but received {type(lora_config)}")
        if lora_config is None and quantization_config is None:
            # Everything disabled, fall back to normal nn.Linear
            self = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)

        elif lora_config:
            # lora enabled, quantization may or may not be
            self = LoRAOptimizedLinear(input_dim=input_dim,
                                       output_dim=output_dim,
                                       bias=bias,
                                       lora_config=lora_config,
                                       quantization_config=quantization_config,
                                       dtype=dtype)

        elif quantization_config:
            # only quantization enabled, no lora
            self = QuantizedLinear(input_dim=input_dim,
                                   output_dim=output_dim,
                                   bias=bias,
                                   quantization_config=quantization_config,
                                   dtype=dtype)
        return self


class LoRAOptimizedLinear(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bias: bool = False,
                 lora_config: LoRAConfig = None,
                 quantization_config: QuantizationConfig = None,
                 device=None,
                 dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.lora_config = lora_config
        self.quantization_config = quantization_config
        device = get_accelerator().current_device_name() if device is None else device
        assert self.lora_config is not None, "DSOptimizedLinear requires a LoRA config"

        self.zero_shards = self.lora_config.base_weight_sharding
        self.sharded_weight_size = int(float(self.input_dim) // self.zero_shards)
        w = torch.nn.Parameter(torch.empty((self.output_dim, self.sharded_weight_size), dtype=dtype))
        torch.nn.init.xavier_uniform_(w)

        if self.quantization_config is not None:
            assert dtype == torch.bfloat16, "only bfloat16 is supported when using quantization"
            self.base_weight = QuantizedParameter(w, quantization_config=quantization_config)
        else:
            self.base_weight = w

        self.base_weight.requires_grad = False

        # Use RS lora for now.
        self.lora_scaling_factor = self.lora_config.lora_alpha / math.sqrt(self.lora_config.lora_r)
        # Keeping lora weights in bf16 precision for ease of training.
        self.lora_weight_1 = nn.Linear(self.input_dim,
                                       self.lora_config.lora_r,
                                       bias=self.bias,
                                       device=device,
                                       dtype=dtype)
        self.lora_weight_2 = nn.Linear(self.lora_config.lora_r,
                                       self.output_dim,
                                       bias=self.bias,
                                       device=device,
                                       dtype=dtype)
        self.lora_weight_1.weight.requires_grad = True
        self.lora_weight_2.weight.requires_grad = True

    def full_weight(self):
        # This assumes weights are evenly sharded across gpus. which might not be correct.
        # in that case, we should flatten before all_gather.
        local_weight = self.base_weight.dequantized() if isinstance(self.base_weight,
                                                                    QuantizedParameter) else self.base_weight
        tensor_list = [
            torch.zeros_like(local_weight, device=local_weight.device, dtype=local_weight.dtype)
            for _ in range(self.zero_shards)
        ]
        dist.all_gather(tensor_list, local_weight)
        weight = nn.Parameter(torch.cat([tensor for tensor in tensor_list], dim=1))
        return weight

    def linear_without_F_linear(self, input, weight):
        output = torch.mm(input.reshape(-1, input.shape[-1]), weight)
        output = output.view(*input.shape[:-1], weight.shape[1])
        return output

    def forward(self, input_tensor):
        # Gather the sharded base weight
        if self.zero_shards > 1:
            with torch.no_grad():
                base_weight = self.full_weight()
        elif self.quantization_config:
            base_weight = self.base_weight.dequantized()
        else:
            base_weight = self.base_weight

        base_weight_output = F.linear(input_tensor, base_weight)
        lora_output = self.lora_weight_2(self.lora_weight_1(input_tensor))
        return base_weight_output + self.lora_scaling_factor * lora_output
