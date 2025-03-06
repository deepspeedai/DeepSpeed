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
                device=None,
                dtype=torch.bfloat16,
                linear_cls=nn.Linear):

        if quantization_config is not None and not is_dataclass(quantization_config):
            raise ValueError(f"Expecting QuantizationConfig but received {type(quantization_config)}")
        if lora_config is not None and not is_dataclass(lora_config):
            raise ValueError(f"Expecting LoRAConfig but received {type(lora_config)}")
        if lora_config is None and quantization_config is None:
            # Everything disabled, fall back to normal nn.Linear
            self = linear_cls(input_dim, output_dim, bias=bias, dtype=dtype, device=device)

        elif lora_config:
            # lora enabled, quantization may or may not be
            self = LoRAOptimizedLinear(input_dim=input_dim,
                                       output_dim=output_dim,
                                       bias=bias,
                                       lora_config=lora_config,
                                       quantization_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       linear_cls=linear_cls)

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
                 dtype=torch.bfloat16,
                 linear_cls=nn.Linear):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.lora_config = lora_config
        self.quantization_config = quantization_config
        self.device = get_accelerator().current_device_name() if device is None else device
        self.linear_cls = linear_cls
        self.dtype = dtype
        assert self.lora_config is not None, "DSOptimizedLinear requires a LoRA config"
        assert not self.bias, "bias=True is not supported by LoRAOptimizedLinear"
        self.zero_shards = self.lora_config.base_weight_sharding
        self.sharded_weight_size = int(float(self.input_dim) // self.zero_shards)
        if self.zero_shards > 1:
            assert self.zero_shards == dist.get_world_size(
            ), "base weight sharding is only supported across world size"
            w = torch.nn.Parameter(torch.empty(self.output_dim * self.sharded_weight_size, dtype=dtype),
                                   requires_grad=False)
        else:
            w = torch.nn.Parameter(torch.empty((self.output_dim, self.input_dim), dtype=dtype), requires_grad=False)
        torch.nn.init.xavier_uniform_(w.reshape(self.sharded_weight_size, self.output_dim))

        if self.quantization_config is not None:
            assert dtype == torch.bfloat16, "only bfloat16 is supported when using quantization"
            self.weight = QuantizedParameter(w, quantization_config=quantization_config)
        else:
            self.weight = w

        self.disabled = False
        self._initialized = False
        if not self.lora_config.delay_lora_init:
            self.init_lora()

    def disable(self):
        self.disabled = True
        self.weight = torch.nn.Parameter(torch.empty((self.output_dim, self.input_dim), dtype=self.dtype),
                                         requires_grad=False)

    def init_lora(self):
        if self.disabled:
            return

        if self.quantization_config is not None:
            # ensure quant-param wasn't stripped, in some cases transformers will do this during model init
            if not isinstance(self.weight, QuantizedParameter):
                self.weight = QuantizedParameter(self.weight, quantization_config=self.quantization_config)

        self._initialized = True
        self.weight.requires_grad = False

        # Mark base weight to prevent broadcast and ensure proper offload behavior
        self.weight.ds_optim_param = True

        self.lora_scaling_factor = self.lora_config.lora_alpha / self.lora_config.lora_r

        # Keeping lora weights in bf16 precision for ease of training.
        self.lora_weight_1 = self.linear_cls(self.input_dim,
                                             self.lora_config.lora_r,
                                             bias=self.bias,
                                             device=self.device,
                                             dtype=self.dtype)
        self.lora_weight_2 = self.linear_cls(self.lora_config.lora_r,
                                             self.output_dim,
                                             bias=self.bias,
                                             device=self.device,
                                             dtype=self.dtype)

        # initialize "A" with kaiming uniform and "B" with zeros following this
        # https://github.com/huggingface/peft/blob/62122b5add8d6892f70c82eaef2147a6ba33b90b/src/peft/tuners/lora/layer.py#L155
        nn.init.kaiming_uniform_(self.lora_weight_1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_weight_2.weight)
        self.lora_weight_1.weight.requires_grad = True
        self.lora_weight_2.weight.requires_grad = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        if not any([target in prefix for target in self.lora_config.target_mods]):
            # module does not match any target_mods, we must revert to normal nn.Linear via disable
            self.disable()
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                 unexpected_keys, error_msgs)

        if self.zero_shards > 1:
            if not dist.is_initialized():
                raise RuntimeError(
                    "attempting to use optimized linear base weight sharding but torch-distributed is not initialized, please init first."
                )
            rank = dist.get_rank()
            shape_local = self.output_dim * self.sharded_weight_size
            base_weight_name = f"{prefix}weight"
            incoming_param = state_dict[base_weight_name]
            state_dict[base_weight_name] = incoming_param.flatten().narrow(0, rank * shape_local, shape_local)

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

    def full_weight(self):
        base_weight = self.weight
        if getattr(base_weight, 'ds_offload', False):
            # move to gpu so we can dequant and all-gather
            assert base_weight.device == torch.device('cpu'), \
                f"expected base weight on cpu but found {base_weight.device}"
            base_weight.offload(revert=True)
            local_weight = base_weight.dequantized() if isinstance(base_weight, QuantizedParameter) else base_weight
            base_weight.offload()
        else:
            local_weight = base_weight.dequantized() if isinstance(base_weight, QuantizedParameter) else base_weight

        tensor_out = torch.empty(self.output_dim * self.input_dim,
                                 dtype=local_weight.dtype,
                                 device=local_weight.device)
        dist.all_gather_into_tensor(tensor_out, local_weight)
        return tensor_out.reshape(self.output_dim, self.input_dim)

    def linear_without_F_linear(self, input, weight):
        output = torch.mm(input.reshape(-1, input.shape[-1]), weight)
        output = output.view(*input.shape[:-1], weight.shape[1])
        return output

    def forward(self, input_tensor):
        if self.disabled:
            return F.linear(input_tensor, self.weight)
        assert self._initialized, "init_lora was never called, please initialize before proceeding"

        # Gather the sharded base weight
        if self.zero_shards > 1:
            with torch.no_grad():
                base_weight = self.full_weight()
        elif self.quantization_config:
            base_weight = self.weight.dequantized()
        else:
            base_weight = self.weight

        base_weight_output = F.linear(input_tensor, base_weight)
        lora_output = self.lora_weight_2(self.lora_weight_1(input_tensor))
        return base_weight_output + self.lora_scaling_factor * lora_output
