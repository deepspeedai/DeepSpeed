# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from .utils import Quantizer, DeQuantizer, concat_to_compat_param
from typing import Tuple, Callable, Dict
from deepspeed.runtime.zero import register_external_parameter

quantized_weight_registry = {}
is_zero3_enabled = False


# deal with weight sharing
def get_quantized_weight_wrapper(model, pre_quant_weight: nn.Parameter, quantize_weight_fn: Callable) -> nn.Parameter:
    if id(pre_quant_weight) in quantized_weight_registry:
        compat_tensor = quantized_weight_registry[id(pre_quant_weight)]
        if is_zero3_enabled:
            register_external_parameter(model, compat_tensor)

        return quantized_weight_registry[id(pre_quant_weight)]
    else:
        quantized_weights, quant_scale, quant_min = quantize_weight_fn()
        quantized_weight_registry[id(pre_quant_weight)] = concat_to_compat_param(quantized_weights, quant_scale,
                                                                                 quant_min)
        return quantized_weight_registry[id(pre_quant_weight)]


def get_quantize_weight_fn(quantizer: Quantizer, pre_quant_weight: nn.Parameter) -> Callable:

    def func() -> Tuple[nn.Parameter, Tensor, Tensor]:
        quantized_weights, quant_scale, quant_min = quantizer.quantize(pre_quant_weight.data)
        # A temporary hack as zero Zero3 assume all model weights has the same type. in all_gather_coalesced.get_only_unique_item
        quantized_weights = quantized_weights.view(pre_quant_weight.dtype)
        quant_scale = quant_scale.type(pre_quant_weight.dtype)
        quant_min = quant_min.type(pre_quant_weight.dtype)
        return quantized_weights, quant_scale, quant_min

    return func


class QuantizedLinear(nn.Linear):

    def __init__(self, config: Dict, pre_quant_layer: nn.Linear) -> None:
        super(QuantizedLinear, self).__init__(in_features=pre_quant_layer.in_features,
                                              out_features=pre_quant_layer.out_features,
                                              bias=pre_quant_layer.bias is not None,
                                              device=pre_quant_layer.weight.device,
                                              dtype=pre_quant_layer.weight.dtype)
        self.config = config

        self.quantizer = Quantizer(config=config)
        self.bias = pre_quant_layer.bias
        self.weight = get_quantized_weight_wrapper(self, pre_quant_layer.weight,
                                                   get_quantize_weight_fn(self.quantizer, pre_quant_layer.weight))

        self.weight.dequantizer = DeQuantizer(config, pre_quant_layer.weight.dtype)

    def forward(self, input: Tensor) -> Tensor:
        quantized_weight, quant_scale, quant_min = self.weight.deconcat(self.weight)
        temp_dequantized_weight = self.weight.dequantizer.dequantize(quantized_weight.view(torch.uint8), quant_scale,
                                                                     quant_min)

        # !!! Do not use torch.functional.linear(input, temp_dequantized_weight, self.bias) here as in zero3 torch.functional.linear is
        # replaced by LinearFunctionForZeroStage3. Which assume weight is non-temporary.
        # If weight is temp buffer there will be memory leak.
        return torch._C._nn.linear(input, temp_dequantized_weight, self.bias)


class QuantizedEmbedding(nn.Embedding):

    def __init__(self, config: Dict, pre_quant_layer: nn.Embedding) -> None:
        super(QuantizedEmbedding, self).__init__(num_embeddings=pre_quant_layer.num_embeddings,
                                                 embedding_dim=pre_quant_layer.embedding_dim,
                                                 padding_idx=pre_quant_layer.padding_idx,
                                                 max_norm=pre_quant_layer.max_norm,
                                                 norm_type=pre_quant_layer.norm_type,
                                                 scale_grad_by_freq=pre_quant_layer.scale_grad_by_freq,
                                                 sparse=pre_quant_layer.sparse,
                                                 _weight=pre_quant_layer.weight,
                                                 device=pre_quant_layer.weight.device,
                                                 dtype=pre_quant_layer.weight.dtype)

        assert pre_quant_layer.max_norm == None, 'Not supported'
        assert pre_quant_layer.norm_type == 2, 'Not supported'
        assert pre_quant_layer.scale_grad_by_freq == False, 'Not supported'
        assert pre_quant_layer.sparse == False, 'Not supported'

        self.config = config
        quantizer = Quantizer(config=config)

        self.weight = get_quantized_weight_wrapper(self, pre_quant_layer.weight,
                                                   get_quantize_weight_fn(quantizer, pre_quant_layer.weight))

        self.weight.dequantizer = DeQuantizer(config, pre_quant_layer.weight.dtype)

    def forward(self, input: Tensor) -> Tensor:
        quantized_weight, quant_scale, quant_min = self.weight.deconcat(self.weight)
        temp_dequantized_weight = self.weight.dequantizer.dequantize(quantized_weight.view(torch.uint8), quant_scale,
                                                                     quant_min)

        return F.embedding(input, temp_dequantized_weight, self.padding_idx, self.max_norm, self.norm_type,
                           self.scale_grad_by_freq, self.sparse)


QUANTIZATION_LAYER_MAPPINGS = {
    nn.Linear: QuantizedLinear,
    nn.Embedding: QuantizedEmbedding,
}
