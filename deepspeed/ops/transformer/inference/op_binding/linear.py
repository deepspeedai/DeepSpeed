# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class LinearOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(LinearOp, self).__init__(config)
        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.linear_func = self.inference_module.linear_layer_fp16
            elif self.config.dtype == torch.bfloat16:
                self.linear_func = self.inference_module.linear_layer_bf16
            else:
                self.linear_func = self.inference_module.linear_layer_fp32
        except AttributeError:
            self.linear_func = self.linear_fallback

    def linear_fallback(self, input, weight, bias, add_bias, do_flash_attn, num_heads, transpose):
        raise NotImplementedError

    def forward(self,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor,
                add_bias: bool,
                do_flash_attn: bool,
                num_heads: int,
                external_cache: bool = None,
                num_layers: int = None):
        qkv_out = self.linear_func(input, weight, bias, add_bias, do_flash_attn, num_heads,
                                   self.config.transposed_mode)
        return qkv_out
