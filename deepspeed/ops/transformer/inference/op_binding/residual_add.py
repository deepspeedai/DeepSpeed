# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Optional

from .vector_add import VectorAddOp
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class ResidualAddOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig):
        super(ResidualAddOp, self).__init__(config)
        try:
            if self.config.dtype in [torch.float16, torch.int8]:
                self.residual_add_func = self.inference_module.residual_add_bias_fp16
            elif self.config.dtype == torch.bfloat16:
                self.residual_add_func = self.inference_module.residual_add_bias_bf16
            else:
                self.residual_add_func = self.inference_module.residual_add_bias_fp32
        except AttributeError:
            self.residual_add_func = self.residual_add_fallback
        self.vector_add = VectorAddOp()

    @staticmethod
    def res_add_bias(hidden_state, residual, attn_output, attn_bias, final_bias, add_attn_bias, mp_size):
        hidden_state += attn_output + (residual + final_bias) / mp_size
        if add_attn_bias:
            hidden_state += attn_bias / mp_size

        return hidden_state

    @staticmethod
    def residual_add_fallback(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size,
                              mlp_after_attn, add_bias, pre_layer_norm):
        if mlp_after_attn:
            if pre_layer_norm:
                tmp = (residual.float() + attention_output.float() + attention_bias.float() +
                       final_bias.float()) / mp_size + hidden_state.float()
            else:
                tmp = residual.float() + hidden_state.float() + final_bias.float()
        else:
            tmp = ResidualAddOp.res_add_bias(hidden_state, residual, attention_output, attention_bias, final_bias,
                                             add_bias, mp_size)
        residual.copy_(tmp.to(hidden_state.dtype))

        return residual

    def forward(self,
                hidden_state: torch.Tensor,
                residual: torch.Tensor,
                add_bias: bool,
                attention_output: Optional[torch.Tensor] = None,
                residual_add: Optional[torch.Tensor] = None,
                attention_bias: Optional[torch.Tensor] = None,
                final_bias: Optional[torch.Tensor] = None):

        if final_bias is None and attention_bias is None:
            residual = self.vector_add(residual + attention_output, hidden_state, 1.0 / self.config.mp_size)
        else:
            if not self.config.pre_layer_norm and residual_add is not None:
                # only use residual add if its set and we are not pre layer norm
                residual = residual_add

            self.residual_add_func(hidden_state, residual, attention_output, attention_bias, final_bias,
                                   self.config.mp_size, self.config.mlp_after_attn, add_bias,
                                   self.config.pre_layer_norm)

        return residual
