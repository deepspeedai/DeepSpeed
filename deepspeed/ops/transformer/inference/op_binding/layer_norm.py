# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class LayerNormOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig = None):
        super(LayerNormOp, self).__init__(config)
        try:
            if config is None:
                self.layer_norm_func = self.inference_module.layer_norm
            elif self.config.dtype in [torch.float16, torch.int8]:
                self.layer_norm_func = self.inference_module.layer_norm_fp16
            else:
                self.layer_norm_func = self.inference_module.layer_norm_fp32
        except AttributeError:
            self.layer_norm_func = self.layer_norm_fallback

    @classmethod
    def layer_norm_residual(cls, vals, bias, res, gamma, beta, epsilon):
        channels = gamma.shape[0]
        dtype = gamma.dtype
        vals_f = vals.to(torch.float32)
        bias_f = bias.to(torch.float32).reshape(1, 1, -1)
        res_f = res.to(torch.float32)
        gamma_f = gamma.to(torch.float32)
        beta_f = beta.to(torch.float32)
        return F.layer_norm(vals_f + bias_f + res_f, (channels, ), weight=gamma_f, bias=beta_f, eps=epsilon).to(dtype)

    @classmethod
    def layer_norm_residual_store_pre_ln_res(cls, vals, bias, res, gamma, beta, epsilon):
        channels = gamma.shape[0]
        dtype = gamma.dtype
        vals_f = vals.to(torch.float32)
        bias_f = bias.to(torch.float32).reshape(1, 1, -1)
        res_f = res.to(torch.float32)
        gamma_f = gamma.to(torch.float32)
        beta_f = beta.to(torch.float32)
        res_output = vals_f + bias_f + res_f
        norm_output = F.layer_norm(res_output, (channels, ), weight=gamma_f, bias=beta_f, eps=epsilon).to(dtype)
        return norm_output, res_output.to(dtype)

    @classmethod
    def layer_norm_fallback(cls, vals, gamma, beta, epsilon):
        channels = gamma.shape[0]
        dtype = gamma.dtype
        vals_f = vals.to(torch.float32)
        gamma_f = gamma.to(torch.float32)
        beta_f = beta.to(torch.float32)
        return F.layer_norm(vals_f, (channels, ), weight=gamma_f, bias=beta_f, eps=epsilon).to(dtype)

    def forward(self, vals, gamma, beta, epsilon):
        return self.layer_norm_func(vals, gamma, beta, epsilon)
