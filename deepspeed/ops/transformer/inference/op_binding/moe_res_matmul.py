# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class MoEResMatmulOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig = None):
        if config is None:
            config = DeepSpeedInferenceConfig()
        super(MoEResMatmulOp, self).__init__(config)
        try:
            self.moe_res_matmul_func = self.inference_module.moe_res_matmul
        except AttributeError:
            self.moe_res_matmul_func = self.moe_res_matmul_fallback

    @classmethod
    def moe_res_matmul_fallback(cls, residual, coef, output):
        coef_t = coef.transpose(1, 2).contiguous()
        coef1, coef2 = torch.split(coef_t, split_size_or_sections=coef_t.shape[len(coef_t.shape) - 1] // 2, dim=-1)
        return residual * coef1 + output * coef2

    def forward(self, residual, coef, output):
        return self.moe_res_matmul_func(residual, coef, output)
