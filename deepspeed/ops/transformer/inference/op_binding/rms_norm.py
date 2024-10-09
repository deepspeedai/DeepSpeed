# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class RMSNormOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig = None):
        if config is None:
            config = DeepSpeedInferenceConfig()
        super(RMSNormOp, self).__init__(config)
        try:
            self.rms_norm_func = self.inference_module.rms_norm
        except AttributeError:
            self.rms_norm_func = self.rms_norm_fallback

    @staticmethod
    def rms_norm_fallback(vals, gamma, epsilon):
        variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
        vals = vals * torch.rsqrt(variance + epsilon)

        if gamma.dtype in [torch.float16, torch.bfloat16]:
            vals = vals.to(gamma.dtype)

        return gamma * vals

    def forward(self, vals, gamma, epsilon):
        return self.rms_norm_func(vals, gamma, epsilon)
