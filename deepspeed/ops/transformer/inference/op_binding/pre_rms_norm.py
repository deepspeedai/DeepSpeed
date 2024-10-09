# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp
from .rms_norm import RMSNormOp


class PreRMSNormOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig = None):
        if config is None:
            config = DeepSpeedInferenceConfig()
        super(PreRMSNormOp, self).__init__(config)
        try:
            self.pre_rms_norm_func = self.inference_module.pre_rms_norm
        except AttributeError:
            self.pre_rms_norm_func = self.pre_rms_norm_fallback

    @staticmethod
    def pre_rms_norm_fallback(vals, residual, gamma, epsilon):
        residual = vals.to(torch.float32) + residual.to(torch.float32)
        vals = residual

        return RMSNormOp.rms_norm_fallback(vals, gamma, epsilon), residual.to(gamma.dtype)

    def forward(self, vals, residual, gamma, epsilon):
        return self.pre_rms_norm_func(vals, residual, gamma, epsilon)
