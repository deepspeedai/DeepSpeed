# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class PadTransformOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig = None):
        if config is None:
            config = DeepSpeedInferenceConfig()
        super(PadTransformOp, self).__init__(config)
        try:
            self.pad_transform_func = self.inference_module.pad_transform_fp16
        except AttributeError:
            self.pad_transform_func = self.pad_transform_fallback

    @staticmethod
    def pad_transform_fallback(query, key, value, heads, do_flash_attn):
        raise NotImplementedError("pad_transform fallback is not implemented.")

    def forward(self, query, key, value, heads, do_flash_attn):
        return self.pad_transform_func(query, key, value, heads, do_flash_attn)
