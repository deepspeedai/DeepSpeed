# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ..config import DeepSpeedInferenceConfig
from .base import BaseOp


class VectorAddOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig = None):
        if config is None:
            config = DeepSpeedInferenceConfig()
        super(VectorAddOp, self).__init__(config)
        try:
            self.vector_add_func = self.inference_module._vector_add
        except AttributeError:
            self.vector_add_func = self.vector_add_fallback

    @classmethod
    def vector_add_fallback(cls, a, b, gamma):
        """Based on csrc/transformer/inference/csrc/pt_binding.cpp code of _vector_add"""
        dtype = a.dtype
        return (gamma * a.float() + b.float()).to(dtype)

    def forward(self, a, b, gamma):
        return self.vector_add_func(a, b, gamma)
