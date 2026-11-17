# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder


class FPQuantizerBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FP_QUANTIZER"
    NAME = "fp_quantizer"
    MIN_COMPUTE_CAPABILITY = 8

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.fp_quantizer.{self.NAME}_op'

    def sources(self):
        return [
            "csrc/fp_quantizer/fp_quantize_impl.cu",
            "csrc/fp_quantizer/fp_quantize.cpp",
        ]

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []

    def include_paths(self):
        return ['csrc/fp_quantizer/includes', 'csrc/includes']

    @staticmethod
    def get_default_quant_dtype():
        import torch
        return torch.uint8

    @staticmethod
    def get_quant_range(q_bits=None):
        if q_bits == 8:
            return 480
        elif q_bits == 6:
            return 28.
        elif q_bits == 12:
            return 510.
        else:
            assert (0), \
                "Please specify the right quantization range for the selected precision!"
