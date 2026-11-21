# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder


class InferenceBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER_INFERENCE"
    NAME = "transformer_inference"
    MIN_COMPUTE_CAPABILITY = 6

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.inference.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/transformer/inference/csrc/pt_binding.cpp',
            'csrc/transformer/inference/csrc/gelu.cu',
            'csrc/transformer/inference/csrc/relu.cu',
            'csrc/transformer/inference/csrc/layer_norm.cu',
            'csrc/transformer/inference/csrc/rms_norm.cu',
            'csrc/transformer/inference/csrc/softmax.cu',
            'csrc/transformer/inference/csrc/dequantize.cu',
            'csrc/transformer/inference/csrc/apply_rotary_pos_emb.cu',
            'csrc/transformer/inference/csrc/transform.cu',
            'csrc/transformer/inference/csrc/pointwise_ops.cu',
        ]

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            return ['-lcurand']
        else:
            return []

    def include_paths(self):
        return ['csrc/transformer/inference/includes', 'csrc/includes']

    def nvcc_args(self):
        args = super().nvcc_args()
        """BF16 is supported on AMD, but including `cuda_bf16.h` (`<hip/hip_bfloat16.h>` after hipification)
           in host-only translation units (*.cpp files) fails because GPU-specific builtins are pulled in with the BF16 type.
           This cannot be avoided via forward declarations for this transformer_inference extension,
           since `pt_binding.cpp` code explicitly requires the BF16 header, so disable it for now.
        """
        if self.is_rocm_pytorch():
            self.enable_bf16 = False
        return args
