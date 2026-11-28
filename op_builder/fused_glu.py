# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from .builder import CUDAOpBuilder


class FusedGLUBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_GLU"
    NAME = "fused_glu"
    MIN_COMPUTE_CAPABILITY = 8  # The kernels use BF16 WMMA and cp.async.
    ROCM_SUPPORTED = False  # The kernels use CUDA-only mma.h/cuda_pipeline.h and 32-lane warp assumptions.

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/module_inject/fused_glu.cu']
