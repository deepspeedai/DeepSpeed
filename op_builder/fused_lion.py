# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder

import sys


class FusedLionBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_LION"
    NAME = "fused_lion"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.lion.{self.NAME}_op'

    def sources(self):
        return ['csrc/lion/fused_lion_frontend.cpp', 'csrc/lion/multi_tensor_lion.cu']

    def include_paths(self):
        return ['csrc/includes', 'csrc/lion']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags.extend(
                ['-allow-unsupported-compiler' if sys.platform == "win32" else '', '-lineinfo', '--use_fast_math'] +
                self.compute_capability_args())
        return nvcc_flags
