# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import TorchCPUOpBuilder


class DeepCompileBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_DEEP_COMPILE"
    NAME = "dc"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/compile/deepcompile.cpp', 'csrc/compile/init.cpp', 'csrc/compile/z1.cpp', 'csrc/compile/z2.cpp',
            'csrc/compile/z3.cpp', 'csrc/compile/util.cpp'
        ]

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        import os
        import torch
        if self.build_for_cpu:
            CUDA_INCLUDE = []
        elif not self.is_rocm_pytorch():
            CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        else:
            CUDA_INCLUDE = [
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include"),
            ]
        return ['csrc/includes', 'csrc/compile'] + CUDA_INCLUDE
