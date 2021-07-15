"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import sys
import subprocess
from .builder import CUDAOpBuilder


class CPUAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def is_compatible(self):
        # Disable on Windows.
        return sys.platform != "win32"

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/custom_cuda_kernel.cu']

    def include_paths(self):
        import torch
        CUDA_INCLUDE = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
        return ['csrc/includes', CUDA_INCLUDE]

    def cxx_args(self):
        import torch
        CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()

        return [
            '-O3',
            '-std=c++14',
            f'-L{CUDA_LIB64}',
            '-lcudart',
            '-lcublas',
            '-g',
            '-Wno-reorder',
            CPU_ARCH,
            '-fopenmp',
            SIMD_WIDTH,
        ]
