"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import torch
import subprocess
from .builder import CUDAOpBuilder


class CPUAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/custom_cuda_kernel.cu']

    def include_paths(self):
        CUDA_INCLUDE = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
        return ['csrc/includes', CUDA_INCLUDE]

    def simd_width(self):
        if not self.command_exists('lscpu'):
            self.warning(
                "CPUAdam attempted to query 'lscpu' to detect the existence "
                "of AVX instructions. However, 'lscpu' does not appear to exist on "
                "your system, will fall back to non-vectorized execution.")
            return ''

        result = subprocess.check_output('lscpu', shell=True)
        result = result.decode('utf-8').strip().lower()
        if 'genuineintel' in result:
            if 'avx512' in result:
                return '-D__AVX512__'
            elif 'avx2' in result:
                return '-D__AVX256__'
        return '-D__SCALAR__'

    def cxx_args(self):
        CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        SIMD_WIDTH = self.simd_width()

        return [
            '-O3',
            '-std=c++14',
            f'-L{CUDA_LIB64}',
            '-lcudart',
            '-lcublas',
            '-g',
            '-Wno-reorder',
            '-march=native',
            '-fopenmp',
            SIMD_WIDTH
        ]

    def nvcc_args(self):
        args = [
            '-O3',
            '--use_fast_math',
            '-std=c++14',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__'
        ]
        args += self.compute_capability_args()
        return args
