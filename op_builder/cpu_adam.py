import os
import torch
import warnings
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

    def available_vector_instructions(self):
        try:
            import cpufeature
        except ImportError:
            warnings.warn(
                f'import cpufeature failed - CPU vector optimizations are not available for CPUAdam'
            )
            return {}

        cpu_vector_instructions = {}
        try:
            cpu_vector_instructions = cpufeature.CPUFeature
        except _:
            warnings.warn(
                f'cpufeature.CPUFeature failed - CPU vector optimizations are not available for CPUAdam'
            )
            return {}

        return cpu_vector_instructions

    def cxx_args(self):
        CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        cpu_info = self.available_vector_instructions()
        SIMD_WIDTH = ''
        if 'Intel' in cpu_info.get('VendorId', ''):
            if cpu_info.get('AVX512f', False):
                SIMD_WIDTH = '-D__AVX512__'
            elif cpu_info.get('AVX2', False):
                SIMD_WIDTH = '-D__AVX256__'

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
