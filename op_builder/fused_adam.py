"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import CUDAOpBuilder


class FusedAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/fused_adam_frontend.cpp', 'csrc/adam/multi_tensor_adam.cu']

    def include_paths(self):
        return ['csrc/includes']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        return ['-lineinfo',
                '-O3',
                '--use_fast_math'
                ] + self.version_dependent_macros() + self.compute_capability_args()
