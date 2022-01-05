"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import sys
import subprocess
from .builder import TorchCPUOpBuilder


class CPUAdamBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/cpu_adam.cpp', 'csrc/common/custom_cuda_kernel.cu']

    def include_paths(self):
        import torch
        CUDA_INCLUDE = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
        return ['csrc/includes', CUDA_INCLUDE]
