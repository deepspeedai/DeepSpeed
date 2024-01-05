# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import SYCLOpBuilder


class CPUAdamBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        if self.build_for_cpu:
            return ['csrc/xpu/adam/cpu_adam.cpp', 'csrc/xpu/adam/cpu_adam_impl.cpp']

        return [
            'csrc/xpu/adam/cpu_adam.cpp', 'csrc/xpu/adam/cpu_adam_impl.cpp',
            'csrc/xpu/common/custom_cuda_kernel.dp.cpp'
        ]

    def include_paths(self):
        return ['csrc/xpu/includes']
