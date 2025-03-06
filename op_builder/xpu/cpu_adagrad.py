# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import SYCLOpBuilder


class CPUAdagradBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        return ['csrc/xpu/adagrad/cpu_adagrad.cpp', 'csrc/xpu/common/custom_cuda_kernel.dp.cpp']

    def include_paths(self):
        return ['csrc/xpu/includes']
