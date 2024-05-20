# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import TorchCPUOpBuilder


class CPULionBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_LION"
    NAME = "cpu_lion"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.lion.{self.NAME}_op'

    def sources(self):
        return ['csrc/lion/cpu_lion.cpp', 'csrc/lion/cpu_lion_impl.cpp']

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        return ['csrc/includes']
