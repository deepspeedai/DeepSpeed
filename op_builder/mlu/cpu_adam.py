# Copyright (c) Microsoft Corporation.
# Copyright (c) 2024 Cambricon Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import MLUOpBuilder


class CPUAdamBuilder(MLUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        return ['csrc/includes']
