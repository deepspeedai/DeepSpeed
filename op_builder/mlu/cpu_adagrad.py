# Copyright (c) Microsoft Corporation.
# Copyright (c) 2024 Cambricon Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import MLUOpBuilder


class CPUAdagradBuilder(MLUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        return ['csrc/adagrad/cpu_adagrad.cpp']

    def include_paths(self):
        return ['csrc/includes']
