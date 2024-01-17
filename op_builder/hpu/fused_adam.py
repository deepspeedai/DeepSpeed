# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CPUOpBuilder


class FusedAdamBuilder(CPUOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/cpu/adam/fused_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']

    def cxx_args(self):
        args = super().cxx_args()
        args += ['-DENABLE_BFLOAT16']
        return args

    def include_paths(self):
        return ['csrc/includes']
