# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from .builder import SYCLOpBuilder


class FusedAdamBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/xpu/adam/fused_adam_frontend.cpp', 'csrc/xpu/adam/multi_tensor_adam.dp.cpp']

    def include_paths(self):
        return ['csrc/xpu/includes', 'csrc/xpu/adam']

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()
