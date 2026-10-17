# Copyright (c) Microsoft Corporation
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder


class FusedGLUBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_GLU"
    NAME = "fused_glu"

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/module_inject/fused_glu.cu']
