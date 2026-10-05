# Copyright (c) Microsoft Corporation
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder


class DecodeLoopBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_DECODE_LOOP"
    NAME = "decode_loop"

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)

    def absolute_name(self):
        return f'deepspeed.ops.{self.NAME}_op'

    def sources(self):
        return ['csrc/module_inject/decode_loop.cu']
