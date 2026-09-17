# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""op_builder module for the C++ decode loop (graph replay + step update)."""

from ..op_builder.builder import CUDAOpBuilder


class DecodeLoopBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_DECODE_LOOP"
    NAME = "decode_loop"

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)

    def absolute_name(self):
        return f"deepspeed.ops.{self.NAME}_op"

    def sources(self):
        return ["csrc/module_inject/decode_loop.cu"]


_DECODE_LOOP_OP = None


def get_decode_loop_op():
    """Lazily JIT-build and load the decode_loop CUDA op (cached process-wide)."""
    global _DECODE_LOOP_OP
    if _DECODE_LOOP_OP is None:
        _DECODE_LOOP_OP = DecodeLoopBuilder().load()
    return _DECODE_LOOP_OP
