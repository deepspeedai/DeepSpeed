# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""op_builder module for the segment-KI fused_glu native CUDA kernel."""

from ..op_builder.builder import CUDAOpBuilder


class FusedGLUBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_GLU"
    NAME = "fused_glu"

    def __init__(self, name=None):
        super().__init__(name=self.NAME if name is None else name)

    def absolute_name(self):
        return f"deepspeed.ops.{self.NAME}_op"

    def sources(self):
        return ["csrc/module_inject/fused_glu.cu"]


_FUSED_GLU_OP = None


def get_fused_glu_op():
    """Lazily JIT-build and load the fused_glu CUDA op (cached process-wide)."""
    global _FUSED_GLU_OP
    if _FUSED_GLU_OP is None:
        _FUSED_GLU_OP = FusedGLUBuilder().load()
    return _FUSED_GLU_OP
