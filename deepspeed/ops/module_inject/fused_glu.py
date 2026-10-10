# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Runtime loader for the segment-KI fused_glu native CUDA kernel.

The builder lives in op_builder/ so DS_BUILD_FUSED_GLU=1 installs pick it
up; this module only exposes the lazily JIT-loaded op to callers."""

from deepspeed.ops.op_builder import FusedGLUBuilder

_FUSED_GLU_OP = None


def get_fused_glu_op():
    """Lazily JIT-build and load the fused_glu CUDA op (cached process-wide)."""
    global _FUSED_GLU_OP
    if _FUSED_GLU_OP is None:
        _FUSED_GLU_OP = FusedGLUBuilder().load()
    return _FUSED_GLU_OP
