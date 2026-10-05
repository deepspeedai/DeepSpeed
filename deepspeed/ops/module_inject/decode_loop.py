# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Runtime loader for the C++ decode loop (graph replay + step update).

The builder lives in op_builder/ so DS_BUILD_DECODE_LOOP=1 installs pick
it up; this module only exposes the lazily JIT-loaded op to callers."""

from deepspeed.ops.op_builder import DecodeLoopBuilder

_DECODE_LOOP_OP = None


def get_decode_loop_op():
    """Lazily JIT-build and load the decode_loop CUDA op (cached process-wide)."""
    global _DECODE_LOOP_OP
    if _DECODE_LOOP_OP is None:
        _DECODE_LOOP_OP = DecodeLoopBuilder().load()
    return _DECODE_LOOP_OP
