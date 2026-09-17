# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Triton-backed DeepSpeed ops.

A new Triton op added to this folder should reuse the shared flag::

    from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, triton, tl

    if _TRITON_AVAILABLE:

        @triton.jit
        def _my_kernel(...):
            ...
"""

from ._triton import _TRITON_AVAILABLE, is_triton_available, triton, tl
from .group_gemm_triton import group_gemm_triton
from .swiglu_triton import swiglu

__all__ = ["_TRITON_AVAILABLE", "is_triton_available", "triton", "tl", "group_gemm_triton", "swiglu"]
