# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Shared Triton availability check for the ``triton_ops`` package.
"""

try:
    import triton
    import triton.language as tl
    from triton.backends import backends

    # Same test triton.runtime.driver.active uses to pick a backend, without get_current_target(), which initializes
    # CUDA and breaks GPU use in fork()-ed children. For CUDA, is_active() asks torch whether a GPU is available, and
    # torch answers from NVML (deepspeed sets PYTORCH_NVML_BASED_CUDA_CHECK=1), so CPU-only machines get False.
    _TRITON_AVAILABLE = any(backend.driver.is_active() for backend in backends.values())
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """Return True if Triton is importable and its kernels can be used."""
    return _TRITON_AVAILABLE
