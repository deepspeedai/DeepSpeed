# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Shared Triton availability check for the ``triton_ops`` package.
"""

try:
    import triton
    import triton.language as tl
    triton.runtime.driver.active.get_current_target()
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """Return True if Triton is importable and its kernels can be used."""
    return _TRITON_AVAILABLE
