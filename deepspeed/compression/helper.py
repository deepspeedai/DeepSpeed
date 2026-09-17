# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Deprecated compatibility shim.

DeepSpeed-Chat and other callers imported ``recursive_getattr`` /
``recursive_setattr`` from this module. The compression library is gone;
these helpers now live in ``deepspeed.utils.module_utils``.
"""

import warnings

from deepspeed.utils.module_utils import recursive_getattr, recursive_setattr

warnings.warn(
    "deepspeed.compression.helper is deprecated; import recursive_getattr and "
    "recursive_setattr from deepspeed.utils.module_utils instead. "
    "See https://github.com/deepspeedai/DeepSpeed/issues/8489",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["recursive_getattr", "recursive_setattr"]
