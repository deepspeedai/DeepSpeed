# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""DeepSpeed integration for torch_npu HiFloat8 training."""

from .hifloat8_linear import (
    assert_hifloat8_training_available,
    convert_to_hifloat8_training,
    get_hifloat8_linear_class,
    is_hifloat8_available,
)

__all__ = [
    "assert_hifloat8_training_available",
    "convert_to_hifloat8_training",
    "get_hifloat8_linear_class",
    "is_hifloat8_available",
]
