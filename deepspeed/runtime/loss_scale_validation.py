# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from numbers import Integral, Real


def _to_finite_float(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite, got {numeric_value}")
    return numeric_value


def validate_loss_scale_value(value, *, name: str = "loss_scale", allow_dynamic_zero: bool = False) -> float:
    """
    Validate static loss scale values.

    A value of 0 is accepted only when it represents dynamic loss scaling mode.
    """
    numeric_value = _to_finite_float(value, name=name)
    if allow_dynamic_zero and numeric_value == 0.0:
        return numeric_value
    if numeric_value <= 0.0:
        raise ValueError(f"{name} must be greater than 0, got {numeric_value}")
    return numeric_value


def validate_positive_finite(value, *, name: str) -> float:
    numeric_value = _to_finite_float(value, name=name)
    if numeric_value <= 0.0:
        raise ValueError(f"{name} must be greater than 0, got {numeric_value}")
    return numeric_value


def validate_positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f"{name} must be greater than 0, got {int_value}")
    return int_value
