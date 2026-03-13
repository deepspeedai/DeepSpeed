# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any


def ensure_nonzero_divisor(divisor: Any, *, name: str = "divisor") -> None:
    """
    Validate that a divisor is non-zero before modulo/division math.
    """
    if divisor == 0:
        raise ValueError(f"{name} must be non-zero")
