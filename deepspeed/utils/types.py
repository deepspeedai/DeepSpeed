# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import IntEnum


class ActivationFuncType(IntEnum):
    UNKNOWN = 0
    GELU = 1
    ReLU = 2
    GATED_GELU = 3
    GATED_SILU = 4


GATED_ACTIVATION_TYPES = [
    ActivationFuncType.GATED_GELU,
    ActivationFuncType.GATED_SILU,
]


class NormType(IntEnum):
    UNKNOWN = 0
    LayerNorm = 1
    GroupNorm = 2
    RMSNorm = 3
