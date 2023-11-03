# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict

import torch

from enum import Enum, IntEnum


class NormTypeEnum(Enum):
    LayerNorm: str = "layer_norm"
    RMSNorm: str = "rms_norm"


class DtypeEnum(Enum):
    # The torch dtype must always be the first value (so we return torch.dtype)
    fp16 = torch.float16, "torch.float16", "fp16", "float16", "half"
    fp32 = torch.float32, "torch.float32", "fp32", "float32", "float"
    bf16 = torch.bfloat16, "torch.bfloat16", "bf16", "bfloat16", "bfloat"
    int8 = torch.int8, "torch.int8", "int8"

    # Copied from https://stackoverflow.com/a/43210118
    # Allows us to use multiple values for each Enum index and returns first
    # listed value when Enum is called
    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return "<%s.%s: %s>" % (
            self.__class__.__name__,
            self._name_,
            ", ".join([repr(v) for v in self._all_values]),
        )


ELEM_SIZES: Dict[torch.dtype, int] = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
}


class ActivationType(IntEnum):
    """
    Types of activations supported by DS-Inference
    """

    GELU = 0

    RELU = 1

    SILU = 2

    GEGLU = 3

    ReGLU = 4

    SiGLU = 5

    IDENTITY = 6

    InvalidType = -1


def is_gated(act_fn: ActivationType) -> bool:
    """
    Return True if the given activation function is gated.
    """
    if not isinstance(act_fn, ActivationType):
        act_fn = ActivationType(act_fn)

    return act_fn in [ActivationType.GEGLU, ActivationType.ReGLU, ActivationType.SiGLU]


def elem_size(dtype: torch.dtype) -> int:
    """
    Return size in bytes of the given dtype.
    """
    try:
        return ELEM_SIZES[dtype]
    except KeyError:
        raise ValueError("Unknown dtype size for {}".format(dtype))


def ceil_div(a: int, b: int) -> int:
    """
    Return ceil(a / b).
    """
    return -(-a // b)
