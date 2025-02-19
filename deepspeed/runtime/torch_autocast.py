# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Set

import torch

LOWER_PRECISION_SAFE_MODULES = [
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
]

TORCH_AUTOCAST_INITIALIZED = False


def _validate_auto_cast_settings(engine):

    assert not engine.fp16_enabled(), "Cannot enable both torch autocast and fp16"
    assert not engine.bfloat16_enabled(), "Cannot enable both torch autocast and bfloat16"

    assert all(p.dtype == torch.float32
               for p in engine.parameters()), "All parameters must be float32 for torch autocast"
    assert engine.communication_data_type == torch.float32, "Communication data type must be float32 for torch autocast"


def init_autocast_params(engine, dtype: torch.dtype) -> None:

    _validate_auto_cast_settings(engine)
    model = engine.module

    for module in model.modules():
        if module.__class__ in LOWER_PRECISION_SAFE_MODULES:
            for p in module.parameters(recurse=False):
                p.autocast_dtype = dtype

    global TORCH_AUTOCAST_INITIALIZED
    TORCH_AUTOCAST_INITIALIZED = True


def is_autocast_initialized() -> bool:
    return TORCH_AUTOCAST_INITIALIZED


def get_autocast_dtype(param: torch.nn.Parameter) -> torch.dtype:
    return param.autocast_dtype if hasattr(param, "autocast_dtype") else param.dtype


def get_all_autocast_dtypes(params: Iterable) -> Set[torch.dtype]:
    return {get_autocast_dtype(p) for p in params}
