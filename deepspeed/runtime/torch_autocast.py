# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Set, List, Union
import importlib

import torch
from deepspeed.utils import logger

LOWER_PRECISION_SAFE_MODULES = [
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
]

PARAM_COMM_DTYPE_ATTR_NAME = "comm_dtype"
_WARNED_NESTED_AUTOCAST = False

# TODO: Avoid using global variables
TORCH_AUTOCAST_INITIALIZED = False
TORCH_AUTOCAST_DTYPE = None


def _validate_auto_cast_settings(engine):

    assert not engine.fp16_enabled(), "Cannot enable both torch autocast and fp16"
    assert not engine.bfloat16_enabled(), "Cannot enable both torch autocast and bfloat16"
    assert not engine.zero_quantized_weights(), "Cannot enable both torch autocast and zero quantized weights"

    assert all(p.dtype == torch.float32
               for p in engine.parameters()), "All parameters must be float32 for torch autocast"
    assert engine.communication_data_type == torch.float32, "Communication data type must be float32 for torch autocast"


def init_autocast_params(engine, dtype: torch.dtype,
                         torch_autocast_lower_precision_safe_modules: Union[None, List[str]]) -> None:

    _validate_auto_cast_settings(engine)
    model = engine.module

    if torch_autocast_lower_precision_safe_modules is None:
        lower_precision_safe_module_classes = LOWER_PRECISION_SAFE_MODULES
    else:
        lower_precision_safe_module_classes = []
        for module_name in torch_autocast_lower_precision_safe_modules:
            try:
                package_name, class_name = module_name.rsplit('.', 1)
                module = importlib.import_module(package_name)
                class_ = getattr(module, class_name)
                lower_precision_safe_module_classes.append(class_)
            except Exception as e:
                raise ValueError(f"Failed to import lower precision safe module {module_name}: {e}")

    for module in model.modules():
        if module.__class__ in lower_precision_safe_module_classes:
            for p in module.parameters(recurse=False):
                setattr(p, PARAM_COMM_DTYPE_ATTR_NAME, dtype)

    global TORCH_AUTOCAST_INITIALIZED
    TORCH_AUTOCAST_INITIALIZED = True
    global TORCH_AUTOCAST_DTYPE
    TORCH_AUTOCAST_DTYPE = dtype


def is_autocast_initialized() -> bool:
    return TORCH_AUTOCAST_INITIALIZED


def get_default_autocast_lower_precision_modules() -> List[str]:
    return [f"{cls.__module__}.{cls.__name__}" for cls in LOWER_PRECISION_SAFE_MODULES]


def get_autocast_dtype() -> torch.dtype:
    return TORCH_AUTOCAST_DTYPE


def has_comm_dtype(param: torch.nn.Parameter) -> bool:
    return hasattr(param, PARAM_COMM_DTYPE_ATTR_NAME)


def get_comm_dtype(param: torch.nn.Parameter) -> torch.dtype:
    return getattr(param, PARAM_COMM_DTYPE_ATTR_NAME, param.dtype)


def get_all_comm_dtypes(params: Iterable) -> Set[torch.dtype]:
    return {get_comm_dtype(p) for p in params}


def sort_dtypes(dtypes: List[torch.dtype]) -> List[torch.dtype]:
    return sorted(dtypes, key=str)


def validate_nested_autocast(engine):
    global _WARNED_NESTED_AUTOCAST

    if torch.is_autocast_enabled():
        if engine.torch_autocast_enabled():
            if not _WARNED_NESTED_AUTOCAST:
                logger.warning(
                    "DeepSpeed detected torch.autocast context outside the engine. "
                    "This is unnecessary when torch.autocast is already enabled through the DeepSpeed config.")
                _WARNED_NESTED_AUTOCAST = True
        else:
            raise AssertionError(
                "torch.autocast is enabled outside DeepSpeed, but not in the DeepSpeed config. "
                "Please enable torch.autocast through the DeepSpeed config to ensure the correct communication dtype is used."
            )
