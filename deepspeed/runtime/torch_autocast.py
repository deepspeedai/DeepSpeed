# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Set, List, Union
import importlib
from contextlib import contextmanager

import torch
import deepspeed.comm as dist
from deepspeed.utils import logger
from deepspeed.accelerator import get_accelerator

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

    assert not engine.zero_quantized_weights(), "Cannot enable both torch autocast and zero quantized weights"


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


@contextmanager
def autocast_if_enabled(engine):
    """Context manager for DeepSpeed autocast with conditional support.

    This function manages `torch.autocast` contexts under DeepSpeed, allowing
    autocast to be enabled or disabled dynamically based on runtime conditions.
    It ensures consistency when autocast is already active outside of DeepSpeed,
    or when it is configured within the DeepSpeed engine.

    Args:
        engine: DeepSpeed engine instance.
    """
    global _WARNED_NESTED_AUTOCAST

    if torch.is_autocast_enabled():
        if engine.torch_autocast_enabled():
            if not _WARNED_NESTED_AUTOCAST:
                if dist.get_rank() == 0:
                    logger.info(
                        "torch.autocast is already enabled outside DeepSpeed. "
                        "Switching to the configuration defined in `torch_autocast` section of DeepSpeed config.")
                _WARNED_NESTED_AUTOCAST = True
            with torch.autocast(device_type=get_accelerator().device_name(),
                                dtype=engine.torch_autocast_dtype(),
                                enabled=True):
                yield
        else:
            if not _WARNED_NESTED_AUTOCAST:
                if dist.get_rank() == 0:
                    logger.warning(
                        "torch.autocast is enabled outside DeepSpeed but disabled within the DeepSpeed engine. "
                        "If you are using DeepSpeed's built-in mixed precision, the engine will follow the settings in bf16/fp16 section. "
                        "To use torch's native autocast instead, configure the `torch_autocast` section in the DeepSpeed config."
                    )
                _WARNED_NESTED_AUTOCAST = True
            with torch.autocast(device_type=get_accelerator().device_name(), enabled=False):
                yield
    else:
        if engine.torch_autocast_enabled():
            with torch.autocast(device_type=get_accelerator().device_name(),
                                dtype=engine.torch_autocast_dtype(),
                                enabled=True):
                yield
        else:
            yield
