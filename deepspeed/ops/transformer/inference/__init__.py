# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from importlib import import_module
from typing import TYPE_CHECKING

from .config import DeepSpeedInferenceConfig
from ....model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference

if TYPE_CHECKING:
    from .moe_inference import DeepSpeedMoEInference, DeepSpeedMoEInferenceConfig

_MOE_LAZY_IMPORTS = {
    "DeepSpeedMoEInferenceConfig": ".moe_inference",
    "DeepSpeedMoEInference": ".moe_inference",
}

__all__ = [
    "DeepSpeedInferenceConfig",
    "DeepSpeedTransformerInference",
    *_MOE_LAZY_IMPORTS,
]


def __getattr__(name):
    module_path = _MOE_LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
