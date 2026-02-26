# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from importlib import import_module
from typing import TYPE_CHECKING

from .transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from .inference.config import DeepSpeedInferenceConfig
from ...model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference

if TYPE_CHECKING:
    from .inference.moe_inference import DeepSpeedMoEInference, DeepSpeedMoEInferenceConfig

_MOE_LAZY_IMPORTS = {
    "DeepSpeedMoEInferenceConfig": ".inference.moe_inference",
    "DeepSpeedMoEInference": ".inference.moe_inference",
}

__all__ = [
    "DeepSpeedTransformerLayer",
    "DeepSpeedTransformerConfig",
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
