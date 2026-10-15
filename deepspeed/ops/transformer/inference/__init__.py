# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .config import DeepSpeedInferenceConfig
from .moe_inference import DeepSpeedMoEInferenceConfig, DeepSpeedMoEInference

__all__ = [
    "DeepSpeedInferenceConfig",
    "DeepSpeedMoEInferenceConfig",
    "DeepSpeedMoEInference",
    "DeepSpeedTransformerInference",
]


def __getattr__(name: str):
    # Lazy import breaks the circular dependency between this package and
    # `deepspeed.model_implementations.transformers.ds_transformer`, which
    # imports `deepspeed.ops.transformer.inference.triton.*` at module load
    # time when Triton is installed. Accessing the symbol on demand keeps the
    # package import path acyclic.
    if name == "DeepSpeedTransformerInference":
        from ....model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference
        return DeepSpeedTransformerInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
