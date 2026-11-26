# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# The names are loaded on first access (see __getattr__). Listing them in __all__ keeps them in
# "from deepspeed.inference import *", which otherwise only exports names that are already loaded.
__all__ = [
    "RaggedInferenceEngineConfig", "DeepSpeedTPConfig", "InferenceEngineV2", "build_hf_engine",
    "build_engine_from_ds_checkpoint"
]


def __getattr__(name):
    # These names come from FastGen (inference v2), which loads ~175 modules that import deepspeed does not need, so
    # v2 is imported on first access. from deepspeed.inference import build_hf_engine still works.
    if name in __all__:
        from . import v2
        globals()[name] = getattr(v2, name)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
