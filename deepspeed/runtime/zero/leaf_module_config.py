# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List
from pydantic import Field, model_validator

from deepspeed.runtime.config_utils import DeepSpeedConfigModel

DEFAULT_LEAF_MODULE_CLASSES: List[str] = [
    "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock",
    "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock",
    "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
]
DEFAULT_LEAF_MODULE_NAMES: List[str] = []
DEFAULT_LEAF_MODULE_NAME_SUFFIXES: List[str] = []


class DeepSpeedZeroLeafModuleConfig(DeepSpeedConfigModel):
    """Configuration for ZeRO leaf modules that should bypass hook installation."""

    classes: List[str] = Field(default_factory=lambda: list(DEFAULT_LEAF_MODULE_CLASSES))
    names: List[str] = Field(default_factory=lambda: list(DEFAULT_LEAF_MODULE_NAMES))
    name_suffixes: List[str] = Field(default_factory=lambda: list(DEFAULT_LEAF_MODULE_NAME_SUFFIXES))

    @model_validator(mode="before")
    def _coerce_container_types(cls, values):
        if values is None:
            return {}
        if isinstance(values, dict):
            coerced = dict(values)
            for key in ("classes", "names", "name_suffixes"):
                if key in coerced and isinstance(coerced[key], str):
                    coerced[key] = [coerced[key]]
            return coerced
        raise TypeError("leaf_module configuration must be a mapping of fields to values")

    @model_validator(mode="after")
    def _validate_entries(self):
        normalized_classes = [str(cls) for cls in self.classes]
        normalized_names = [str(name) for name in self.names]
        normalized_suffixes = [str(suffix) for suffix in self.name_suffixes]

        deduped_classes = list(dict.fromkeys(normalized_classes))
        deduped_names = list(dict.fromkeys(normalized_names))
        deduped_suffixes = list(dict.fromkeys(normalized_suffixes))

        object.__setattr__(self, "classes", deduped_classes)
        object.__setattr__(self, "names", deduped_names)
        object.__setattr__(self, "name_suffixes", deduped_suffixes)
        return self
