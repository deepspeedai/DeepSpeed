# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
from typing import Any, List, Literal

from pydantic import Field, field_validator, model_validator

from deepspeed.runtime.config import DeepSpeedConfigError
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

QUANTIZATION = "quantization"

SUBCHANNEL_BLOCK_SIZES = (32, 64, 128, 256)


class GemmQuantizationConfig(DeepSpeedConfigModel):
    """How one of the three GEMMs of a linear layer (forward, input gradient, weight gradient) is quantized.

    Both operands are quantized along the GEMM's reduction dimension in blocks of ``block_size`` elements,
    each block with its own FP32 scale; ``block_size`` 0 gives every operand row a single scale.
    """

    format: Literal["e4m3", "e5m2", "int8", "none"] = "e4m3"
    """``e4m3``/``e5m2`` (FP8), ``int8``, or ``none`` to run this GEMM unquantized in the compute dtype."""

    block_size: int = 128
    """Elements of the reduction dimension that share one scale: 32, 64, 128 or 256, or 0 for the whole
    reduction dimension (per AutoEP expert in the weight-gradient GEMM, which reduces over tokens).

    Scales that change inside the reduction have to be applied between tensor-core dot products, which holds a
    block-scaled GEMM below BF16 speed. With 0 the scales are applied once per output element and the GEMM
    itself runs faster than BF16. 0 requires an FP8 format and no Hadamard rotation."""

    hadamard_size: int = 0
    """Rotate every ``hadamard_size`` chunk of each block by an orthonormal Hadamard matrix before quantizing,
    spreading outliers over the chunk. The rotations of the two operands cancel in the product. A power of two
    no larger than ``block_size``; 0 disables it."""

    stochastic_rounding: bool = False
    """Round to a neighbouring grid point at random, with probabilities that make the rounding unbiased."""

    @field_validator("block_size")
    @classmethod
    def _check_block_size(cls, value):
        if value != 0 and value not in SUBCHANNEL_BLOCK_SIZES:
            raise ValueError(f"block_size must be 0 or one of {SUBCHANNEL_BLOCK_SIZES}, got {value}")
        return value

    @model_validator(mode="after")
    def _check_row_scaling(self):
        if self.block_size != 0 or not self.quantized:
            return self
        if self.format == "int8":
            raise ValueError("block_size 0 (one scale per row) supports the FP8 formats only; INT8 needs a "
                             f"block_size from {SUBCHANNEL_BLOCK_SIZES}")
        if self.hadamard_size:
            raise ValueError("hadamard_size requires a block_size to rotate within; it cannot be combined with "
                             "block_size 0")
        return self

    @model_validator(mode="after")
    def _check_hadamard_size(self):
        size = self.hadamard_size
        if size == 0 or self.block_size == 0:
            return self
        if size < 2 or size & (size - 1) or size > self.block_size:
            raise ValueError(f"hadamard_size must be 0 or a power of two between 2 and block_size "
                             f"({self.block_size}), got {size}")
        return self

    @property
    def quantized(self) -> bool:
        return self.format != "none"


def _gradient_gemm_default():
    return GemmQuantizationConfig(stochastic_rounding=True)


class SubchannelQuantizedMatmulConfig(DeepSpeedConfigModel):
    """Subchannel (block-scaled) FP8/INT8 training for ``nn.Linear`` layers and AutoEP grouped experts.

    The forward GEMM, the input-gradient GEMM (``dx``) and the weight-gradient GEMM (``dw``) are configured
    independently. Parameters, optimizer state and module outputs keep their usual precision; only the GEMM
    operands are quantized, on the fly, at each call.
    """

    enabled: bool = False

    modules: List[str] = Field(default_factory=lambda: ["*"])
    """``fnmatch`` patterns over qualified module names. ``nn.Linear`` modules and AutoEP grouped experts
    whose names match are converted."""

    exclude_modules: List[str] = Field(default_factory=lambda: ["*lm_head*", "*router*", "*.gate"])
    """``fnmatch`` patterns of modules to leave unconverted even when ``modules`` matches them. The defaults keep
    the output head and MoE routers (AutoEP's ``router.gate``, Hugging Face's ``mlp.gate``) in full precision,
    since quantizing router logits changes which experts tokens reach."""

    forward: GemmQuantizationConfig = Field(default_factory=GemmQuantizationConfig)
    dx: GemmQuantizationConfig = Field(default_factory=_gradient_gemm_default)
    dw: GemmQuantizationConfig = Field(default_factory=_gradient_gemm_default)


HIFLOAT8 = "hifloat8"
HIFLOAT8_DEFAULT = {
    "enabled": False,
    "backend": "torchao",
    "module_name_patterns": (),
    "min_numel": 0,
    "expected_module_count": None,
    "config": None,
    "module_configs": None,
}


def _parse_hifloat8_config(config):
    if not isinstance(config, dict):
        raise DeepSpeedConfigError(f"'{HIFLOAT8}' must be an object, got {type(config).__name__}")

    unknown = sorted(set(config) - set(HIFLOAT8_DEFAULT))
    if unknown:
        raise DeepSpeedConfigError(f"Unknown '{HIFLOAT8}' fields: {unknown}")

    result = dict(HIFLOAT8_DEFAULT)
    result.update(config)
    if not isinstance(result["enabled"], bool):
        raise DeepSpeedConfigError("'hifloat8.enabled' must be a boolean")
    if result["backend"] == "torchao_npu":
        result["backend"] = "torchao"
    if result["backend"] != "torchao":
        raise DeepSpeedConfigError("'hifloat8.backend' must be 'torchao'; the legacy torch_npu helper was removed")
    policy = result["config"]
    if policy is not None:
        if not isinstance(policy, dict):
            raise DeepSpeedConfigError("'hifloat8.config' must be an object or null")
    module_configs = result["module_configs"]
    if module_configs is not None:
        if not isinstance(module_configs, dict):
            raise DeepSpeedConfigError("'hifloat8.module_configs' must be an object or null")
        for selector, recipe in module_configs.items():
            if not isinstance(selector, str):
                raise DeepSpeedConfigError("'hifloat8.module_configs' keys must be module names or re: patterns")
            if recipe is not None and not isinstance(recipe, dict):
                raise DeepSpeedConfigError(f"HiFloat8 recipe for {selector!r} must be an object or null")
            if selector.startswith("re:"):
                try:
                    re.compile(selector[3:])
                except re.error as error:
                    raise DeepSpeedConfigError(
                        f"Invalid HiFloat8 module_configs regex {selector!r}: {error}") from error
    patterns = result["module_name_patterns"]
    if not isinstance(patterns,
                      (list, tuple)) or not all(isinstance(pattern, str) and pattern for pattern in patterns):
        raise DeepSpeedConfigError("'hifloat8.module_name_patterns' must be a list of non-empty strings")
    if result["enabled"] and not patterns and not module_configs:
        raise DeepSpeedConfigError("HiFloat8 requires module_name_patterns or module_configs when enabled")
    min_numel = result["min_numel"]
    if isinstance(min_numel, bool) or not isinstance(min_numel, int) or min_numel < 0:
        raise DeepSpeedConfigError("'hifloat8.min_numel' must be a non-negative integer")
    expected_count = result["expected_module_count"]
    if expected_count is not None and (isinstance(expected_count, bool) or not isinstance(expected_count, int)
                                       or expected_count <= 0):
        raise DeepSpeedConfigError("'hifloat8.expected_module_count' must be a positive integer or null")
    result["module_name_patterns"] = tuple(patterns)
    return result


class DeepSpeedQuantizationConfig(DeepSpeedConfigModel):
    """The ``quantization`` section: low-precision training recipes, one subsection per recipe."""

    subchannel: SubchannelQuantizedMatmulConfig = Field(default_factory=SubchannelQuantizedMatmulConfig)
    hifloat8: dict[str, Any] = Field(default_factory=dict)

    @field_validator("hifloat8", mode="before")
    @classmethod
    def _validate_hifloat8(cls, value):
        return _parse_hifloat8_config(value)

    @model_validator(mode="after")
    def _validate_recipe_combination(self):
        if self.subchannel.enabled and self.hifloat8["enabled"]:
            raise DeepSpeedConfigError("Enable only one of quantization.subchannel and quantization.hifloat8")
        return self


def get_quantization_config(param_dict):
    if "low_precision" in param_dict:
        raise DeepSpeedConfigError("Use quantization.hifloat8 instead of low_precision")
    options = param_dict.get(QUANTIZATION, {})
    if not isinstance(options, dict):
        raise DeepSpeedConfigError("'quantization' must be an object")
    if HIFLOAT8 in param_dict:
        if HIFLOAT8 in options:
            raise DeepSpeedConfigError("Set HiFloat8 options only under quantization.hifloat8")
        options = {**options, HIFLOAT8: param_dict[HIFLOAT8]}
    return DeepSpeedQuantizationConfig(**options)
