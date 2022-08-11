"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from pydantic import Field, validator
import sys
from typing import Optional
from enum import Enum
from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigModel
from deepspeed.utils import logger
from .offload_config import DeepSpeedZeroOffloadParamConfig, DeepSpeedZeroOffloadOptimizerConfig, OffloadDeviceEnum

# ZeRO optimization. By default, this optimization is not enabled.
# Users have to configure the desired optimization (0 means disabled) in params.json as below example:
ZERO_FORMAT = """
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": {
    "stage": [0|1|2],
    "stage3_max_live_parameters" : 1000000000,
    "stage3_max_reuse_distance" : 1000000000,
    "allgather_partitions": [true|false],
    "allgather_bucket_size": 500000000,
    "reduce_scatter": [true|false],
    "contiguous_gradients" : [true|false]
    "overlap_comm": [true|false],
    "reduce_bucket_size": 500000000,
    "load_from_fp32_weights": [true|false],
    "cpu_offload": [true|false] (deprecated),
    "cpu_offload_params" : [true|false] (deprecated),
    "cpu_offload_use_pin_memory": [true|false] (deprecated),
    "sub_group_size" : 1000000000000,
    "offload_param": {...},
    "offload_optimizer": {...},
    "ignore_unused_parameters": [true|false],
    "round_robin_gradients": [true|false]
    }
}
"""

ZERO_OPTIMIZATION = "zero_optimization"


def read_zero_config_deprecated(param_dict):
    zero_config_dict = {}
    zero_config_dict["stage"] = 1 if param_dict[ZERO_OPTIMIZATION] else 0
    if zero_config_dict["stage"] > 0:
        zero_config_dict["allgather_bucket_size"] = get_scalar_param(
            param_dict,
            "allgather_size",
            5e8)
    logger.warning(
        "DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}"
        .format(ZERO_FORMAT))
    return zero_config_dict


def get_zero_config(param_dict):
    if ZERO_OPTIMIZATION in param_dict:
        zero_config_dict = param_dict[ZERO_OPTIMIZATION]
        if isinstance(zero_config_dict, bool):
            zero_config_dict = read_zero_config_deprecated(param_dict)
    else:
        zero_config_dict = {}
    return DeepSpeedZeroConfig(**zero_config_dict)


class ZeroStageEnum(int, Enum):
    disabled = 0
    optimizer_states = 1
    gradients = 2
    weights = 3
    max_stage = 3


class DeepSpeedZeroConfig(DeepSpeedConfigModel):
    stage: ZeroStageEnum = ZeroStageEnum.disabled
    contiguous_gradients: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: int = Field(5e8, ge=0)
    allgather_partitions: bool = True
    allgather_bucket_size: int = Field(5e8, ge=0)
    overlap_comm: bool = None  # None for dynamic default value
    load_from_fp32_weights: bool = True

    elastic_checkpoint: bool = False

    # Offload Specific Parameters
    offload_param: Optional[DeepSpeedZeroOffloadParamConfig] = None
    offload_optimizer: Optional[DeepSpeedZeroOffloadOptimizerConfig] = None
    sub_group_size: int = Field(1e9, ge=0)
    cpu_offload_param: bool = Field(
        None,
        deprecated=True,
        new_param="offload_param",
        new_param_fn=(
            lambda val: DeepSpeedZeroOffloadParamConfig(device=OffloadDeviceEnum.cpu)
            if val else None),
    )
    cpu_offload_use_pin_memory: bool = Field(
        None,
        deprecated=True,
        new_param="offload_param or offload_optimizer",
        set_new_param=False,
    )
    cpu_offload: bool = Field(
        None,
        deprecated=True,
        new_param="offload_optimizer",
        new_param_fn=(
            lambda val: DeepSpeedZeroOffloadOptimizerConfig(device=OffloadDeviceEnum.cpu)
            if val else None),
    )

    # Stage3 Specific Parameters
    prefetch_bucket_size: int = Field(5e7, ge=0, alias="stage3_prefetch_bucket_size")
    param_persistence_threshold: int = Field(1e5,
                                             ge=0,
                                             alias="stage3_param_persistence_threshold")
    model_persistence_threshold: int = Field(sys.maxsize,
                                             ge=0,
                                             alias="stage3_model_persistence_threshold")
    max_live_parameters: int = Field(1e9, ge=0, alias="stage3_max_live_parameters")
    max_reuse_distance: int = Field(1e9, ge=0, alias="stage3_max_reuse_distance")
    gather_16bit_weights_on_model_save: bool = Field(
        False,
        alias="stage3_gather_16bit_weights_on_model_save")
    stage3_gather_fp16_weights_on_model_save: bool = Field(
        False,
        deprecated=True,
        new_param="gather_16bit_weights_on_model_save")

    ignore_unused_parameters: bool = True
    legacy_stage1: bool = False
    round_robin_gradients: bool = False

    @validator("overlap_comm")
    def overlap_comm_valid(cls, field_value, values):
        if field_value is None:
            assert (
                "stage" in values
            ), "DeepSpeedZeroConfig: 'stage' must be defined before 'overlap_comm'"
            field_value = values["stage"] == ZeroStageEnum.weights
        return field_value
