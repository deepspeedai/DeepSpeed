# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.pydantic_v1 import Field

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from .ragged import DSStateManagerConfig


class DeepSpeedTPConfig(DeepSpeedConfigModel):
    """ Configure tensor parallelism settings """

    tp_size: int = 1
    """ Number of devices to split the model across using tensor parallelism. """


class RaggedInferenceEngineConfig(DeepSpeedConfigModel):
    """ Sets parameters for DeepSpeed Inference Engine. """

    tensor_parallel: DeepSpeedTPConfig = Field({}, alias="tp")
    """
    Configuration for tensor parallelism used to split the model across several
    GPUs. Expects a dictionary containing values for :any:`DeepSpeedTPConfig`.
    """

    state_manager: DSStateManagerConfig = Field({}, alias="manager")
    """
    Configuration for managing persistent state
    """
