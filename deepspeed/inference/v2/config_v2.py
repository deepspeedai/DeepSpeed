# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional
from deepspeed.pydantic_v1 import Field
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from .ragged import DSStateManagerConfig


class DeepSpeedTPConfig(DeepSpeedConfigModel):
    """ Configure tensor parallelism settings """

    tp_size: int = 1
    """ Number of devices to split the model across using tensor parallelism. """


class QuantizationConfig(DeepSpeedConfigModel):
    """ Configure tensor parallelism settings """

    quantization_mode: Optional[str] = None
    """ The quantization mode in string format. The supported modes are as follows:
        - 'wf6af16', weight-only quantization with FP6 weight and FP16 activation.
    """
    # TODO: may reuse the constants in deepspeed/compression/constants.py


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

    quantization: QuantizationConfig = {}
