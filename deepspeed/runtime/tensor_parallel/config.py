# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
from deepspeed.runtime.config_utils import DeepSpeedConfigModel
import torch
from pydantic import Field
from typing import Optional


class AUTOTP_MODE(Enum):
    TRAINING = "TRAINING"
    INFERENCE = "INFERENCE"


class TPConfig(DeepSpeedConfigModel):
    """ Configure tensor parallelism settings """

    tp_size: int = 1
    """ Number of devices to split the model across using tensor parallelism. """

    tp_grain_size: int = 1
    "The variable required by the autoTP parser has not been activated in training yet"
    "as it depends on the gather logic that supports uneven partitioning. "
    "Desired MLP/lm_head tp size granularity. DNN library favors tensor size in granularity of power of 2, we pick 64 as a default size."

    mpu: object = None
    """
    A model parallelism unit object that implements
    ``get_{model,data}_parallel_{rank,group,world_size}()``.
    """

    tp_group: object = None


class TPTrainingConfig(DeepSpeedConfigModel):

    dtype: torch.dtype = torch.float16
    """
    Desired model data type, will convert model to this type.
    """

    autotp_size: int = 0
    """
    In automatic tensor-parallelism training, 'tensor_parallel_size'
    When set to 0, indicates that it is disabled.
    """
    tp_overlap_comm: bool = False
    """ Whether to overlap communication with computation. Currently, only allreduce supports overlap. """

    tensor_parallel: TPConfig = Field({}, alias="tp")
    """
    Configuration for tensor parallelism used to split the model across several
    GPUs. Expects a dictionary containing values for :any:`DeepSpeedTPConfig`.
    """

    injection_policy_tuple: Optional[tuple] = None
    #The following parameters are required by autoTP parser.
    ########################################
    keep_module_on_host: bool = False
    """
    When loading checkpoints to model parameters, they are moved to the device. In very large models
    this might fill the device and cause OOM. Setting this flag to true, will keep checkpoints on
    host and not move them directly to the device (giving an option to quantize checkpoint data before
    moving it to the device for example).
    """

    replace_with_kernel_inject: bool = Field(False, alias="kernel_inject")
    """
    Set to true to inject inference kernels for models such as, Bert, GPT2,
    GPT-Neo and GPT-J.  Otherwise, the injection_dict provides the names of two
    linear layers as a tuple:
    `(attention_output projection, transformer output projection)`
    """
    ########################################


def get_tensor_parallel_config(ds_config):

    if 'tensor_parallel' in ds_config:
        return TPTrainingConfig(**ds_config['tensor_parallel'])
    return TPTrainingConfig()
