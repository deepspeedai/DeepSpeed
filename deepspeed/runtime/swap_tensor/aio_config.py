# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.accelerator import get_accelerator

AIO_DEFAULT_DICT = {
    AIO_BLOCK_SIZE: AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH: AIO_QUEUE_DEPTH_DEFAULT,
    AIO_THREAD_COUNT: AIO_THREAD_COUNT_DEFAULT,
    AIO_SINGLE_SUBMIT: AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS: AIO_OVERLAP_EVENTS_DEFAULT,
    AIO_USE_GDS: AIO_USE_GDS_DEFAULT
}


def get_aio_config(param_dict):
    if AIO in param_dict.keys() and param_dict[AIO] is not None:
        aio_dict = param_dict[AIO]
        aio_config = {
            AIO_BLOCK_SIZE: get_scalar_param(aio_dict, AIO_BLOCK_SIZE, AIO_BLOCK_SIZE_DEFAULT),
            AIO_QUEUE_DEPTH: get_scalar_param(aio_dict, AIO_QUEUE_DEPTH, AIO_QUEUE_DEPTH_DEFAULT),
            AIO_THREAD_COUNT: get_scalar_param(aio_dict, AIO_THREAD_COUNT, AIO_THREAD_COUNT_DEFAULT),
            AIO_SINGLE_SUBMIT: get_scalar_param(aio_dict, AIO_SINGLE_SUBMIT, AIO_SINGLE_SUBMIT_DEFAULT),
            AIO_OVERLAP_EVENTS: get_scalar_param(aio_dict, AIO_OVERLAP_EVENTS, AIO_OVERLAP_EVENTS_DEFAULT),
            AIO_USE_GDS: get_scalar_param(aio_dict, AIO_USE_GDS, AIO_USE_GDS_DEFAULT)
        }

        if aio_config[AIO_USE_GDS]:
            assert get_accelerator().device_name() == 'cuda', 'GDS currently only supported for CUDA accelerator'

        return aio_config

    return AIO_DEFAULT_DICT
