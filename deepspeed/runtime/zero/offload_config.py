'''
Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
'''

from deepspeed.runtime.config_utils import get_scalar_param
from .offload_constants import *

OFFLOAD_PARAM_KEY_DEFAULT_DICT = {
    OFFLOAD_PARAM_DEVICE: OFFLOAD_PARAM_DEVICE_DEFAULT,
    OFFLOAD_PARAM_NVME_PATH: OFFLOAD_PARAM_NVME_PATH_DEFAULT,
    OFFLOAD_PARAM_BUFFER_COUNT: OFFLOAD_PARAM_BUFFER_COUNT_DEFAULT,
    OFFLOAD_PARAM_BUFFER_SIZE: OFFLOAD_PARAM_BUFFER_SIZE_DEFAULT,
    OFFLOAD_PARAM_MAX_IN_CPU: OFFLOAD_PARAM_MAX_IN_CPU_DEFAULT,
    OFFLOAD_PARAM_PIN_MEMORY: OFFLOAD_PARAM_PIN_MEMORY_DEFAULT
}

OFFLOAD_OPTIMIZER_KEY_DEFAULT_DICT = {
    OFFLOAD_OPTIMIZER_DEVICE: OFFLOAD_OPTIMIZER_DEVICE_DEFAULT,
    OFFLOAD_OPTIMIZER_NVME_PATH: OFFLOAD_OPTIMIZER_NVME_PATH_DEFAULT,
    OFFLOAD_OPTIMIZER_BUFFER_COUNT: OFFLOAD_OPTIMIZER_BUFFER_COUNT_DEFAULT,
    OFFLOAD_OPTIMIZER_PIN_MEMORY: OFFLOAD_OPTIMIZER_PIN_MEMORY_DEFAULT,
    OFFLOAD_OPTIMIZER_PIPELINE_READ: OFFLOAD_OPTIMIZER_PIPELINE_READ_DEFAULT,
    OFFLOAD_OPTIMIZER_PIPELINE_WRITE: OFFLOAD_OPTIMIZER_PIPELINE_WRITE_DEFAULT,
    OFFLOAD_OPTIMIZER_FAST_INIT: OFFLOAD_OPTIMIZER_FAST_INIT_DEFAULT
}


def _get_offload_config(param_dict, key_default_dict):
    offload_config = {}
    for key, default_value in key_default_dict.items():
        offload_config[key] = get_scalar_param(param_dict, key, default_value)

    return offload_config


def get_offload_param_config(param_dict):
    if OFFLOAD_PARAM in param_dict and param_dict[OFFLOAD_PARAM] is not None:
        return _get_offload_config(param_dict=param_dict[OFFLOAD_PARAM],
                                   key_default_dict=OFFLOAD_PARAM_KEY_DEFAULT_DICT)

    return None


def get_default_offload_param_config():
    return OFFLOAD_PARAM_KEY_DEFAULT_DICT


def get_offload_optimizer_config(param_dict):
    if OFFLOAD_OPTIMIZER in param_dict and param_dict[OFFLOAD_OPTIMIZER] is not None:
        offload_config = _get_offload_config(
            param_dict=param_dict[OFFLOAD_OPTIMIZER],
            key_default_dict=OFFLOAD_OPTIMIZER_KEY_DEFAULT_DICT)
        offload_config[OFFLOAD_OPTIMIZER_PIPELINE] = offload_config[
            OFFLOAD_OPTIMIZER_PIPELINE_READ] or offload_config[
                OFFLOAD_OPTIMIZER_PIPELINE_WRITE]
        return offload_config

    return None


def get_default_offload_optimizer_config():
    return OFFLOAD_OPTIMIZER_KEY_DEFAULT_DICT
