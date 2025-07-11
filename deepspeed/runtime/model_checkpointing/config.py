# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import get_scalar_param
from .constants import *

VALID_VALUES = {
    CHECKPOINT_TAG_VALIDATION: CHECKPOINT_TAG_VALIDATION_MODES,
    CHECKPOINT_WRITER_TYPE: CHECKPOINT_WRITER_TYPES,
    CHECKPOINT_DATA_PARALLEL: CHECKPOINT_DATA_PARALLEL_UNITS
}

CHECKPOINT_DEFAULT_DICT = {
    CHECKPOINT_TAG_VALIDATION: CHECKPOINT_TAG_VALIDATION_DEFAULT,
    CHECKPOINT_SERIALIZATION: CHECKPOINT_SERIALIZATION_DEFAULT,
    CHECKPOINT_WRITER: CHECKPOINT_WRITER_DEFAULT
}


def _validate_config_values(config_name, config_dict, valid_values):
    for key, value in config_dict.items():
        if value is None:
            continue
        if key in valid_values.keys():
            assert value in valid_values[key], \
                f"{config_name} contains invalid value {value} for {key}, expecting one of {valid_values[key]}"


def _make_upper_case(value):
    return value if value is None else value.upper()


def get_checkpoint_writer_config(param_dict):
    writer_dict = param_dict.get(CHECKPOINT_WRITER, None)
    if writer_dict is None:
        return CHECKPOINT_WRITER_DEFAULT

    writer_config = {
        CHECKPOINT_WRITER_TYPE:
        _make_upper_case(get_scalar_param(writer_dict, CHECKPOINT_WRITER_TYPE, CHECKPOINT_WRITER_TYPE_DEFAULT)),
        CHECKPOINT_IO_BUFFER_SIZE:
        get_scalar_param(writer_dict, CHECKPOINT_IO_BUFFER_SIZE, CHECKPOINT_IO_BUFFER_SIZE_DEFAULT),
        CHECKPOINT_IO_BUFFER_DOUBLE:
        get_scalar_param(writer_dict, CHECKPOINT_IO_BUFFER_DOUBLE, CHECKPOINT_IO_BUFFER_DOUBLE_DEFAULT),
        CHECKPOINT_IO_STATISTICS:
        get_scalar_param(writer_dict, CHECKPOINT_IO_STATISTICS, CHECKPOINT_IO_STATISTICS_DEFAULT),
        CHECKPOINT_DATA_PARALLEL:
        _make_upper_case(get_scalar_param(writer_dict, CHECKPOINT_DATA_PARALLEL, CHECKPOINT_DATA_PARALLEL_DEFAULT)),
        CHECKPOINT_WRITER_DECOUPLED:
        get_scalar_param(writer_dict, CHECKPOINT_WRITER_DECOUPLED, CHECKPOINT_WRITER_DECOUPLED_DEFAULT),
        CHECKPOINT_IO_MULTIPLIER:
        get_scalar_param(writer_dict, CHECKPOINT_IO_MULTIPLIER, CHECKPOINT_IO_MULTIPLIER_DEFAULT),
    }
    _validate_config_values(CHECKPOINT_WRITER, writer_config, VALID_VALUES)

    return writer_config


def get_checkpoint_config(param_dict):
    checkpoint_dict = param_dict.get(CHECKPOINT, None)
    if checkpoint_dict is None:
        return CHECKPOINT_DEFAULT_DICT

    checkpoint_config = {
        CHECKPOINT_TAG_VALIDATION:
        get_scalar_param(checkpoint_dict, CHECKPOINT_TAG_VALIDATION, CHECKPOINT_TAG_VALIDATION_DEFAULT).upper(),
        CHECKPOINT_SERIALIZATION:
        get_scalar_param(checkpoint_dict, CHECKPOINT_SERIALIZATION, CHECKPOINT_SERIALIZATION_DEFAULT),
        CHECKPOINT_WRITER:
        get_checkpoint_writer_config(checkpoint_dict)
    }

    _validate_config_values(CHECKPOINT, checkpoint_config, VALID_VALUES)

    return checkpoint_config
