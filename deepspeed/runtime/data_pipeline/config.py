# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .constants import *
import copy
from ..config_utils import get_scalar_param


# TODO: Reducing config verbosity by returning None or {} when disabled.
# One challenge is that we still need to somehow include the default values,
# for example the *_ENABLED has default of false.
def get_data_efficiency_config(param_dict):
    output = {}
    output[DATA_EFFICIENCY_ENABLED] = get_data_efficiency_enabled(param_dict)
    output[DATA_EFFICIENCY_SEED] = get_data_efficiency_seed(param_dict)
    if DATA_EFFICIENCY not in param_dict.keys():
        param_dict[DATA_EFFICIENCY] = {}
    sub_param_dict = param_dict[DATA_EFFICIENCY]
    output[DATA_SAMPLING] = get_data_sampling(sub_param_dict)
    output[DATA_ROUTING] = get_data_routing(sub_param_dict)

    return output


def get_data_efficiency_enabled(param_dict):
    if DATA_EFFICIENCY in param_dict.keys():
        return get_scalar_param(param_dict[DATA_EFFICIENCY], DATA_EFFICIENCY_ENABLED, DATA_EFFICIENCY_ENABLED_DEFAULT)
    else:
        return False


def get_data_efficiency_seed(param_dict):
    if DATA_EFFICIENCY in param_dict.keys():
        return get_scalar_param(param_dict[DATA_EFFICIENCY], DATA_EFFICIENCY_SEED, DATA_EFFICIENCY_SEED_DEFAULT)
    else:
        return DATA_EFFICIENCY_SEED_DEFAULT


def get_data_sampling(param_dict):
    output = {}
    output[DATA_SAMPLING_ENABLED] = get_data_sampling_enabled(param_dict)
    output[DATA_SAMPLING_NUM_EPOCHS] = get_data_sampling_num_epochs(param_dict)
    output[DATA_SAMPLING_NUM_WORKERS] = get_data_sampling_num_workers(param_dict)
    if DATA_SAMPLING not in param_dict.keys():
        param_dict[DATA_SAMPLING] = {}
    sub_param_dict = param_dict[DATA_SAMPLING]
    output[CURRICULUM_LEARNING] = get_curriculum_learning(sub_param_dict)

    return output


def get_data_sampling_enabled(param_dict):
    if DATA_SAMPLING in param_dict.keys():
        return get_scalar_param(param_dict[DATA_SAMPLING], DATA_SAMPLING_ENABLED, DATA_SAMPLING_ENABLED_DEFAULT)
    else:
        return False


def get_data_sampling_num_epochs(param_dict):
    if DATA_SAMPLING in param_dict.keys():
        return get_scalar_param(param_dict[DATA_SAMPLING], DATA_SAMPLING_NUM_EPOCHS, DATA_SAMPLING_NUM_EPOCHS_DEFAULT)
    else:
        return DATA_SAMPLING_NUM_EPOCHS_DEFAULT


def get_data_sampling_num_workers(param_dict):
    if DATA_SAMPLING in param_dict.keys():
        return get_scalar_param(param_dict[DATA_SAMPLING], DATA_SAMPLING_NUM_WORKERS,
                                DATA_SAMPLING_NUM_WORKERS_DEFAULT)
    else:
        return DATA_SAMPLING_NUM_WORKERS_DEFAULT


def get_curriculum_learning(param_dict):
    output = {}
    output[CURRICULUM_LEARNING_ENABLED] = get_curriculum_learning_enabled(param_dict)
    if CURRICULUM_LEARNING not in param_dict.keys():
        param_dict[CURRICULUM_LEARNING] = {}
    sub_param_dict = param_dict[CURRICULUM_LEARNING]
    if output[CURRICULUM_LEARNING_ENABLED]:
        assert CURRICULUM_LEARNING_METRICS in sub_param_dict.keys(
        ), f"Curriculum learning is enabled, {CURRICULUM_LEARNING_METRICS} must be specified"
        for key, val in get_curriculum_learning_params(param_dict).items():
            output[key] = val
    return output


def get_curriculum_learning_enabled(param_dict):
    if CURRICULUM_LEARNING in param_dict.keys():
        return get_scalar_param(param_dict[CURRICULUM_LEARNING], CURRICULUM_LEARNING_ENABLED,
                                CURRICULUM_LEARNING_ENABLED_DEFAULT)
    else:
        return False


def get_curriculum_learning_params(param_dict):
    if CURRICULUM_LEARNING in param_dict.keys():
        curriculum_learning_params = copy.copy(param_dict[CURRICULUM_LEARNING])
        curriculum_learning_params.pop(CURRICULUM_LEARNING_ENABLED)
        return curriculum_learning_params
    else:
        return {}


def get_curriculum_enabled_legacy(param_dict):
    if CURRICULUM_LEARNING_LEGACY in param_dict.keys():
        return get_scalar_param(param_dict[CURRICULUM_LEARNING_LEGACY], CURRICULUM_ENABLED_LEGACY,
                                CURRICULUM_ENABLED_DEFAULT_LEGACY)
    else:
        return False


def get_curriculum_params_legacy(param_dict):
    if CURRICULUM_LEARNING_LEGACY in param_dict.keys():
        curriculum_params = copy.copy(param_dict[CURRICULUM_LEARNING_LEGACY])
        curriculum_params.pop(CURRICULUM_ENABLED_LEGACY)
        return curriculum_params
    else:
        return False


def get_data_routing(param_dict):
    output = {}
    output[DATA_ROUTING_ENABLED] = get_data_routing_enabled(param_dict)
    if DATA_ROUTING not in param_dict.keys():
        param_dict[DATA_ROUTING] = {}
    sub_param_dict = param_dict[DATA_ROUTING]
    output[RANDOM_LTD] = get_random_ltd(sub_param_dict)

    return output


def get_data_routing_enabled(param_dict):
    if DATA_ROUTING in param_dict.keys():
        return get_scalar_param(param_dict[DATA_ROUTING], DATA_ROUTING_ENABLED, DATA_ROUTING_ENABLED_DEFAULT)
    else:
        return False


def get_random_ltd(param_dict):
    output = {}
    output[RANDOM_LTD_ENABLED] = RANDOM_LTD_ENABLED_DEFAULT
    output[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE] = {}
    output[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE][
        RANDOM_LTD_LAYER_TOKEN_LR_ENABLED] = RANDOM_LTD_LAYER_TOKEN_LR_ENABLED_DEFAULT
    if get_random_ltd_enabled(param_dict):
        output[RANDOM_LTD_ENABLED] = get_random_ltd_enabled(param_dict)
        for key, val in get_random_ltd_params(param_dict).items():
            output[key] = val
    return output


def get_random_ltd_enabled(param_dict):
    if RANDOM_LTD in param_dict.keys():
        return get_scalar_param(param_dict[RANDOM_LTD], RANDOM_LTD_ENABLED, RANDOM_LTD_ENABLED_DEFAULT)
    else:
        return False


def get_random_ltd_params(param_dict):
    if RANDOM_LTD in param_dict.keys():
        random_ltd_params = copy.copy(param_dict[RANDOM_LTD])
        random_ltd_params.pop(RANDOM_LTD_ENABLED)
        return random_ltd_params
    else:
        return {}
