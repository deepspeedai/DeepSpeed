# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import get_scalar_param, get_dict_param, DeepSpeedConfigObject
from deepspeed.autotuning.constants import *


class DeepSpeedAutotuningConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedAutotuningConfig, self).__init__()

        self.enabled = None
        self.start_step = None
        self.end_step = None
        self.metric_path = None
        self.arg_mappings = None
        self.metric = None
        self.model_info = None
        self.results_dir = None
        self.exps_dir = None
        self.overwrite = None

        if param_dict and AUTOTUNING in param_dict.keys():
            autotuning_dict = param_dict[AUTOTUNING]
        else:
            autotuning_dict = {}

        self._initialize(autotuning_dict)

    def _initialize(self, autotuning_dict):
        self.enabled = get_scalar_param(autotuning_dict, AUTOTUNING_ENABLED, AUTOTUNING_ENABLED_DEFAULT)

        self.fast = get_scalar_param(autotuning_dict, AUTOTUNING_FAST, AUTOTUNING_FAST_DEFAULT)

        self.results_dir = get_scalar_param(autotuning_dict, AUTOTUNING_RESULTS_DIR, AUTOTUNING_RESULTS_DIR_DEFAULT)
        assert self.results_dir, "results_dir cannot be empty"
        self.exps_dir = get_scalar_param(autotuning_dict, AUTOTUNING_EXPS_DIR, AUTOTUNING_EXPS_DIR_DEFAULT)
        assert self.exps_dir, "exps_dir cannot be empty"
        self.overwrite = get_scalar_param(autotuning_dict, AUTOTUNING_OVERWRITE, AUTOTUNING_OVERWRITE_DEFAULT)

        self.start_profile_step = get_scalar_param(autotuning_dict, AUTOTUNING_START_PROFILE_STEP,
                                                   AUTOTUNING_START_PROFILE_STEP_DEFAULT)

        self.end_profile_step = get_scalar_param(autotuning_dict, AUTOTUNING_END_PROFILE_STEP,
                                                 AUTOTUNING_END_PROFILE_STEP_DEFAULT)

        self.metric = get_scalar_param(autotuning_dict, AUTOTUNING_METRIC, AUTOTUNING_METRIC_DEFAULT)

        self.metric_path = get_scalar_param(autotuning_dict, AUTOTUNING_METRIC_PATH, AUTOTUNING_METRIC_PATH_DEFAULT)

        self.tuner_type = get_scalar_param(autotuning_dict, AUTOTUNING_TUNER_TYPE, AUTOTUNING_TUNER_TYPE_DEFAULT)

        self.tuner_early_stopping = get_scalar_param(autotuning_dict, AUTOTUNING_TUNER_EARLY_STOPPING,
                                                     AUTOTUNING_TUNER_EARLY_STOPPING_DEFAULT)

        self.tuner_num_trials = get_scalar_param(autotuning_dict, AUTOTUNING_TUNER_NUM_TRIALS,
                                                 AUTOTUNING_TUNER_NUM_TRIALS_DEFAULT)

        self.arg_mappings = get_dict_param(autotuning_dict, AUTOTUNING_ARG_MAPPINGS, AUTOTUNING_ARG_MAPPINGS_DEFAULT)

        self.model_info = get_model_info_config(autotuning_dict)

        self.model_info_path = get_scalar_param(autotuning_dict, AUTOTUNING_MODEL_INFO_PATH,
                                                AUTOTUNING_MODEL_INFO_PATH_DEFAULT)
        self.mp_size = get_scalar_param(autotuning_dict, AUTOTUNING_MP_SIZE, AUTOTUNING_MP_SIZE_DEFAULT)

        self.max_train_batch_size = get_dict_param(autotuning_dict, AUTOTUNING_MAX_TRAIN_BATCH_SIZE,
                                                   AUTOTUNING_MAX_TRAIN_BATCH_SIZE_DEFAULT)

        self.min_train_batch_size = get_dict_param(autotuning_dict, AUTOTUNING_MIN_TRAIN_BATCH_SIZE,
                                                   AUTOTUNING_MIN_TRAIN_BATCH_SIZE_DEFAULT)

        self.max_train_micro_batch_size_per_gpu = get_dict_param(
            autotuning_dict, AUTOTUNING_MAX_TRAIN_MICRO_BATCH_SIZE_PER_GPU,
            AUTOTUNING_MAX_TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT)

        self.min_train_micro_batch_size_per_gpu = get_dict_param(
            autotuning_dict, AUTOTUNING_MIN_TRAIN_MICRO_BATCH_SIZE_PER_GPU,
            AUTOTUNING_MIN_TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT)

        self.num_tuning_micro_batch_sizes = get_dict_param(autotuning_dict, AUTOTUNING_NUM_TUNING_MICRO_BATCH_SIZES,
                                                           AUTOTUNING_NUM_TUNING_MICRO_BATCH_SIZES_DEFAULT)


def get_model_info_config(param_dict):
    if MODEL_INFO in param_dict and param_dict[MODEL_INFO] is not None:
        model_info_config = {}
        for key, default_value in MODEL_INFO_KEY_DEFAULT_DICT.items():
            model_info_config[key] = get_scalar_param(param_dict[MODEL_INFO], key, default_value)
        return model_info_config
    return None


def get_default_model_info_config():
    return MODEL_INFO_KEY_DEFAULT_DICT
