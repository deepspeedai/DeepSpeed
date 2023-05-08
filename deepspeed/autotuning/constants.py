# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

#########################################
# autotuner implementation constants
#########################################

import os

DEFAULT_TEMPLATE_PATH_ZERO_0 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_templates",
                                            "template_zero0.json")
DEFAULT_TEMPLATE_PATH_ZERO_1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_templates",
                                            "template_zero1.json")
DEFAULT_TEMPLATE_PATH_ZERO_2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_templates",
                                            "template_zero2.json")
DEFAULT_TEMPLATE_PATH_ZERO_3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_templates",
                                            "template_zero3.json")

METRIC_PERCENT_DIFF_CONST = 0.05
DS_CONFIG = "ds_config"
BUFSIZE = 1  # line buffer size for writing files

#########################################
# autotuner configuration constants
#########################################
# Autotuner. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
AUTOTUNING_FORMAT = """
autotuner should be enabled as:
"session_params": {
  "autotuning": {
    "enabled": true,
    "start_step": 5,
    "end_step": 15
    }
}
"""

AUTOTUNING = "autotuning"

AUTOTUNING_ENABLED = "enabled"
AUTOTUNING_ENABLED_DEFAULT = False

AUTOTUNING_FAST = "fast"
AUTOTUNING_FAST_DEFAULT = True

AUTOTUNING_RESULTS_DIR = "results_dir"
AUTOTUNING_RESULTS_DIR_DEFAULT = "autotuning_results"

AUTOTUNING_EXPS_DIR = "exps_dir"
AUTOTUNING_EXPS_DIR_DEFAULT = "autotuning_exps"

AUTOTUNING_OVERWRITE = "overwrite"
AUTOTUNING_OVERWRITE_DEFAULT = True

AUTOTUNING_START_PROFILE_STEP = "start_profile_step"
AUTOTUNING_START_PROFILE_STEP_DEFAULT = 3

AUTOTUNING_END_PROFILE_STEP = "end_profile_step"
AUTOTUNING_END_PROFILE_STEP_DEFAULT = 5
AUTOTUNING_METRIC_PATH = "metric_path"
AUTOTUNING_METRIC_PATH_DEFAULT = None

AUTOTUNING_TUNER_TYPE = "tuner_type"
AUTOTUNING_TUNER_GRIDSEARCH = "gridsearch"
AUTOTUNING_TUNER_RANDOM = "random"
AUTOTUNING_TUNER_MODELBASED = "model_based"
AUTOTUNING_TUNER_TYPE_DEFAULT = AUTOTUNING_TUNER_GRIDSEARCH
AUTOTUNING_TUNER_EARLY_STOPPING = "tuner_early_stopping"
AUTOTUNING_TUNER_EARLY_STOPPING_DEFAULT = 5
AUTOTUNING_TUNER_NUM_TRIALS = "tuner_num_trials"
AUTOTUNING_TUNER_NUM_TRIALS_DEFAULT = 50

AUTOTUNING_ARG_MAPPINGS = "arg_mappings"
AUTOTUNING_ARG_MAPPINGS_DEFAULT = None

AUTOTUNING_MAX_TRAIN_BATCH_SIZE = "max_train_batch_size"
AUTOTUNING_MAX_TRAIN_BATCH_SIZE_DEFAULT = None
AUTOTUNING_MIN_TRAIN_BATCH_SIZE = "min_train_batch_size"
AUTOTUNING_MIN_TRAIN_BATCH_SIZE_DEFAULT = 1
AUTOTUNING_MAX_TRAIN_MICRO_BATCH_SIZE_PER_GPU = "max_train_micro_batch_size_per_gpu"
AUTOTUNING_MAX_TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = 1024
AUTOTUNING_MIN_TRAIN_MICRO_BATCH_SIZE_PER_GPU = "min_train_micro_batch_size_per_gpu"
AUTOTUNING_MIN_TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = 1
AUTOTUNING_NUM_TUNING_MICRO_BATCH_SIZES = "num_tuning_micro_batch_sizes"
AUTOTUNING_NUM_TUNING_MICRO_BATCH_SIZES_DEFAULT = 3

AUTOTUNING_MP_SIZE = "mp_size"
AUTOTUNING_MP_SIZE_DEFAULT = 1

AUTOTUNING_METRIC = "metric"
AUTOTUNING_METRIC_LATENCY = "latency"
AUTOTUNING_METRIC_THROUGHPUT = "throughput"
AUTOTUNING_METRIC_FLOPS = "flops"
AUTOTUNING_METRIC_FORWARD = "forward"
AUTOTUNING_METRIC_BACKWRAD = "flops"
AUTOTUNING_METRIC_STEPS = "step"
AUTOTUNING_METRIC_DEFAULT = AUTOTUNING_METRIC_THROUGHPUT

#########################################
# MODEL INFO
#########################################
AUTOTUNING_MODEL_INFO_PATH = "model_info_path"
AUTOTUNING_MODEL_INFO_PATH_DEFAULT = None

MODEL_INFO_FORMAT = '''
"model_info": {
  "num_params": 1000000000,
  "hidden_size": 10,
  "num_layers": 12,
}
'''
MODEL_INFO = "model_info"
MODEL_INFO_PROFILE = "profile"
MODEL_INFO_PROFILE_DEFAULT = False
MODEL_INFO_NUM_PARAMS = "num_params"
MODEL_INFO_NUM_PARAMS_DEFAULT = None
MODEL_INFO_HIDDEN_SIZE = "hidden_size"
MODEL_INFO_HIDDEN_SIZE_DEFAULT = None
MODEL_INFO_NUM_LAYERS = "num_layers"
MODEL_INFO_NUM_LAYERS_DEFAULT = None

MODEL_INFO_KEY_DEFAULT_DICT = {
    MODEL_INFO_PROFILE: MODEL_INFO_PROFILE_DEFAULT,
    MODEL_INFO_NUM_PARAMS: MODEL_INFO_NUM_PARAMS_DEFAULT,
    MODEL_INFO_HIDDEN_SIZE: MODEL_INFO_HIDDEN_SIZE_DEFAULT,
    MODEL_INFO_NUM_LAYERS: MODEL_INFO_NUM_LAYERS_DEFAULT
}

#########################################
# autotuner search space constants
#########################################

DEFAULT_HF_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
}

DEFAULT_MIN_MEM_CONFIG = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {
        "stage": 3
    },
    "memory_break_down": False
}

DEFAULT_TUNING_SPACE_ZERO_0 = {"zero_optimization": {"stage": 0}}

DEFAULT_TUNING_SPACE_ZERO_1 = {
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": [5e7, 5e8, 1e9],
        "allgather_bucket_size": [5e7, 5e8, 1e9],
    }
}

DEFAULT_TUNING_SPACE_ZERO_2 = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": [True, False],
        "reduce_scatter": [False, True],
        "reduce_bucket_size": [5e7, 5e8, 1e9],
        "allgather_bucket_size": [5e7, 5e8, 1e9],
        "contiguous_gradients": [False, True]
    },
}

DEFAULT_TUNING_SPACE_ZERO_3 = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": [True, False],
        "reduce_scatter": [False, True],
        "reduce_bucket_size": [5e7, 5e8, 1e9],
        "allgather_partitions": [True, False],
        "allgather_bucket_size": [5e7, 5e8, 1e9],
        "contiguous_gradients": [False, True]
    },
}

GLOBAL_TUNING_SPACE = 'global'
# TUNING_MICRO_BATCH_SIZE_PREFIX="tune_micro_batch_size_z"
TUNING_MICRO_BATCH_SIZE_PREFIX = "z"
