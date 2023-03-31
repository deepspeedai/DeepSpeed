# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Data efficiency library
 See sample config at https://www.deepspeed.ai/docs/config-json/data-efficiency
"""
DATA_EFFICIENCY = "data_efficiency"
DATA_EFFICIENCY_ENABLED = "enabled"
DATA_EFFICIENCY_ENABLED_DEFAULT = False
DATA_EFFICIENCY_SEED = "seed"
DATA_EFFICIENCY_SEED_DEFAULT = 1234

#########################################
# Data efficiency - Data Sampling
#########################################
DATA_SAMPLING = "data_sampling"
DATA_SAMPLING_ENABLED = "enabled"
DATA_SAMPLING_ENABLED_DEFAULT = False
DATA_SAMPLING_NUM_EPOCHS = "num_epochs"
DATA_SAMPLING_NUM_EPOCHS_DEFAULT = 1000
DATA_SAMPLING_NUM_WORKERS = "num_workers"
DATA_SAMPLING_NUM_WORKERS_DEFAULT = 0

#########################################
# Data efficiency - Data Sampling - Curriculum Learning
#########################################
CURRICULUM_LEARNING = "curriculum_learning"
CURRICULUM_LEARNING_ENABLED = "enabled"
CURRICULUM_LEARNING_ENABLED_DEFAULT = False
CURRICULUM_LEARNING_CLUSTER_PATH = "data_cluster_path"
CURRICULUM_LEARNING_METRICS = "curriculum_metrics"
CURRICULUM_LEARNING_SAMPLE_PATH = "index_to_sample_path"
CURRICULUM_LEARNING_METRIC_PATH = "index_to_metric_path"
CURRICULUM_LEARNING_CLUSTERING_TYPE = "clustering_type"
CURRICULUM_LEARNING_SINGLE_CLUSTER = "single_cluster"
CURRICULUM_LEARNING_CLUSTER_PREFIX = "cluster"
CURRICULUM_LEARNING_DIFFICULTY_TYPE = "difficulty_type"
CURRICULUM_LEARNING_VALUE_BASED = "value"
CURRICULUM_LEARNING_PERCENTILE_BASED = "percentile"
CURRICULUM_LEARNING_MIN_DIFFICULTY = "min_difficulty"
CURRICULUM_LEARNING_MAX_DIFFICULTY = "max_difficulty"
CURRICULUM_LEARNING_SCHEDULE_TYPE = "schedule_type"
CURRICULUM_LEARNING_SCHEDULE_CONFIG = "schedule_config"
CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY = "difficulty"
CURRICULUM_LEARNING_SCHEDULE_MAX_STEP = "max_step"
CURRICULUM_LEARNING_SCHEDULE_TOTAL_STEP = "total_curriculum_step"
CURRICULUM_LEARNING_SCHEDULE_DIFFICULTY_STEP = "difficulty_step"
CURRICULUM_LEARNING_SCHEDULE_ROOT_DEGREE = "root_degree"
CURRICULUM_LEARNING_SCHEDULE_FIXED_DISCRETE = "fixed_discrete"
CURRICULUM_LEARNING_SCHEDULE_FIXED_ROOT = "fixed_root"
CURRICULUM_LEARNING_SCHEDULE_FIXED_LINEAR = "fixed_linear"
CURRICULUM_LEARNING_SCHEDULE_CUSTOM = "custom"
CURRICULUM_LEARNING_CURRENT_DIFFICULTY = "current_difficulty"

CURRICULUM_LEARNING_BATCH = "batch"
CURRICULUM_LEARNING_CONSUMED_SAMPLES = "consumed_samples"
CURRICULUM_LEARNING_STEP = "curriculum_step"
CURRICULUM_LEARNING_CURRENT_DIFFICULTIES = "current_difficulties"
CURRICULUM_LEARNING_DATA_CLUSTER_PATHS = "data_cluster_paths"
CURRICULUM_LEARNING_DATA_CLUSTER_CURRENT_POSITION = "data_cluster_current_position"
CURRICULUM_LEARNING_NP_RNG_STATE = "np_rng_state"

#########################################
# Curriculum Learning legacy implementation
#########################################
CURRICULUM_LEARNING_LEGACY = "curriculum_learning"

CURRICULUM_ENABLED_LEGACY = "enabled"
CURRICULUM_ENABLED_DEFAULT_LEGACY = False

#########################################
# Data efficiency - Data Routing
#########################################
DATA_ROUTING = "data_routing"
DATA_ROUTING_ENABLED = "enabled"
DATA_ROUTING_ENABLED_DEFAULT = False

#########################################
# Data efficiency - Data Routing - Random LTD
#########################################
RANDOM_LTD = "random_ltd"
RANDOM_LTD_ENABLED = "enabled"
RANDOM_LTD_ENABLED_DEFAULT = False

RANDOM_LTD_MODEL_MASK_NAME = "model_mask_name"
RANDOM_LTD_MODEL_TYPE = "model_type"
RANDOM_LTD_MICRO_BATCH_SIZE = "micro_batch_size"
RANDOM_LTD_GLOBAL_BATCH_SIZE = "global_batch_size"
RANDOM_LTD_SAMPLE_INDEX = "sample_idx"
RANDOM_LTD_ATTENTION_MASK = "attention_mask"
RANDOM_LTD_HIDDEN_STATE_ORDER = "hidden_state_order"
RANDOM_LTD_LAYER_NUM = "random_ltd_layer_num"
RANDOM_LTD_LAYER_ID = "random_ltd_layer_id"
RANDOM_LTD_TOTAL_LAYER_NUM = "total_layer_num"
RANDOM_LTD_CONSUMED_LAYER_TOKENS = "consumed_layer_tokens"

# scheduler
RANDOM_LTD_SCHEDULER = "random_ltd_schedule"
RANDOM_LTD_MAX_VALUE = "max_value"
RANDOM_LTD_MIN_VALUE = "min_value"
RANDOM_LTD_CURRENT_VALUE = "current_value"
RANDOM_LTD_SCHEDULE_CONFIG = "schedule_config"
RANDOM_LTD_INCREASE_STEP = "seq_per_step"
RANDOM_LTD_REQUIRE_STEP = "require_steps"
RANDOM_LTD_SCHEDULER_TYPE = "schedule_type"
RANDOM_LTD_CURR_STEP = "current_steps"

# learning rate schedulers
RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE = "layer_token_lr_schedule"
RANDOM_LTD_LAYER_TOKEN_LR_ENABLED = "enabled"
RANDOM_LTD_LAYER_TOKEN_LR_ENABLED_DEFAULT = False
RANDOM_LTD_TOTAL_LAYER_TOKENS = "total_layer_tokens"
RANDOM_LTD_WARMUP_TYPE = "warmup_type"
RANDOM_LTD_WARMUP_LAYER_TOKENS = "warmup_layer_tokens"
