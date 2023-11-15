# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Various symbolic constants used for model checkpointing
"""

#########################################
# Optimizer checkpoint keys
#########################################
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
FP32_GROUPS = "fp32_groups"
FP32_FLAT_GROUPS = 'fp32_flat_groups'

BASE_OPTIMIZER_STATE = 'base_optimizer_state'
BASE_OPTIMIZER_STATE_STEP = 'base_optimizer_state_step'
SINGLE_PARTITION_OF_FP32_GROUPS = "single_partition_of_fp32_groups"
GROUP_PADDINGS = 'group_paddings'
PARTITION_COUNT = 'partition_count'
ZERO_STAGE = 'zero_stage'
CLIP_GRAD = 'clip_grad'
FP32_WEIGHT_KEY = "fp32"
LOSS_SCALER = 'loss_scaler'

#########################################
# Module checkpoint keys
#########################################
PARAM = 'param'
PARAM_SHAPES = 'param_shapes'
BUFFER_NAMES = 'buffer_names'
FROZEN_PARAM_SHAPES = 'frozen_param_shapes'
FROZEN_PARAM_FRAGMENTS = 'frozen_param_fragments'

#########################################
# Checkpoint naming constants
#########################################
MODEL_FILE_PREFIX = 'mp_rank_'
ZERO_FILE_PREFIX = 'zero_pp_rank_'
OPTIM_FILE_SUFFIX = '_optim_states.pt'
MODEL_FILE_SUFFIX = '_model_states.pt'
LAYER_FILE_PREFIX = 'layer_'
BF16_ZERO_FILE_PREFIX = 'bf16_' + ZERO_FILE_PREFIX
FP16_ZERO_FILE_PREFIX = 'fp16_' + ZERO_FILE_PREFIX

#########################################
# Checkpoint utility keys
#########################################
DS_VERSION = 'ds_version'

#########################################
# Universal Checkpoint keys
#########################################
UNIVERSAL_CHECKPOINT_INFO = 'universal_checkpoint_info'
UNIVERSAL_CHECKPOINT_VERSION_KEY = 'universal_checkpoint_version'
# Reserve version 0.1  for the hardcoded logic used in BLOOM-176B training
UNIVERSAL_CHECKPOINT_VERSION_VALUE = 0.2

# Vocabulary padding
VOCAB_TENSOR = 'vocab_tensor'
PADDED_VOCAB_SIZE = 'padded_vocab_size'
ORIGINAL_VOCAB_SIZE = 'original_vocab_size'

# Parameter splitting/merging
PARAM_SLICE_MAPPINGS = 'param_slice_mappings'
CAT_DIM = "cat_dim"
# Following is a special case where a parameter effectively contains sub parameters.
# As an example, consider Megatron-DeepSpeed GPT SWIGLU implementation (mlp.h_to_4h).
# In this case, a single parameter ia allocated contiguously, but used as separate parameters.
# When using universal checkpoint, we have to normalize the representation of the full parameter.
# We normalize it by concatenating all slices of the sub params and then concatenating the sub params.
# All concat operations are done on CAT_DIM (currently, no support for different concat dims sub params and TP slicing).
# Similarly, load_hp_checkpoint_state has to take the needed actions when loading from universal.
PARAM_N_SUB_PARAMS = "param_n_sub_params"

# Regex list of parameters that require special handling
VOCABULARY_PARAMETER_PATTERNS = 'vocabulary_parameter_patterns'
PIPELINE_REPLICATED_PARAMETER_PATTERNS = 'pipeline_replicated_parameter_patterns'
PARAMETER_TO_AVERAGE_PATTERNS = 'parameter_to_average_patterns'
PARAMETER_WITH_ROW_PARALLELISM_PATTERNS = 'parameter_with_row_parallelism_patterns'
TP_REPLICATED_PARAMETER_PATTERNS = 'tp_replicated_parameter_patterns'
PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0 = 'parameter_with_2_sub_params_cat_dim_0'
