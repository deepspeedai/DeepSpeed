'''
    Various symbolic constants used for model checkpointing
'''

#########################################
# Optimizer checkpoint keys
#########################################
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
FP32_GROUPS = "fp32_groups"
FP32_FLAT_GROUPS = 'fp32_flat_groups'

BASE_OPTIMIZER_STATE = 'base_optimizer_state'
SINGLE_PARTITION_OF_FP32_GROUPS = "single_partition_of_fp32_groups"
GROUP_PADDINGS = 'group_paddings'
PARTITION_COUNT = 'partition_count'
ZERO_STAGE = 'zero_stage'
CLIP_GRAD = 'clip_grad'
PARAM_SLICE_MAPPINGS = 'param_slice_mappings'
FP32_WEIGHT_KEY = "fp32"

#########################################
# Module checkpoint keys
#########################################
PARAM = 'param'
PARAM_SHAPES = 'param_shapes'
BUFFER_NAMES = 'buffer_names'
VOCAB_DIVISIBILITY_PADDING_TENSOR = 'vocab_divisibility_padding_tensor'
CAT_DIM = "cat_dim"

#########################################
# Checkpoint naming constants
#########################################
MODEL_FILE_PREFIX = 'mp_rank_'
ZERO_FILE_PREFIX = 'bf16_' + 'zero_pp_rank_'
OPTIM_FILE_SUFFIX = '_optim_states.pt'
MODEL_FILE_SUFFIX = '_model_states.pt'
LAYER_FILE_PREFIX = 'layer_'
BF16_ZERO_FILE_PREFIX = ZERO_FILE_PREFIX

#########################################
# Checkpoint utility keys
#########################################
DS_VERSION = 'ds_version'
