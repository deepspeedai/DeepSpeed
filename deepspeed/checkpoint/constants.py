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

PARTITION_COUNT = 'partition_count'
ZERO_STAGE = 'zero_stage'

#########################################
# Module checkpoint keys
#########################################
PARAM_SHAPES = 'param_shapes'
BUFFER_NAMES = 'buffer_names'
DS_VERSION = 'ds_version'
