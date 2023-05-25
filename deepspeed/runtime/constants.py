# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

#############################################
# Routes
#############################################
ROUTE_TRAIN = "train"
ROUTE_EVAL = "eval"
ROUTE_PREDICT = "predict"
ROUTE_ENCODE = "encode"

#############################################
# Batch size
#############################################
TRAIN_BATCH_SIZE = "train_batch_size"
TRAIN_BATCH_SIZE_DEFAULT = None

#############################################
# Sparse attention
#############################################
SPARSE_ATTENTION = "sparse_attention"
SPARSE_DENSE_MODE = "dense"
SPARSE_FIXED_MODE = "fixed"
SPARSE_VARIABLE_MODE = "variable"
SPARSE_BIGBIRD_MODE = "bigbird"
SPARSE_BSLONGFORMER_MODE = "bslongformer"
SPARSE_MODE = "mode"
SPARSE_MODE_DEFAULT = SPARSE_FIXED_MODE
SPARSE_BLOCK = "block"
SPARSE_BLOCK_DEFAULT = 16
SPARSE_DIFFERENT_LAYOUT_PER_HEAD = "different_layout_per_head"
SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT = False
SPARSE_NUM_LOCAL_BLOCKS = "num_local_blocks"
SPARSE_NUM_LOCAL_BLOCKS_DEFAULT = 4
SPARSE_NUM_GLOBAL_BLOCKS = "num_global_blocks"
SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT = 1
SPARSE_ATTENTION_TYPE = "attention"
SPARSE_ATTENTION_TYPE_DEFAULT = "bidirectional"
SPARSE_HORIZONTAL_GLOBAL_ATTENTION = "horizontal_global_attention"
SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT = False
SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS = "num_different_global_patterns"
SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS_DEFAULT = 1
SPARSE_NUM_RANDOM_BLOCKS = "num_random_blocks"
SPARSE_NUM_RANDOM_BLOCKS_DEFAULT = 0
SPARSE_LOCAL_WINDOW_BLOCKS = "local_window_blocks"
SPARSE_LOCAL_WINDOW_BLOCKS_DEFAULT = [4]
SPARSE_GLOBAL_BLOCK_INDICES = "global_block_indices"
SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT = [0]
SPARSE_GLOBAL_BLOCK_END_INDICES = "global_block_end_indices"
SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT = None
SPARSE_NUM_SLIDING_WINDOW_BLOCKS = "num_sliding_window_blocks"
SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT = 3

#############################################
# Optimizer and lr scheduler
#############################################
OPTIMIZER = "optimizer"
OPTIMIZER_TYPE_DEFAULT = None
OPTIMIZER_PARAMS = "params"
TYPE = "type"
LEGACY_FUSION = "legacy_fusion"
LEGACY_FUSION_DEFAULT = False
SCHEDULER = "scheduler"
SCHEDULER_TYPE_DEFAULT = None
SCHEDULER_PARAMS = "params"
MAX_GRAD_NORM = 'max_grad_norm'

#############################################
# Optimizer and lr scheduler
#############################################
ZERO_ALLOW_UNTESTED_OPTIMIZER = "zero_allow_untested_optimizer"
ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT = False
ZERO_FORCE_DS_CPU_OPTIMIZER = "zero_force_ds_cpu_optimizer"
ZERO_FORCE_DS_CPU_OPTIMIZER_DEFAULT = True

# Steps
STEPS_PER_PRINT = "steps_per_print"
STEPS_PER_PRINT_DEFAULT = 10

#########################################
# Training micro batch size per GPU
#########################################
# Batch size for one training step. This is used when the
# TRAIN_BATCH_SIZE cannot fit in GPU memory to determine
# the number of gradient accumulation steps. By default, this
# is set to None. Users can configure in ds_config.json as below example:
TRAIN_MICRO_BATCH_SIZE_PER_GPU = '''
TRAIN_MICRO_BATCH_SIZE_PER_GPU is defined in this format:
"train_micro_batch_size_per_gpu": 1
'''
TRAIN_MICRO_BATCH_SIZE_PER_GPU = "train_micro_batch_size_per_gpu"
TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT = None

#########################################
# Gradient Accumulation
#########################################
# Gradient accumulation feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
GRADIENT_ACCUMULATION_FORMAT = '''
Gradient Accumulation should be of the format:
"gradient_accumulation_steps": 1
'''
GRADIENT_ACCUMULATION_STEPS = "gradient_accumulation_steps"
GRADIENT_ACCUMULATION_STEPS_DEFAULT = None

# DeepSpeed CSR gradient sparsity
SPARSE_GRADIENTS = "sparse_gradients"
SPARSE_GRADIENTS_DEFAULT = False

#########################################
# BFLOAT16 support
#########################################
# BFLOAT16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
BFLOAT16_FORMAT = '''
BFLOAT16 parameters should be of the format:
"bf16": {
  "enabled": true
}
'''
BFLOAT16 = "bf16"
BFLOAT16_OLD = "bfloat16"  # keeping for backwards compatibility

BFLOAT16_ENABLED = "enabled"
BFLOAT16_ENABLED_DEFAULT = False

#########################################
# FP16 support
#########################################
# FP16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FP16_FORMAT = '''
FP16 parameters should be of the format:
"fp16": {
  "enabled": true,
  "auto_cast": false,
  "loss_scale": 0,
  "initial_scale_power": 16,
  "loss_scale_window": 1000,
  "hysteresis": 2,
  "consecutive_hysteresis": false,
  "min_loss_scale": 1
}
'''
FP16 = "fp16"

FP16_ENABLED = "enabled"
FP16_ENABLED_DEFAULT = False

# FP16 loss scale, zero means using dynamic scaling
FP16_LOSS_SCALE = "loss_scale"
FP16_LOSS_SCALE_DEFAULT = 0

FP16_AUTO_CAST = "auto_cast"
FP16_AUTO_CAST_DEFAULT = False

# FP16 initial dynamic scale loss power
FP16_INITIAL_SCALE_POWER = "initial_scale_power"
FP16_INITIAL_SCALE_POWER_DEFAULT = 16

# FP16 loss scale window
FP16_LOSS_SCALE_WINDOW = "loss_scale_window"
FP16_LOSS_SCALE_WINDOW_DEFAULT = 1000

# FP16 hysteresis
FP16_HYSTERESIS = "hysteresis"
FP16_HYSTERESIS_DEFAULT = 2

# FP16 consecutive hysteresis
FP16_CONSECUTIVE_HYSTERESIS = "consecutive_hysteresis"
FP16_CONSECUTIVE_HYSTERESIS_DEFAULT = False

# FP16 min loss scale
FP16_MIN_LOSS_SCALE = "min_loss_scale"
FP16_MIN_LOSS_SCALE_DEFAULT = 1

# FP16 master and grads
FP16_MASTER_WEIGHTS_AND_GRADS = "fp16_master_weights_and_grads"
FP16_MASTER_WEIGHTS_AND_GRADS_DEFAULT = False

#########################################
# Apex AMP support
#########################################
# Use Apex AMP for mixed precision support, all parameters (other than 'enabled') will be passed to
# amp.initialize(model, optimizer, **amp_params)
# See apex documentation for supported parameters/features: https://nvidia.github.io/apex/amp.html#apex.amp.initialize
AMP_FORMAT = '''
"amp" {
  "enabled: true,
  "opt_level": "O1",
  ...
}
'''
AMP = "amp"

AMP_ENABLED = "enabled"
AMP_ENABLED_DEFAULT = False

#########################################
# Gradient clipping
#########################################
# Gradient clipping. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
GRADIENT_CLIPPING_FORMAT = '''
Gradient clipping should be enabled as:
"gradient_clipping": 1.0
'''
GRADIENT_CLIPPING = 'gradient_clipping'
GRADIENT_CLIPPING_DEFAULT = 0.

#########################################
# Communication data type
#########################################
# Supported types: ['none', 'fp16', 'fp32']
# By default, this feature is not enabled ('none' value)
# Users can configure in ds_config.json as below example:
COMMUNICATION_DATA_TYPE_FORMAT = '''
Communication data type should be set as:
"communication_data_type": "fp32"
'''
COMMUNICATION_DATA_TYPE = "communication_data_type"
COMMUNICATION_DATA_TYPE_DEFAULT = None

#########################################
# Scale/predivide gradients before allreduce
#########################################
# Prescale gradients. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
PRESCALE_GRADIENTS_FORMAT = '''
Gradient prescaling should be enabled as:
"prescale_gradients": true
'''
PRESCALE_GRADIENTS = "prescale_gradients"
PRESCALE_GRADIENTS_DEFAULT = False

GRADIENT_PREDIVIDE_FACTOR_FORMAT = '''
Gradient predivide factor should be enabled as:
"gradient_predivide_factor": 1.0
'''
GRADIENT_PREDIVIDE_FACTOR = "gradient_predivide_factor"
GRADIENT_PREDIVIDE_FACTOR_DEFAULT = 1.0

#########################################
# Disable AllGather
#########################################
# Disable AllGather. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
DISABLE_ALLGATHER_FORMAT = '''
Disable AllGather should be enabled as:
"disable_allgather": true
'''
DISABLE_ALLGATHER = "disable_allgather"
DISABLE_ALLGATHER_DEFAULT = False

#########################################
# Dump DeepSpeed state
#########################################
# Dump State. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
DUMP_STATE_FORMAT = '''
Dump state should be enabled as:
"dump_state": true
'''
DUMP_STATE = 'dump_state'
DUMP_STATE_DEFAULT = False

#########################################
# Vocabulary size
#########################################
# Vocabulary size.
# Users can configure in ds_config.json as below example:
VOCABULARY_SIZE_FORMAT = '''
Vocabulary size can be specified as:
"vocabulary_size": 1024
'''
VOCABULARY_SIZE = 'vocabulary_size'
VOCABULARY_SIZE_DEFAULT = None

#########################################
# Wall block breakdown
#########################################
# Wall clock breakdown. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
WALL_CLOCK_BREAKDOWN_FORMAT = '''
Wall block breakdown should be enabled as:
"wall_clock_breakdown": true
'''
WALL_CLOCK_BREAKDOWN = 'wall_clock_breakdown'
WALL_CLOCK_BREAKDOWN_DEFAULT = False

MEMORY_BREAKDOWN = 'memory_breakdown'
MEMORY_BREAKDOWN_DEFAULT = False

#########################################
# Eigenvalue
#########################################
# Eigenvalue computation. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
EIGENVALUE_FORMAT = '''
Tensorboard can be specified as:
"eigenvalue": {
  "enabled": true,
  "verbose": true,
  "max_iter": 100,
  "tol": 1e-2,
  "stability": 1e-6
}
'''
EIGENVALUE = "eigenvalue"

# Tensorboard enable signal
EIGENVALUE_ENABLED = "enabled"
EIGENVALUE_ENABLED_DEFAULT = False

EIGENVALUE_VERBOSE = "verbose"
EIGENVALUE_VERBOSE_DEFAULT = False

EIGENVALUE_MAX_ITER = "max_iter"
EIGENVALUE_MAX_ITER_DEFAULT = 100

EIGENVALUE_TOL = "tol"
EIGENVALUE_TOL_DEFAULT = 1e-2

EIGENVALUE_STABILITY = "stability"
EIGENVALUE_STABILITY_DEFAULT = 1e-6

EIGENVALUE_GAS_BOUNDARY_RESOLUTION = "gas_boundary_resolution"
EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT = 1

EIGENVALUE_LAYER_NAME = "layer_name"
EIGENVALUE_LAYER_NAME_DEFAULT = "bert.encoder.layer"

EIGENVALUE_LAYER_NUM = "layer_num"
EIGENVALUE_LAYER_NUM_DEFAULT = 0

#########################################
# Progressive Layer Drop (PLD)
#########################################
PROGRESSIVE_LAYER_DROP = "progressive_layer_drop"

# PLD enable signal
PLD_ENABLED = "enabled"
PLD_ENABLED_DEFAULT = False

PLD_THETA = "theta"
PLD_THETA_DEFAULT = 1.0

PLD_GAMMA = "gamma"
PLD_GAMMA_DEFAULT = 0.001


#########################################
# Validation modes
#########################################
class ValidationMode:
    WARN = "WARN"
    IGNORE = "IGNORE"
    FAIL = "FAIL"


#########################################
# Checkpoint config params
#########################################
# "checkpoint": {
#   tag_validation=["Ignore"|"Warn"|"Fail"]
#   load_universal=false
#   use_node_local_storage=false
#   parallel_write: {
#     pipeline_stage: [True|False]
#   }
# }
CHECKPOINT = "checkpoint"
CHECKPOINT_TAG_VALIDATION = "tag_validation"
CHECKPOINT_TAG_VALIDATION_DEFAULT = ValidationMode.WARN
CHECKPOINT_TAG_VALIDATION_MODES = [ValidationMode.WARN, ValidationMode.IGNORE, ValidationMode.FAIL]

LOAD_UNIVERSAL_CHECKPOINT = "load_universal"
LOAD_UNIVERSAL_CHECKPOINT_DEFAULT = False

USE_NODE_LOCAL_STORAGE_CHECKPOINT = "use_node_local_storage"
USE_NODE_LOCAL_STORAGE_CHECKPOINT_DEFAULT = False

CHECKPOINT_PARALLEL_WRITE = "parallel_write"
CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE = "pipeline_stage"
CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE_DEFAULT = False

#########################################
# Data types config params
#########################################
# "data_types": {
#   grad_accum_dtype=["bf16"|"fp16"|"fp32"]
#   }
# }

DATA_TYPES = "data_types"
GRAD_ACCUM_DTYPE = "grad_accum_dtype"
GRAD_ACCUM_DTYPE_DEFAULT = None

#########################################
# Drop the last incomplete Batch
# #########################################
# dataloader_drop_last. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
DATALOADER_DROP_LAST_FORMAT = '''
The last incomplete batch can be dropped by setting:
"dataloader_drop_last": True
'''
DATALOADER_DROP_LAST = "dataloader_drop_last"
DATALOADER_DROP_LAST_DEFAULT = False

#########################################
# PIPELINE PARALLELISM
#########################################
PIPE_REPLICATED = 'ds_pipe_replicated'

#########################################
# DATA PARALLELISM
#########################################
DATA_PARALLEL_GROUP = "data_parallel_group"
GLOBAL_RANK = "global_rank"
