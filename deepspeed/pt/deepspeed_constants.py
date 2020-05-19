"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

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

#############################################
# Torch distributed constants
#############################################
TORCH_DISTRIBUTED_DEFAULT_PORT = "29500"

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
# FP16 support
#########################################
# FP16 feature. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FP16_FORMAT = '''
FP16 parameters should be of the format:
"fp16": {
  "enabled": true,
  "loss_scale": 0,
  "initial_scale_power": 32,
  "loss_scale_window": 1000,
  "hysteresis": 2,
  "min_loss_scale": 1
}
'''
FP16 = "fp16"

FP16_ENABLED = "enabled"
FP16_ENABLED_DEFAULT = False

# FP16 loss scale, zero means using dynamic scaling
FP16_LOSS_SCALE = "loss_scale"
FP16_LOSS_SCALE_DEFAULT = 0

# FP16 initial dynamic scale loss power
FP16_INITIAL_SCALE_POWER = "initial_scale_power"
FP16_INITIAL_SCALE_POWER_DEFAULT = 32

# FP16 loss scale window
FP16_LOSS_SCALE_WINDOW = "loss_scale_window"
FP16_LOSS_SCALE_WINDOW_DEFAULT = 1000

# FP16 hysteresis
FP16_HYSTERESIS = "hysteresis"
FP16_HYSTERESIS_DEFAULT = 2

# FP16 min loss scale
FP16_MIN_LOSS_SCALE = "min_loss_scale"
FP16_MIN_LOSS_SCALE_DEFAULT = 1

#########################################
# Gradient clipping
#########################################
# Gradient clipping. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
GRADIENT_CLIPPING_FORMAT = '''
Dump state should be enabled as:
"gradient_clipping": 1.0
'''
GRADIENT_CLIPPING = 'gradient_clipping'
GRADIENT_CLIPPING_DEFAULT = 0.

#########################################
# ZeRO optimization
#########################################
# ZeRO optimization. By default, this optimization is not enabled.
# Users have to configure the desired optimization (0 means disabled) in params.json as below example:
ZERO_FORMAT = '''
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": [0|1|2],
  "zero_all_gather_size": 200
}
'''

ZERO_OPTIMIZATION = 'zero_optimization'
ZERO_OPTIMIZATION_DEFAULT = 0
ZERO_OPTIMIZATION_OPTIMIZER_STATES = 1
ZERO_OPTIMIZATION_GRADIENTS = 2
ZERO_OPTIMIZATION_WEIGHTS = 3
MAX_STAGE_ZERO_OPTIMIZATION = ZERO_OPTIMIZATION_GRADIENTS

ZERO_REDUCE_SCATTER = "zero_reduce_scatter"
ZERO_REDUCE_SCATTER_DEFAULT = True

ZERO_MAX_ELEMENTS_PER_COMM = "zero_max_elements_per_comm"
ZERO_MAX_ELEMENTS_PER_COMM_DEFAULT = 5e8

ALLGATHER_SIZE = 'allgather_size'
ALLGATHER_SIZE_DEFAULT = 500000000

#########################################
# FP32 AllReduce
#########################################
# FP32 All reduce. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
FP32_ALLREDUCE_FORMAT = '''
FP32 Allreduce should be enabled as:
"fp32_allreduce": true
'''
FP32_ALLREDUCE = "fp32_allreduce"
FP32_ALLREDUCE_DEFAULT = False

#########################################
# Scale gradients before allreduce
#########################################
# Prescale gradients. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
PRESCALE_GRADIENTS_FORMAT = '''
Gradient prescaling should be enabled as:
"prescale_gradients": true
'''
PRESCALE_GRADIENTS = "prescale_gradients"
PRESCALE_GRADIENTS_DEFAULT = False

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
# Tensorboard
#########################################
# Tensorboard. By default, this feature is not enabled.
# Users can configure in ds_config.json as below example:
TENSORBOARD_FORMAT = '''
Tensorboard can be specified as:
"tensorboard": {
  "enabled": true,
  "output_path": "/home/myname/foo",
  "job_name": "model_lr2e-5_epoch3_seed2_seq64"
}
'''
TENSORBOARD = "tensorboard"

# Tensorboard enable signal
TENSORBOARD_ENABLED = "enabled"
TENSORBOARD_ENABLED_DEFAULT = False

# Tensorboard output path
TENSORBOARD_OUTPUT_PATH = "output_path"
TENSORBOARD_OUTPUT_PATH_DEFAULT = ""

# Tensorboard job name
TENSORBOARD_JOB_NAME = "job_name"
TENSORBOARD_JOB_NAME_DEFAULT = "DeepSpeedJobName"
