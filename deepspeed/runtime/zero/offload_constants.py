"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""
#########################################
# TENSOR OFFLOADING
#########################################
OFFLOAD_CPU_DEVICE = "cpu"
OFFLOAD_NVME_DEVICE = "nvme"

#########################################
# PARAM TENSOR OFFLOADING
#########################################
OFFLOAD_PARAM_FORMAT = '''
"offload_param": {
  "device": [cpu|nvme],
  "nvme_path": "/local_nvme",
  "buffer_count": 5,
  "buffer_size": 1e8,
  "max_in_cpu": 1e9,
  "pin_memory": [true|false]
}
'''
OFFLOAD_PARAM = "offload_param"
OFFLOAD_PARAM_DEVICE = "device"
OFFLOAD_PARAM_DEVICE_DEFAULT = OFFLOAD_CPU_DEVICE
OFFLOAD_PARAM_NVME_PATH = "nvme_path"
OFFLOAD_PARAM_NVME_PATH_DEFAULT = None
OFFLOAD_PARAM_BUFFER_COUNT = "buffer_count"
OFFLOAD_PARAM_BUFFER_COUNT_DEFAULT = 5
OFFLOAD_PARAM_BUFFER_SIZE = "buffer_size"
OFFLOAD_PARAM_BUFFER_SIZE_DEFAULT = 1e8
OFFLOAD_PARAM_MAX_IN_CPU = "max_in_cpu"
OFFLOAD_PARAM_MAX_IN_CPU_DEFAULT = 1e9
OFFLOAD_PARAM_PIN_MEMORY = "pin_memory"
OFFLOAD_PARAM_PIN_MEMORY_DEFAULT = False

#########################################
# OPTIMIZER TENSOR OFFLOADING
#########################################
OFFLOAD_OPTIMIZER_FORMAT = '''
"offload_optimizer": {
  "device": [cpu|nvme],
  "nvme_path": "/local_nvme",
  "buffer_count": 4,
  "pin_memory": [true|false],
  "pipeline_read": false,
  "pipeline_write": false,
  "fast_init": false
}
'''
OFFLOAD_OPTIMIZER = "offload_optimizer"
OFFLOAD_OPTIMIZER_DEVICE = "device"
OFFLOAD_OPTIMIZER_DEVICE_DEFAULT = OFFLOAD_CPU_DEVICE
OFFLOAD_OPTIMIZER_NVME_PATH = "nvme_path"
OFFLOAD_OPTIMIZER_NVME_PATH_DEFAULT = None
OFFLOAD_OPTIMIZER_BUFFER_COUNT = "buffer_count"
OFFLOAD_OPTIMIZER_BUFFER_COUNT_DEFAULT = 4
OFFLOAD_OPTIMIZER_PIN_MEMORY = "pin_memory"
OFFLOAD_OPTIMIZER_PIN_MEMORY_DEFAULT = False
OFFLOAD_OPTIMIZER_PIPELINE_READ = "pipeline_read"
OFFLOAD_OPTIMIZER_PIPELINE_READ_DEFAULT = False
OFFLOAD_OPTIMIZER_PIPELINE_WRITE = "pipeline_write"
OFFLOAD_OPTIMIZER_PIPELINE_WRITE_DEFAULT = False
OFFLOAD_OPTIMIZER_PIPELINE = "pipeline"
OFFLOAD_OPTIMIZER_FAST_INIT = "fast_init"
OFFLOAD_OPTIMIZER_FAST_INIT_DEFAULT = False
