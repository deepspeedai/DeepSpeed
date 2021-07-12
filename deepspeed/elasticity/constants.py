"""
Copyright 2020 The Microsoft DeepSpeed Team
"""

#########################################
# Elasticity
#########################################
''' Elasticity Utility in DeepSpeed can be used to create highly elastic jobs compatible
with a large number of GPUs. For elastic jobs, DeepSpeed will provide a batch size that
can support a large number of GPUs based on the user specified parameters
'''
FORMAT = '''
Elasticity should be enabled as:
"elasticity": {
  "enabled": true,
  "max_train_batch_size": 2000,
  "micro_batch_sizes": [2,4,6],
  "min_gpus": 1,
  "max_gpus" : 10000,
  "min_time": 20,
  "prefer_larger_batch": true,
  "ignore_non_elastic_batch_info": false,
  "version": 0.1
}
'''

ELASTICITY = 'elasticity'

# Current elasticity version
LATEST_ELASTICITY_VERSION = 0.1

ENABLED = 'enabled'
ENABLED_DEFAULT = False

# Max acceptable train_batch_size
MAX_ACCEPTABLE_BATCH_SIZE = 'max_train_batch_size'
MAX_ACCEPTABLE_BATCH_SIZE_DEFAULT = 2000

# Acceptable micro batch sizes, same as train_micro_batch_size_per_gpu
MICRO_BATCHES = 'micro_batch_sizes'
MICRO_BATCHES_DEFAULT = [2, 4, 6]

# Min/max of GPUs to search over
MIN_GPUS = 'min_gpus'
MIN_GPUS_DEFAULT = 1
MAX_GPUS = 'max_gpus'
MAX_GPUS_DEFAULT = 10000

# Minimum running time (minutes) before the scheduler will scale us, 0 implies it's unknown
MIN_TIME = "min_time"
MIN_TIME_DEFAULT = 0

# When finding a suitable batch size, attempt to find one that is closest
# to the max train batch size given.
PREFER_LARGER_BATCH = 'prefer_larger_batch'
PREFER_LARGER_BATCH_DEFAULT = True

# In order to reduce confusion, if elastic mode is enabled we
# require (via assert) that no batch info is set outside of the
# elastic config. You can turn off this assert via this config
# but keep in mind that all batch info defined outside the
# elastic mode *will be ignored*.
IGNORE_NON_ELASTIC_BATCH_INFO = 'ignore_non_elastic_batch_info'
IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT = False

# Version of elastic logic to use
VERSION = "version"
VERSION_DEFAULT = LATEST_ELASTICITY_VERSION

# Minimum deepspeed version to use elasticity
MINIMUM_DEEPSPEED_VERSION = "0.3.8"

# Environment variable storing elastic config from resource scheduler
DEEPSPEED_ELASTICITY_CONFIG = "DEEPSPEED_ELASTICITY_CONFIG"
