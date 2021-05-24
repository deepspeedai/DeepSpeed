'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
from datetime import timedelta

#############################################
# Torch distributed constants
#############################################
TORCH_DISTRIBUTED_DEFAULT_PORT = 29500

# Default process group wide timeout, if applicable.
# This only applies to the gloo and nccl backends
# (only if NCCL_BLOCKING_WAIT or NCCL_ASYNC_ERROR_HANDLING is set to 1).
# To make an attempt at backwards compatibility with THD, we use an
# extraordinarily high default timeout, given that THD did not have timeouts.
default_pg_timeout = timedelta(minutes=30)

INFERENCE_GENERIC_MODE = 'generic'
INFERENCE_SPECIALIZED_MODE = 'specialized'
