'''Copyright The Microsoft DeepSpeed Team'''
from deepspeed.accelerator import get_accelerator

DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_TYPE = 'float'
DEFAULT_BACKEND = get_accelerator().communication_backend_name()
DEFAULT_UNIT = 'Gbps'
DEFAULT_DIST = 'deepspeed'
DEFAULT_MAXSIZE = 24
