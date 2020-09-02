"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import get_scalar_param
from deepspeed.utils import logger

#########################################
# ZeRO optimization
#########################################
# ZeRO optimization. By default, this optimization is not enabled.
# Users have to configure the desired optimization (0 means disabled) in params.json as below example:
ZERO_FORMAT = '''
ZeRO optimization should be enabled as:
"session_params": {
  "zero_optimization": {
    "stage": [0|1|2],
    "allgather_partitions": [true|false],
    "allgather_bucket_size": 500000000,
    "reduce_scatter": [true|false],
    "contiguous_gradients" : [true|false]
    "overlap_comm": [true|false],
    "reduce_bucket_size": 500000000
    "load_from_fp32_weights": [true|false]
    }
}
'''

ZERO_OPTIMIZATION = 'zero_optimization'
ZERO_OPTIMIZATION_DISABLED = 0
ZERO_OPTIMIZATION_OPTIMIZER_STATES = 1
ZERO_OPTIMIZATION_GRADIENTS = 2
ZERO_OPTIMIZATION_WEIGHTS = 3
MAX_STAGE_ZERO_OPTIMIZATION = ZERO_OPTIMIZATION_GRADIENTS

ZERO_OPTIMIZATION_STAGE = 'stage'
ZERO_OPTIMIZATION_STAGE_1 = 'stage_1'
ZERO_OPTIMIZATION_STAGE_2 = 'stage_2'
ZERO_OPTIMIZATION_STAGE_3 = 'stage_3'

ZERO_OPTIMIZATION_STAGE_DEFAULT = ZERO_OPTIMIZATION_DISABLED

ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS = 'allgather_partitions'
ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT = True

ZERO_OPTIMIZATION_REDUCE_SCATTER = 'reduce_scatter'
ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT = True

ZERO_OPTIMIZATION_OVERLAP_COMM = 'overlap_comm'
ZERO_OPTIMIZATION_OVERLAP_COMM_DEFAULT = False

ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS = 'contiguous_gradients'
ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT = False

ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE = 'reduce_bucket_size'
ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT = 500000000

ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE = 'allgather_bucket_size'
ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT = 500000000
ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEPRECATED = 'allgather_size'
ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS = 'load_from_fp32_weights'
ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS_DEFAULT = True

ZERO_OPTIMIZATION_DEFAULT = {
    ZERO_OPTIMIZATION_STAGE:
    ZERO_OPTIMIZATION_STAGE_DEFAULT,
    ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS:
    ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT,
    ZERO_OPTIMIZATION_REDUCE_SCATTER:
    ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT,
    ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE:
    ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT,
    ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS:
    ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT,
    ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE:
    ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT,
    ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS:
    ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS_DEFAULT
}


class DeepSpeedZeroConfig(object):
    def __init__(self, param_dict):
        super(DeepSpeedZeroConfig, self).__init__()

        self.stage = None
        self.contiguous_gradients = None
        self.reduce_scatter = None
        self.reduce_bucket_size = None
        self.allgather_partitions = None
        self.allgather_bucket_size = None
        self.overlap_comm = None
        self.load_from_fp32_weights = None

        if ZERO_OPTIMIZATION in param_dict.keys():
            zero_config_dict = param_dict[ZERO_OPTIMIZATION]
            if type(zero_config_dict) is bool:
                zero_config_dict = self.read_zero_config_deprecated(param_dict)
        else:
            zero_config_dict = ZERO_OPTIMIZATION_DEFAULT

        self._initialize(zero_config_dict)

    def read_zero_config_deprecated(self, param_dict):
        zero_config_dict = {}
        zero_config_dict[
            ZERO_OPTIMIZATION_STAGE] = 1 if param_dict[ZERO_OPTIMIZATION] else 0
        if zero_config_dict[ZERO_OPTIMIZATION_STAGE] > 0:
            zero_config_dict[ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE] = get_scalar_param(
                param_dict,
                ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEPRECATED,
                ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)

        logger.warning(
            'DeepSpeedConfig: this format of ZeRO optimization setup is deprecated. Please use the following format: {}'
            .format(ZERO_FORMAT))
        return zero_config_dict

    """
    For json serialization
    """

    def repr(self):
        return self.__dict__

    def _initialize(self, zero_config_dict):
        self.stage = get_scalar_param(zero_config_dict,
                                      ZERO_OPTIMIZATION_STAGE,
                                      ZERO_OPTIMIZATION_STAGE_DEFAULT)

        self.contiguous_gradients = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS,
            ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT)

        self.reduce_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT)

        self.reduce_scatter = get_scalar_param(zero_config_dict,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT)

        self.overlap_comm = get_scalar_param(zero_config_dict,
                                             ZERO_OPTIMIZATION_OVERLAP_COMM,
                                             ZERO_OPTIMIZATION_OVERLAP_COMM_DEFAULT)

        self.allgather_partitions = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS,
            ZERO_OPTIMIZATION_ALLGATHER_PARTITIONS_DEFAULT)

        self.allgather_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE,
            ZERO_OPTIMIZATION_ALLGATHER_BUCKET_SIZE_DEFAULT)
        self.load_from_fp32_weights = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS,
            ZERO_OPTIMIZATION_LOAD_FROM_FP32_WEIGHTS_DEFAULT)
