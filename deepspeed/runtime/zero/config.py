"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject
from deepspeed.utils import logger
from .constants import *
from .offload_constants import *
from .offload_config import get_offload_param_config, get_default_offload_param_config, \
    get_offload_optimizer_config, get_default_offload_optimizer_config


class DeepSpeedZeroConfig(DeepSpeedConfigObject):
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

        self.elastic_checkpoint = None

        #Offload Specific Parameters
        self.offload_param = None
        self.offload_optimizer = None
        self.sub_group_size = None

        #Stage3 Specific Parameters
        self.prefetch_bucket_size = None
        self.param_persistence_threshold = None
        self.max_live_parameters = None
        self.max_reuse_distance = None
        self.gather_fp16_weights_on_model_save = None

        self.ignore_unused_parameters = None

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

    def _sanity_check(self, zero_config_dict):
        deprecated_dict = dict(
            ZERO_OPTIMIZATION_CPU_OFFLOAD=ZERO_OPTIMIZATION_OFFLOAD_OPTIMIZER,
            ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS=ZERO_OPTIMIZATION_OFFLOAD_PARAM,
            ZERO_OPTIMIZATION_CPU_OFFLOAD_USE_PIN_MEMORY=
            f'{ZERO_OPTIMIZATION_OFFLOAD_PARAM} or {ZERO_OPTIMIZATION_OFFLOAD_OPTIMIZER}'
        )

        for old_key, new_key in deprecated_dict.items():
            if old_key in zero_config_dict:
                logger.warning(
                    f'DeepSpeedConfig: {old_key} is deprecated. Please use {new_key}.')

    def _initialize(self, zero_config_dict):
        self._sanity_check(zero_config_dict)

        self.stage = get_scalar_param(zero_config_dict,
                                      ZERO_OPTIMIZATION_STAGE,
                                      ZERO_OPTIMIZATION_STAGE_DEFAULT)

        self.contiguous_gradients = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS,
            ZERO3_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT
            if self.stage == ZERO_OPTIMIZATION_WEIGHTS else
            ZERO_OPTIMIZATION_CONTIGUOUS_GRADIENTS_DEFAULT)

        self.reduce_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE,
            ZERO_OPTIMIZATION_REDUCE_BUCKET_SIZE_DEFAULT)

        self.reduce_scatter = get_scalar_param(zero_config_dict,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER,
                                               ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT)

        self.overlap_comm = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_OVERLAP_COMM,
            ZERO3_OPTIMIZATION_OVERLAP_COMM_DEFAULT
            if self.stage == ZERO_OPTIMIZATION_WEIGHTS else
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

        self.elastic_checkpoint = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_ELASTIC_CHECKPOINT,
            ZERO_OPTIMIZATION_ELASTIC_CHECKPOINT_DEFAULT)

        if ZERO_OPTIMIZATION_CPU_OFFLOAD in zero_config_dict:
            cpu_offload_optimizer = get_scalar_param(
                zero_config_dict,
                ZERO_OPTIMIZATION_CPU_OFFLOAD,
                ZERO_OPTIMIZATION_CPU_OFFLOAD_DEFAULT)
            if cpu_offload_optimizer:
                self.offload_optimizer = get_default_offload_optimizer_config()
        else:
            self.offload_optimizer = get_offload_optimizer_config(zero_config_dict)

        if ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS in zero_config_dict:
            cpu_offload_params = get_scalar_param(
                zero_config_dict,
                ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS,
                ZERO_OPTIMIZATION_CPU_OFFLOAD_PARAMS_DEFAULT)
            if cpu_offload_params:
                self.offload_param = get_default_offload_param_config()
        else:
            self.offload_param = get_offload_param_config(zero_config_dict)

        self.sub_group_size = get_scalar_param(zero_config_dict,
                                               ZERO_OPTIMIZATION_SUB_GROUP_SIZE,
                                               ZERO_OPTIMIZATION_SUB_GROUP_SIZE_DEFAULT)

        self.max_live_parameters = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS,
            ZERO_OPTIMIZATION_MAX_LIVE_PARAMETERS_DEFAULT)

        self.max_reuse_distance = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE,
            ZERO_OPTIMIZATION_MAX_REUSE_DISTANCE_DEFAULT)

        self.prefetch_bucket_size = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE,
            ZERO_OPTIMIZATION_PREFETCH_BUCKET_SIZE_DEFAULT)

        self.param_persistence_threshold = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD,
            ZERO_OPTIMIZATION_PARAM_PERSISTENCE_THRESHOLD_DEFAULT)

        self.gather_fp16_weights_on_model_save = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_GATHER_FP16_WEIGHTS_ON_MODEL_SAVE,
            ZERO_OPTIMIZATION_GATHER_FP16_WEIGHTS_ON_MODEL_SAVE_DEFAULT)

        self.ignore_unused_parameters = get_scalar_param(
            zero_config_dict,
            ZERO_OPTIMIZATION_IGNORE_UNUSED_PARAMETERS,
            ZERO_OPTIMIZATION_IGNORE_UNUSED_PARAMETERS_DEFAULT)

        self.legacy_stage1 = get_scalar_param(zero_config_dict,
                                              ZERO_OPTIMIZATION_LEGACY_STAGE1,
                                              ZERO_OPTIMIZATION_LEGACY_STAGE1_DEFAULT)
