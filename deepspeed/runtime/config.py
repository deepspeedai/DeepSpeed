# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from typing import Union
from enum import Enum

import torch
import json
import hjson
import copy
import base64

from .constants import *
from .fp16.loss_scaler import (
    INITIAL_LOSS_SCALE,
    SCALE_WINDOW,
    DELAYED_SHIFT,
    CONSECUTIVE_HYSTERESIS,
    MIN_LOSS_SCALE,
)
from .config_utils import (
    get_scalar_param,
    dict_raise_error_on_duplicate_keys,
    ScientificNotationEncoder,
)
from .zero.config import get_zero_config, ZeroStageEnum
from .activation_checkpointing.config import DeepSpeedActivationCheckpointingConfig
from ..comm.config import DeepSpeedCommsConfig
from ..monitor.config import get_monitor_config
from ..inference.config import WeightQuantConfig

from deepspeed import comm as dist
from deepspeed.runtime.config_utils import DeepSpeedConfigModel

from ..git_version_info import version as __version__
from ..utils import logger

from ..elasticity import (
    elasticity_enabled,
    compute_elastic_config,
    ensure_immutable_elastic_config,
)
from ..elasticity.config import ElasticityConfigError
from ..elasticity.constants import (
    ELASTICITY,
    IGNORE_NON_ELASTIC_BATCH_INFO,
    IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT,
    MODEL_PARALLEL_SIZE,
    MODEL_PARALLEL_SIZE_DEFAULT,
    NUM_GPUS_PER_NODE,
    NUM_GPUS_PER_NODE_DEFAULT,
)

from ..profiling.config import DeepSpeedFlopsProfilerConfig
from ..autotuning.config import DeepSpeedAutotuningConfig
from ..nebula.config import DeepSpeedNebulaConfig

from ..compression.config import get_compression_config, get_quantize_enabled
from ..compression.constants import *
from .swap_tensor.aio_config import get_aio_config

from .data_pipeline.config import get_data_efficiency_enabled, get_data_efficiency_config, get_curriculum_enabled_legacy, get_curriculum_params_legacy
from .data_pipeline.constants import *

TENSOR_CORE_ALIGN_SIZE = 8

ADAGRAD_OPTIMIZER = 'adagrad'
ADAM_OPTIMIZER = 'adam'
ADAMW_OPTIMIZER = 'adamw'
LAMB_OPTIMIZER = 'lamb'
ONEBIT_ADAM_OPTIMIZER = 'onebitadam'
ZERO_ONE_ADAM_OPTIMIZER = 'zerooneadam'
ONEBIT_LAMB_OPTIMIZER = 'onebitlamb'
MUADAM_OPTIMIZER = 'muadam'
MUADAMW_OPTIMIZER = 'muadamw'
MUSGD_OPTIMIZER = 'musgd'
LION_OPTIMIZER = 'lion'
DEEPSPEED_OPTIMIZERS = [
    ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER,
    ZERO_ONE_ADAM_OPTIMIZER, MUADAM_OPTIMIZER, MUADAMW_OPTIMIZER, MUSGD_OPTIMIZER, LION_OPTIMIZER
]

# extra optimizer parameters for adam/adamw
TORCH_ADAM_PARAM = "torch_adam"

# default to adamw logic for adam/adamw optimizers unless user explicitly opts out
ADAM_W_MODE = "adam_w_mode"
ADAM_W_MODE_DEFAULT = True


class DeepSpeedConfigError(Exception):
    pass


class DtypeEnum(Enum):
    # The torch dtype must always be the first value (so we return torch.dtype)
    fp16 = torch.float16, "torch.float16", "fp16", "float16", "half"
    fp32 = torch.float32, "torch.float32", "fp32", "float32", "float"
    int8 = torch.int8, "torch.int8", "int8"
    bf16 = torch.bfloat16, "torch.bfloat16", "bf16", "bfloat16"

    # Copied from https://stackoverflow.com/a/43210118
    # Allows us to use multiple values for each Enum index and returns first
    # listed value when Enum is called
    def __new__(cls, *values):
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self):
        return "<%s.%s: %s>" % (
            self.__class__.__name__,
            self._name_,
            ", ".join([repr(v) for v in self._all_values]),
        )


def get_pld_enabled(param_dict):
    if PROGRESSIVE_LAYER_DROP in param_dict.keys():
        return get_scalar_param(param_dict[PROGRESSIVE_LAYER_DROP], PLD_ENABLED, PLD_ENABLED_DEFAULT)
    else:
        return False


def get_pld_params(param_dict):
    if PROGRESSIVE_LAYER_DROP in param_dict.keys():
        pld_params = copy.copy(param_dict[PROGRESSIVE_LAYER_DROP])
        pld_params.pop(PLD_ENABLED)
        return pld_params
    else:
        return False


def get_amp_enabled(param_dict):
    if AMP in param_dict.keys():
        return get_scalar_param(param_dict[AMP], AMP_ENABLED, AMP_ENABLED_DEFAULT)
    else:
        return False


def get_amp_params(param_dict):
    if AMP in param_dict.keys():
        amp_params = copy.copy(param_dict[AMP])
        amp_params.pop(AMP_ENABLED)
        return amp_params
    else:
        return False


def get_fp16_enabled(param_dict):
    if FP16 in param_dict.keys():
        return get_scalar_param(param_dict[FP16], FP16_ENABLED, FP16_ENABLED_DEFAULT)
    else:
        return False


def get_bfloat16_enabled(param_dict):
    for key in [BFLOAT16, BFLOAT16_OLD]:
        if key in param_dict.keys():
            return get_scalar_param(param_dict[key], BFLOAT16_ENABLED, BFLOAT16_ENABLED_DEFAULT)
    return False


def get_fp16_master_weights_and_grads_enabled(param_dict):
    if get_fp16_enabled(param_dict):
        return get_scalar_param(param_dict[FP16], FP16_MASTER_WEIGHTS_AND_GRADS, FP16_MASTER_WEIGHTS_AND_GRADS_DEFAULT)
    else:
        return False


def get_fp16_auto_cast(param_dict):
    if get_fp16_enabled(param_dict):
        return get_scalar_param(param_dict[FP16], FP16_AUTO_CAST, FP16_AUTO_CAST_DEFAULT)


def get_loss_scale(param_dict):
    if get_fp16_enabled(param_dict):
        return get_scalar_param(param_dict[FP16], FP16_LOSS_SCALE, FP16_LOSS_SCALE_DEFAULT)
    elif get_bfloat16_enabled(param_dict):
        return 1.0
    else:
        return FP16_LOSS_SCALE_DEFAULT


def get_initial_dynamic_scale(param_dict):
    if get_fp16_enabled(param_dict):
        initial_scale_power = get_scalar_param(param_dict[FP16], FP16_INITIAL_SCALE_POWER,
                                               FP16_INITIAL_SCALE_POWER_DEFAULT)
    elif get_bfloat16_enabled(param_dict):
        initial_scale_power = 0
    else:
        initial_scale_power = FP16_INITIAL_SCALE_POWER_DEFAULT

    return 2**initial_scale_power


def get_dynamic_loss_scale_args(param_dict):
    loss_scale_args = None
    if get_fp16_enabled(param_dict):
        fp16_dict = param_dict[FP16]
        dynamic_loss_args = [
            FP16_INITIAL_SCALE_POWER,
            FP16_LOSS_SCALE_WINDOW,
            FP16_MIN_LOSS_SCALE,
            FP16_HYSTERESIS,
            FP16_CONSECUTIVE_HYSTERESIS,
        ]
        if any(arg in list(fp16_dict.keys()) for arg in dynamic_loss_args):
            init_scale = get_scalar_param(fp16_dict, FP16_INITIAL_SCALE_POWER, FP16_INITIAL_SCALE_POWER_DEFAULT)
            scale_window = get_scalar_param(fp16_dict, FP16_LOSS_SCALE_WINDOW, FP16_LOSS_SCALE_WINDOW_DEFAULT)
            delayed_shift = get_scalar_param(fp16_dict, FP16_HYSTERESIS, FP16_HYSTERESIS_DEFAULT)
            consecutive_hysteresis = get_scalar_param(fp16_dict, FP16_CONSECUTIVE_HYSTERESIS,
                                                      FP16_CONSECUTIVE_HYSTERESIS_DEFAULT)
            min_loss_scale = get_scalar_param(fp16_dict, FP16_MIN_LOSS_SCALE, FP16_MIN_LOSS_SCALE_DEFAULT)
            loss_scale_args = {
                INITIAL_LOSS_SCALE: 2**init_scale,
                SCALE_WINDOW: scale_window,
                DELAYED_SHIFT: delayed_shift,
                CONSECUTIVE_HYSTERESIS: consecutive_hysteresis,
                MIN_LOSS_SCALE: min_loss_scale,
            }

    return loss_scale_args


def get_gradient_accumulation_steps(param_dict):
    return get_scalar_param(param_dict, GRADIENT_ACCUMULATION_STEPS, GRADIENT_ACCUMULATION_STEPS_DEFAULT)


def get_sparse_gradients_enabled(param_dict):
    return get_scalar_param(param_dict, SPARSE_GRADIENTS, SPARSE_GRADIENTS_DEFAULT)


def get_communication_data_type(param_dict):
    val = get_scalar_param(param_dict, COMMUNICATION_DATA_TYPE, COMMUNICATION_DATA_TYPE_DEFAULT)
    val = val.lower() if val is not None else val
    if val is None:
        return val  # we must determine it by other parameters
    elif val == "fp32":
        return torch.float32
    elif val == "fp16":
        return torch.float16
    elif val == "bfp16":
        return torch.bfloat16

    raise ValueError(f"Invalid communication_data_type. Supported data types: ['fp16', 'bfp16', 'fp32']. Got: {val}")


def get_prescale_gradients(param_dict):
    return get_scalar_param(param_dict, PRESCALE_GRADIENTS, PRESCALE_GRADIENTS_DEFAULT)


def get_gradient_predivide_factor(param_dict):
    return get_scalar_param(param_dict, GRADIENT_PREDIVIDE_FACTOR, GRADIENT_PREDIVIDE_FACTOR_DEFAULT)


def get_steps_per_print(param_dict):
    return get_scalar_param(param_dict, STEPS_PER_PRINT, STEPS_PER_PRINT_DEFAULT)


def get_disable_allgather(param_dict):
    return get_scalar_param(param_dict, DISABLE_ALLGATHER, DISABLE_ALLGATHER_DEFAULT)


def get_dump_state(param_dict):
    return get_scalar_param(param_dict, DUMP_STATE, DUMP_STATE_DEFAULT)


def get_gradient_clipping(param_dict):
    return get_scalar_param(param_dict, GRADIENT_CLIPPING, GRADIENT_CLIPPING_DEFAULT)


def get_sparse_attention(param_dict):
    if SPARSE_ATTENTION in param_dict.keys():
        sparsity = param_dict[SPARSE_ATTENTION]
        mode = get_sparse_attention_mode(sparsity)

        if mode == SPARSE_DENSE_MODE:
            return get_sparse_dense_config(sparsity)
        elif mode == SPARSE_FIXED_MODE:
            return get_sparse_fixed_config(sparsity)
        elif mode == SPARSE_VARIABLE_MODE:
            return get_sparse_variable_config(sparsity)
        elif mode == SPARSE_BIGBIRD_MODE:
            return get_sparse_bigbird_config(sparsity)
        elif mode == SPARSE_BSLONGFORMER_MODE:
            return get_sparse_bslongformer_config(sparsity)
        else:
            raise NotImplementedError(f"Given sparsity mode, {mode}, has not been implemented yet!")

    else:
        return None


def get_sparse_dense_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    return {SPARSE_MODE: SPARSE_DENSE_MODE, SPARSE_BLOCK: block}


def get_sparse_fixed_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT,
    )
    num_local_blocks = get_scalar_param(sparsity, SPARSE_NUM_LOCAL_BLOCKS, SPARSE_NUM_LOCAL_BLOCKS_DEFAULT)
    num_global_blocks = get_scalar_param(sparsity, SPARSE_NUM_GLOBAL_BLOCKS, SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT)
    attention = get_scalar_param(sparsity, SPARSE_ATTENTION_TYPE, SPARSE_ATTENTION_TYPE_DEFAULT)
    horizontal_global_attention = get_scalar_param(
        sparsity,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT,
    )
    num_different_global_patterns = get_scalar_param(
        sparsity,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS_DEFAULT,
    )

    return {
        SPARSE_MODE: SPARSE_FIXED_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_LOCAL_BLOCKS: num_local_blocks,
        SPARSE_NUM_GLOBAL_BLOCKS: num_global_blocks,
        SPARSE_ATTENTION_TYPE: attention,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION: horizontal_global_attention,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS: num_different_global_patterns,
    }


def get_sparse_variable_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT,
    )
    num_random_blocks = get_scalar_param(sparsity, SPARSE_NUM_RANDOM_BLOCKS, SPARSE_NUM_RANDOM_BLOCKS_DEFAULT)
    local_window_blocks = get_scalar_param(sparsity, SPARSE_LOCAL_WINDOW_BLOCKS, SPARSE_LOCAL_WINDOW_BLOCKS_DEFAULT)
    global_block_indices = get_scalar_param(sparsity, SPARSE_GLOBAL_BLOCK_INDICES, SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT)
    global_block_end_indices = get_scalar_param(
        sparsity,
        SPARSE_GLOBAL_BLOCK_END_INDICES,
        SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT,
    )
    attention = get_scalar_param(sparsity, SPARSE_ATTENTION_TYPE, SPARSE_ATTENTION_TYPE_DEFAULT)
    horizontal_global_attention = get_scalar_param(
        sparsity,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT,
    )

    return {
        SPARSE_MODE: SPARSE_VARIABLE_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_RANDOM_BLOCKS: num_random_blocks,
        SPARSE_LOCAL_WINDOW_BLOCKS: local_window_blocks,
        SPARSE_GLOBAL_BLOCK_INDICES: global_block_indices,
        SPARSE_GLOBAL_BLOCK_END_INDICES: global_block_end_indices,
        SPARSE_ATTENTION_TYPE: attention,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION: horizontal_global_attention,
    }


def get_sparse_bigbird_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT,
    )
    num_random_blocks = get_scalar_param(sparsity, SPARSE_NUM_RANDOM_BLOCKS, SPARSE_NUM_RANDOM_BLOCKS_DEFAULT)
    num_sliding_window_blocks = get_scalar_param(
        sparsity,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT,
    )
    num_global_blocks = get_scalar_param(sparsity, SPARSE_NUM_GLOBAL_BLOCKS, SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT)

    return {
        SPARSE_MODE: SPARSE_BIGBIRD_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_RANDOM_BLOCKS: num_random_blocks,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS: num_sliding_window_blocks,
        SPARSE_NUM_GLOBAL_BLOCKS: num_global_blocks,
    }


def get_sparse_bslongformer_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT,
    )
    num_sliding_window_blocks = get_scalar_param(
        sparsity,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT,
    )
    global_block_indices = get_scalar_param(sparsity, SPARSE_GLOBAL_BLOCK_INDICES, SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT)
    global_block_end_indices = get_scalar_param(
        sparsity,
        SPARSE_GLOBAL_BLOCK_END_INDICES,
        SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT,
    )

    return {
        SPARSE_MODE: SPARSE_BSLONGFORMER_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS: num_sliding_window_blocks,
        SPARSE_GLOBAL_BLOCK_INDICES: global_block_indices,
        SPARSE_GLOBAL_BLOCK_END_INDICES: global_block_end_indices,
    }


def get_sparse_attention_mode(param_dict):
    if SPARSE_MODE in param_dict.keys():
        return param_dict[SPARSE_MODE]
    else:
        return SPARSE_MODE_DEFAULT


def get_sparse_attention_type(param_dict):
    if SPARSE_ATTENTION_TYPE in param_dict.keys():
        return param_dict[SPARSE_ATTENTION_TYPE]
    else:
        return SPARSE_ATTENTION_TYPE_DEFAULT


def get_pipeline_config(param_dict):
    """Parses pipeline engine configuration. """
    default_pipeline = {
        "stages": "auto",
        "partition": "best",
        "seed_layers": False,
        "activation_checkpoint_interval": 0,
    }
    config = default_pipeline
    for key, val in param_dict.get("pipeline", {}).items():
        config[key] = val
    return config


def get_optimizer_name(param_dict):
    if OPTIMIZER in param_dict.keys() and TYPE in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][TYPE]
    else:
        return OPTIMIZER_TYPE_DEFAULT


def get_optimizer_params(param_dict):
    if (get_optimizer_name(param_dict) is not None and OPTIMIZER_PARAMS in param_dict[OPTIMIZER].keys()):
        return param_dict[OPTIMIZER][OPTIMIZER_PARAMS]
    else:
        return None


def get_optimizer_gradient_clipping(param_dict):
    optimizer_params = get_optimizer_params(param_dict)
    if optimizer_params is not None and MAX_GRAD_NORM in optimizer_params.keys():
        return optimizer_params[MAX_GRAD_NORM]
    else:
        return None


def get_optimizer_legacy_fusion(param_dict):
    if OPTIMIZER in param_dict.keys() and LEGACY_FUSION in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][LEGACY_FUSION]
    else:
        return LEGACY_FUSION_DEFAULT


def get_zero_allow_untested_optimizer(param_dict):
    return get_scalar_param(param_dict, ZERO_ALLOW_UNTESTED_OPTIMIZER, ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT)


def get_zero_force_ds_cpu_optimizer(param_dict):
    return get_scalar_param(param_dict, ZERO_FORCE_DS_CPU_OPTIMIZER, ZERO_FORCE_DS_CPU_OPTIMIZER_DEFAULT)


def get_scheduler_name(param_dict):
    if SCHEDULER in param_dict.keys() and TYPE in param_dict[SCHEDULER].keys():
        return param_dict[SCHEDULER][TYPE]
    else:
        return SCHEDULER_TYPE_DEFAULT


def get_scheduler_params(param_dict):
    if (get_scheduler_name(param_dict) is not None and SCHEDULER_PARAMS in param_dict[SCHEDULER].keys()):
        return param_dict[SCHEDULER][SCHEDULER_PARAMS]
    else:
        return None


def get_train_batch_size(param_dict):
    return get_scalar_param(param_dict, TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE_DEFAULT)


def get_train_micro_batch_size_per_gpu(param_dict):
    return get_scalar_param(
        param_dict,
        TRAIN_MICRO_BATCH_SIZE_PER_GPU,
        TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT,
    )


def get_wall_clock_breakdown(param_dict):
    return get_scalar_param(param_dict, WALL_CLOCK_BREAKDOWN, WALL_CLOCK_BREAKDOWN_DEFAULT)


def get_memory_breakdown(param_dict):
    return get_scalar_param(param_dict, MEMORY_BREAKDOWN, MEMORY_BREAKDOWN_DEFAULT)


class HybridEngineConfig(DeepSpeedConfigModel):
    enabled: bool = False
    max_out_tokens: int = 512
    inference_tp_size: int = 1
    release_inference_cache: bool = False
    pin_parameters: bool = True
    tp_gather_partition_size: int = 8


def get_hybrid_engine_config(param_dict):
    hybrid_engine_config_dict = param_dict.get("hybrid_engine", {})
    hybrid_engine_config = HybridEngineConfig(**hybrid_engine_config_dict)
    return hybrid_engine_config


def get_eigenvalue_config(param_dict):
    if get_quantize_enabled(param_dict):
        param_dict = param_dict[QUANTIZE_TRAINING]
        assert not get_eigenvalue_enabled(param_dict), "Eigenvalue based MoQ is temporarily disabled"
        return (
            get_eigenvalue_enabled(param_dict),
            get_eigenvalue_verbose(param_dict),
            get_eigenvalue_max_iter(param_dict),
            get_eigenvalue_tol(param_dict),
            get_eigenvalue_stability(param_dict),
            get_eigenvalue_gas_boundary_resolution(param_dict),
            get_eigenvalue_layer_name(param_dict),
            get_eigenvalue_layer_num(param_dict),
        )
    else:
        return (
            EIGENVALUE_ENABLED_DEFAULT,
            EIGENVALUE_VERBOSE_DEFAULT,
            EIGENVALUE_MAX_ITER_DEFAULT,
            EIGENVALUE_TOL_DEFAULT,
            EIGENVALUE_STABILITY_DEFAULT,
            EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT,
            EIGENVALUE_LAYER_NAME_DEFAULT,
            EIGENVALUE_LAYER_NUM_DEFAULT,
        )


def get_eigenvalue_enabled(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_ENABLED, EIGENVALUE_ENABLED_DEFAULT)
    else:
        return EIGENVALUE_ENABLED_DEFAULT


def get_eigenvalue_verbose(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_VERBOSE, EIGENVALUE_VERBOSE_DEFAULT)
    else:
        return EIGENVALUE_VERBOSE_DEFAULT


def get_eigenvalue_max_iter(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_MAX_ITER, EIGENVALUE_MAX_ITER_DEFAULT)
    else:
        return EIGENVALUE_MAX_ITER_DEFAULT


def get_eigenvalue_tol(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_TOL, EIGENVALUE_TOL_DEFAULT)
    else:
        return EIGENVALUE_TOL_DEFAULT


def get_eigenvalue_stability(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_STABILITY, EIGENVALUE_STABILITY_DEFAULT)
    else:
        return EIGENVALUE_STABILITY_DEFAULT


def get_eigenvalue_gas_boundary_resolution(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(
            param_dict[EIGENVALUE],
            EIGENVALUE_GAS_BOUNDARY_RESOLUTION,
            EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT,
        )
    else:
        return EIGENVALUE_GAS_BOUNDARY_RESOLUTION_DEFAULT


def get_eigenvalue_layer_name(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_LAYER_NAME, EIGENVALUE_LAYER_NAME_DEFAULT)
    else:
        return EIGENVALUE_LAYER_NAME_DEFAULT


def get_eigenvalue_layer_num(param_dict):
    if EIGENVALUE in param_dict.keys():
        return get_scalar_param(param_dict[EIGENVALUE], EIGENVALUE_LAYER_NUM, EIGENVALUE_LAYER_NUM_DEFAULT)
    else:
        return EIGENVALUE_LAYER_NUM_DEFAULT


def get_checkpoint_params(param_dict):
    return param_dict.get(CHECKPOINT, {})


def get_data_types_params(param_dict):
    return param_dict.get(DATA_TYPES, {})


def get_checkpoint_tag_validation_mode(checkpoint_params):
    tag_validation_mode = checkpoint_params.get(CHECKPOINT_TAG_VALIDATION, CHECKPOINT_TAG_VALIDATION_DEFAULT)
    tag_validation_mode = tag_validation_mode.upper()
    if tag_validation_mode in CHECKPOINT_TAG_VALIDATION_MODES:
        return tag_validation_mode
    else:
        raise DeepSpeedConfigError(
            "Checkpoint config contains invalid tag_validation "
            f"value of {tag_validation_mode}, expecting one of {CHECKPOINT_TAG_VALIDATION_MODES}")


def get_checkpoint_parallel_write_pipeline(checkpoint_params):
    par_write_params = checkpoint_params.get(CHECKPOINT_PARALLEL_WRITE, {})
    par_write_pipeline = par_write_params.get(CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE,
                                              CHECKPOINT_PARALLEL_WRITE_PIPELINE_STAGE_DEFAULT)
    if par_write_pipeline in [True, False]:
        return par_write_pipeline
    else:
        raise DeepSpeedConfigError("checkpoint::parallel_write::pipeline_stage "
                                   f"value of '{par_write_pipeline}' is invalid, expecting: true or false")


def get_dataloader_drop_last(param_dict):
    return get_scalar_param(param_dict, DATALOADER_DROP_LAST, DATALOADER_DROP_LAST_DEFAULT)


'''Write deepspeed config files by modifying basic templates.
Can be used for quickly changing parameters via command line parameters.'''


class DeepSpeedConfigWriter:

    def __init__(self, data=None):
        self.data = data if data is not None else {}

    def add_config(self, key, value):
        self.data[key] = value

    def load_config(self, filename):
        self.data = json.load(open(filename, "r"), object_pairs_hook=dict_raise_error_on_duplicate_keys)

    def write_config(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.data, outfile)


class DeepSpeedConfig(object):

    def __init__(self, config: Union[str, dict], mpu=None):
        super(DeepSpeedConfig, self).__init__()
        if isinstance(config, dict):
            self._param_dict = config
        elif os.path.exists(config):
            self._param_dict = hjson.load(open(config, "r"), object_pairs_hook=dict_raise_error_on_duplicate_keys)
        else:
            try:
                config_decoded = base64.urlsafe_b64decode(config).decode('utf-8')
                self._param_dict = hjson.loads(config_decoded)
            except (UnicodeDecodeError, AttributeError):
                raise ValueError(
                    f"Expected a string path to an existing deepspeed config, or a dictionary or a valid base64. Received: {config}"
                )
        try:
            self.global_rank = dist.get_rank()
            if mpu is None:
                self.world_size = dist.get_world_size()
            else:
                self.world_size = mpu.get_data_parallel_world_size()
        except:
            self.global_rank = 0
            self.world_size = 1

        # If elastic-mode enabled, update compute + update _param_dict
        self.elasticity_enabled = elasticity_enabled(self._param_dict)
        if self.elasticity_enabled:
            logger.info("DeepSpeed elasticity support enabled")
            final_batch_size, valid_gpus, micro_batch_size = compute_elastic_config(
                ds_config=self._param_dict,
                target_deepspeed_version=__version__,
                world_size=self.world_size,
            )

            elastic_dict = self._param_dict[ELASTICITY]

            # Ensure the resource scheduler saw the same elastic config we are using at runtime
            ensure_immutable_elastic_config(runtime_elastic_config_dict=elastic_dict)

            self.elastic_model_parallel_size = elastic_dict.get(MODEL_PARALLEL_SIZE, MODEL_PARALLEL_SIZE_DEFAULT)
            if self.elastic_model_parallel_size < 1:
                raise ElasticityConfigError("Model-Parallel size cannot be less than 1, "
                                            f"given model-parallel size: {self.elastic_model_parallel_size}")

            self.num_gpus_per_node = elastic_dict.get(NUM_GPUS_PER_NODE, NUM_GPUS_PER_NODE_DEFAULT)
            if self.num_gpus_per_node < 1:
                raise ElasticityConfigError("NUmber of GPUs per node cannot be less than 1, "
                                            f"given number of GPUs per node: {self.num_gpus_per_node}")

            ignore_non_elastic_batch_info = elastic_dict.get(IGNORE_NON_ELASTIC_BATCH_INFO,
                                                             IGNORE_NON_ELASTIC_BATCH_INFO_DEFAULT)

            if not ignore_non_elastic_batch_info:
                batch_params = [
                    TRAIN_BATCH_SIZE,
                    TRAIN_MICRO_BATCH_SIZE_PER_GPU,
                    GRADIENT_ACCUMULATION_STEPS,
                ]
                if any(map(lambda t: t in self._param_dict, batch_params)):
                    raise ElasticityConfigError("One or more batch related parameters were found in your " \
                        f"ds_config ({TRAIN_BATCH_SIZE}, {TRAIN_MICRO_BATCH_SIZE_PER_GPU}, and/or " \
                        f"{GRADIENT_ACCUMULATION_STEPS}). These parameters *will not be used* since " \
                        "elastic training is enabled, which takes control of these parameters. " \
                        "If you want to suppress this error (the parameters will be silently ignored) " \
                        f"please set {IGNORE_NON_ELASTIC_BATCH_INFO}':true in your elasticity config.")

            # micro_bsz * world_size * gas = total_batch_size
            # gas = total_batch_size // (micro_bsz * world_size)
            gradient_accu_steps = final_batch_size // (micro_batch_size * self.world_size)

            if TRAIN_BATCH_SIZE in self._param_dict:
                logger.warning("[Elasticity] overriding training_batch_size: "
                               f"{self._param_dict[TRAIN_BATCH_SIZE]} -> {final_batch_size}")
            if TRAIN_MICRO_BATCH_SIZE_PER_GPU in self._param_dict:
                logger.warning("[Elasticity] overriding train_micro_batch_size_per_gpu: "
                               f"{self._param_dict[TRAIN_MICRO_BATCH_SIZE_PER_GPU]} -> {micro_batch_size}")
            if GRADIENT_ACCUMULATION_STEPS in self._param_dict:
                logger.warning("[Elasticity] overriding gradient_accumulation_steps: "
                               f"{self._param_dict[GRADIENT_ACCUMULATION_STEPS]} -> {gradient_accu_steps}")

            logger.info(f"[Elasticity] valid GPU counts: {valid_gpus}")

            self._param_dict[TRAIN_BATCH_SIZE] = final_batch_size
            self._param_dict[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = micro_batch_size
            self._param_dict[GRADIENT_ACCUMULATION_STEPS] = gradient_accu_steps

        # Pass a copy so that user json is unmodified, e.g. for logging
        self._initialize_params(copy.copy(self._param_dict))
        self._configure_train_batch_size()
        self._do_sanity_check()

    def _initialize_params(self, param_dict):
        self.train_batch_size = get_train_batch_size(param_dict)
        #print(f"beginning get_train_batch_size = {get_train_batch_size}")
        self.train_micro_batch_size_per_gpu = get_train_micro_batch_size_per_gpu(param_dict)
        self.gradient_accumulation_steps = get_gradient_accumulation_steps(param_dict)
        self.steps_per_print = get_steps_per_print(param_dict)
        self.dump_state = get_dump_state(param_dict)

        self.disable_allgather = get_disable_allgather(param_dict)
        self.communication_data_type = get_communication_data_type(param_dict)
        self.prescale_gradients = get_prescale_gradients(param_dict)
        self.gradient_predivide_factor = get_gradient_predivide_factor(param_dict)
        self.sparse_gradients_enabled = get_sparse_gradients_enabled(param_dict)

        self.zero_config = get_zero_config(param_dict)
        self.mics_shard_size = self.zero_config.mics_shard_size
        self.mics_hierarchial_params_gather = self.zero_config.mics_hierarchical_params_gather
        self.zero_optimization_stage = self.zero_config.stage
        self.zero_enabled = self.zero_optimization_stage > 0

        self.activation_checkpointing_config = DeepSpeedActivationCheckpointingConfig(param_dict)

        self.comms_config = DeepSpeedCommsConfig(param_dict)
        self.monitor_config = get_monitor_config(param_dict)

        self.gradient_clipping = get_gradient_clipping(param_dict)
        self.fp16_enabled = get_fp16_enabled(param_dict)
        self.fp16_auto_cast = get_fp16_auto_cast(param_dict)
        self.bfloat16_enabled = get_bfloat16_enabled(param_dict)
        assert not (self.fp16_enabled
                    and self.bfloat16_enabled), 'bfloat16 and fp16 modes cannot be simultaneously enabled'
        self.fp16_master_weights_and_gradients = get_fp16_master_weights_and_grads_enabled(param_dict)
        self.amp_enabled = get_amp_enabled(param_dict)
        self.amp_params = get_amp_params(param_dict)
        self.loss_scale = get_loss_scale(param_dict)
        self.initial_dynamic_scale = get_initial_dynamic_scale(param_dict)
        self.dynamic_loss_scale_args = get_dynamic_loss_scale_args(param_dict)

        self.compression_config = get_compression_config(param_dict)

        self.optimizer_name = get_optimizer_name(param_dict)
        if (self.optimizer_name is not None and self.optimizer_name.lower() in DEEPSPEED_OPTIMIZERS):
            self.optimizer_name = self.optimizer_name.lower()

        self.optimizer_params = get_optimizer_params(param_dict)
        self.optimizer_legacy_fusion = get_optimizer_legacy_fusion(param_dict)

        self.zero_allow_untested_optimizer = get_zero_allow_untested_optimizer(param_dict)

        self.zero_force_ds_cpu_optimizer = get_zero_force_ds_cpu_optimizer(param_dict)

        self.scheduler_name = get_scheduler_name(param_dict)
        self.scheduler_params = get_scheduler_params(param_dict)

        self.flops_profiler_config = DeepSpeedFlopsProfilerConfig(param_dict)
        self.wall_clock_breakdown = (get_wall_clock_breakdown(param_dict) | self.flops_profiler_config.enabled)
        self.memory_breakdown = get_memory_breakdown(param_dict)
        self.autotuning_config = DeepSpeedAutotuningConfig(param_dict)

        (
            self.eigenvalue_enabled,
            self.eigenvalue_verbose,
            self.eigenvalue_max_iter,
            self.eigenvalue_tol,
            self.eigenvalue_stability,
            self.eigenvalue_gas_boundary_resolution,
            self.eigenvalue_layer_name,
            self.eigenvalue_layer_num,
        ) = get_eigenvalue_config(param_dict)

        self.hybrid_engine = get_hybrid_engine_config(param_dict)

        self.sparse_attention = get_sparse_attention(param_dict)
        self.pipeline = get_pipeline_config(param_dict)

        self.pld_enabled = get_pld_enabled(param_dict)
        self.pld_params = get_pld_params(param_dict)

        self.curriculum_enabled_legacy = get_curriculum_enabled_legacy(param_dict)
        self.curriculum_params_legacy = get_curriculum_params_legacy(param_dict)

        self.data_efficiency_enabled = get_data_efficiency_enabled(param_dict)
        self.data_efficiency_config = get_data_efficiency_config(param_dict)

        checkpoint_params = get_checkpoint_params(param_dict)
        validation_mode = get_checkpoint_tag_validation_mode(checkpoint_params)
        self.checkpoint_tag_validation_enabled = (validation_mode != ValidationMode.IGNORE)
        self.checkpoint_tag_validation_fail = validation_mode == ValidationMode.FAIL
        self.load_universal_checkpoint = checkpoint_params.get(LOAD_UNIVERSAL_CHECKPOINT,
                                                               LOAD_UNIVERSAL_CHECKPOINT_DEFAULT)

        self.use_node_local_storage = checkpoint_params.get(USE_NODE_LOCAL_STORAGE_CHECKPOINT,
                                                            USE_NODE_LOCAL_STORAGE_CHECKPOINT_DEFAULT)

        data_types_params = get_data_types_params(param_dict)
        self.grad_accum_dtype = data_types_params.get(GRAD_ACCUM_DTYPE, GRAD_ACCUM_DTYPE_DEFAULT)

        par_write_pipe = get_checkpoint_parallel_write_pipeline(checkpoint_params)
        self.checkpoint_parallel_write_pipeline = par_write_pipe

        self.aio_config = get_aio_config(param_dict)

        self.dataloader_drop_last = get_dataloader_drop_last(param_dict)

        self.nebula_config = DeepSpeedNebulaConfig(param_dict)

        self.weight_quantization_config = WeightQuantConfig(
            **param_dict['weight_quantization']) if 'weight_quantization' in param_dict else None

    def _batch_assertion(self):

        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps

        assert (train_batch > 0), f"Train batch size: {train_batch} has to be greater than 0"

        assert (micro_batch > 0), f"Micro batch size per gpu: {micro_batch} has to be greater than 0"

        assert (grad_acc > 0), f"Gradient accumulation steps: {grad_acc} has to be greater than 0"

        assert train_batch == micro_batch * grad_acc * self.world_size, (
            f"Check batch related parameters. train_batch_size is not equal "
            "to micro_batch_per_gpu * gradient_acc_step * world_size "
            f"{train_batch} != {micro_batch} * {grad_acc} * {self.world_size}")

    def _set_batch_related_parameters(self):

        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps

        #print(f"train_batch = {train_batch}, micro_batch={micro_batch}")

        # all values are provided nothing needs to be set
        if train_batch is not None and micro_batch is not None and grad_acc is not None:
            return

        # global_accumulation_steps needs to be set
        elif train_batch is not None and micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= self.world_size
            self.gradient_accumulation_steps = grad_acc

        # micro_batch_per_gpu needs to be set
        elif train_batch is not None and grad_acc is not None:
            micro_batch = train_batch // self.world_size
            micro_batch //= grad_acc
            self.train_micro_batch_size_per_gpu = micro_batch

        # train_batch_size needs to be set
        elif micro_batch is not None and grad_acc is not None:
            train_batch_size = micro_batch * grad_acc
            train_batch_size *= self.world_size
            self.train_batch_size = train_batch_size

        # gradient_accumulation_steps and micro_batch_per_gpus is set
        elif train_batch is not None:
            self.gradient_accumulation_steps = 1
            self.train_micro_batch_size_per_gpu = train_batch // self.world_size

        # train_batch_size and gradient_accumulation_step is set
        elif micro_batch is not None:
            self.train_batch_size = micro_batch * self.world_size
            self.gradient_accumulation_steps = 1

        # either none of the three parameters are provided or just gradient_accumulation_step is provided
        else:
            assert False, \
                'Either train_batch_size or train_micro_batch_size_per_gpu needs to be provided'

    def _configure_train_batch_size(self):
        self._set_batch_related_parameters()
        self._batch_assertion()

    def _do_sanity_check(self):
        self._do_error_check()

        self._do_warning_check()

    def print_user_config(self):
        logger.info("  json = {}".format(
            json.dumps(
                self._param_dict,
                sort_keys=True,
                indent=4,
                cls=ScientificNotationEncoder,
                separators=(",", ":"),
            )))

    def print(self, name):
        logger.info("{}:".format(name))
        for arg in sorted(vars(self)):
            if arg != "_param_dict":
                dots = "." * (29 - len(arg))
                logger.info("  {} {} {}".format(arg, dots, getattr(self, arg)))

        self.print_user_config()

    def _do_error_check(self):
        assert (self.train_micro_batch_size_per_gpu
                ), "DeepSpeedConfig: {} is not defined".format(TRAIN_MICRO_BATCH_SIZE_PER_GPU)

        assert (
            self.gradient_accumulation_steps), "DeepSpeedConfig: {} is not defined".format(GRADIENT_ACCUMULATION_STEPS)

        if self.zero_enabled:
            assert (self.zero_optimization_stage <=
                    ZeroStageEnum.max_stage), "DeepSpeedConfig: Maximum supported ZeRO stage is {}".format(
                        ZeroStageEnum.max_stage)

        if self.fp16_master_weights_and_gradients:
            assert self.zero_enabled and self.zero_optimization_stage == ZeroStageEnum.gradients, "Fp16_master_weights_and_grads is only supported with ZeRO Stage 2 for now."

    def _do_warning_check(self):
        fp16_enabled = self.fp16_enabled

        vocabulary_size = self._param_dict.get(VOCABULARY_SIZE, VOCABULARY_SIZE_DEFAULT)
        if vocabulary_size and vocabulary_size % TENSOR_CORE_ALIGN_SIZE != 0:
            logger.warning(
                "DeepSpeedConfig: vocabulary size {} is not aligned to {}, may import tensor core utilization.".format(
                    vocabulary_size, TENSOR_CORE_ALIGN_SIZE))

        if (self.optimizer_params is not None and MAX_GRAD_NORM in self.optimizer_params.keys()
                and self.optimizer_params[MAX_GRAD_NORM] > 0):
            if fp16_enabled:
                if self.global_rank == 0:
                    logger.warning("DeepSpeedConfig: In FP16 mode, DeepSpeed will pass {}:{} to FP16 wrapper".format(
                        MAX_GRAD_NORM, self.optimizer_params[MAX_GRAD_NORM]))
            else:
                if self.global_rank == 0:
                    logger.warning(
                        "DeepSpeedConfig: In FP32 mode, DeepSpeed does not permit MAX_GRAD_NORM ({}) > 0, setting to zero"
                        .format(self.optimizer_params[MAX_GRAD_NORM]))
                self.optimizer_params[MAX_GRAD_NORM] = 0.0
