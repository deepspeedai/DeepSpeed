"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

import torch
import json
import copy
from deepspeed.runtime.constants import *
from deepspeed.runtime.fp16.loss_scaler import INITIAL_LOSS_SCALE, SCALE_WINDOW, DELAYED_SHIFT, MIN_LOSS_SCALE
from deepspeed.runtime.config_utils import get_scalar_param, dict_raise_error_on_duplicate_keys
from deepspeed.runtime.zero.config import DeepSpeedZeroConfig
from deepspeed.runtime.zero.constants import *
from deepspeed.runtime.activation_checkpointing.config import DeepSpeedActivationCheckpointingConfig
from deepspeed.utils import logger

TENSOR_CORE_ALIGN_SIZE = 8

ADAM_OPTIMIZER = 'adam'
LAMB_OPTIMIZER = 'lamb'
ONEBIT_ADAM_OPTIMIZER = 'onebitadam'
DEEPSPEED_OPTIMIZERS = [
    ADAM_OPTIMIZER,
    LAMB_OPTIMIZER,
    ONEBIT_ADAM_OPTIMIZER,
]

# extra optimizer parameters for adam
TORCH_ADAM_PARAM = "torch_adam"
ADAM_W_MODE_PARAM = "adam_w_mode"


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


def get_loss_scale(param_dict):
    if get_fp16_enabled(param_dict):
        return get_scalar_param(param_dict[FP16],
                                FP16_LOSS_SCALE,
                                FP16_LOSS_SCALE_DEFAULT)
    else:
        return FP16_LOSS_SCALE_DEFAULT


def get_initial_dynamic_scale(param_dict):
    if get_fp16_enabled(param_dict):
        initial_scale_power = get_scalar_param(param_dict[FP16],
                                               FP16_INITIAL_SCALE_POWER,
                                               FP16_INITIAL_SCALE_POWER_DEFAULT)
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
            FP16_HYSTERESIS
        ]
        if any(arg in list(fp16_dict.keys()) for arg in dynamic_loss_args):
            init_scale = get_scalar_param(fp16_dict,
                                          FP16_INITIAL_SCALE_POWER,
                                          FP16_INITIAL_SCALE_POWER_DEFAULT)
            scale_window = get_scalar_param(fp16_dict,
                                            FP16_LOSS_SCALE_WINDOW,
                                            FP16_LOSS_SCALE_WINDOW_DEFAULT)
            delayed_shift = get_scalar_param(fp16_dict,
                                             FP16_HYSTERESIS,
                                             FP16_HYSTERESIS_DEFAULT)
            min_loss_scale = get_scalar_param(fp16_dict,
                                              FP16_MIN_LOSS_SCALE,
                                              FP16_MIN_LOSS_SCALE_DEFAULT)
            loss_scale_args = {
                INITIAL_LOSS_SCALE: 2**init_scale,
                SCALE_WINDOW: scale_window,
                DELAYED_SHIFT: delayed_shift,
                MIN_LOSS_SCALE: min_loss_scale
            }

    return loss_scale_args


def get_gradient_accumulation_steps(param_dict):
    return get_scalar_param(param_dict,
                            GRADIENT_ACCUMULATION_STEPS,
                            GRADIENT_ACCUMULATION_STEPS_DEFAULT)


def get_sparse_gradients_enabled(param_dict):
    return get_scalar_param(param_dict, SPARSE_GRADIENTS, SPARSE_GRADIENTS_DEFAULT)


def get_zero_optimization(param_dict):
    return get_scalar_param(param_dict, ZERO_OPTIMIZATION, ZERO_OPTIMIZATION_DEFAULT)


def get_zero_reduce_scatter(param_dict):
    return get_scalar_param(param_dict,
                            ZERO_OPTIMIZATION_REDUCE_SCATTER,
                            ZERO_OPTIMIZATION_REDUCE_SCATTER_DEFAULT)


def get_allreduce_always_fp32(param_dict):
    return get_scalar_param(param_dict, FP32_ALLREDUCE, FP32_ALLREDUCE_DEFAULT)


def get_prescale_gradients(param_dict):
    return get_scalar_param(param_dict, PRESCALE_GRADIENTS, PRESCALE_GRADIENTS_DEFAULT)


def get_gradient_predivide_factor(param_dict):
    return get_scalar_param(param_dict,
                            GRADIENT_PREDIVIDE_FACTOR,
                            GRADIENT_PREDIVIDE_FACTOR_DEFAULT)


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

        if (mode == SPARSE_DENSE_MODE):
            return get_sparse_dense_config(sparsity)
        elif (mode == SPARSE_FIXED_MODE):
            return get_sparse_fixed_config(sparsity)
        elif (mode == SPARSE_VARIABLE_MODE):
            return get_sparse_variable_config(sparsity)
        elif (mode == SPARSE_BIGBIRD_MODE):
            return get_sparse_bigbird_config(sparsity)
        elif (mode == SPARSE_BSLONGFORMER_MODE):
            return get_sparse_bslongformer_config(sparsity)
        else:
            raise NotImplementedError(
                f'Given sparsity mode, {mode}, has not been implemented yet!')

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
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT)
    num_local_blocks = get_scalar_param(sparsity,
                                        SPARSE_NUM_LOCAL_BLOCKS,
                                        SPARSE_NUM_LOCAL_BLOCKS_DEFAULT)
    num_global_blocks = get_scalar_param(sparsity,
                                         SPARSE_NUM_GLOBAL_BLOCKS,
                                         SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT)
    attention = get_scalar_param(sparsity,
                                 SPARSE_ATTENTION_TYPE,
                                 SPARSE_ATTENTION_TYPE_DEFAULT)
    horizontal_global_attention = get_scalar_param(
        sparsity,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT)
    num_different_global_patterns = get_scalar_param(
        sparsity,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS_DEFAULT)

    return {
        SPARSE_MODE: SPARSE_FIXED_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_LOCAL_BLOCKS: num_local_blocks,
        SPARSE_NUM_GLOBAL_BLOCKS: num_global_blocks,
        SPARSE_ATTENTION_TYPE: attention,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION: horizontal_global_attention,
        SPARSE_NUM_DIFFERENT_GLOBAL_PATTERNS: num_different_global_patterns
    }


def get_sparse_variable_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT)
    num_random_blocks = get_scalar_param(sparsity,
                                         SPARSE_NUM_RANDOM_BLOCKS,
                                         SPARSE_NUM_RANDOM_BLOCKS_DEFAULT)
    local_window_blocks = get_scalar_param(sparsity,
                                           SPARSE_LOCAL_WINDOW_BLOCKS,
                                           SPARSE_LOCAL_WINDOW_BLOCKS_DEFAULT)
    global_block_indices = get_scalar_param(sparsity,
                                            SPARSE_GLOBAL_BLOCK_INDICES,
                                            SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT)
    global_block_end_indices = get_scalar_param(sparsity,
                                                SPARSE_GLOBAL_BLOCK_END_INDICES,
                                                SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT)
    attention = get_scalar_param(sparsity,
                                 SPARSE_ATTENTION_TYPE,
                                 SPARSE_ATTENTION_TYPE_DEFAULT)
    horizontal_global_attention = get_scalar_param(
        sparsity,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION_DEFAULT)

    return {
        SPARSE_MODE: SPARSE_VARIABLE_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_RANDOM_BLOCKS: num_random_blocks,
        SPARSE_LOCAL_WINDOW_BLOCKS: local_window_blocks,
        SPARSE_GLOBAL_BLOCK_INDICES: global_block_indices,
        SPARSE_GLOBAL_BLOCK_END_INDICES: global_block_end_indices,
        SPARSE_ATTENTION_TYPE: attention,
        SPARSE_HORIZONTAL_GLOBAL_ATTENTION: horizontal_global_attention
    }


def get_sparse_bigbird_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT)
    num_random_blocks = get_scalar_param(sparsity,
                                         SPARSE_NUM_RANDOM_BLOCKS,
                                         SPARSE_NUM_RANDOM_BLOCKS_DEFAULT)
    num_sliding_window_blocks = get_scalar_param(
        sparsity,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT)
    num_global_blocks = get_scalar_param(sparsity,
                                         SPARSE_NUM_GLOBAL_BLOCKS,
                                         SPARSE_NUM_GLOBAL_BLOCKS_DEFAULT)

    return {
        SPARSE_MODE: SPARSE_BIGBIRD_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_RANDOM_BLOCKS: num_random_blocks,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS: num_sliding_window_blocks,
        SPARSE_NUM_GLOBAL_BLOCKS: num_global_blocks
    }


def get_sparse_bslongformer_config(sparsity):
    block = get_scalar_param(sparsity, SPARSE_BLOCK, SPARSE_BLOCK_DEFAULT)
    different_layout_per_head = get_scalar_param(
        sparsity,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD_DEFAULT)
    num_sliding_window_blocks = get_scalar_param(
        sparsity,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS_DEFAULT)
    global_block_indices = get_scalar_param(sparsity,
                                            SPARSE_GLOBAL_BLOCK_INDICES,
                                            SPARSE_GLOBAL_BLOCK_INDICES_DEFAULT)
    global_block_end_indices = get_scalar_param(sparsity,
                                                SPARSE_GLOBAL_BLOCK_END_INDICES,
                                                SPARSE_GLOBAL_BLOCK_END_INDICES_DEFAULT)

    return {
        SPARSE_MODE: SPARSE_BSLONGFORMER_MODE,
        SPARSE_BLOCK: block,
        SPARSE_DIFFERENT_LAYOUT_PER_HEAD: different_layout_per_head,
        SPARSE_NUM_SLIDING_WINDOW_BLOCKS: num_sliding_window_blocks,
        SPARSE_GLOBAL_BLOCK_INDICES: global_block_indices,
        SPARSE_GLOBAL_BLOCK_END_INDICES: global_block_end_indices
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
    '''Parses pipeline engine configuration. '''
    default_pipeline = {
        'stages': 'auto',
        'partition': 'best',
        'seed_layers': False,
        'activation_checkpoint_interval': 0
    }
    config = default_pipeline
    for key, val in param_dict.get('pipeline', {}).items():
        config[key] = val
    return config


def get_optimizer_name(param_dict):
    if OPTIMIZER in param_dict.keys() and \
            TYPE in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][TYPE]
    else:
        return OPTIMIZER_TYPE_DEFAULT


def get_optimizer_params(param_dict):
    if get_optimizer_name(param_dict) is not None and \
            OPTIMIZER_PARAMS in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][OPTIMIZER_PARAMS]
    else:
        return None


def get_optimizer_gradient_clipping(param_dict):
    optimizer_params = get_optimizer_params(param_dict)
    if optimizer_params is not None and \
            MAX_GRAD_NORM in optimizer_params.keys():
        return optimizer_params[MAX_GRAD_NORM]
    else:
        return None


def get_optimizer_legacy_fusion(param_dict):
    if OPTIMIZER in param_dict.keys() and \
        LEGACY_FUSION in param_dict[OPTIMIZER].keys():
        return param_dict[OPTIMIZER][LEGACY_FUSION]
    else:
        return LEGACY_FUSION_DEFAULT


def get_zero_allow_untested_optimizer(param_dict):
    return get_scalar_param(param_dict,
                            ZERO_ALLOW_UNTESTED_OPTIMIZER,
                            ZERO_ALLOW_UNTESTED_OPTIMIZER_DEFAULT)


def get_scheduler_name(param_dict):
    if SCHEDULER in param_dict.keys() and \
            TYPE in param_dict[SCHEDULER].keys():
        return param_dict[SCHEDULER][TYPE]
    else:
        return SCHEDULER_TYPE_DEFAULT


def get_scheduler_params(param_dict):
    if get_scheduler_name(param_dict) is not None and \
            SCHEDULER_PARAMS in param_dict[SCHEDULER].keys():
        return param_dict[SCHEDULER][SCHEDULER_PARAMS]
    else:
        return None


def get_train_batch_size(param_dict):
    return get_scalar_param(param_dict, TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE_DEFAULT)


def get_train_micro_batch_size_per_gpu(param_dict):
    return get_scalar_param(param_dict,
                            TRAIN_MICRO_BATCH_SIZE_PER_GPU,
                            TRAIN_MICRO_BATCH_SIZE_PER_GPU_DEFAULT)


def get_wall_clock_breakdown(param_dict):
    return get_scalar_param(param_dict,
                            WALL_CLOCK_BREAKDOWN,
                            WALL_CLOCK_BREAKDOWN_DEFAULT)


def get_memory_breakdown(param_dict):
    return get_scalar_param(param_dict, MEMORY_BREAKDOWN, MEMORY_BREAKDOWN_DEFAULT)


def get_tensorboard_enabled(param_dict):
    if TENSORBOARD in param_dict.keys():
        return get_scalar_param(param_dict[TENSORBOARD],
                                TENSORBOARD_ENABLED,
                                TENSORBOARD_ENABLED_DEFAULT)
    else:
        return False


def get_tensorboard_output_path(param_dict):
    if get_tensorboard_enabled(param_dict):
        return get_scalar_param(param_dict[TENSORBOARD],
                                TENSORBOARD_OUTPUT_PATH,
                                TENSORBOARD_OUTPUT_PATH_DEFAULT)
    else:
        return TENSORBOARD_OUTPUT_PATH_DEFAULT


def get_tensorboard_job_name(param_dict):
    if get_tensorboard_enabled(param_dict):
        return get_scalar_param(param_dict[TENSORBOARD],
                                TENSORBOARD_JOB_NAME,
                                TENSORBOARD_JOB_NAME_DEFAULT)
    else:
        return TENSORBOARD_JOB_NAME_DEFAULT


'''Write deepspeed config files by modifying basic templates.
Can be used for quicly changing parameters via command line parameters.'''


class DeepSpeedConfigWriter:
    def __init__(self, data=None):
        self.data = data if data is not None else {}

    def add_config(self, key, value):
        self.data[key] = value

    def load_config(self, filename):
        self.data = json.load(open(filename,
                                   'r'),
                              object_pairs_hook=dict_raise_error_on_duplicate_keys)

    def write_config(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.data, outfile)


class DeepSpeedConfig(object):
    def __init__(self, json_file, mpu=None, param_dict=None):
        super(DeepSpeedConfig, self).__init__()

        if param_dict is None:
            self._param_dict = json.load(
                open(json_file,
                     'r'),
                object_pairs_hook=dict_raise_error_on_duplicate_keys)
        else:
            self._param_dict = param_dict

        try:
            self.global_rank = torch.distributed.get_rank()
            if mpu is None:
                self.world_size = torch.distributed.get_world_size()
            else:
                self.world_size = mpu.get_data_parallel_world_size()
        except:
            self.global_rank = 0
            self.world_size = 1

        self._initialize_params(self._param_dict)
        self._configure_train_batch_size()
        self._do_sanity_check()

    def _initialize_params(self, param_dict):
        self.train_batch_size = get_train_batch_size(param_dict)
        self.train_micro_batch_size_per_gpu = get_train_micro_batch_size_per_gpu(
            param_dict)
        self.gradient_accumulation_steps = get_gradient_accumulation_steps(param_dict)
        self.steps_per_print = get_steps_per_print(param_dict)
        self.dump_state = get_dump_state(param_dict)

        self.disable_allgather = get_disable_allgather(param_dict)
        self.allreduce_always_fp32 = get_allreduce_always_fp32(param_dict)
        self.prescale_gradients = get_prescale_gradients(param_dict)
        self.gradient_predivide_factor = get_gradient_predivide_factor(param_dict)
        self.sparse_gradients_enabled = get_sparse_gradients_enabled(param_dict)

        self.zero_config = DeepSpeedZeroConfig(param_dict)
        self.zero_optimization_stage = self.zero_config.stage
        self.zero_enabled = self.zero_optimization_stage > 0

        self.activation_checkpointing_config = DeepSpeedActivationCheckpointingConfig(
            param_dict)

        self.gradient_clipping = get_gradient_clipping(param_dict)
        self.fp16_enabled = get_fp16_enabled(param_dict)
        self.amp_enabled = get_amp_enabled(param_dict)
        self.amp_params = get_amp_params(param_dict)
        self.loss_scale = get_loss_scale(param_dict)
        self.initial_dynamic_scale = get_initial_dynamic_scale(param_dict)
        self.dynamic_loss_scale_args = get_dynamic_loss_scale_args(param_dict)

        self.optimizer_name = get_optimizer_name(param_dict)
        if self.optimizer_name is not None and \
            self.optimizer_name.lower() in DEEPSPEED_OPTIMIZERS:
            self.optimizer_name = self.optimizer_name.lower()

        self.optimizer_params = get_optimizer_params(param_dict)
        self.optimizer_legacy_fusion = get_optimizer_legacy_fusion(param_dict)

        self.zero_allow_untested_optimizer = get_zero_allow_untested_optimizer(
            param_dict)

        self.scheduler_name = get_scheduler_name(param_dict)
        self.scheduler_params = get_scheduler_params(param_dict)

        self.wall_clock_breakdown = get_wall_clock_breakdown(param_dict)
        self.memory_breakdown = get_memory_breakdown(param_dict)
        self.tensorboard_enabled = get_tensorboard_enabled(param_dict)
        self.tensorboard_output_path = get_tensorboard_output_path(param_dict)
        self.tensorboard_job_name = get_tensorboard_job_name(param_dict)

        self.sparse_attention = get_sparse_attention(param_dict)
        self.pipeline = get_pipeline_config(param_dict)

    def _batch_assertion(self):

        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps

        assert train_batch > 0, \
            f'Train batch size: {train_batch} has to be greater than 0'

        assert micro_batch > 0, \
            f'Micro batch size per gpu: {micro_batch} has to be greater than 0'

        assert grad_acc > 0, \
            f'Gradient accumulation steps: {grad_acc} has to be greater than 0'

        assert train_batch == micro_batch * grad_acc * self.world_size, \
                (f'Check batch related parameters. train_batch_size is not equal'
                ' to micro_batch_per_gpu * gradient_acc_step * world_size'
                f'{train_batch} != {micro_batch} * {grad_acc} * {self.world_size}')

    def _set_batch_related_parameters(self):

        train_batch = self.train_batch_size
        micro_batch = self.train_micro_batch_size_per_gpu
        grad_acc = self.gradient_accumulation_steps

        #all values are provided nothing needs to be set
        if train_batch is not None and \
            micro_batch is not None and \
            grad_acc is not None:
            return

        #global_accumulation_steps needs to be set
        elif train_batch is not None and \
            micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= self.world_size
            self.gradient_accumulation_steps = grad_acc

        #micro_batch_per_gpu needs to be set
        elif train_batch is not None and \
            grad_acc is not None:
            micro_batch = train_batch // self.world_size
            micro_batch //= grad_acc
            self.train_micro_batch_size_per_gpu = micro_batch

        #train_batch_size needs to be set
        elif micro_batch is not None and \
            grad_acc is not None:
            train_batch_size = micro_batch * grad_acc
            train_batch_size *= self.world_size
            self.train_batch_size = train_batch_size

        #gradient_accumulation_steps and micro_batch_per_gpus is set
        elif train_batch is not None:
            self.gradient_accumulation_steps = 1
            self.train_micro_batch_size_per_gpu = train_batch // self.world_size

        #train_batch_size and gradient_accumulation_step is set
        elif micro_batch is not None:
            self.train_batch_size = micro_batch * self.world_size
            self.gradient_accumulation_steps = 1

        #either none of the three parameters are provided or just gradient_accumulation_step is provided
        else:
            assert False, \
                'Either train_batch_size or micro_batch_per_gpu needs to be provided'

    def _configure_train_batch_size(self):
        self._set_batch_related_parameters()
        self._batch_assertion()

    def _do_sanity_check(self):
        self._do_error_check()

        self._do_warning_check()

    def print(self, name):
        logger.info('{}:'.format(name))
        for arg in sorted(vars(self)):
            if arg != '_param_dict':
                dots = '.' * (29 - len(arg))
                logger.info('  {} {} {}'.format(arg, dots, getattr(self, arg)))

        logger.info('  json = {}'.format(
            json.dumps(self._param_dict,
                       sort_keys=True,
                       indent=4,
                       separators=(',',
                                   ':'))))

    def _do_error_check(self):
        assert self.train_micro_batch_size_per_gpu, "DeepSpeedConfig: {} is not defined".format(TRAIN_MICRO_BATCH_SIZE_PER_GPU)

        assert self.gradient_accumulation_steps, "DeepSpeedConfig: {} is not defined".format(
            GRADIENT_ACCUMULATION_STEPS)

        if self.zero_enabled:
            assert self.fp16_enabled, "DeepSpeedConfig: ZeRO is only supported if fp16 is enabled"
            assert self.zero_optimization_stage <= MAX_STAGE_ZERO_OPTIMIZATION, "DeepSpeedConfig: Maximum supported ZeRO stage is {}".format(MAX_STAGE_ZERO_OPTIMIZATION)
            if self.zero_config.cpu_offload is True:
                assert self.zero_optimization_stage == ZERO_OPTIMIZATION_GRADIENTS, "DeepSpeedConfig: cpu-offload supported ZeRO stage is {}".format(ZERO_OPTIMIZATION_GRADIENTS)
                #assert self.gradient_accumulation_steps == 1, "DeepSpeedConfig: {}is not supported for {}".format(GRADIENT_ACCUMULATION_STEPS, ZERO_OPTIMIZATION_CPU_OFFLOAD)

    def _do_warning_check(self):
        fp16_enabled = self.fp16_enabled or self.zero_enabled

        vocabulary_size = self._param_dict.get(VOCABULARY_SIZE, VOCABULARY_SIZE_DEFAULT)
        if vocabulary_size and vocabulary_size % TENSOR_CORE_ALIGN_SIZE != 0:
            logger.warning(
                "DeepSpeedConfig: vocabulary size {} is not aligned to {}, may import tensor core utilization."
                .format(vocabulary_size,
                        TENSOR_CORE_ALIGN_SIZE))

        if self.optimizer_params is not None and \
            MAX_GRAD_NORM in self.optimizer_params.keys() and \
                self.optimizer_params[MAX_GRAD_NORM] > 0:
            if fp16_enabled:
                if self.global_rank == 0:
                    logger.warning(
                        'DeepSpeedConfig: In FP16 mode, DeepSpeed will pass {}:{} to FP16 wrapper'
                        .format(MAX_GRAD_NORM,
                                self.optimizer_params[MAX_GRAD_NORM]))
            else:
                if self.global_rank == 0:
                    logger.warning(
                        'DeepSpeedConfig: In FP32 mode, DeepSpeed does not permit MAX_GRAD_NORM ({}) > 0, setting to zero'
                        .format(self.optimizer_params[MAX_GRAD_NORM]))
                self.optimizer_params[MAX_GRAD_NORM] = 0.0
