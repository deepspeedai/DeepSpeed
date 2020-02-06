"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""

import torch
import logging
import json
from deepspeed.pt.deepspeed_constants import *

TENSOR_CORE_ALIGN_SIZE = 8
ADAM_OPTIMIZER = 'adam'
LAMB_OPTIMIZER = 'lamb'
DEEPSPEED_OPTIMIZERS = [ADAM_OPTIMIZER, LAMB_OPTIMIZER]


def get_scalar_param(param_dict, param_name, param_default_value):
    if param_name in param_dict.keys():
        return param_dict[param_name]
    else:
        return param_default_value


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
                'init_scale': 2**init_scale,
                'scale_window': scale_window,
                'delayed_shift': delayed_shift,
                'min_scale': min_loss_scale
            }

    return loss_scale_args


def get_gradient_accumulation_steps(param_dict):
    return get_scalar_param(param_dict,
                            GRADIENT_ACCUMULATION_STEPS,
                            GRADIENT_ACCUMULATION_STEPS_DEFAULT)


def get_sparse_gradients_enabled(param_dict):
    return get_scalar_param(param_dict, SPARSE_GRADIENTS, SPARSE_GRADIENTS_DEFAULT)


def get_zero_enabled(param_dict):
    return get_scalar_param(param_dict, ZERO_OPTIMIZATION, ZERO_OPTIMIZATION_DEFAULT)


def get_allgather_size(param_dict):
    return get_scalar_param(param_dict,
                            ALLGATHER_SIZE,
                            ALLGATHER_SIZE_DEFAULT) if get_scalar_param(
                                param_dict,
                                ALLGATHER_SIZE,
                                ALLGATHER_SIZE_DEFAULT) > 0 else ALLGATHER_SIZE_DEFAULT


def get_allreduce_always_fp32(param_dict):
    return get_scalar_param(param_dict, FP32_ALLREDUCE, FP32_ALLREDUCE_DEFAULT)


def get_prescale_gradients(param_dict):
    return get_scalar_param(param_dict, PRESCALE_GRADIENTS, PRESCALE_GRADIENTS_DEFAULT)


def get_steps_per_print(param_dict):
    return get_scalar_param(param_dict, STEPS_PER_PRINT, STEPS_PER_PRINT_DEFAULT)


def get_disable_allgather(param_dict):
    return get_scalar_param(param_dict, DISABLE_ALLGATHER, DISABLE_ALLGATHER_DEFAULT)


def get_dump_state(param_dict):
    return get_scalar_param(param_dict, DUMP_STATE, DUMP_STATE_DEFAULT)


def get_gradient_clipping(param_dict):
    grad_clip = get_optimizer_gradient_clipping(param_dict)
    if grad_clip is not None:
        return grad_clip
    else:
        return get_scalar_param(param_dict, GRADIENT_CLIPPING, GRADIENT_CLIPPING_DEFAULT)


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


class DeepSpeedConfig(object):
    def __init__(self, json_file, mpu=None):
        super(DeepSpeedConfig, self).__init__()
        self._param_dict = json.load(open(json_file, 'r'))
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
        self.sparse_gradients_enabled = get_sparse_gradients_enabled(param_dict)

        self.allgather_size = get_allgather_size(param_dict)
        self.zero_enabled = get_zero_enabled(param_dict)
        self.gradient_clipping = get_gradient_clipping(param_dict)
        self.fp16_enabled = get_fp16_enabled(param_dict)
        self.loss_scale = get_loss_scale(param_dict)
        self.initial_dynamic_scale = get_initial_dynamic_scale(param_dict)
        self.dynamic_loss_scale_args = get_dynamic_loss_scale_args(param_dict)

        self.optimizer_name = get_optimizer_name(param_dict)
        if self.optimizer_name is not None and \
            self.optimizer_name.lower() in DEEPSPEED_OPTIMIZERS:
            self.optimizer_name = self.optimizer_name.lower()

        self.optimizer_params = get_optimizer_params(param_dict)

        self.scheduler_name = get_scheduler_name(param_dict)
        self.scheduler_params = get_scheduler_params(param_dict)

        self.wall_clock_breakdown = get_wall_clock_breakdown(param_dict)
        self.tensorboard_enabled = get_tensorboard_enabled(param_dict)
        self.tensorboard_output_path = get_tensorboard_output_path(param_dict)
        self.tensorboard_job_name = get_tensorboard_job_name(param_dict)

    def _do_batch_size_sanity_check(self):
        assert self.train_batch_size >= self.world_size, \
            'DeepSpeedConfig: {} {} is smaller than device count {}' \
            .format(TRAIN_BATCH_SIZE, self.train_batch_size, self.world_size)

        assert self.train_batch_size % self.world_size == 0, \
            'DeepSpeedConfig: {} {} is not divisible by device count {}' \
            .format(TRAIN_BATCH_SIZE, self.train_batch_size, self.world_size)

        per_device_batch_size = self.train_batch_size // self.world_size

        if self.train_micro_batch_size_per_gpu is not None:
            assert self.gradient_accumulation_steps is None, \
                'DeepSpeedConfig: {} and {} should not be defined together' \
                .format(TRAIN_MICRO_BATCH_SIZE_PER_GPU, GRADIENT_ACCUMULATION_STEPS)

            assert self.train_micro_batch_size_per_gpu <= self.train_batch_size, \
                'DeepSpeedConfig: {} {} is greater than {} {}' \
                .format(TRAIN_MICRO_BATCH_SIZE_PER_GPU, self.train_micro_batch_size_per_gpu, TRAIN_BATCH_SIZE, self.train_batch_size)

            assert self.train_batch_size % self.train_micro_batch_size_per_gpu == 0, \
                'DeepSpeedConfig: {} {} is not divisible by {} {}' \
                .format(TRAIN_BATCH_SIZE, self.train_batch_size, TRAIN_MICRO_BATCH_SIZE_PER_GPU, self.train_micro_batch_size_per_gpu)

            if per_device_batch_size > self.train_micro_batch_size_per_gpu:
                assert per_device_batch_size % self.train_micro_batch_size_per_gpu == 0, \
                    'DeepSpeedConfig: Per device batch size {} is not divisible by {} {}' \
                    .format(per_device_batch_size, TRAIN_MICRO_BATCH_SIZE_PER_GPU, self.train_micro_batch_size_per_gpu)

        if self.gradient_accumulation_steps is not None:
            assert self.train_batch_size % self.gradient_accumulation_steps == 0, \
                'DeepSpeedConfig: {} {} is not divisible by {} {}' \
                .format(TRAIN_BATCH_SIZE, self.train_batch_size, GRADIENT_ACCUMULATION_STEPS, self.gradient_accumulation_steps)

            assert per_device_batch_size % self.gradient_accumulation_steps == 0, \
                'DeepSpeedConfig: Per device batch size {} is not divisible by {} {}' \
                .format(per_device_batch_size, GRADIENT_ACCUMULATION_STEPS, self.gradient_accumulation_steps)

    def _configure_train_batch_size(self):
        self._do_batch_size_sanity_check()
        if self.train_micro_batch_size_per_gpu is None and \
                self.gradient_accumulation_steps is None:
            self.train_micro_batch_size_per_gpu = self.train_batch_size
            self.gradient_accumulation_steps = 1
        elif self.train_micro_batch_size_per_gpu is not None:
            per_device_batch_size = self.train_batch_size // self.world_size
            if self.train_micro_batch_size_per_gpu > per_device_batch_size:
                self.train_micro_batch_size_per_gpu = per_device_batch_size
                self.gradient_accumulation_steps = 1
            else:
                self.gradient_accumulation_steps = per_device_batch_size // self.train_micro_batch_size_per_gpu
        else:
            self.train_micro_batch_size_per_gpu = self.train_batch_size // (
                self.gradient_accumulation_steps * self.world_size)

    def _do_sanity_check(self):
        self._do_error_check()

        self._do_warning_check()

    def print(self, name):
        print('{}:'.format(name), flush=True)
        for arg in sorted(vars(self)):
            if arg != '_param_dict':
                dots = '.' * (29 - len(arg))
                print('  {} {} {}'.format(arg, dots, getattr(self, arg)), flush=True)

        print('  json = {}'.format(
            json.dumps(self._param_dict,
                       sort_keys=True,
                       indent=4,
                       separators=(',',
                                   ':'))))

    def _do_error_check(self):
        if self.zero_enabled:
            assert self.fp16_enabled, "DeepSpeedConfig: ZeRO is only supported if fp16 is enabled"

        assert self.train_micro_batch_size_per_gpu, "DeepSpeedConfig: {} is not defined".format(TRAIN_MICRO_BATCH_SIZE_PER_GPU)

        assert self.gradient_accumulation_steps, 'DeepSpeedConfig: {} is not defined'.format(
            GRADIENT_ACCUMULATION_STEPS)

    def _do_warning_check(self):
        fp16_enabled = self.fp16_enabled or self.zero_enabled
        if self.gradient_clipping > 0. and not fp16_enabled:
            logging.warning(
                'DeepSpeedConfig: gradient clipping enabled without FP16 enabled.')

        vocabulary_size = self._param_dict.get(VOCABULARY_SIZE, VOCABULARY_SIZE_DEFAULT)
        if vocabulary_size and vocabulary_size % TENSOR_CORE_ALIGN_SIZE != 0:
            logging.warning(
                "DeepSpeedConfig: vocabulary size {} is not aligned to {}, may import tensor core utilization."
                .format(vocabulary_size,
                        TENSOR_CORE_ALIGN_SIZE))

        if self.optimizer_params is not None and \
            MAX_GRAD_NORM in self.optimizer_params.keys() and \
                self.optimizer_params[MAX_GRAD_NORM] > 0:
            if fp16_enabled:
                logging.warning(
                    'DeepSpeedConfig: In FP16 mode, DeepSpeed will pass {}:{} to FP16 wrapper'
                    .format(MAX_GRAD_NORM,
                            self.optimizer_params[MAX_GRAD_NORM]))
            else:
                logging.warning(
                    'DeepSpeedConfig: In FP32 mode, DeepSpeed does not permit MAX_GRAD_NORM ({}) > 0, setting to zero'
                    .format(self.optimizer_params[MAX_GRAD_NORM]))
                self.optimizer_params[MAX_GRAD_NORM] = 0.0
