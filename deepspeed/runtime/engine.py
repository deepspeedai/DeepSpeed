# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import stat
import torch
import hashlib
from collections import defaultdict, OrderedDict, deque
from shutil import copyfile
import gc

from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from typing import Callable, Dict, Union, Iterable

import deepspeed

from deepspeed import comm as dist
from deepspeed.runtime.utils import see_memory_usage, DummyOptim
from .zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.runtime.zero.utils import is_zero_supported_optimizer, ZeRORuntimeException
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.config import ZERO_OPTIMIZATION

from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS, \
    ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER, \
    TORCH_ADAM_PARAM, ADAM_W_MODE, ADAM_W_MODE_DEFAULT, ZERO_ONE_ADAM_OPTIMIZER, MUADAM_OPTIMIZER, MUADAMW_OPTIMIZER, \
    MUSGD_OPTIMIZER, LION_OPTIMIZER

from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.constants import \
    ROUTE_TRAIN, ROUTE_PREDICT, ROUTE_EVAL, \
    PLD_THETA, PLD_GAMMA, BFLOAT16, FP16, AMP, GRADIENT_ACCUMULATION_STEPS, \
    DATA_PARALLEL_GROUP, GLOBAL_RANK
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.compression import compression_scheduler
from deepspeed.compression.constants import \
    WEIGHT_QUANTIZE_IN_FORWARD_ENABLED, \
    WEIGHT_QUANTIZATION, SHARED_PARAMETERS, \
    WEIGHT_QUANTIZE_ENABLED, \
    WEIGHT_QUANTIZE_GROUPS, \
    WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE, \
    WEIGHT_QUANTIZE_CHANGE_RATIO, \
    WEIGHT_QUANTIZE_TYPE, \
    WEIGHT_QUANTIZE_ROUNDING, \
    WEIGHT_QUANTIZE_VERBOSE, \
    WEIGHT_QUANTIZE_KERNEL
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, FROZEN_PARAM_FRAGMENTS
from deepspeed.runtime.sparse_tensor import SparseTensor

from deepspeed.runtime import lr_schedules
from deepspeed.utils import groups
from deepspeed.utils import logger, log_dist, instrument_w_nvtx
from deepspeed.utils.timer import NoopTimer, ThroughputTimer, SynchronizedWallClockTimer, \
    FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER, \
    STEP_MICRO_TIMER, \
    FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER, \
    STEP_GLOBAL_TIMER
from deepspeed.utils.debug import debug_extract_module_and_param_names
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.progressive_layer_drop import ProgressiveLayerDrop
from deepspeed.runtime.utils import clip_grad_norm_
from deepspeed.runtime.eigenvalue import Eigenvalue
from deepspeed.runtime.data_pipeline.constants import DATA_SAMPLING, \
    DATA_ROUTING, DATA_SAMPLING_ENABLED, CURRICULUM_LEARNING, \
    CURRICULUM_LEARNING_ENABLED, DATA_SAMPLING_NUM_WORKERS, RANDOM_LTD, \
    RANDOM_LTD_ENABLED, RANDOM_LTD_LAYER_ID, RANDOM_LTD_LAYER_NUM, \
    RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE, RANDOM_LTD_LAYER_TOKEN_LR_ENABLED, \
    RANDOM_LTD_GLOBAL_BATCH_SIZE, RANDOM_LTD_MICRO_BATCH_SIZE, DATA_EFFICIENCY
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler
from deepspeed.runtime.data_pipeline.data_routing.scheduler import RandomLTDScheduler
from deepspeed.runtime.data_pipeline.data_routing.helper import remove_random_ltd_state_dict
from deepspeed.runtime.data_pipeline.data_routing.basic_layer import RandomLayerTokenDrop

from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from .pipe.module import PipelineModule
from .utils import get_ma_status
from ..ops.adam import FusedAdam
from ..moe.sharded_moe import TopKGate, MOELayer
from ..moe.layer import MoE
from ..moe.utils import is_moe_param
from ..git_version_info import version

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.utils.logging import print_json_dist, print_configuration

from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.config import DtypeEnum

MEMORY_OPT_ALLREDUCE_SIZE = 500000000

DeepSpeedOptimizerCallable = \
    Callable[[Union[Iterable[Parameter], Dict[str, Iterable]]], Optimizer]
DeepSpeedSchedulerCallable = Callable[[Optimizer], _LRScheduler]

try:
    import apex
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    APEX_INSTALLED = False


def split_half_float_double_sparse(tensors):
    device_type = get_accelerator().device_name()
    supported_types = [
        "torch.{}.HalfTensor".format(device_type), "torch.{}.FloatTensor".format(device_type),
        "torch.{}.DoubleTensor".format(device_type), "torch.{}.BFloat16Tensor".format(device_type),
        SparseTensor.type()
    ]

    for t in tensors:
        assert t.type() in supported_types, f"attempting to reduce an unsupported grad type: {t.type()}"

    buckets = []
    for i, dtype in enumerate(supported_types):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


class EngineTimers(object):
    r"""Wallclock timers for DeepSpeedEngine"""

    def __init__(self, enable_micro_timers, enable_global_timers):
        self.forward_timers = []
        self.backward_timers = []
        self.backward_inner_timers = []
        self.backward_reduce_timers = []
        self.step_timers = []
        self.global_timers = []
        self.micro_timers = []

        if enable_micro_timers:
            self.forward_timers += [FORWARD_MICRO_TIMER]
            self.backward_timers += [BACKWARD_MICRO_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_MICRO_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_MICRO_TIMER]
            self.step_timers += [STEP_MICRO_TIMER]
            self.micro_timers += [
                FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER,
                STEP_MICRO_TIMER
            ]

        if enable_global_timers:
            self.forward_timers += [FORWARD_GLOBAL_TIMER]
            self.backward_timers += [BACKWARD_GLOBAL_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_GLOBAL_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_GLOBAL_TIMER]
            self.step_timers += [STEP_GLOBAL_TIMER]
            self.global_timers += [
                FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER,
                STEP_GLOBAL_TIMER
            ]


class DeepSpeedEngine(Module):
    r"""DeepSpeed engine for training."""

    def __init__(
        self,
        args,
        model,
        optimizer=None,
        model_parameters=None,
        training_data=None,
        lr_scheduler=None,
        mpu=None,
        dist_init_required=None,
        collate_fn=None,
        config=None,
        config_class=None,
        dont_change_device=False,
    ):
        super(DeepSpeedEngine, self).__init__()
        self.dont_change_device = dont_change_device
        self.client_optimizer = optimizer
        self.client_lr_scheduler = lr_scheduler
        self.training_data = training_data
        self.collate_fn = collate_fn
        self.mpu = mpu
        self.all_to_all_group = None
        self.data_parallel_group = None
        self.global_steps = 0
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gradient_average = True
        self.warn_unscaled_loss = True
        self.config = config
        self._config = config_class
        self.loaded_checkpoint_mp_world_size = None
        self.loaded_checkpoint_dp_world_size = None
        self.enable_backward_allreduce = True
        self.progressive_layer_drop = None
        self.eigenvalue = None
        self.block_eigenvalue = None
        self.gas_boundary_ctr = 0
        self.dist_backend = get_accelerator().communication_backend_name()
        self.has_moe_layers = False
        self.num_experts = []
        self.gate_modules = []
        self.moe_layers = []
        self._step_applied = False
        self._global_grad_norm = None
        self.use_ds_comm = False  # False --> Use torch.dist, True --> Use ds.comm backend.

        self.checkpoint_engine = None

        self._is_gradient_accumulation_boundary = None
        self.scale_wrt_gas = None
        self.losses = 0.0

        # for debug purposes - can then debug print: debug_get_module_name(module)
        debug_extract_module_and_param_names(model)

        # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
        self.param_names = {param: name for name, param in model.named_parameters()}

        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)
        self._do_sanity_check()
        see_memory_usage(f"DeepSpeed Engine: After args sanity test", force=self.memory_breakdown())
        if mpu is not None:
            if self.elasticity_enabled():
                if not self.is_elastic_model_parallel_supported():
                    assert not self.elasticity_enabled(), ("Elasticity is not currently supported"
                                                           " with model parallelism.")

        self._set_distributed_vars(args)

        dist.configure(self._config)

        self.monitor = MonitorMaster(self._config.monitor_config)

        see_memory_usage(
            f"DeepSpeed Engine: Before configure distributed model",
            force=self.memory_breakdown(),
        )

        self.pipeline_parallelism = isinstance(model, PipelineModule)

        # Configure distributed model
        self._configure_distributed_model(model)

        self._get_model_parameters()

        see_memory_usage(f"DeepSpeed Engine: After configure distributed model")

        # Configure wall clock timers
        self.timers = SynchronizedWallClockTimer()
        # Throughput timer
        self.tput_timer = ThroughputTimer(
            batch_size=self.train_batch_size(),
            steps_per_output=self.steps_per_print(),
            monitor_memory=False,
        )

        log_dist(f"DeepSpeed Flops Profiler Enabled: {self.flops_profiler_enabled()}", ranks=[0])

        if self.flops_profiler_enabled():
            self.flops_profiler = FlopsProfiler(self.module, self, self.flops_profiler_recompute_fwd_factor())

        if training_data:
            self.training_dataloader = self.deepspeed_io(training_data)
        else:
            self.training_dataloader = None

        # Configure optimizer and scheduler
        self.optimizer = None
        self.basic_optimizer = None
        self.lr_scheduler = None
        has_optimizer = False

        if optimizer or self.optimizer_name():
            has_optimizer = True
        # If no parameters given by init default to module parameters
        if model_parameters is None:
            model_parameters = self.module.parameters()

        # Convert model parameters from generator to list
        if not isinstance(model_parameters, list):
            model_parameters = list(model_parameters)

        if has_optimizer:
            self._configure_optimizer(optimizer, model_parameters)
            self._configure_lr_scheduler(lr_scheduler)
            self._report_progress(0)
        elif self.zero_optimization():
            # no optim selected but zero is enabled
            self.optimizer = self._configure_zero_optimizer(optimizer=None)
        elif self.bfloat16_enabled():
            self.optimizer = self._configure_bf16_optimizer(optimizer=None)

        # Hook optimizer for snip_momentum pruning
        if hasattr(model, 'pruners'):
            from ..compression.helper import rewrite_optimizer_step
            self.optimizer.pruners = model.pruners
            rewrite_optimizer_step(self.optimizer)

        # Bookkeeping for sparse support
        self.sparse_tensor_module_names = set()
        # if self.sparse_gradients_enabled():
        for name, module in self.module.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and self.sparse_gradients_enabled():
                self.sparse_tensor_module_names.add(name + ".weight")
                logger.info("Will convert {} to sparse tensor during training".format(name))

        self.save_non_zero_checkpoint = False
        self.save_zero_checkpoint = False
        if not isinstance(self.optimizer, DeepSpeedZeRoOffload):
            self._configure_checkpointing(dist_init_required)

        if self.eigenvalue_enabled():
            self.eigenvalue = self._configure_eigenvalue()

        if self.pld_enabled():
            self.progressive_layer_drop = self._configure_progressive_layer_drop()

        if self.curriculum_enabled_legacy():
            self.curriculum_scheduler_legacy = self._configure_curriculum_scheduler_legacy()

        if self.random_ltd_enabled():
            random_ltd_config = self.random_ltd_config()
            random_ltd_config[RANDOM_LTD_GLOBAL_BATCH_SIZE] = self.train_batch_size()
            random_ltd_config[RANDOM_LTD_MICRO_BATCH_SIZE] = self.train_micro_batch_size_per_gpu()
            self.random_ltd_scheduler = self._configure_random_ltd_scheduler(random_ltd_config)

        # Engine timers

        self.engine_timers = EngineTimers(enable_micro_timers=self.wall_clock_breakdown(),
                                          enable_global_timers=self.wall_clock_breakdown()
                                          or self.flops_profiler_enabled())

        if self.global_rank == 0:
            self._config.print("DeepSpeedEngine configuration")
            if self.dump_state():
                print_configuration(self, "DeepSpeedEngine")

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

    def destroy(self):
        if self.optimizer is not None and hasattr(self.optimizer, 'destroy'):
            self.optimizer.destroy()

    def _get_model_parameters(self):
        if self.autotuning_profile_model_info():
            self.autotuning_model_info = {}
            num_params = 0
            trainable_num_params = 0

            for p in self.module.parameters():
                # since user code might call deepspeed.zero.Init() before deepspeed.initialize(), need to check the attribute to check if the parameter is partitioned in zero 3 already or not
                n = 0
                if hasattr(p, "ds_tensor"):  # if the parameter is partitioned in zero 3
                    n += p.ds_numel
                else:  # if the parameter is not partitioned in zero 3 yet
                    n += p.numel()
                num_params += n
                if p.requires_grad:
                    trainable_num_params += n
            if self.global_rank == 0:
                self.autotuning_model_info["num_params"] = num_params * self.mp_world_size
                self.autotuning_model_info["trainable_num_params"] = trainable_num_params * self.mp_world_size

            logger.info(f"model parameter = {num_params}")

    def get_batch_info(self):
        """Get all training batch related settings.
        Returns:
            train_batch_size (int): The effective training batch size. This is the amount of data
                samples that leads to one step of model update.
            train_micro_batch_size_per_gpu (int): Batch size to be processed by one GPU in one
                step (without gradient accumulation).
            gradient_accumulation_steps (int): Number of training steps to accumulate gradients
                before averaging and applying them.
        """
        return (
            self.train_batch_size,
            self.train_micro_batch_size_per_gpu,
            self.gradient_accumulation_steps,
        )

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        Raises:
            ValueError: if ``train_batch_size`` is not divisible by the
                configured micro-batch size and data parallelism.
        """
        if train_batch_size % (self.train_micro_batch_size_per_gpu() * self.dp_world_size) != 0:
            #print(f'{train_batch_size=} {self.train_micro_batch_size_per_gpu()=} {self.dp_world_size=}')
            raise ValueError(f'Train batch size must be divisible by micro-batch data parallelism')
        new_gas = train_batch_size // (self.train_micro_batch_size_per_gpu() * self.dp_world_size)
        # overwrite config
        self._config.train_batch_size = train_batch_size
        self._config.gradient_accumulation_steps = new_gas

    def set_train_micro_batch_size(self, micro_batch_size):
        """Adjust the micro batch size(i.e., the micro batch size in every data parallel group),
        while keep the gradient accumulation steps the same.
        Args:
            micro_batch_size (int): The new micro batch size for training.
        """
        # overwrite config
        new_global_batch_size = micro_batch_size * self._config.gradient_accumulation_steps * self.dp_world_size
        self._config.train_batch_size = new_global_batch_size
        self._config.train_micro_batch_size_per_gpu = micro_batch_size

    def set_data_post_process_func(self, post_process_func):
        if self.training_dataloader is not None:
            self.training_dataloader.post_process_func = post_process_func

    def set_custom_curriculum_learning_schedule(self, schedule_func_dict):
        if self.training_dataloader is not None and self.curriculum_learning_enabled():
            self.training_dataloader.data_sampler.set_custom_curriculum_learning_schedule(schedule_func_dict)

    def get_global_grad_norm(self) -> float:
        """Return the 2-norm of all gradients. If there is model parallelism,
        the norm will be global.
        The computed norm will be cached and reused until the next step() pass.
        .. note::
            In the presence of model parallelism, this is a collective call
            and acts as a barrier among ``mpu.get_model_parallel_group()``.
        Returns:
            float: norm
        """
        return self._global_grad_norm

    def __getattr__(self, name):
        """
        Pass through attributes defined in the model if they are not overridden by ds-engine.
        """

        _module = {}
        if "module" in self.__dict__:
            _module = self.__dict__['module']
        if name in dir(self):
            return getattr(self, name)
        elif name in dir(_module):
            return getattr(_module, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def checkpoint_tag_validation_enabled(self):
        return self._config.checkpoint_tag_validation_enabled

    def checkpoint_tag_validation_fail(self):
        return self._config.checkpoint_tag_validation_fail

    def elasticity_enabled(self):
        return self._config.elasticity_enabled

    def is_elastic_model_parallel_supported(self):
        if self.elasticity_enabled():
            # Add code for finding number of GPUs per node automatically
            if self._config.num_gpus_per_node % self._config.elastic_model_parallel_size == 0:
                return True
            else:
                return False

    def pld_enabled(self):
        return self._config.pld_enabled

    def pld_params(self):
        return self._config.pld_params

    def pld_theta(self):
        return self.pld_params()[PLD_THETA]

    def pld_gamma(self):
        return self.pld_params()[PLD_GAMMA]

    def eigenvalue_enabled(self):
        return self._config.eigenvalue_enabled

    def eigenvalue_verbose(self):
        return self._config.eigenvalue_verbose

    def eigenvalue_max_iter(self):
        return self._config.eigenvalue_max_iter

    def eigenvalue_tol(self):
        return self._config.eigenvalue_tol

    def eigenvalue_stability(self):
        return self._config.eigenvalue_stability

    def eigenvalue_gas_boundary_resolution(self):
        return self._config.eigenvalue_gas_boundary_resolution

    def eigenvalue_layer_name(self):
        return self._config.eigenvalue_layer_name

    def eigenvalue_layer_num(self):
        return self._config.eigenvalue_layer_num

    def curriculum_enabled_legacy(self):
        return self._config.curriculum_enabled_legacy

    def curriculum_params_legacy(self):
        return self._config.curriculum_params_legacy

    def data_efficiency_enabled(self):
        return self._config.data_efficiency_enabled

    def data_efficiency_config(self):
        return self._config.data_efficiency_config

    def data_sampling_enabled(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][DATA_SAMPLING_ENABLED]

    def data_sampling_config(self):
        return self._config.data_efficiency_config[DATA_SAMPLING]

    def curriculum_learning_enabled(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]

    def curriculum_learning_config(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING]

    def random_ltd_enabled(self):
        return self._config.data_efficiency_config[DATA_ROUTING][RANDOM_LTD][RANDOM_LTD_ENABLED]

    def random_ltd_config(self):
        return self._config.data_efficiency_config[DATA_ROUTING][RANDOM_LTD]

    def random_ltd_initialize(self):
        assert self.random_ltd_enabled()
        random_ltd_config = self.random_ltd_config()
        random_ltd_queue = deque([x for x in sorted(random_ltd_config[RANDOM_LTD_LAYER_ID])])
        count = 0
        for name, layer in self.module.named_modules():
            if isinstance(layer, RandomLayerTokenDrop):
                if len(random_ltd_queue) != 0 and str(random_ltd_queue[0]) in name:  ###[1,2,3]
                    layer.init_config(random_ltd_config, self.random_ltd_scheduler, count)
                    random_ltd_queue.popleft()
                    count += 1

        if random_ltd_config[RANDOM_LTD_LAYER_NUM] != count:
            raise ValueError(f'random_ltd_layer_num {random_ltd_config[RANDOM_LTD_LAYER_NUM]} must be \
                equivalent to the len of random_ltd_layer_id {count}')

        if random_ltd_config[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE][RANDOM_LTD_LAYER_TOKEN_LR_ENABLED]:
            assert self.client_lr_scheduler is None
            raise ValueError(f'not yet support')
            #self.lr_scheduler = lr_schedules.WarmupLayerTokenDecayLR(self.optimizer, self.random_ltd_scheduler)

    def wall_clock_breakdown(self):
        return self._config.wall_clock_breakdown

    def flops_profiler_enabled(self):
        return self._config.flops_profiler_config.enabled or self.autotuning_enabled()

    def flops_profiler_recompute_fwd_factor(self):
        return self._config.flops_profiler_config.recompute_fwd_factor

    def flops_profiler_profile_step(self):
        step = self._config.flops_profiler_config.profile_step
        if self._config.autotuning_config.enabled:
            step = self.autotuning_start_profile_step()
        return step

    def flops_profiler_module_depth(self):
        return self._config.flops_profiler_config.module_depth

    def flops_profiler_top_modules(self):
        return self._config.flops_profiler_config.top_modules

    def flops_profiler_detailed(self):
        if self._config.autotuning_config.enabled:
            return False
        return self._config.flops_profiler_config.detailed

    def flops_profiler_output_file(self):
        return self._config.flops_profiler_config.output_file

    def memory_breakdown(self):
        return self._config.memory_breakdown

    def autotuning_enabled(self):
        return self._config.autotuning_config.enabled

    def autotuning_start_profile_step(self):
        return self._config.autotuning_config.start_profile_step

    def autotuning_end_profile_step(self):
        return self._config.autotuning_config.end_profile_step

    def autotuning_metric_path(self):
        path = self._config.autotuning_config.metric_path
        if not path:
            path = os.path.join(os.getcwd(), "autotuning_metric.json")
        return path

    def autotuning_model_info_path(self):
        path = self._config.autotuning_config.model_info_path
        if not path:
            path = os.path.join(os.getcwd(), "autotuning_model_info.json")
        return path

    def autotuning_metric(self):
        return self._config.autotuning_config.metric

    def autotuning_profile_model_info(self):
        return self.autotuning_enabled(
        ) and self._config.autotuning_config.model_info and self._config.autotuning_config.model_info.get(
            "profile", False)

    def sparse_gradients_enabled(self):
        return self._config.sparse_gradients_enabled

    def train_batch_size(self):
        return self._config.train_batch_size

    def train_micro_batch_size_per_gpu(self):
        return self._config.train_micro_batch_size_per_gpu

    def optimizer_name(self):
        return (self.client_optimizer.__class__.__name__ if self.client_optimizer else self._config.optimizer_name)

    def optimizer_params(self):
        return self._config.optimizer_params

    def optimizer_legacy_fusion(self):
        return self._config.optimizer_legacy_fusion

    def scheduler_name(self):
        return self._config.scheduler_name

    def scheduler_params(self):
        return self._config.scheduler_params

    def quantize_training(self):
        return (
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS]
            [WEIGHT_QUANTIZE_IN_FORWARD_ENABLED],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_ENABLED],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_GROUPS],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS]
            [WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_CHANGE_RATIO],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_TYPE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_ROUNDING],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_VERBOSE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_KERNEL],
        )

    def zero_optimization(self):
        return self._config.zero_enabled

    def zero_allow_untested_optimizer(self):
        return self._config.zero_allow_untested_optimizer

    def zero_force_ds_cpu_optimizer(self):
        return self._config.zero_force_ds_cpu_optimizer

    def zero_reduce_scatter(self):
        return self._config.zero_config.reduce_scatter

    def zero_overlap_comm(self):
        return self._config.zero_config.overlap_comm

    def zero_offload_optimizer(self):
        return self._config.zero_config.offload_optimizer

    def zero_offload_param(self):
        return self._config.zero_config.offload_param

    def zero_use_cpu_optimizer(self):
        if self._config.zero_config.offload_optimizer is not None:
            return self._config.zero_config.offload_optimizer.device in [OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme]
        return False

    def zero_cpu_offload(self):
        if self._config.zero_config.offload_optimizer is not None:
            return self._config.zero_config.offload_optimizer.device == OffloadDeviceEnum.cpu
        return False

    def zero_sub_group_size(self):
        return self._config.zero_config.sub_group_size

    def zero_optimization_stage(self):
        return self._config.zero_optimization_stage

    def mics_shard_size(self):
        return self._config.mics_shard_size

    def zero_reduce_bucket_size(self):
        return self._config.zero_config.reduce_bucket_size

    def zero_allgather_bucket_size(self):
        return self._config.zero_config.allgather_bucket_size

    def zero_optimization_partition_gradients(self):
        return self.zero_optimization_stage() >= ZeroStageEnum.gradients

    def zero_optimization_partition_weights(self):
        return self.zero_optimization_stage() >= ZeroStageEnum.weights

    def is_first_weights_partition_group(self):
        ret = True if self.mics_shard_size() < 0 \
            and self.zero_optimization_partition_weights() else False
        if self.mics_shard_size() > 0 and self.global_rank < self.mics_shard_size():
            ret = True
        return ret

    def zero_contiguous_gradients(self):
        return self._config.zero_config.contiguous_gradients

    def zero_load_from_fp32_weights(self):
        return self._config.zero_config.load_from_fp32_weights

    def zero_elastic_checkpoint(self):
        return self._config.zero_config.elastic_checkpoint

    def zero_max_live_parameters(self):
        return self._config.zero_config.max_live_parameters

    def zero_max_reuse_distance(self):
        return self._config.zero_config.max_reuse_distance

    def zero_prefetch_bucket_size(self):
        return self._config.zero_config.prefetch_bucket_size

    def zero_param_persistence_threshold(self):
        return self._config.zero_config.param_persistence_threshold

    def zero_model_persistence_threshold(self):
        return self._config.zero_config.model_persistence_threshold

    def zero_gather_16bit_weights_on_model_save(self):
        return self._config.zero_config.gather_16bit_weights_on_model_save

    def zero_grad_hooks(self):
        return self._config.zero_config.grad_hooks

    def zero_legacy_stage1(self):
        return self._config.zero_config.legacy_stage1

    def zero_ignore_unused_parameters(self):
        return self._config.zero_config.ignore_unused_parameters

    def fp16_enabled(self):
        return self._config.fp16_enabled

    def bfloat16_enabled(self):
        return self._config.bfloat16_enabled

    def fp16_master_weights_and_gradients(self):
        return self._config.fp16_master_weights_and_gradients

    def amp_enabled(self):
        return self._config.amp_enabled

    def amp_params(self):
        return self._config.amp_params

    def fp16_auto_cast(self):
        return self._config.fp16_auto_cast

    def loss_scale(self):
        return self._config.loss_scale

    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    def use_node_local_storage(self):
        return self._config.use_node_local_storage

    def load_universal_checkpoint(self):
        return self._config.load_universal_checkpoint

    @property
    def communication_data_type(self):
        res = self._config.communication_data_type
        if res is not None:
            return res

        if self.fp16_enabled():
            return torch.float16

        if self.bfloat16_enabled():
            return torch.bfloat16

        return torch.float32

    def postscale_gradients(self):
        return not self._config.prescale_gradients

    def gradient_predivide_factor(self):
        return self._config.gradient_predivide_factor

    def steps_per_print(self):
        return self._config.steps_per_print

    def zero_allgather_partitions(self):
        return self._config.zero_config.allgather_partitions

    def zero_round_robin_gradients(self):
        return self._config.zero_config.round_robin_gradients

    def zero_hpz_partition_size(self):
        return self._config.zero_config.zero_hpz_partition_size

    def zero_quantized_weights(self):
        return self._config.zero_config.zero_quantized_weights

    def zero_quantized_nontrainable_weights(self):
        return self._config.zero_config.zero_quantized_nontrainable_weights

    def zero_quantized_gradients(self):
        return self._config.zero_config.zero_quantized_gradients

    def dump_state(self):
        return self._config.dump_state

    def gradient_clipping(self):
        return self._config.gradient_clipping

    def dynamic_loss_scale(self):
        return self._config.loss_scale == 0

    def initial_dynamic_scale(self):
        return self._config.initial_dynamic_scale

    def dynamic_loss_scale_args(self):
        return self._config.dynamic_loss_scale_args

    def swap_tensor_config(self):
        return self._config.swap_tensor_config

    def aio_config(self):
        return self._config.aio_config

    def get_data_types(self):
        model_dtype = torch.float32
        if self.fp16_enabled():
            model_dtype = torch.float16
        elif self.bfloat16_enabled():
            model_dtype = torch.bfloat16

        if self._config.grad_accum_dtype is None:
            if model_dtype == torch.bfloat16 and not self.zero_optimization():
                grad_accum_dtype = torch.float32
            else:
                grad_accum_dtype = model_dtype
        else:
            grad_accum_dtype = DtypeEnum(self._config.grad_accum_dtype).value

        return (model_dtype, grad_accum_dtype)

    def _optimizer_has_ckpt_event_prologue(self):
        return self.optimizer is not None and hasattr(self.optimizer, 'checkpoint_event_prologue')

    def _optimizer_has_ckpt_event_epilogue(self):
        return self.optimizer is not None and hasattr(self.optimizer, 'checkpoint_event_epilogue')

    def _configure_lr_scheduler(self, client_lr_scheduler):
        # First check for scheduler in json configuration
        lr_scheduler = self._scheduler_from_config(self.optimizer)
        if lr_scheduler:
            log_dist(f"DeepSpeed using configured LR scheduler = {self.scheduler_name()}", ranks=[0])
            self.lr_scheduler = lr_scheduler
        else:
            if isinstance(client_lr_scheduler, Callable):
                log_dist('DeepSpeed using client callable to create LR scheduler', ranks=[0])
                self.lr_scheduler = client_lr_scheduler(self.basic_optimizer)
            else:
                log_dist('DeepSpeed using client LR scheduler', ranks=[0])
                self.lr_scheduler = client_lr_scheduler

        log_dist(f'DeepSpeed LR Scheduler = {self.lr_scheduler}', ranks=[0])

    def _configure_checkpointing(self, dist_init_required):
        self.checkpoint_engine = TorchCheckpointEngine()

        if self._config is not None and self._config.nebula_config.enabled:
            try:
                from deepspeed.runtime.checkpoint_engine.nebula_checkpoint_engine import \
                    NebulaCheckpointEngine
                self.checkpoint_engine = NebulaCheckpointEngine(config_params=self._config.nebula_config)
            except ImportError as err:
                logger.error(f"No torch_nebula was found! Will fall back to torch.save. Details: {err}")
                self.checkpoint_engine = TorchCheckpointEngine()

        dp_rank = groups._get_sequence_data_parallel_rank()

        rank = self.local_rank if self.use_node_local_storage() else dp_rank

        # only the first data parallel process needs to store the model checkpoint
        # if you want to use node local storage this must be done by rank 0 on each
        # node
        self.save_non_zero_checkpoint = (rank == 0) or (self.zero_optimization_partition_weights()
                                                        and self.is_first_weights_partition_group())

        if self.zero_optimization() or self.bfloat16_enabled():
            param_rank = dist.get_rank(group=self.optimizer.dp_process_group)

            # Only the first parameter parallel process needs to store the
            # optimizer state checkpoints for zero
            self.save_zero_checkpoint = param_rank == dp_rank

    def _scheduler_from_config(self, optimizer):
        scheduler_name = self.scheduler_name()
        if scheduler_name is not None:
            if hasattr(lr_schedules, scheduler_name):
                scheduler = getattr(lr_schedules, scheduler_name)
            else:
                assert hasattr(torch.optim.lr_scheduler,
                               scheduler_name), f"DeepSpeed does not recognize LR scheduler {scheduler_name}"

                scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)

            scheduler_params = self.scheduler_params()
            instantiated_scheduler = scheduler(optimizer, **scheduler_params)
            return instantiated_scheduler
        else:
            return None

    def _set_distributed_vars(self, args):
        device_rank = args.device_rank if args is not None and hasattr(args, 'device_rank') else self.local_rank
        if device_rank >= 0:
            get_accelerator().set_device(device_rank)
            self.device = torch.device(get_accelerator().device_name(), device_rank)
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device(get_accelerator().device_name())

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        # After the distributed backend is initialized we are guaranteed the LOCAL_RANK
        # environment variable is set. We must align args.local_rank to this value for
        # backwards compatibility with scripts relying on [args|self].local_rank containing
        # the correct local rank info. _do_args_sanity_check will ensure this is the case.

        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            ompi_local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            local_rank = os.environ.get('LOCAL_RANK', ompi_local_rank)
            assert ompi_local_rank == local_rank, f"LOCAL_RANK ({local_rank}) != OMPI_COMM_WORLD_LOCAL_RANK ({ompi_local_rank}), " \
                "not sure how to proceed as we're seeing conflicting local rank info."
            os.environ['LOCAL_RANK'] = local_rank

        self.local_rank = int(os.environ['LOCAL_RANK'])
        if hasattr(args, 'local_rank'):
            args.local_rank = self.local_rank

    # Validate command line arguments
    def _do_args_sanity_check(self, args):
        assert "LOCAL_RANK" in os.environ or "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ, "DeepSpeed requires the LOCAL_RANK environment " \
            "variable, it is set by the deepspeed launcher, deepspeed.init_distributed, or the torch's launcher. If using a " \
            "different launcher please ensure LOCAL_RANK is set prior to initializing deepspeed."

        if hasattr(args, 'local_rank') and args.local_rank is not None:
            assert isinstance(args.local_rank,
                              int), f"args.local_rank of {args.local_rank} is an unknown type {type(args.local_rank)}"
            if args.local_rank >= 0:
                env_local_rank = int(os.environ.get("LOCAL_RANK"))
                assert (
                    env_local_rank == args.local_rank
                ), f"Mismatch in local rank setting, args.local_rank={args.local_rank} but env['LOCAL_RANK']={env_local_rank}."

    def _is_supported_optimizer(self, optimizer_name):
        return (optimizer_name in DEEPSPEED_OPTIMIZERS or getattr(torch.optim, optimizer_name, None) is not None)

    def _supported_optims(self):
        FairseqOptimizer = None
        try:
            from fairseq.optim.fairseq_optimizer import FairseqOptimizer
        except ImportError:
            pass

        expected_optim_types = [Optimizer]
        if FairseqOptimizer:
            # fairseq optims are not torch.optim objects
            expected_optim_types.append(FairseqOptimizer)
        return expected_optim_types

    # Validate configuration based on command line arguments
    def _do_sanity_check(self):
        expected_optim_types = self._supported_optims()
        expected_optim_types += [type(None), Callable]
        assert isinstance(self.client_optimizer, tuple(expected_optim_types)), \
            f'Client Optimizer is of unexpected type {type(self.client_optimizer)}'

        if not self.client_optimizer:
            if self.optimizer_name() is not None:
                assert self._is_supported_optimizer(
                    self.optimizer_name()), "{} is not a supported DeepSpeed Optimizer".format(self.optimizer_name())

        if (self.optimizer_name() == LAMB_OPTIMIZER or self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER):
            assert (self.dynamic_loss_scale()), "DeepSpeed {} optimizer requires dynamic loss scaling".format(
                self.optimizer_name())

        # Detect invalid combinations of client optimizer and client scheduler
        if isinstance(self.client_lr_scheduler, _LRScheduler):
            assert isinstance(self.client_optimizer, Optimizer), \
                f'Client Optimizer (type = {type(self.client_optimizer)} is not instantiated but Client LR Scheduler is instantiated'

    def _broadcast_model(self):

        def is_replicated(p):
            if hasattr(p, "ds_status") and p.ds_status is not ZeroParamStatus.AVAILABLE:
                return False
            return True

        for p in self.module.parameters():
            # Broadcast the model for different parameters
            if is_moe_param(p):
                if torch.is_tensor(p) and is_replicated(p):
                    dist.broadcast(p,
                                   groups._get_expert_broadcast_src_rank(p.group_name),
                                   group=self.expert_data_parallel_group[p.group_name])
            else:
                if torch.is_tensor(p) and is_replicated(p):
                    dist.broadcast(p, groups._get_broadcast_src_rank(), group=self.seq_data_parallel_group)

    @staticmethod
    def __check_params(model: Module, dtype: torch.dtype) -> None:
        return
        if not all(param.dtype == dtype for param in model.parameters()) and dist.get_rank() == 0:
            raise ValueError(f"{dtype} is enabled but the following parameters have dtype that is "
                             f"not {dtype}: "
                             f"{[(n, p.dtype) for n, p in model.named_parameters() if p.dtype != dtype]}")

    def _set_client_model(self, model):
        # register client model in _modules so that nn.module methods work correctly
        modules = self.__dict__.get('_modules')
        modules['module'] = model
        # register module attribute in engine but avoid getattr
        self.__dict__['module'] = model

    def _configure_distributed_model(self, model):
        self._set_client_model(model)
        is_zero_init_model = self.zero_optimization_partition_weights() and any(
            [hasattr(param, "ds_id") for param in self.module.parameters()])

        if self.fp16_enabled():
            if is_zero_init_model:
                self.__check_params(self.module, torch.half)
            self.module.half()
        elif self.bfloat16_enabled():
            if is_zero_init_model:
                self.__check_params(self.module, torch.bfloat16)
            self.module.bfloat16()
        else:
            self.__check_params(self.module, torch.float)

        # zero.Init() handles device placement of model
        if not (self.dont_change_device or is_zero_init_model):
            self.module.to(self.device)

        # MoE related initialization
        for _, module in self.module.named_modules():
            if isinstance(module, MoE):
                self.has_moe_layers = True
                self.num_experts.append(module.num_experts)

        if self.has_moe_layers:
            for _, module in self.module.named_modules():
                if isinstance(module, TopKGate):
                    self.gate_modules.append(module)
                    if self.wall_clock_breakdown():
                        module.wall_clock_breakdown = True
                if isinstance(module, MOELayer):
                    self.moe_layers.append(module)
                    if self.wall_clock_breakdown():
                        module.wall_clock_breakdown = True

        # Pass the mpu from here to groups. For subsequent use, just query groups
        if self.mpu is not None:
            groups.mpu = self.mpu

        # Set deepspeed parallelism spec. for the model including expert parallelism
        for _, module in self.module.named_modules():
            if hasattr(module, 'set_deepspeed_parallelism'):
                module.set_deepspeed_parallelism()

        # Query the groups module to get information about various parallel groups
        self.local_all_to_all_group = None
        if self.zero_quantized_gradients():
            log_dist("Using quantized gradients", ranks=[0])
            self.local_all_to_all_group = groups._get_local_all_to_all_group()
        self.data_parallel_group = groups._get_data_parallel_group()
        self.dp_world_size = groups._get_data_parallel_world_size()
        self.seq_data_parallel_group = groups._get_sequence_data_parallel_group()
        self.seq_dp_world_size = groups._get_sequence_data_parallel_world_size()
        self.mp_world_size = groups._get_model_parallel_world_size()
        self.expert_parallel_group = groups._get_expert_parallel_group_dict()
        self.expert_data_parallel_group = groups._get_expert_data_parallel_group_dict()

        if not (self.amp_enabled() or is_zero_init_model):
            self._broadcast_model()

    # check if parameters are duplicated in optimizer param_groups
    def _check_for_duplicates(self, optimizer):
        for name, param in self.module.named_parameters():
            param_id = id(param)

            def ids_list(group):
                return [id(param) for param in group]

            occurrence = sum([
                ids_list(group['params']).count(param_id) if param_id in ids_list(group['params']) else 0
                for group in optimizer.param_groups
            ])
            assert occurrence <= 1, f"Parameter with name: {name} occurs multiple times in optimizer.param_groups. Make sure it only appears once to prevent undefined behavior."

    def _do_optimizer_sanity_check(self, basic_optimizer):
        model_dtype, grad_accum_dtype = self.get_data_types()
        zero_enabled = self.zero_optimization()
        amp_enabled = self.amp_enabled()
        # config based assertions
        assert (
            not (amp_enabled and zero_enabled)
        ), "Amp and ZeRO are not currently compatible, please use (legacy) fp16 mode which performs similar to amp opt_mode=O2"
        if zero_enabled:
            if not is_zero_supported_optimizer(basic_optimizer):
                assert (
                    self.zero_allow_untested_optimizer()
                ), 'You are using an untested ZeRO Optimizer. Please add <"zero_allow_untested_optimizer": true> in the configuration file to use it.'

                if self.global_rank == 0:
                    logger.warning("**** You are using ZeRO with an untested optimizer, proceed with caution *****")
            if model_dtype == torch.bfloat16 and grad_accum_dtype == torch.float32 and self.zero_optimization_stage(
            ) == 1 and not self.zero_cpu_offload():
                return BFLOAT16
            return ZERO_OPTIMIZATION
        elif amp_enabled:
            if model_dtype != grad_accum_dtype:
                raise NotImplementedError(
                    "Model data type and gradient accumulation data type must be equal to use Amp")
            if model_dtype == torch.bfloat16 or model_dtype == torch.float16:
                raise NotImplementedError("Cannot enable both amp with (legacy) fp16 or bfloat16 mode")
            try:
                logger.info("Initializing Apex amp from: {}".format(amp.__path__))
            except NameError:
                # If apex/amp is available it will be imported above
                raise RuntimeError("Unable to import apex/amp, please make sure it is installed")
            return AMP
        # data type checks
        elif model_dtype == grad_accum_dtype:
            if model_dtype == torch.bfloat16:
                raise NotImplementedError(
                    "Bfloat16 wrapper must use a gradient accumulation type of fp32, enable ZeRO to use Bfloat16 gradient accumulation"
                )
            if model_dtype == torch.float16:
                return FP16
            # else optimizer_wrapper = None
        elif model_dtype == torch.bfloat16 and grad_accum_dtype == torch.float32:
            return BFLOAT16
        else:
            raise NotImplementedError("unsupported mix of model dtype and gradient accumulation type")

        return None

    # Configure optimizer
    def _configure_optimizer(self, client_optimizer, model_parameters):
        if client_optimizer is None:
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            log_dist(f"Using DeepSpeed Optimizer param name {self.optimizer_name()} as basic optimizer", ranks=[0])
        else:
            if isinstance(client_optimizer, tuple(self._supported_optims())):
                basic_optimizer = client_optimizer
                log_dist('Using client Optimizer as basic optimizer', ranks=[0])
            else:
                basic_optimizer = client_optimizer(model_parameters)
                log_dist('Using client callable to create basic optimizer', ranks=[0])

            if self.zero_use_cpu_optimizer() and not isinstance(basic_optimizer, deepspeed.ops.adam.DeepSpeedCPUAdam):
                if self.zero_force_ds_cpu_optimizer():
                    msg = f'You are using ZeRO-Offload with a client provided optimizer ({type(basic_optimizer)}) which in most cases will yield poor performance. Please either use deepspeed.ops.adam.DeepSpeedCPUAdam or set an optimizer in your ds-config (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters). If you really want to use a custom optimizer w. ZeRO-Offload and understand the performance impacts you can also set <"zero_force_ds_cpu_optimizer": false> in your configuration file.'
                    raise ZeRORuntimeException(msg)

        basic_optimizer.param_groups[:] = [pg for pg in basic_optimizer.param_groups if len(pg["params"]) != 0]
        log_dist("Removing param_group that has no 'params' in the basic Optimizer", ranks=[0])

        self._check_for_duplicates(basic_optimizer)

        self.basic_optimizer = basic_optimizer
        log_dist("DeepSpeed Basic Optimizer = {}".format(basic_optimizer.__class__.__name__), ranks=[0])

        optimizer_wrapper = self._do_optimizer_sanity_check(basic_optimizer)

        if optimizer_wrapper == ZERO_OPTIMIZATION:
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)
        elif optimizer_wrapper == AMP:
            amp_params = self.amp_params()
            log_dist(f"Initializing AMP with these params: {amp_params}", ranks=[0])
            model, self.optimizer = amp.initialize(self.module, basic_optimizer, **amp_params)
            self._set_client_model(model)
            self._broadcast_model()
            # TODO: maybe need to broadcast experts differently?
        elif optimizer_wrapper == FP16:
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer)
        elif optimizer_wrapper == BFLOAT16:
            self.optimizer = self._configure_bf16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer

        log_dist("DeepSpeed Final Optimizer = {}".format(self.optimizer_name()), ranks=[0])

        self.compression_scheduler = self._configure_compression_scheduler()
        self.quantizer = self._configure_quantization()

    def _configure_basic_optimizer(self, model_parameters):
        optimizer_parameters = self.optimizer_params()
        if optimizer_parameters is None:
            optimizer_parameters = {}
        # print(optimizer_parameters.keys())
        if "max_grad_norm" in optimizer_parameters.keys():
            raise ValueError(
                "'max_grad_norm' is not supported as an optimizer parameter, please switch to using the deepspeed parameter 'gradient_clipping' see: https://www.deepspeed.ai/docs/config-json/#gradient-clipping for more details"
            )

        if self.optimizer_name() in [ADAM_OPTIMIZER, ADAMW_OPTIMIZER]:
            torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
            adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)

            # Optimizer name of Adam forces AdamW logic unless adam_w_mode is explicitly set
            effective_adam_w_mode = self.optimizer_name() == ADAMW_OPTIMIZER or adam_w_mode

            if torch_adam:
                if not effective_adam_w_mode:
                    optimizer = torch.optim.Adam(model_parameters, **optimizer_parameters)
                else:
                    optimizer = torch.optim.AdamW(model_parameters, **optimizer_parameters)
            else:
                if self.zero_use_cpu_optimizer():
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    optimizer = DeepSpeedCPUAdam(model_parameters,
                                                 **optimizer_parameters,
                                                 adamw_mode=effective_adam_w_mode)
                else:
                    from deepspeed.ops.adam import FusedAdam

                    optimizer = FusedAdam(
                        model_parameters,
                        **optimizer_parameters,
                        adam_w_mode=effective_adam_w_mode,
                    )

        elif self.optimizer_name() == ADAGRAD_OPTIMIZER:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
                optimizer = DeepSpeedCPUAdagrad(model_parameters, **optimizer_parameters)
            else:
                optimizer = torch.optim.Adagrad(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == LAMB_OPTIMIZER:
            from deepspeed.ops.lamb import FusedLamb

            optimizer = FusedLamb(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.adam import OnebitAdam

            optimizer = OnebitAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f"Currently the convergence of 1-bit Adam is only verified under FP16")
        elif self.optimizer_name() == ZERO_ONE_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "0/1 Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

            optimizer = ZeroOneAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f'Currently the convergence of 0/1 Adam is only verified under FP16')
        elif self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Lamb is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb

            optimizer = OnebitLamb(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f"Currently the convergence of 1-bit Lamb is only verified under FP16")
        elif self.optimizer_name() == LION_OPTIMIZER:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.lion import DeepSpeedCPULion
                optimizer = DeepSpeedCPULion(model_parameters, **optimizer_parameters)
            else:
                from deepspeed.ops.lion import FusedLion
                optimizer = FusedLion(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUADAM_OPTIMIZER:
            try:
                from mup import MuAdam
            except ImportError:
                logger.error(f"Install mup to use MuAdam optimizer")
            optimizer = MuAdam(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUADAMW_OPTIMIZER:
            try:
                from mup import MuAdamW
            except ImportError:
                logger.error(f"Install mup to use MuAdamW optimizer")
            optimizer = MuAdamW(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUSGD_OPTIMIZER:
            try:
                from mup import MuSGD
            except ImportError:
                logger.error(f"Install mup to use MuSGD optimizer")
            optimizer = MuSGD(model_parameters, **optimizer_parameters)
        else:
            torch_optimizer = getattr(torch.optim, self.optimizer_name())
            optimizer = torch_optimizer(model_parameters, **optimizer_parameters)
        return optimizer

    def _configure_compression_scheduler(self):
        return compression_scheduler(self.module, self._config.compression_config)

    def _configure_random_ltd_scheduler(self, configs):
        return RandomLTDScheduler(configs)

    def _configure_quantization(self):
        (
            quantize_weight_in_forward,
            quantize_enabled,
            q_groups,
            q_mixed_fp16,
            q_change_ratio,
            q_type,
            q_rounding,
            q_verbose,
            use_quantizer_kernel,
        ) = self.quantize_training()
        if quantize_enabled and not quantize_weight_in_forward:
            assert self.fp16_enabled(
            ), "MoQ (quantize in optimization step) weight quantization is only supported for FP16"
        quantizer = None
        if quantize_enabled and not quantize_weight_in_forward:
            from deepspeed.runtime.quantize import Quantizer

            quantizer = Quantizer(
                q_groups,
                q_mixed_fp16,
                q_change_ratio,
                q_type,
                q_rounding,
                q_verbose,
                self.eigenvalue_enabled(),
                use_quantizer_kernel,
                self.eigenvalue_layer_num() if self.eigenvalue_enabled() else 0,
            )
        return quantizer

    def _configure_fp16_optimizer(self, optimizer):
        initial_dynamic_scale = self.initial_dynamic_scale()
        dynamic_loss_args = self.dynamic_loss_scale_args()
        clip_grad = self.gradient_clipping()
        if APEX_INSTALLED:
            fused_opts = (apex.optimizers.FusedAdam, FusedAdam)
        else:
            fused_opts = FusedAdam
        if isinstance(optimizer, fused_opts) \
                or self.optimizer_name() in [ONEBIT_ADAM_OPTIMIZER, ZERO_ONE_ADAM_OPTIMIZER]:
            if self.dynamic_loss_scale():
                log_dist(f'Creating fp16 optimizer with dynamic loss scale', ranks=[0])
                timers = self.timers if self.wall_clock_breakdown() else NoopTimer()
                optimizer = FP16_Optimizer(
                    optimizer,
                    deepspeed=self,
                    dynamic_loss_scale=True,
                    initial_dynamic_scale=initial_dynamic_scale,
                    dynamic_loss_args=dynamic_loss_args,
                    mpu=self.mpu,
                    clip_grad=clip_grad,
                    fused_adam_legacy=self.optimizer_legacy_fusion(),
                    timers=timers,
                    has_moe_layers=self.has_moe_layers,
                )
            else:
                log_dist(f'Creating fp16 optimizer with static loss scale: {self.loss_scale()}', ranks=[0])
                optimizer = FP16_Optimizer(
                    optimizer,
                    deepspeed=self,
                    static_loss_scale=self.loss_scale(),
                    mpu=self.mpu,
                    clip_grad=clip_grad,
                    fused_adam_legacy=self.optimizer_legacy_fusion(),
                    has_moe_layers=self.has_moe_layers,
                )
        else:
            log_dist(f'Creating fp16 unfused optimizer with dynamic loss scale', ranks=[0])
            optimizer = FP16_UnfusedOptimizer(
                optimizer,
                deepspeed=self,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_lamb_legacy=self.optimizer_name() == LAMB_OPTIMIZER,
            )

        return optimizer

    def _configure_bf16_optimizer(self, optimizer):
        clip_grad = self.gradient_clipping()

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        log_dist('Creating BF16 optimizer', ranks=[0])

        timers = self.timers if self.wall_clock_breakdown() else NoopTimer()
        optimizer = BF16_Optimizer(optimizer,
                                   self.param_names,
                                   mpu=self.mpu,
                                   clip_grad=clip_grad,
                                   allgather_bucket_size=self.zero_allgather_bucket_size(),
                                   dp_process_group=self.seq_data_parallel_group,
                                   timers=timers)

        return optimizer

    def _configure_zero_optimizer(self, optimizer):
        zero_stage = self.zero_optimization_stage()

        mics_shard_size = self.mics_shard_size()
        model_dtype, gradient_accumulation_dtype = self.get_data_types()

        timers = self.timers if self.wall_clock_breakdown() else NoopTimer()

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        if self.zero_legacy_stage1():
            raise Exception(
                "The deprecated version of ZeRO Stage 1 is not supported in deepspeed >= 0.5.9. Please downgrade to a version less than 0.5.9 if you need to use this deprecated version of ZeRO."
            )

        if zero_stage <= ZeroStageEnum.gradients:
            overlap_comm = self.zero_overlap_comm()
            contiguous_gradients = self.zero_contiguous_gradients()
            round_robin_gradients = self.zero_round_robin_gradients()
            assert not isinstance(optimizer, DummyOptim), "zero stage {} requires an optimizer".format(zero_stage)

            log_dist(f'Creating {model_dtype} ZeRO stage {zero_stage} optimizer', ranks=[0])
            # Overlap and contiguous grads are meaningless in stage 1 and are ignored
            if zero_stage == ZeroStageEnum.optimizer_states:
                overlap_comm = False
                round_robin_gradients = False
                # Non-MoE requires contiguous grads to be disabled w. stage 1
                if not self.has_moe_layers:
                    contiguous_gradients = False

            if isinstance(self.module, PipelineModule):
                if overlap_comm:
                    logger.warning("Pipeline parallelism does not support overlapped communication, will be disabled.")
                    overlap_comm = False
            optimizer = DeepSpeedZeroOptimizer(
                optimizer,
                self.param_names,
                timers=timers,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                contiguous_gradients=contiguous_gradients,
                reduce_bucket_size=self.zero_reduce_bucket_size(),
                allgather_bucket_size=self.zero_allgather_bucket_size(),
                dp_process_group=self.seq_data_parallel_group,
                expert_parallel_group=self.expert_parallel_group if self.has_moe_layers else None,
                expert_data_parallel_group=self.expert_data_parallel_group if self.has_moe_layers else None,
                reduce_scatter=self.zero_reduce_scatter(),
                overlap_comm=overlap_comm,
                offload_optimizer_config=self.zero_offload_optimizer(),
                mpu=self.mpu,
                postscale_gradients=self.postscale_gradients(),
                gradient_predivide_factor=self.gradient_predivide_factor(),
                gradient_accumulation_steps=self.gradient_accumulation_steps(),
                ignore_unused_parameters=self.zero_ignore_unused_parameters(),
                partition_grads=zero_stage == ZeroStageEnum.gradients,
                round_robin_gradients=round_robin_gradients,
                has_moe_layers=self.has_moe_layers,
                fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(),
                gradient_accumulation_dtype=gradient_accumulation_dtype,
                communication_data_type=self.communication_data_type,
                elastic_checkpoint=self.zero_elastic_checkpoint())

        elif zero_stage == ZeroStageEnum.weights:
            assert not self.has_moe_layers, "MoE not supported with Stage 3"
            if isinstance(optimizer, DummyOptim):
                log_dist("Creating ZeRO Offload", ranks=[0])
                zero_param_parallel_group = groups._get_zero_param_intra_parallel_group()
                if self.zero_hpz_partition_size() > 1 and zero_param_parallel_group is None:
                    self._set_zero_group_parallelism()
                    zero_param_parallel_group = groups._get_zero_param_intra_parallel_group()
                optimizer = DeepSpeedZeRoOffload(
                    self.module,
                    timers=timers,
                    ds_config=self.config,
                    overlap_comm=self.zero_overlap_comm(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    offload_param_config=self.zero_offload_param(),
                    mpu=self.mpu,
                    zero_param_parallel_group=zero_param_parallel_group,
                    zero_quantized_weights=self.zero_quantized_weights(),
                    zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights(),
                )
            else:
                log_dist(
                    f'Creating fp16 ZeRO stage {zero_stage} optimizer,'
                    f' MiCS is enabled {mics_shard_size>0},'
                    f' Hierarchical params gather {self._config.mics_hierarchial_params_gather}',
                    ranks=[0])
                if mics_shard_size > 0:
                    return self._return_mics_optimizer(optimizer, timers)

                log_dist(f'Creating {model_dtype} ZeRO stage {zero_stage} optimizer', ranks=[0])
                from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
                optimizer = DeepSpeedZeroOptimizer_Stage3(
                    self.module,
                    optimizer,
                    timers=timers,
                    ds_config=self.config,
                    static_loss_scale=self.loss_scale(),
                    dynamic_loss_scale=self.dynamic_loss_scale(),
                    dynamic_loss_args=self.dynamic_loss_scale_args(),
                    clip_grad=self.gradient_clipping(),
                    contiguous_gradients=self.zero_contiguous_gradients(),
                    reduce_bucket_size=self.zero_reduce_bucket_size(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    dp_process_group=self.seq_data_parallel_group,
                    all2all_process_group=self.local_all_to_all_group,
                    reduce_scatter=self.zero_reduce_scatter(),
                    overlap_comm=self.zero_overlap_comm(),
                    offload_optimizer_config=self.zero_offload_optimizer(),
                    offload_param_config=self.zero_offload_param(),
                    sub_group_size=self.zero_sub_group_size(),
                    mpu=self.mpu,
                    postscale_gradients=self.postscale_gradients(),
                    gradient_predivide_factor=self.gradient_predivide_factor(),
                    gradient_accumulation_steps=self.gradient_accumulation_steps(),
                    aio_config=self.aio_config(),
                    gradient_accumulation_dtype=gradient_accumulation_dtype,
                    communication_data_type=self.communication_data_type,
                    zero_hpz_partition_size=self.zero_hpz_partition_size(),
                    zero_quantized_weights=self.zero_quantized_weights(),
                    zero_quantized_nontrainable_weights=self.zero_quantized_nontrainable_weights(),
                )

        else:
            raise NotImplementedError("ZeRO stage {} not implemented".format(zero_stage))

        return optimizer

    def _return_mics_optimizer(self, basic_optimizer, timers):
        from deepspeed.runtime.zero.mics import MiCS_Optimizer
        model_dtype, gradient_accumulation_dtype = self.get_data_types()
        optimizer = MiCS_Optimizer(self.module,
                                   basic_optimizer,
                                   timers=timers,
                                   ds_config=self.config,
                                   static_loss_scale=self.loss_scale(),
                                   dynamic_loss_scale=self.dynamic_loss_scale(),
                                   dynamic_loss_args=self.dynamic_loss_scale_args(),
                                   clip_grad=self.gradient_clipping(),
                                   contiguous_gradients=self.zero_contiguous_gradients(),
                                   reduce_bucket_size=self.zero_reduce_bucket_size(),
                                   prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                                   max_reuse_distance=self.zero_max_reuse_distance(),
                                   max_live_parameters=self.zero_max_live_parameters(),
                                   param_persistence_threshold=self.zero_param_persistence_threshold(),
                                   model_persistence_threshold=self.zero_model_persistence_threshold(),
                                   dp_process_group=self.seq_data_parallel_group,
                                   reduce_scatter=self.zero_reduce_scatter(),
                                   overlap_comm=self.zero_overlap_comm(),
                                   offload_optimizer_config=self.zero_offload_optimizer(),
                                   offload_param_config=self.zero_offload_param(),
                                   sub_group_size=self.zero_sub_group_size(),
                                   mpu=self.mpu,
                                   postscale_gradients=self.postscale_gradients(),
                                   gradient_predivide_factor=self.gradient_predivide_factor(),
                                   gradient_accumulation_steps=self.gradient_accumulation_steps(),
                                   aio_config=self.aio_config(),
                                   gradient_accumulation_dtype=gradient_accumulation_dtype,
                                   communication_data_type=self.communication_data_type)
        return optimizer

    def _configure_eigenvalue(self):
        eigenvalue = Eigenvalue(
            verbose=self.eigenvalue_verbose(),
            max_iter=self.eigenvalue_max_iter(),
            tol=self.eigenvalue_tol(),
            stability=self.eigenvalue_stability(),
            gas_boundary_resolution=self.eigenvalue_gas_boundary_resolution(),
            layer_name=self.eigenvalue_layer_name(),
            layer_num=self.eigenvalue_layer_num(),
        )

        return eigenvalue

    def _configure_progressive_layer_drop(self):
        pld = ProgressiveLayerDrop(theta=self.pld_theta(), gamma=self.pld_gamma())

        return pld

    def _configure_curriculum_scheduler_legacy(self):
        scheduler = CurriculumScheduler(self.curriculum_params_legacy())
        return scheduler

    @staticmethod
    def is_map_style_dataset(obj):
        return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")

    @staticmethod
    def is_iterable_style_dataset(obj):
        return isinstance(obj, torch.utils.data.IterableDataset)  # hasattr(obj, "__iter__") should work as well

    def dataloader_drop_last(self):
        return self._config.dataloader_drop_last

    def was_step_applied(self) -> bool:
        """Returns True if the latest ``step()`` produced in parameter updates.
        Note that a ``False`` return is not an error condition. Steps are frequently
        no-ops, such as between gradient accumulation boundaries or when overflows
        occur.
        Returns:
            bool: Whether the latest ``step()`` modified model parameters.
        """
        return self._step_applied

    def deepspeed_io(self,
                     dataset,
                     batch_size=None,
                     route=ROUTE_TRAIN,
                     pin_memory=True,
                     data_sampler=None,
                     collate_fn=None,
                     num_local_io_workers=None):
        if not (self.is_map_style_dataset(dataset) or self.is_iterable_style_dataset(dataset)):
            raise ValueError("Training data must be a torch Dataset")

        if batch_size is None:
            batch_size = self.train_micro_batch_size_per_gpu()

        if collate_fn is None:
            collate_fn = self.collate_fn

        # Currently we only use timer in train route
        deepspeed_io_timer = None
        if route == ROUTE_TRAIN:
            deepspeed_io_timer = self.tput_timer

        # If mpu is provided, forward world size and parallel rank to sampler.
        data_parallel_world_size = self.dp_world_size
        data_parallel_rank = self.global_rank
        if self.mpu is not None:
            data_parallel_world_size = self.mpu.get_data_parallel_world_size()
            data_parallel_rank = self.mpu.get_data_parallel_rank()

        if data_sampler is None and (route == ROUTE_PREDICT or route == ROUTE_EVAL):
            data_sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=data_parallel_world_size,
                rank=data_parallel_rank,
                shuffle=False,
            )

        deepspeed_dataloader_config = {}
        if self.curriculum_learning_enabled():
            deepspeed_dataloader_config = {
                CURRICULUM_LEARNING: self.curriculum_learning_enabled(),
                DATA_EFFICIENCY: self.data_efficiency_config(),
                DATA_PARALLEL_GROUP: self.data_parallel_group,
                GRADIENT_ACCUMULATION_STEPS: self.gradient_accumulation_steps(),
                GLOBAL_RANK: self.global_rank,
                DATA_SAMPLING_NUM_WORKERS: self.data_sampling_config()[DATA_SAMPLING_NUM_WORKERS]
            }

        return DeepSpeedDataLoader(dataset=dataset,
                                   batch_size=batch_size,
                                   pin_memory=pin_memory,
                                   collate_fn=collate_fn,
                                   local_rank=self.local_rank,
                                   tput_timer=deepspeed_io_timer,
                                   num_local_io_workers=num_local_io_workers,
                                   data_sampler=data_sampler,
                                   data_parallel_world_size=data_parallel_world_size,
                                   data_parallel_rank=data_parallel_rank,
                                   dataloader_drop_last=self.dataloader_drop_last(),
                                   deepspeed_dataloader_config=deepspeed_dataloader_config)

    def train(self, mode=True):
        r""""""

        self.warn_unscaled_loss = True
        self.module.train(mode)

    def eval(self):
        r""""""

        self.warn_unscaled_loss = True
        self.module.train(False)

    def _scale_loss_by_gas(self, prescaled_loss):
        if isinstance(prescaled_loss, torch.Tensor):
            scaled_loss = prescaled_loss / self.gradient_accumulation_steps()
        elif isinstance(prescaled_loss, tuple) or isinstance(prescaled_loss, list):
            scaled_loss = []
            for l in prescaled_loss:
                if isinstance(l, torch.Tensor):
                    scaled_loss.append(l / self.gradient_accumulation_steps())
                else:
                    scaled_loss.append(l)
        else:
            scaled_loss = prescaled_loss
            if self.warn_unscaled_loss:
                logger.warning(f"DeepSpeed unable to scale loss because of type: {type(prescaled_loss)}")
                self.warn_unscaled_loss = False

        return scaled_loss

    @instrument_w_nvtx
    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation
        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """

        if self.autotuning_profile_model_info():
            ma = get_ma_status()
        else:
            see_memory_usage("Engine before forward", force=self.memory_breakdown())

        flops_profiler_active = (self.flops_profiler_enabled()
                                 and self.global_steps == self.flops_profiler_profile_step() and self.global_rank == 0)

        # used to check quantization happens at step 0!
        if self.global_steps == 0 and hasattr(self, "compression_scheduler"):
            self.compression_scheduler.step(step_zero_check=True)
            if self.quantizer:
                tensor_to_quantize = self.optimizer.bit16_groups if self.zero_optimization_stage(
                ) == 2 else self.optimizer.fp16_groups
                if self.compression_scheduler.weight_quantization_enabled:
                    self.quantizer.quantize(
                        tensor_to_quantize,
                        (self.optimizer.overflow if self.fp16_enabled() else False),
                        self.eigenvalue_enabled(),
                        None,
                    )

        if flops_profiler_active:
            self.flops_profiler.start_profile(ignore_list=None)

        if self.module.training:
            if self.progressive_layer_drop:
                kwargs.update(self.progressive_layer_drop.get_state())

        if self.__class__.__name__ != "PipelineEngine":
            # TODO: The above if condition is a HACK since for PipelineEngine
            # it's difficult to inject argument in forward pass.
            if self.module.training and self.curriculum_enabled_legacy():
                self.curriculum_scheduler_legacy.update_difficulty(self.global_steps + 1)
                if self.curriculum_params_legacy()["curriculum_type"] == "seqlen":
                    kwargs.update({"curriculum_seqlen": self.curriculum_scheduler_legacy.get_current_difficulty()})

        if self.module.training and self.random_ltd_enabled():
            self.random_ltd_scheduler.update_seq(self.global_steps)

        if self.zero_optimization_partition_weights():
            # Enable automated discovery of external parameters by indicating that
            # we are in a forward pass.
            for module in self.module.modules():
                module._parameters._in_forward = True

        self._start_timers(self.engine_timers.forward_timers)

        if self.training_dataloader is None:
            self.tput_timer.start()

        if self.fp16_auto_cast():
            inputs = self._cast_inputs_half(inputs)

        loss = self.module(*inputs, **kwargs)

        if self.zero_optimization_partition_weights():
            # Disable automated discovery of external parameters
            for module in self.module.modules():
                module._parameters._in_forward = False

        self._stop_timers(self.engine_timers.forward_timers)

        if flops_profiler_active:
            self.flops_profiler.stop_profile()

        if self.autotuning_profile_model_info():
            activation_mem = get_ma_status() - ma
            self.autotuning_model_info["activation_mem_per_gpu"] = activation_mem
            print_json_dist(self.autotuning_model_info, [0], path=self.autotuning_model_info_path())
            exit()
        else:
            see_memory_usage("Engine after forward", force=self.memory_breakdown())
        return loss

    def _cast_inputs_half(self, inputs):
        if isinstance(inputs, (list, tuple)):
            new_inputs = []
            for v in inputs:
                new_inputs.append(self._cast_inputs_half(v))
            return inputs.__class__(new_inputs)
        elif isinstance(inputs, dict):
            new_inputs = {}
            for k, v in inputs.items():
                new_inputs[k] = self._cast_inputs_half(v)
            return new_inputs
        elif hasattr(inputs, 'half'):
            return inputs.half()
        else:
            return inputs

    def print_forward_breakdown(self, fwd_time):
        gate_time = 0.0
        moe_time = 0.0
        falltoall = 0.0
        salltoall = 0.0

        for gate in self.gate_modules:
            #logger.info(f"Individual TopK gate time: {gate.gate_time:.2f} ms")
            gate_time += gate.gate_time

        for l in self.moe_layers:
            #logger.info(f"MoE layer; total: {l.time_moe:.2f} ms, first alltoall: {l.time_falltoall:.2f}, second alltoall: {l.time_salltoall:.2f}")
            moe_time += l.time_moe
            falltoall += l.time_falltoall
            salltoall += l.time_salltoall

        # TODO: Allreduce/average them across ranks for more accurate timing.

        # if deepspeed.comm.get_rank() == 0:
        log_dist(
            f"time (ms) | fwd: {fwd_time:.2f} (fwd_moe: {moe_time:.2f}, 1st_a2a: {falltoall:.2f}, 2nd_a2a: {salltoall:.2f}, top_k: {gate_time:.2f})",
            ranks=[0])

    @instrument_w_nvtx
    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        assert not (self.bfloat16_enabled() and self.pipeline_parallelism), \
            f'allreduce_gradients() is not valid when bfloat+pipeline_parallelism is enabled'

        # Pass (PP) gas boundary flag to optimizer (required for zero)
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        # ZeRO stage >= 2 communicates during non gradient accumulation boundaries as well
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        # Communicate only at gradient accumulation boundaries
        elif self.is_gradient_accumulation_boundary():
            if self.zero_optimization_stage() == ZeroStageEnum.optimizer_states and hasattr(
                    self.optimizer, 'reduce_gradients'):
                self.optimizer.reduce_gradients(pipeline_parallel=self.pipeline_parallelism)
            else:
                self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    @instrument_w_nvtx
    def backward(self, loss, allreduce_gradients=True, release_loss=False, retain_graph=False, scale_wrt_gas=True):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: is deprecated, ignored, and will soon be removed'
            retain_graph: bool, default: false
                forward on user defined choice of retain_graph
        """

        see_memory_usage("Engine before backward", force=self.memory_breakdown())

        if self.scale_wrt_gas is not None:
            scale_wrt_gas = self.scale_wrt_gas

        if not allreduce_gradients:
            logger.warning(f"Argument `allreduce_gradients` is deprecated, ignored, and will soon be removed")

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps() > 1 and scale_wrt_gas:
            loss = self._scale_loss_by_gas(loss.float())

        # Log training loss
        self.losses += loss.mean().item()
        if self.monitor.enabled:
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [(
                        f"Train/Samples/train_loss",
                        self.losses,
                        self.global_samples,
                    )]
                    self.monitor.write_events(self.summary_events)

        self._start_timers(self.engine_timers.backward_timers)

        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            "must provide optimizer during init in order to use backward"

        self._start_timers(self.engine_timers.backward_inner_timers)

        if self.zero_optimization():
            self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
            self.optimizer.backward(loss, retain_graph=retain_graph)
        elif self.amp_enabled():
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not self.is_gradient_accumulation_boundary()
            with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        elif self.fp16_enabled():
            if self.eigenvalue_enabled():
                self.optimizer.backward(loss, create_graph=True, retain_graph=True)
            else:
                self.optimizer.backward(loss, retain_graph=retain_graph)
        elif self.bfloat16_enabled():
            self.optimizer.backward(loss)
        else:
            if self.eigenvalue_enabled():
                loss.backward(create_graph=True, retain_graph=True)
            else:
                loss.backward(retain_graph=retain_graph)

        self._stop_timers(self.engine_timers.backward_inner_timers)

        self._start_timers(self.engine_timers.backward_reduce_timers)

        if allreduce_gradients and self.enable_backward_allreduce:
            # Traditional code path that allreduces the module parameter grads
            self.allreduce_gradients()

        self._stop_timers(self.engine_timers.backward_reduce_timers)

        self._stop_timers(self.engine_timers.backward_timers)

        if release_loss:
            # loss.data = None
            pass

        see_memory_usage("Engine after backward", force=self.memory_breakdown())

        return loss

    def is_gradient_accumulation_boundary(self):
        """
        Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.

        Returns:
            bool: if the current step is a gradient accumulation boundary.

        """
        if self._is_gradient_accumulation_boundary is None:
            return (self.micro_steps + 1) % \
                self.gradient_accumulation_steps() == 0
        else:
            return self._is_gradient_accumulation_boundary

    def set_gradient_accumulation_boundary(self, is_boundary):
        """
        Manually overrides the DeepSpeed engine's gradient accumulation boundary state, this is an optional
        feature and should be used with care. The state should be set before to the intended
        value before each forward/backward. The final forward/backward should have the
        boundary state set to True. This style allows client code to only call engine.step() once after all
        the gradient accumulation passes are complete. See example below:
        .. code-block:: python
        engine.set_gradient_accumulation_boundary(False)
        for _ in range(gradient_accumulation_steps - 1):
            micro_batch = next(data_loader)
            loss = engine(micro_batch)
            engine.backward(loss)
        engine.set_gradient_accumulation_boundary(True)
        micro_batch = next(data_loader)
        loss = engine(micro_batch)
        engine.backward(loss)
        engine.step()
        Arguments:
            is_boundary (bool): are we at a gradient accumulation boundary or not?
        """
        self._is_gradient_accumulation_boundary = is_boundary
        self.optimizer.is_gradient_accumulation_boundary = is_boundary

    def zero_grad(self):
        """
        Zero parameter grads.
        """
        for param_name, param in self.module.named_parameters():
            param.grad = None

    def clip_fp32_gradients(self):
        clip_grad_norm_(parameters=self.module.parameters(), max_norm=self.gradient_clipping(), mpu=self.mpu)

    def _take_model_step(self, lr_kwargs, block_eigenvalue={}):
        if self.gradient_clipping() > 0.0:
            if not (self.fp16_enabled() or self.bfloat16_enabled() or self.amp_enabled() or self.zero_optimization()):
                self.clip_fp32_gradients()
            elif self.amp_enabled():
                # AMP's recommended way of doing clipping
                # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                master_params = amp.master_params(self.optimizer)
                clip_grad_norm_(parameters=master_params, max_norm=self.gradient_clipping(), mpu=self.mpu)
        self.optimizer.step()

        if hasattr(self.optimizer, '_global_grad_norm'):
            self._global_grad_norm = self.optimizer._global_grad_norm

        # Quantize the updated parameter if there is no overflow
        if self.quantizer:
            tensor_to_quantize = self.optimizer.bit16_groups if self.zero_optimization_stage(
            ) == 2 else self.optimizer.fp16_groups
            if self.compression_scheduler.weight_quantization_enabled:
                self.quantizer.quantize(
                    tensor_to_quantize,
                    (self.optimizer.overflow if self.fp16_enabled() else False),
                    self.eigenvalue_enabled(),
                    block_eigenvalue,
                )
        # zero grad in basic optimizer could be unreliable and may not exhibit
        # the behavior that we want
        if self.bfloat16_enabled():
            # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
            if self.zero_optimization() and hasattr(self.optimizer, "zero_grad"):
                self.optimizer.zero_grad()
            else:
                pass
        elif self.zero_optimization() or self.fp16_enabled() or self.amp_enabled():
            self.optimizer.zero_grad()
        else:
            self.zero_grad()

        report_progress = self.global_rank == 0 if self.global_rank else True

        # Check overflow here since in DS fp16 optimizer, the overflow is updated in above step() function.
        overflow = False
        if hasattr(self.optimizer, "overflow"):
            overflow = self.optimizer.overflow
        self._step_applied = not overflow

        if overflow:
            self.skipped_steps += 1
        else:
            self.compression_scheduler.step()
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step(**(lr_kwargs or {}))
                except TypeError:
                    # XXX Hack to work with Megatron 2.0 and DeepSpeed pipelines.
                    # We don't currently have a way to specify lr_kwargs from
                    # pipe_engine.train_batch()
                    self.lr_scheduler.step(self.train_batch_size())

        if report_progress and (self.global_steps + 1) % self.steps_per_print() == 0:
            self._report_progress(self.global_steps + 1)

        self.losses = 0.0
        self.global_steps += 1
        self.global_samples += self.train_batch_size()

    def step(self, lr_kwargs=None):
        r"""Execute the weight update step after forward and backward propagation
        on effective_train_batch.
        """
        see_memory_usage("Engine before step", force=self.memory_breakdown())

        # Check early because self.global_steps is incremented at some point here.
        # TODO: Delay self.global_steps increment until very end of this function.
        flops_profiler_active = self.flops_profiler_enabled(
        ) and self.global_steps == self.flops_profiler_profile_step() and self.global_rank == 0

        self._start_timers(self.engine_timers.step_timers)

        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            "must provide optimizer during init in order to use step"

        report_progress = False

        self._step_applied = False  # assume False, will flip to True

        # Update the model when we reach gradient accumulation boundaries
        if self.is_gradient_accumulation_boundary():
            self.gas_boundary_ctr += 1

            if (self.eigenvalue_enabled() and (self.gas_boundary_ctr % self.eigenvalue_gas_boundary_resolution() == 0)
                    and self.quantizer.any_precision_switch()):
                log_dist(f"computing eigenvalue...", ranks=[0])
                self.block_eigenvalue = self.eigenvalue.compute_eigenvalue(self.module, self.device,
                                                                           self.optimizer.cur_scale)

            if self.progressive_layer_drop:
                self.progressive_layer_drop.update_state(self.global_steps)

            if (self.eigenvalue_enabled() and not self.gas_boundary_ctr % self.eigenvalue_gas_boundary_resolution()
                    and self.quantizer.any_precision_switch()):
                self._take_model_step(lr_kwargs, self.block_eigenvalue)
            else:
                self._take_model_step(lr_kwargs)

            report_progress = self.global_rank == 0 if self.global_rank else True

        self.tput_timer.stop(global_step=self.is_gradient_accumulation_boundary(), report_speed=report_progress)

        self._stop_timers(self.engine_timers.step_timers)

        # Log learning rate
        if self.monitor.enabled:
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [(f"Train/Samples/lr", self.get_lr()[0], self.global_samples)]

                    if self.fp16_enabled() and hasattr(self.optimizer, "cur_scale"):
                        self.summary_events.append((
                            f"Train/Samples/loss_scale",
                            self.optimizer.cur_scale,
                            self.global_samples,
                        ))

                    if (self.eigenvalue_enabled()
                            and not self.gas_boundary_ctr % self.eigenvalue_gas_boundary_resolution()):
                        ev_values = self.block_eigenvalue.values()
                        for i in range(len(ev_values)):
                            self.summary_events.append((
                                f"Train/Eigenvalues/ModelBlockParam_{i}",
                                self.ev_values[i][0],
                                self.global_samples,
                            ))
                    self.monitor.write_events(self.summary_events)

        # Check flops profiling
        if flops_profiler_active:
            if self.autotuning_enabled():
                self.flops = self.flops_profiler.get_total_flops() * 3
                self.fwd_duration = self.flops_profiler.get_total_duration()
            else:
                self.flops_profiler.print_model_profile(
                    profile_step=self.global_steps,
                    module_depth=self.flops_profiler_module_depth(),
                    top_modules=self.flops_profiler_top_modules(),
                    detailed=self.flops_profiler_detailed(),
                    output_file=self.flops_profiler_output_file(),
                )
            self.flops_profiler.end_profile()

        if self.autotuning_enabled() and self.global_steps == (self.autotuning_end_profile_step() + 1):
            self._autotuning_exit()

        if self.wall_clock_breakdown():
            # Log micro timing and reset
            self.timers.log(names=self.engine_timers.micro_timers, memory_breakdown=self.memory_breakdown())

        if self.wall_clock_breakdown() or self.flops_profiler_enabled():
            # Log global timing and reset
            if self.is_gradient_accumulation_boundary():
                if self.monitor.enabled:
                    self._write_monitor()

                if self.has_moe_layers:
                    fwd_time = self.timers(FORWARD_GLOBAL_TIMER).elapsed(reset=False)
                    self.print_forward_breakdown(fwd_time=fwd_time)

                self.timers.log(self.engine_timers.global_timers)

        self.micro_steps += 1
        see_memory_usage("Engine after step", force=self.memory_breakdown())

    def _start_timers(self, timer_names):
        for name in timer_names:
            self.timers(name).start()

    def _stop_timers(self, timer_names):
        record = self.is_gradient_accumulation_boundary() and \
            self.flops_profiler_enabled() and \
                (self.global_steps >= self.flops_profiler_profile_step())
        for name in timer_names:
            self.timers(name).stop(record=record)

    def _autotuning_exit(self):
        if self.global_rank == 0:
            msg = self.timers.get_mean([
                FORWARD_GLOBAL_TIMER,
                BACKWARD_GLOBAL_TIMER,
                STEP_GLOBAL_TIMER,
            ], reset=False)
            titer = 0.0
            titer += msg[FORWARD_GLOBAL_TIMER] if FORWARD_GLOBAL_TIMER in msg else 0
            titer += msg[BACKWARD_GLOBAL_TIMER] if BACKWARD_GLOBAL_TIMER in msg else 0
            titer += msg[STEP_GLOBAL_TIMER] if STEP_GLOBAL_TIMER in msg else 0
            titer *= self.gradient_accumulation_steps()
            msg["latency"] = titer
            msg["FLOPS_per_gpu"] = self.flops * 1_000_000 * self.gradient_accumulation_steps() / titer
            msg["throughput"] = self.train_batch_size() * 1_000_000 / \
                msg["latency"]
            print_json_dist(msg, [0], path=self.autotuning_metric_path())
            log_dist(
                f"Wrote metrics to {self.autotuning_metric_path()}, {os.path.abspath(self.autotuning_metric_path())}",
                ranks=[0])
            import atexit
            atexit.register(print, "Autotuning: done with running current ds config.")
        exit()

    def _write_monitor(self):
        if self.global_rank == 0:
            self.summary_events = [
                (
                    f"Train/Samples/elapsed_time_ms_forward",
                    self.timers(FORWARD_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward",
                    self.timers(BACKWARD_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward_inner",
                    self.timers(BACKWARD_INNER_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward_allreduce",
                    self.timers(BACKWARD_REDUCE_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_step",
                    self.timers(STEP_GLOBAL_TIMER).elapsed(reset=False),
                    self.global_samples,
                ),
            ]
            self.monitor.write_events(self.summary_events)

    def _get_optimizer_param(self, param_name):
        result = []
        if not self.optimizer:
            return result
        for group in self.optimizer.param_groups:
            if param_name in group:
                result.append(group[param_name])
            else:
                result.append(0.0)
        return result

    def get_lr(self):
        return self._get_optimizer_param("lr")

    def get_type(self):
        return self._get_optimizer_param("type")

    def get_mom(self):
        if self.optimizer_name() in ["SGD", "RMSprop"]:
            return self._get_optimizer_param("momentum")
        else:
            return self._get_optimizer_param("betas")

    def get_pld_theta(self):
        if self.progressive_layer_drop:
            return self.progressive_layer_drop.get_theta()
        else:
            return None

    def _report_progress(self, step):
        lr = self.get_lr()
        mom = self.get_mom()
        log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr}, mom={mom}", ranks=[0])

    def allreduce_bucket(self, bucket, dp_group):
        tensor = self.flatten(bucket)

        tensor_to_allreduce = tensor

        if self.communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(self.communication_data_type)

        if self.postscale_gradients():
            if self.gradient_predivide_factor() != 1.0:
                tensor_to_allreduce.mul_(1.0 / self.gradient_predivide_factor())

            dist.all_reduce(tensor_to_allreduce, group=dp_group)
            if self.gradient_average:
                if self.gradient_predivide_factor() != dist.get_world_size(group=dp_group):
                    tensor_to_allreduce.mul_(self.gradient_predivide_factor() / dist.get_world_size(group=dp_group))
        else:
            tensor_to_allreduce.mul_(1. / dist.get_world_size(group=dp_group))
            dist.all_reduce(tensor_to_allreduce, group=dp_group)

        if self.communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy(self, small_bucket, dp_group):
        allreduced = self.allreduce_bucket(small_bucket, dp_group)
        for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def allreduce_no_retain(self, bucket, dp_group, numel_per_bucket=500000000):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, dp_group)
                small_bucket = []
                numel = 0
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, dp_group)

    def _get_gradients_for_reduction(self):
        non_expert_grads = []
        expert_grads = {}
        if self.has_moe_layers:
            for key in self.expert_data_parallel_group.keys():
                expert_grads[key] = []

        for param_name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(param.size(), dtype=param.dtype, device=param.device)

            grad_data = param.grad.data
            if param_name in self.sparse_tensor_module_names or grad_data.is_sparse:
                # Call param.grad without data to avoid problem with setting of updated grads
                grad_data = SparseTensor(param.grad)

            if is_moe_param(param):
                expert_grads[param.group_name].append(grad_data)
            else:
                non_expert_grads.append(grad_data)

        return non_expert_grads, expert_grads

    def _reduce_non_expert_gradients(self, grads, elements_per_buffer):
        split_buckets = split_half_float_double_sparse(grads)
        for _, bucket_tuple in enumerate(split_buckets):
            bucket_type, bucket = bucket_tuple

            if self.pipeline_parallelism:
                dp_group = self.mpu.get_data_parallel_group()
            else:
                dp_group = groups._get_data_parallel_group()

            if bucket_type == SparseTensor.type():
                self.sparse_allreduce_no_retain(bucket, dp_group=dp_group)
            else:
                self.allreduce_no_retain(bucket, dp_group=dp_group, numel_per_bucket=elements_per_buffer)

    def _reduce_expert_gradients(self, expert_grads, elements_per_buffer):
        for ep_name, expert_grads_group in expert_grads.items():
            expert_split_buckets = split_half_float_double_sparse(expert_grads_group)
            for i, bucket_tuple in enumerate(expert_split_buckets):
                bucket_type, bucket = bucket_tuple
                if bucket_type == SparseTensor.type():
                    self.sparse_allreduce_no_retain(bucket, groups._get_expert_data_parallel_group(ep_name))
                else:
                    # Separate between diff groups
                    self.allreduce_no_retain(bucket,
                                             dp_group=groups._get_expert_data_parallel_group(ep_name),
                                             numel_per_bucket=elements_per_buffer)

    def buffered_allreduce_fallback(self, grads=None, elements_per_buffer=500000000):
        if grads is None:
            non_expert_grads, expert_grads = self._get_gradients_for_reduction()
        else:
            assert not self.has_moe_layers, "attempting to reduce grads in unsupported way w.r.t. MoE"
            non_expert_grads = grads

        self._reduce_non_expert_gradients(non_expert_grads, elements_per_buffer)

        if self.has_moe_layers:
            self._reduce_expert_gradients(expert_grads, elements_per_buffer)

    def sparse_allreduce_no_retain(self, bucket, dp_group):
        allreduced_sparses = self.sparse_allreduce_bucket(bucket, dp_group)
        # Densify sparse tensor and copy back to original location
        for tensor in allreduced_sparses:
            if tensor.is_sparse:
                tensor.orig_dense_tensor.data = tensor.to_coo_tensor()
            else:
                tensor.orig_dense_tensor.copy_(tensor.to_dense())

    def sparse_allreduce_bucket(self, bucket, dp_group):
        sparse_list = []
        for sparse in bucket:
            sparse_list.append(self.sparse_allreduce(sparse, dp_group))
        return sparse_list

    def sparse_allreduce(self, sparse, dp_group):
        original_data_type = sparse.values.dtype
        if self.communication_data_type != sparse.values.dtype:
            if self.communication_data_type in (torch.float16, torch.bfloat16):
                indices = sparse.indices.to(torch.int32)
            else:
                indices = sparse.indices
            values = sparse.values.to(self.communication_data_type)
        else:
            indices = sparse.indices
            values = sparse.values

        if self.postscale_gradients():
            if self.gradient_average:
                values.mul_(self.gradient_predivide_factor() / dist.get_world_size(group=dp_group))
        else:
            values.mul_(1. / dist.get_world_size(group=dp_group))

        indices_device_list = self.sparse_all_gather(indices, dp_group)
        values_device_list = self.sparse_all_gather(values, dp_group)

        sparse.indices = torch.cat(indices_device_list).to(torch.long)
        sparse.values = torch.cat(values_device_list).to(original_data_type)
        return sparse

    def sparse_all_gather(self, value, dp_group):
        my_size = torch.LongTensor([value.size()[0]]).to(self.device)
        all_sizes = self.all_gather_scalar(my_size, dp_group)
        max_size = torch.cat(all_sizes).max()
        fill_size = max_size - my_size

        assert value.dim() in [1, 2]
        if value.dim() == 1:
            if fill_size > 0:
                value = torch.cat([value, value.new_empty(fill_size)])
            tensor_list = [value.new_empty(max_size) for _ in range(dist.get_world_size(group=dp_group))]
        else:
            if fill_size > 0:
                value = torch.cat([value, value.new_empty(fill_size, value.size()[1])])
            tensor_list = [
                value.new_empty(max_size,
                                value.size()[1]) for _ in range(dist.get_world_size(group=dp_group))
            ]

        dist.all_gather(tensor_list, value, group=dp_group)
        tensors = []
        for dev_idx, t in enumerate(tensor_list):
            size = all_sizes[dev_idx][0]
            tensors.append(t.index_select(0, torch.arange(size, dtype=torch.long, device=self.device)))

        return tensors

    def all_gather_scalar(self, value, dp_group):
        tensor_list = [value.new_zeros(value.size()) for _ in range(dist.get_world_size(group=dp_group))]
        dist.all_gather(tensor_list, value, group=dp_group)
        return tensor_list

    def module_state_dict(self, destination=None, prefix="", keep_vars=False, exclude_frozen_parameters=False):
        sd = self.module.state_dict(destination, prefix, keep_vars)

        # Remove frozen parameter weights from state_dict if specified
        if exclude_frozen_parameters:
            for n, p in self.module.named_parameters():
                if not p.requires_grad:
                    del sd[n]

        if self.random_ltd_enabled():
            sd = remove_random_ltd_state_dict(sd)
        return sd

    @staticmethod
    def load_moe_state_dict(checkpoint_path,
                            tag,
                            state_dict,
                            old_moe_load,
                            model=None,
                            mpu=None,
                            num_experts=1,
                            checkpoint_engine=TorchCheckpointEngine()):
        if old_moe_load:
            expp_rank = groups._get_expert_data_parallel_rank(groups._get_max_expert_size_name())

            num_local_experts = max(num_experts) // groups._get_expert_parallel_world_size(
                groups._get_max_expert_size_name())
            for local_expert_id in range(num_local_experts):
                global_expert_id = expp_rank * num_local_experts + local_expert_id
                expert_state_dict = checkpoint_engine.load(
                    DeepSpeedEngine._get_expert_ckpt_name(
                        checkpoint_path,
                        -1,  # -1 means ignore layer_id
                        global_expert_id,
                        tag,
                        mpu),
                    map_location=torch.device('cpu'))

                # Updating global -> local expert ids
                moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
                for key in list(expert_state_dict.keys()):
                    local_key = key.replace(f'{moe_str_prefix}{global_expert_id}',
                                            f'{moe_str_prefix}{local_expert_id}')
                    expert_state_dict[local_key] = expert_state_dict.pop(key)
                state_dict.update(expert_state_dict)

        else:
            moe_layer_id = 0
            for n_module, module in model.named_modules():
                if isinstance(module, MoE):  # and deepspeed.comm.get_rank() == 0:
                    group_name = module.expert_group_name
                    num_local_experts = module.num_local_experts
                    expp_rank = groups._get_expert_parallel_rank(group_name)
                    # loop all local_experts
                    for local_expert_id in range(num_local_experts):
                        global_expert_id = expp_rank * num_local_experts + local_expert_id
                        expert_state_dict = checkpoint_engine.load(DeepSpeedEngine._get_expert_ckpt_name(
                            checkpoint_path, moe_layer_id, global_expert_id, tag, mpu),
                                                                   map_location=torch.device('cpu'))
                        # print(expert_state_dict.keys())
                        # Updating global -> local expert ids
                        moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
                        for key in list(expert_state_dict.keys()):
                            local_key = key.replace(f'{moe_str_prefix}{global_expert_id}',
                                                    f'{moe_str_prefix}{local_expert_id}')
                            expert_state_dict[local_key] = expert_state_dict.pop(key)
                        state_dict.update(expert_state_dict)
                    moe_layer_id += 1

    def load_module_state_dict(self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False):
        if fetch_z3_params:
            params_to_fetch = [
                p for p in self.module.parameters()
                if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
            ]
        else:
            params_to_fetch = []

        with deepspeed.zero.GatheredParameters(params_to_fetch, modifier_rank=0):
            module_state_dict = checkpoint['module']
            if custom_load_fn:
                custom_load_fn(src=module_state_dict, dst=self.module)
            else:
                self.module.load_state_dict(
                    module_state_dict,  # TODO
                    strict=strict)

        if checkpoint.get(FROZEN_PARAM_FRAGMENTS, None) is not None:
            saved_frozen_params = checkpoint[FROZEN_PARAM_FRAGMENTS]
            for param in self.module.parameters():
                if param.requires_grad:
                    continue
                if param not in self.param_names:
                    raise ValueError(f"failed to find frozen {param} in named params")
                name = self.param_names[param]
                if hasattr(param, 'ds_id'):
                    param.ds_tensor.data.copy_(saved_frozen_params[name].data)
                else:
                    param.data.copy_(saved_frozen_params[name].data)

    def _get_zero_ckpt_prefix(self, dp_rank, bf16_mode):
        return f'{"bf16_" if bf16_mode else ""}zero_pp_rank_{dp_rank}'

    def _get_rank_zero_ckpt_name(self, checkpoints_path, tag, mp_rank, dp_rank, bf16_mode):
        file_prefix = self._get_zero_ckpt_prefix(dp_rank, bf16_mode=bf16_mode)
        zero_ckpt_name = os.path.join(
            checkpoints_path,
            str(tag),
            f"{file_prefix}_mp_rank_{mp_rank:02d}_optim_states.pt",
        )
        return zero_ckpt_name

    def _get_zero_ckpt_name(self, checkpoints_path, tag):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        pp_rank = dist.get_rank(group=self.optimizer.dp_process_group)
        bf16_mode = self.bfloat16_enabled()
        return self._get_rank_zero_ckpt_name(checkpoints_path, tag, mp_rank, pp_rank, bf16_mode)

    def _get_ckpt_name(self, checkpoints_path, tag, mp_placeholder=None):
        if mp_placeholder is not None:
            mp_rank_str = mp_placeholder
        else:
            mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
            mp_rank_str = f"{mp_rank:02d}"

        if self.zero_optimization_partition_weights():
            filename = "zero_pp_rank_{}".format(dist.get_rank(group=self.optimizer.dp_process_group))
            ckpt_name = os.path.join(
                checkpoints_path,
                str(tag),
                f"{filename}_mp_rank_{mp_rank_str}_model_states.pt",
            )
        else:
            ckpt_name = os.path.join(
                checkpoints_path,
                str(tag),
                "mp_rank_" + mp_rank_str + "_model_states.pt",
            )
        return ckpt_name

    def _get_optimizer_ckpt_name(self, checkpoints_path, tag, expp_rank):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        ckpt_name = os.path.join(checkpoints_path, str(tag),
                                 f'expp_rank_{expp_rank}_mp_rank_{mp_rank:02d}_optim_states.pt')
        return ckpt_name

    @staticmethod
    def _get_expert_ckpt_name(checkpoints_path, layer_id, expert_id, tag, mpu=None):
        mp_rank = 0 if mpu is None else mpu.get_model_parallel_rank()
        if layer_id <= -1:
            # Used to support old checkpoint loading
            ckpt_name = os.path.join(checkpoints_path, '' if tag is None else str(tag),
                                     f'expert_{expert_id}_mp_rank_{mp_rank:02d}_model_states.pt')
        else:
            # Used to support new checkpoint loading
            ckpt_name = os.path.join(checkpoints_path, '' if tag is None else str(tag),
                                     f'layer_{layer_id}_expert_{expert_id}_mp_rank_{mp_rank:02d}_model_states.pt')
        return ckpt_name

    def _get_all_ckpt_names(self, checkpoints_path, tag):
        # It is required that (checkpoints_path, tag) are consistent among all ranks.
        ckpt_file_pattern = self._get_ckpt_name(checkpoints_path, tag, mp_placeholder="*")
        import glob

        ckpt_files = glob.glob(ckpt_file_pattern)
        ckpt_files.sort()
        return ckpt_files

    def load_checkpoint(self,
                        load_dir,
                        tag=None,
                        load_module_strict=True,
                        load_optimizer_states=True,
                        load_lr_scheduler_states=True,
                        load_module_only=False,
                        custom_load_fn=None):
        """
        Load training checkpoint

        Arguments:
            load_dir: Required. Directory to load the checkpoint from
            tag: Checkpoint tag used as a unique identifier for checkpoint, if not provided will attempt to load tag in 'latest' file
            load_module_strict: Optional. Boolean to strictly enforce that the keys in state_dict of module and checkpoint match.
            load_optimizer_states: Optional. Boolean to load the training optimizer states from Checkpoint. Ex. ADAM's momentum and variance
            load_lr_scheduler_states: Optional. Boolean to add the learning rate scheduler states from Checkpoint.
            load_module_only: Optional. Boolean to load only the model weights from the checkpoint. Ex. warmstarting.
            custom_load_fn: Optional. Custom model load function.

        Returns:
            A tuple of ``load_path`` and ``client_state``.
            *``load_path``: Path of the loaded checkpoint. ``None`` if loading the checkpoint failed.
            *``client_state``: State dictionary used for loading required training states in the client code.

        Important: under ZeRO3, one cannot load checkpoint with ``engine.load_checkpoint()`` right
        after ``engine.save_checkpoint()``. It is because ``engine.module`` is partitioned, and
        ``load_checkpoint()`` wants a pristine model. If insisting to do so, please reinitialize engine
        before ``load_checkpoint()``.

        """

        if tag is None:
            latest_tag = "latest_universal" if self.load_universal_checkpoint() else "latest"
            latest_path = os.path.join(load_dir, latest_tag)
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    tag = fd.read().strip()
            else:
                if self.load_universal_checkpoint():
                    raise ValueError(f'Invalid for universal checkpoint: {latest_path} does not exist')
                else:
                    logger.warning(
                        f"Unable to find latest file at {latest_path}, if trying to load latest "
                        "checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint."
                    )
                    return None, None

        if self._optimizer_has_ckpt_event_prologue():
            # Prepare for checkpoint load by ensuring all parameters are partitioned
            self.optimizer.checkpoint_event_prologue()

        load_path, client_states = self._load_checkpoint(load_dir,
                                                         tag,
                                                         load_module_strict=load_module_strict,
                                                         load_optimizer_states=load_optimizer_states,
                                                         load_lr_scheduler_states=load_lr_scheduler_states,
                                                         load_module_only=load_module_only,
                                                         custom_load_fn=custom_load_fn)

        load_zero_checkpoint = load_optimizer_states and load_path is not None and (self.zero_optimization()
                                                                                    or self.bfloat16_enabled())
        if load_zero_checkpoint:
            success = self._load_zero_checkpoint(load_dir, tag, load_optimizer_states=load_optimizer_states)
            if not success:
                self.optimizer._restore_from_bit16_weights()

        if self._optimizer_has_ckpt_event_epilogue():
            self.optimizer.checkpoint_event_epilogue()

        return load_path, client_states

    def _load_checkpoint(self,
                         load_dir,
                         tag,
                         load_module_strict=True,
                         load_optimizer_states=True,
                         load_lr_scheduler_states=True,
                         load_module_only=False,
                         custom_load_fn=None):

        from deepspeed.runtime.state_dict_factory import SDLoaderFactory

        ckpt_list = self._get_all_ckpt_names(load_dir, tag)
        sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, checkpoint_engine=self.checkpoint_engine)

        is_pipe_parallel = isinstance(self.module, PipelineModule)

        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        load_path, checkpoint, _ = sd_loader.load(self.mp_world_size, mp_rank, is_pipe_parallel=is_pipe_parallel)

        if checkpoint is None:
            return None, None

        fetch_z3_params = False
        if self.zero_optimization_partition_weights() and not load_optimizer_states:
            checkpoint['module'] = get_fp32_state_dict_from_zero_checkpoint(load_dir)
            fetch_z3_params = True

        if is_pipe_parallel:
            # Pipeline parallelism uses this to load its own checkpoint files.
            self._curr_ckpt_path = os.path.join(load_dir, tag)

        if self.has_moe_layers:
            # print(checkpoint.keys())
            old_moe_load = False
            if not isinstance(checkpoint['num_experts'], list):
                old_moe_load = True
            DeepSpeedEngine.load_moe_state_dict(load_dir,
                                                tag,
                                                state_dict=checkpoint['module'],
                                                old_moe_load=old_moe_load,
                                                model=self.module,
                                                mpu=self.mpu,
                                                num_experts=self.num_experts,
                                                checkpoint_engine=self.checkpoint_engine)
        if not self.load_universal_checkpoint():
            self.load_module_state_dict(checkpoint=checkpoint,
                                        strict=load_module_strict,
                                        custom_load_fn=custom_load_fn,
                                        fetch_z3_params=fetch_z3_params)

        self.loaded_checkpoint_dp_world_size = checkpoint['dp_world_size']

        optim_checkpoint = None
        if load_module_only:
            deepspeed_states = ['module']
            if self.optimizer is not None and self.fp16_enabled():
                self.optimizer.refresh_fp32_params()
        else:
            has_zero_optimizer_state = self.zero_optimization() or self.bfloat16_enabled()
            if load_optimizer_states and self.optimizer is not None and not has_zero_optimizer_state:
                if self.has_moe_layers:
                    largest_group_name = groups._get_max_expert_size_name()
                    expp_rank = groups._get_expert_parallel_rank(largest_group_name)
                    optim_load_path = self._get_optimizer_ckpt_name(load_dir, tag, expp_rank)
                    optim_checkpoint = self.checkpoint_engine.load(optim_load_path, map_location=torch.device('cpu'))
                else:
                    optim_checkpoint = checkpoint

                if self.fp16_enabled() or self.bfloat16_enabled():
                    self.optimizer.load_state_dict(optim_checkpoint['optimizer'],
                                                   load_optimizer_states=load_optimizer_states)
                else:
                    optim_checkpoint = checkpoint

                self.optimizer.load_state_dict(optim_checkpoint['optimizer'])

            if load_lr_scheduler_states and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            if self.random_ltd_enabled() and self.random_ltd_scheduler is not None and 'random_ltd' in checkpoint:
                self.random_ltd_scheduler.load_state_dict(checkpoint['random_ltd'])

            if self.training_dataloader is not None and self.curriculum_learning_enabled(
            ) and 'data_sampler' in checkpoint:
                self.training_dataloader.data_sampler.load_state_dict(checkpoint['data_sampler'])

            def get_sparse_tensor_module_names(original_set, loaded_set, original_parameters, loaded_parameters):
                result = set()

                for name in original_set:
                    if name in loaded_parameters and name not in loaded_set:
                        continue  # parameter existed in previous model and was not sparse
                    result.add(name)

                for name in loaded_set:
                    if name in original_parameters:
                        result.add(name)  # parameter exists in both configs and it was sparse

                return result

            if 'sparse_tensor_module_names' in checkpoint:
                sparse_tensor_module_names = checkpoint['sparse_tensor_module_names']
            elif 'csr_tensor_module_names' in checkpoint:
                sparse_tensor_module_names = checkpoint['csr_tensor_module_names']
            else:
                sparse_tensor_module_names = None
            if sparse_tensor_module_names is not None:
                if load_module_strict:
                    self.sparse_tensor_module_names = sparse_tensor_module_names
                else:
                    self.sparse_tensor_module_names = get_sparse_tensor_module_names(
                        self.sparse_tensor_module_names, sparse_tensor_module_names,
                        dict(self.module.named_parameters()), checkpoint["module"])

            self.global_steps = checkpoint['global_steps']
            self.global_samples = checkpoint.get('global_samples', self.global_steps * self.train_batch_size())
            self.skipped_steps = checkpoint['skipped_steps']
            self.loaded_checkpoint_mp_world_size = checkpoint['mp_world_size']
            deepspeed_states = [
                'module', 'sparse_tensor_module_names', 'skipped_steps', 'global_steps', 'dp_world_size',
                'mp_world_size', 'data_sampler', 'random_ltd'
            ]
        client_state = {}

        if load_lr_scheduler_states:
            deepspeed_states.append('lr_scheduler')
        if load_optimizer_states:
            deepspeed_states.append('optimizer')

        client_state = {key: value for key, value in checkpoint.items() if not key in deepspeed_states}

        if optim_checkpoint is not None:
            client_state['optimizer'] = optim_checkpoint['optimizer']

        return load_path, client_state

    def _load_zero_checkpoint(self, load_dir, tag, load_optimizer_states=True):

        load_serial = None
        # When use loading checkpoint serial, checkpoint loading start from local rank 0,
        # all other local rank would be paused, waiting for its rank-1 peer ready and its notification.
        if self._config.zero_config.pipeline_loading_checkpoint:
            assert self.zero_optimization_stage(
            ) == ZeroStageEnum.weights, "Only stage3 support for pipeline checkpoint loading"
            load_serial = torch.zeros(1).to(self.device)
            if dist.get_local_rank() != 0:
                dist.recv(tensor=load_serial, src=dist.get_rank() - 1)
        if self.load_universal_checkpoint():
            zero_sd_list = None
            checkpoint_folder = f'{os.path.join(load_dir, tag)}'
        else:
            if load_optimizer_states and self.seq_dp_world_size != self.loaded_checkpoint_dp_world_size:
                raise ZeRORuntimeException("The checkpoint being loaded used a DP " \
                    f"world size of {self.loaded_checkpoint_dp_world_size} but the " \
                    f"current world size is {self.seq_dp_world_size}. Automatic adjustment " \
                    "of ZeRO's optimizer state partitioning with a new world size is not " \
                    "currently supported.")
            checkpoint_folder = None
            zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
            if zero_sd_list is None:
                return False

        self.optimizer.load_state_dict(state_dict_list=zero_sd_list,
                                       load_optimizer_states=load_optimizer_states,
                                       load_from_fp32_weights=self.zero_load_from_fp32_weights(),
                                       checkpoint_folder=checkpoint_folder,
                                       load_serial=load_serial)

        if self.load_universal_checkpoint():
            logger.info(f'loaded universal zero checkpoints from {checkpoint_folder} for rank {self.global_rank}')
        else:
            logger.info(f"loading {len(zero_sd_list)} zero partition checkpoints for rank {self.global_rank}")
        return True

    def _get_mp_rank_zero_checkpoint_names(self, load_dir, tag, mp_rank, dp_world_size, bf16_mode):
        zero_ckpt_names = []
        for dp_rank in range(dp_world_size):
            ckpt_name = self._get_rank_zero_ckpt_name(checkpoints_path=load_dir,
                                                      tag=tag,
                                                      mp_rank=mp_rank,
                                                      dp_rank=dp_rank,
                                                      bf16_mode=bf16_mode)
            zero_ckpt_names.append(ckpt_name)

        return zero_ckpt_names

    def _get_all_zero_checkpoint_names(self, load_dir, tag, bf16_mode):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        zero_ckpt_names = self._get_mp_rank_zero_checkpoint_names(load_dir=load_dir,
                                                                  tag=tag,
                                                                  mp_rank=mp_rank,
                                                                  dp_world_size=self.loaded_checkpoint_dp_world_size,
                                                                  bf16_mode=bf16_mode)
        for i, ckpt_name in enumerate(zero_ckpt_names):
            if not os.path.exists(ckpt_name):
                # transparently handle the old file pattern for optim_states
                if "optim_states.pt" in ckpt_name:
                    ckpt_name_try = ckpt_name.replace("_optim_states.pt", "optim_states.pt")
                    if os.path.exists(ckpt_name_try):
                        zero_ckpt_names[i] = ckpt_name_try
                        continue

        return zero_ckpt_names

    def _get_all_zero_checkpoint_state_dicts(self, zero_ckpt_names):
        zero_sd_list = []
        for i, ckpt_name in enumerate(zero_ckpt_names):
            _state = None
            if ckpt_name is None:
                _state = {OPTIMIZER_STATE_DICT: None}
            # Fully load state for current rank
            elif self.zero_elastic_checkpoint() or dist.get_rank(group=self.optimizer.dp_process_group) == i:
                _state = self.checkpoint_engine.load(
                    ckpt_name,
                    map_location='cpu',
                )
            else:
                _state = {OPTIMIZER_STATE_DICT: None}
            zero_sd_list.append(_state)

        zero_optimizer_sd = [sd[OPTIMIZER_STATE_DICT] for sd in zero_sd_list]
        logger.info(f"successfully read {len(zero_optimizer_sd)} ZeRO state_dicts for rank {self.global_rank}")
        return zero_optimizer_sd

    def _get_all_zero_checkpoints(self, load_dir, tag):
        for bf16_mode in [self.bfloat16_enabled(), not self.bfloat16_enabled()]:
            zero_ckpt_names = self._get_all_zero_checkpoint_names(load_dir, tag, bf16_mode)
            if zero_ckpt_names is not None:
                # Warn if loading checkpoint of different bit16 type
                if bf16_mode is not self.bfloat16_enabled():
                    checkpoint_bit16 = BFLOAT16 if bf16_mode else FP16
                    engine_bit16 = BFLOAT16 if self.bfloat16_enabled() else FP16
                    logger.warn(f'Loading {checkpoint_bit16} zero checkpoints into {engine_bit16} training engine')
                return self._get_all_zero_checkpoint_state_dicts(zero_ckpt_names)

        return None

    def _checkpoint_tag_validation(self, tag):
        if self.checkpoint_tag_validation_enabled():
            s_hash = hashlib.sha1(tag.encode())
            bhash = torch.ByteTensor([s_hash.digest()]).flatten().to(self.device)
            max_bhash = bhash.clone()
            min_bhash = bhash.clone()
            dist.all_reduce(max_bhash, op=dist.ReduceOp.MAX)
            dist.all_reduce(min_bhash, op=dist.ReduceOp.MIN)
            valid = all(min_bhash == bhash) and all(max_bhash == bhash)
            msg = (f"[rank={dist.get_rank()}] The checkpoint tag name '{tag}' is not consistent across "
                   "all ranks. Including rank unique information in checkpoint tag could cause issues when "
                   "restoring with different world sizes.")
            if self.checkpoint_tag_validation_fail():
                assert valid, msg
            elif not valid:
                logger.warning(msg)

    def save_checkpoint(self, save_dir, tag=None, client_state={}, save_latest=True, exclude_frozen_parameters=False):
        """Save training checkpoint

        Arguments:
            save_dir: Required. Directory for saving the checkpoint
            tag: Optional. Checkpoint tag used as a unique identifier for the checkpoint, global step is
                used if not provided. Tag name must be the same across all ranks.
            client_state: Optional. State dictionary used for saving required training states in the client code.
            save_latest: Optional. Save a file 'latest' pointing to the latest saved checkpoint.
            exclude_frozen_parameters: Optional. Exclude frozen parameters from checkpointed state.
        Important: all processes must call this method and not just the process with rank 0. It is
        because each process needs to save its master weights and scheduler+optimizer states. This
        method will hang waiting to synchronize with other processes if it's called just for the
        process with rank 0.

        """
        if self._optimizer_has_ckpt_event_prologue():
            # Custom preparation for checkpoint save, if applicable
            self.optimizer.checkpoint_event_prologue()

        rank = self.local_rank if self.use_node_local_storage() else self.global_rank

        # This is to make sure the checkpoint names are created without collision
        # There seems to be issue creating them in parallel

        # Ensure save_dir directory exists
        self.checkpoint_engine.makedirs(save_dir, exist_ok=True)
        dist.barrier()

        if tag is None:
            tag = f"global_step{self.global_steps}"

        # Ensure tag is a string
        tag = str(tag)
        self.checkpoint_engine.create(tag)

        # Ensure checkpoint tag is consistent across ranks
        self._checkpoint_tag_validation(tag)

        if self.has_moe_layers:
            self.save_non_zero_checkpoint = False
            self._create_checkpoint_file(save_dir, tag, False)
            self._save_moe_checkpoint(save_dir,
                                      tag,
                                      client_state=client_state,
                                      exclude_frozen_parameters=exclude_frozen_parameters)

        # We distribute the task of saving layer checkpoint files among
        # data parallel instances, so all procs should call _save_checkpoint.
        # All procs then call module_state_dict(), but only procs of data
        # parallel rank 0 save the general model params.
        if not self.has_moe_layers:
            self._create_checkpoint_file(save_dir, tag, False)
            self._save_checkpoint(save_dir,
                                  tag,
                                  client_state=client_state,
                                  exclude_frozen_parameters=exclude_frozen_parameters)

        if self.save_zero_checkpoint:
            self._create_zero_checkpoint_files(save_dir, tag)
            self._save_zero_checkpoint(save_dir, tag)

        if self._optimizer_has_ckpt_event_epilogue():
            self.optimizer.checkpoint_event_epilogue()

        # Save latest checkpoint tag
        self.checkpoint_engine.commit(tag)
        if save_latest and rank == 0:
            with open(os.path.join(save_dir, 'latest'), 'w') as fd:
                fd.write(tag)

        dist.barrier()

        return True

    def _get_non_moe_state_dict(self, full_state_dict):
        """
            Get the state dict of the non-moe layers
        """
        for key in list(full_state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                full_state_dict.pop(key)

        return full_state_dict

    def _save_moe_checkpoint(self, save_dir, tag, client_state={}, exclude_frozen_parameters=False):
        save_path = self._get_ckpt_name(save_dir, tag)
        # A hack to save the checkpointing directory. Pipeline parallelism overrides
        # module_state_dict() and uses this path to save the model. module_state_dict()
        # then instead just returns None.

        # Using layer_#_export_# to save the model's expert state_dict
        moe_layer_id = 0
        for n_module, module in self.module.named_modules():
            if isinstance(module, MoE):  # and deepspeed.comm.get_rank() == 0:
                group_name = module.expert_group_name
                num_local_experts = module.num_local_experts
                expp_rank = groups._get_expert_parallel_rank(group_name)
                exp_dp_rank = groups._get_expert_data_parallel_rank(group_name)
                # print(expp_rank, exp_dp_rank)
                if exp_dp_rank != 0:
                    moe_layer_id += 1
                    continue

                # get all moe parameters
                moe_state_dict = {}
                for n, p in module.state_dict().items():
                    if 'expert' in n and 'moe.gate.wg.weight' not in n:
                        moe_state_dict[n_module + '.' + n] = p
                moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
                # print(moe_state_dict.keys()) # until now, everything is fine. So the bug happens at next few lines
                # Reorder the moe name rank, so that each checkpoint only has one expert
                experts_state_dict = defaultdict(dict)
                for key in list(moe_state_dict.keys()):
                    m = re.match(f".*{moe_str_prefix}([0-9]+).*", key)

                    local_expert_id = None
                    if not m:
                        logger.warn(f'No expert found in key {key}.')
                    else:
                        local_expert_id = m.group(1)

                    global_expert_id = expp_rank * \
                        num_local_experts + int(local_expert_id)
                    expert_key = key.replace(f'{moe_str_prefix}{local_expert_id}',
                                             f'{moe_str_prefix}{global_expert_id}')
                    # truncating extra tensor (shared) storage
                    truncated = moe_state_dict.pop(key).clone().detach()
                    experts_state_dict[str(global_expert_id)][expert_key] = truncated

                # let save the moe parameters
                for global_expert_id, expert_state_dict in experts_state_dict.items():
                    # save the moe parameters
                    moe_save_path = self._get_expert_ckpt_name(save_dir, moe_layer_id, global_expert_id, tag, self.mpu)
                    if self.random_ltd_enabled():
                        expert_state_dict = remove_random_ltd_state_dict(expert_state_dict)
                    self.checkpoint_engine.save(expert_state_dict, moe_save_path)
                moe_layer_id += 1

        self._curr_ckpt_path = os.path.join(save_dir, tag)

        largest_group_name = groups._get_max_expert_size_name()
        expp_rank = groups._get_expert_parallel_rank(largest_group_name)
        exp_dp_rank = groups._get_expert_data_parallel_rank(largest_group_name)

        # In the case of E + D parallelism, only the
        # first expert parallel group should save the expert weights
        # since each expert parallel group is a copy of the model's experts
        if exp_dp_rank != 0:
            return

        # Save optimizer states. They are different across each exp parallel rank.
        optimizer_state = {
            'optimizer': self.optimizer.state_dict() if self.optimizer and not self.zero_optimization() else None
        }
        # TODO: why use BufferedWriter not the path
        file_path = self._get_optimizer_ckpt_name(save_dir, tag, expp_rank)
        self.checkpoint_engine.save(optimizer_state, file_path)

        # get non-moe parameters
        model_state_dict = self._get_non_moe_state_dict(
            self.module_state_dict(exclude_frozen_parameters=exclude_frozen_parameters))

        if expp_rank == 0:
            # TODO: update num experts info,.. in checkpoint
            state = {
                'module':
                model_state_dict,
                'lr_scheduler':
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                'data_sampler':
                self.training_dataloader.data_sampler.state_dict() if
                (self.training_dataloader is not None and self.curriculum_learning_enabled()) else None,
                'random_ltd':
                self.random_ltd_scheduler.state_dict() if self.random_ltd_enabled() else None,
                'sparse_tensor_module_names':
                self.sparse_tensor_module_names,
                'skipped_steps':
                self.skipped_steps,
                'global_steps':
                self.global_steps,
                'global_samples':
                self.global_samples,
                'dp_world_size':
                self.dp_world_size,
                'mp_world_size':
                self.mp_world_size,
                'num_experts':
                self.num_experts
            }
            state.update(client_state)
            logger.info(f'Saving model checkpoint: {save_path}')
            self.checkpoint_engine.save(state, save_path)
        self._curr_save_path = None

    def _create_checkpoint_file(self, save_dir, tag, zero_checkpoint):
        name_function = (self._get_zero_ckpt_name if zero_checkpoint else self._get_ckpt_name)
        try:
            checkpoint_name = name_function(save_dir, tag)
            path = os.path.dirname(checkpoint_name)
            self.checkpoint_engine.makedirs(path, exist_ok=True)
        except:
            logger.error(f"Failed saving model checkpoint to {save_dir} with tag {tag}")
            return False

        return True

    def _create_zero_checkpoint_files(self, save_dir, tag):
        success = True
        # zero checkpoint files are created sequentially
        for rank in range(dist.get_world_size(self.optimizer.dp_process_group)):
            if rank == self.global_rank:
                success = self._create_checkpoint_file(save_dir, tag, True)

            dist.barrier(group=self.optimizer.dp_process_group)

        return success

    def _save_checkpoint(self, save_dir, tag, client_state={}, exclude_frozen_parameters=False):

        save_path = self._get_ckpt_name(save_dir, tag)

        zero_optimizer_state = self.zero_optimization() or self.bfloat16_enabled()

        save_frozen_param = self.zero_optimization_partition_gradients() and not exclude_frozen_parameters

        # A hack to save the checkpointing directory. Pipeline parallelism overrides
        # module_state_dict() and uses this path to save the model. module_state_dict()
        # then instead just returns None.  The module_state_dict() implementation in
        # PipelineEngine expects the save path to be set in self._curr_ckpt_path.
        self._curr_ckpt_path = os.path.join(save_dir, tag)
        module = self.module_state_dict(exclude_frozen_parameters=exclude_frozen_parameters)
        self._curr_ckpt_path = None

        state = dict(module=module,
                     buffer_names=self._get_buffer_names(),
                     optimizer=self.optimizer.state_dict() if self.optimizer and not zero_optimizer_state else None,
                     param_shapes=self._get_zero_param_shapes() if self.optimizer and zero_optimizer_state else None,
                     frozen_param_shapes=self._get_zero_frozen_param_attributes(self._get_param_shape_func)
                     if save_frozen_param else None,
                     shared_params=self._get_shared_params() if self.optimizer and zero_optimizer_state else None,
                     frozen_param_fragments=self._get_zero_frozen_param_attributes(self._get_param_fragment_func)
                     if save_frozen_param else None,
                     lr_scheduler=self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                     data_sampler=self.training_dataloader.data_sampler.state_dict() if
                     (self.training_dataloader is not None and self.curriculum_learning_enabled()) else None,
                     random_ltd=self.random_ltd_scheduler.state_dict() if self.random_ltd_enabled() else None,
                     sparse_tensor_module_names=self.sparse_tensor_module_names,
                     skipped_steps=self.skipped_steps,
                     global_steps=self.global_steps,
                     global_samples=self.global_samples,
                     dp_world_size=self.seq_dp_world_size,
                     mp_world_size=self.mp_world_size,
                     ds_config=self.config,
                     ds_version=version)
        state.update(client_state)

        if self.save_non_zero_checkpoint:
            log_dist(message=f'Saving model checkpoint: {save_path}', ranks=[0, 1])
            self.checkpoint_engine.save(state, save_path)

    def _get_buffer_names(self):
        buffer_names = []

        # we save buffer names so that we could extract later the real buffers from the saved
        # state_dict["module"] in the non-zero checkpoint - the buffers are already there but they
        # are intermixed with param placeholders

        # have to traverse the tree to be able to skip non-persistent buffers
        def get_layer_named_buffers(module, prefix=""):
            for name, buf in module.named_buffers(recurse=False):
                if buf is not None and name not in module._non_persistent_buffers_set:
                    buffer_names.append(prefix + name)

            for name, child in module.named_children():
                if child is not None:
                    get_layer_named_buffers(child, prefix + name + ".")

        get_layer_named_buffers(self.module, prefix="")

        return buffer_names

    def _get_param_shape_func(self, param):
        return param.ds_shape if hasattr(param, 'ds_id') else param.shape

    def _get_param_fragment_func(self, param):
        return param.ds_tensor.detach().cpu() if hasattr(param, 'ds_id') else param.detach().cpu()

    def _get_zero_frozen_param_attributes(self, attr_func):
        frozen_param_fragments = OrderedDict()

        for param in self.module.parameters():
            if param.requires_grad:
                continue
            if param not in self.param_names:
                raise ValueError(f"failed to find frozen {param} in named params")
            name = self.param_names[param]
            frozen_param_fragments[name] = attr_func(param)

        return frozen_param_fragments

    def _get_zero_param_shapes(self):
        """Returns a dict of name to shape mapping, only for the flattened fp32 weights saved by the
        optimizer. the names are exactly as in state_dict. The order is absolutely important, since
        the saved data is just flattened data with no identifiers and requires reconstruction in the
        same order it was saved.
        We can't rely on self.module.named_parameters() to get the saved tensors, as some params
        will be missing and others unsaved and then it'd be impossible to reconstruct state_dict
        from the flattened weights.
        optimizer.bit16_groups seems to be the easiest to use as it's in all zeroX versions.
        """
        param_group_shapes = []
        cnt = 0
        numel = 0

        # zero2 started using a round_robin_bit16_groups which is a shuffled version of bit16_groups -
        # if we don't use it, we get parameters ordered incorrectly
        if hasattr(self.optimizer, "round_robin_bit16_groups"):
            bit16_groups = self.optimizer.round_robin_bit16_groups
        elif self.bfloat16_enabled() and hasattr(self.optimizer, "bf16_groups"):
            bit16_groups = self.optimizer.bf16_groups
        else:
            bit16_groups = self.optimizer.bit16_groups if self.zero_optimization_stage(
            ) == 2 else self.optimizer.fp16_groups

        for bit16_group in bit16_groups:
            param_shapes = OrderedDict()
            for param in bit16_group:
                cnt += 1
                numel += param.ds_numel if hasattr(param, "ds_numel") else param.numel()
                shape = param.ds_shape if hasattr(param, "ds_shape") else param.shape
                if param not in self.param_names:
                    raise ValueError(f"failed to find optimizer param in named params")
                name = self.param_names[param]
                param_shapes[name] = shape

                # uncomment to debug zero_to_fp32.py problems
                # if self.global_rank == 0: print(f"saving param {name} {shape} (numel={shape.numel()})")
            param_group_shapes.append(param_shapes)
        # if self.global_rank == 0: print(f"Total saved {numel} numels in {cnt} params")

        return param_group_shapes

    def _get_shared_params(self):
        """
        Returns a dict of shared params, which can later be used to reconstruct the original state dict,
        e.g. in `zero_to_fp32`. Each dict entry is a pair of param names, where the key is the name
        of the variable that isn't stored and the value is the actual param holding data.
        """
        shared_index = {}
        shared_params_by_full_name = {}

        is_zero3_model = (self.zero_optimization_partition_weights()
                          and any(hasattr(param, "ds_id") for param in self.module.parameters()))

        def get_layer_state_dict(module, prefix=""):
            # handle params
            for name, param in module.named_parameters(recurse=False):
                if param is None or (is_zero3_model and not hasattr(param, "ds_id")):
                    continue
                key = prefix + name

                # When weights are manged by stage 3, we can't rely on param.data_ptr() as it will be reused
                # as weights get gathered and reduced, but param.ds_id is unique across all zero weights
                # (and shared params will have the same param.ds_id)
                param_id = param.ds_id if is_zero3_model else param.data_ptr()

                if param_id in shared_index:
                    # shared weights
                    #print(f"`{key}` is shared with `{shared_index[param_id]}`")
                    shared_params_by_full_name[key] = shared_index[param_id]
                else:
                    shared_index[param_id] = key

            for name, child in module.named_children():
                if child is not None:
                    get_layer_state_dict(child, prefix + name + ".")

        if dist.get_rank() == 0:
            get_layer_state_dict(self.module, prefix="")

        return shared_params_by_full_name

    def _copy_recovery_script(self, save_path):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        script = "zero_to_fp32.py"
        src = os.path.join(base_dir, "utils", script)
        dst = os.path.join(save_path, script)
        #logger.info(f"creating recovery script {dst}")
        copyfile(src, dst)
        self._change_recovery_script_permissions(dst)

    def _change_recovery_script_permissions(self, dst):
        # make executable (safeguard for file shares - Azure as example)
        try:
            os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC)
        except (FileNotFoundError, PermissionError) as e:
            #this message is used in unit test TestZeRONonDistributed
            logger.info(
                f'Warning: Could not change permissions for {dst} due to error: {e}. Continuing without changing permissions.'
            )

    def _save_zero_checkpoint(self, save_path, tag):
        zero_checkpoint_name = self._get_zero_ckpt_name(save_path, tag)
        zero_sd = dict(optimizer_state_dict=self.optimizer.state_dict(), ds_config=self.config, ds_version=version)
        self.checkpoint_engine.save(zero_sd, zero_checkpoint_name)

        if self.global_rank == 0:
            self._copy_recovery_script(save_path)
        ckpt_type = 'zero' if self.zero_optimization() else 'bf16_zero'
        logger.info(f'{ckpt_type} checkpoint saved {zero_checkpoint_name}')

    def _zero3_consolidated_16bit_state_dict(self):
        """
        Get a full non-partitioned state_dict with fp16 weights on cpu.
        Important: this function must be called on all ranks and not just rank 0.
        This is similar to nn.Module.state_dict (modelled after _save_to_state_dict), but:
        1. consolidates the weights from different partitions on gpu0
        2. works on one layer at a time to require as little gpu0 memory as possible, by
        moving the already consolidated weights to cpu
        3. takes care to keep the shared params shared when gradually copying the params to cpu
        Returns:
            a consolidated fp16 ``state_dict`` on cpu on rank 0, ``None`` on other ranks
        """
        if not self.zero_optimization_partition_weights():
            raise ValueError("this function requires ZeRO-3 mode")

        state_dict = OrderedDict() if dist.get_rank() == 0 else None
        shared_params = {}

        def get_layer_state_dict(module, prefix=""):
            # gather one layer at a time to be memory-efficient
            # must use modifier_rank=0 to release GPU memory after each layer gathered
            #see_memory_usage("before GatheredParameters", force=True)
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if dist.get_rank() == 0:
                    # handle params
                    for name, param in module.named_parameters(recurse=False):
                        if param is None:
                            continue
                        key = prefix + name
                        # can't rely on param.data_ptr() as it will be reused as weights gets
                        # gathered and reduced, but param.ds_id is unique across all zero weights
                        # (and shared params will have the same param.ds_id)
                        if param.ds_id in shared_params:
                            # shared weights
                            #print(f"`{key}` is shared with `{shared_params[param.ds_id]}`")
                            state_dict[key] = state_dict[shared_params[param.ds_id]]
                        else:
                            state_dict[key] = param.detach().cpu()
                            shared_params[param.ds_id] = key
                        #print(f"param {param.ds_id} {param.shape} {key} ")

                    # now buffers - not sure if need to take care of potentially shared weights here
                    for name, buf in module.named_buffers(recurse=False):
                        if (buf is not None and name not in module._non_persistent_buffers_set):
                            state_dict[prefix + name] = buf.detach().cpu()
            #see_memory_usage("after GatheredParameters", force=True)

            for name, child in module.named_children():
                if child is not None:
                    get_layer_state_dict(child, prefix + name + ".")

        # Prepare for checkpoint save by ensuring all parameters are partitioned
        if self._optimizer_has_ckpt_event_prologue():
            self.optimizer.checkpoint_event_prologue()

        see_memory_usage("before get_layer_state_dict", force=False)
        get_layer_state_dict(self.module, prefix="")
        see_memory_usage("after get_layer_state_dict", force=False)

        if self._optimizer_has_ckpt_event_epilogue():
            self.optimizer.checkpoint_event_epilogue()

        return state_dict

    def save_fp16_model(self, save_dir, save_filename="pytorch_model.bin"):
        """has been renamed to save_16bit_model, keeping this around for backwards
        compatibility"""
        return self.save_16bit_model(save_dir, save_filename)

    def save_16bit_model(self, save_dir, save_filename="pytorch_model.bin"):
        """
        Save 16bit model weights

        This method saves the 16bit model weights at the desired destination.

        Arguments:
            save_dir: Required. Directory for saving the model
            save_filename: Optional. Filename to save to. Defaults to ``pytorch_model.bin``

        Returns:
            ``True`` when a model has been saved, ``False`` otherwise. It will not be saved if
            stage3_gather_16bit_weights_on_model_save is ``False``.

        Important: all processes must call this method and not just the process with rank 0. It is
        because the processes need to work in sync to gather the weights. This method will hang
        waiting to synchronize with other processes if it's called just for the process with rank 0.

        """

        path = os.path.join(save_dir, save_filename)

        if self.zero_optimization_partition_weights():
            if self.zero_gather_16bit_weights_on_model_save():
                # consolidation is expensive in time and memory and therefore isn't a default
                state_dict = self._zero3_consolidated_16bit_state_dict()
            else:
                # the model will be bogus if not consolidated so don't confuse the user by saving it
                logger.info(
                    f"Did not save the model {path} because `stage3_gather_16bit_weights_on_model_save` is False")
                return False
        else:
            state_dict = self.module.state_dict()

        tag = f"global_step{self.global_steps}"
        tag = str(tag)
        self.checkpoint_engine.create(tag)

        if dist.get_rank() == 0:
            self.checkpoint_engine.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving model weights to {path}, tag: {tag}")
            self.checkpoint_engine.save(state_dict, path)

        self.checkpoint_engine.commit(tag)

        return True

    def empty_partition_cache(self):
        """
        Release GPU memory consumed by offloaded model parameters.
        """
        if hasattr(self.optimizer, 'empty_partition_cache'):
            self.optimizer.empty_partition_cache()
            gc.collect()
            get_accelerator().empty_cache()
