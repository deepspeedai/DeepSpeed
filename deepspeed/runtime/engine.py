"""
Copyright 2019 The Microsoft DeepSpeed Team
"""

import os
import re
import stat
import math
import torch
import warnings
import hashlib
import torch.distributed as dist
from collections import defaultdict, OrderedDict
from shutil import copyfile

from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.distributed.distributed_c10d import _get_global_rank

from typing import Callable, Dict, Optional, Union, Iterable

from deepspeed.runtime.utils import see_memory_usage, get_ma_status, DummyOptim
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.runtime.zero.utils import is_zero_supported_optimizer, ZeRORuntimeException
from deepspeed.runtime.activation_checkpointing import (
    checkpointing as activation_checkpointing,
)
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer

from deepspeed.runtime.config import DeepSpeedConfig, DEEPSPEED_OPTIMIZERS, \
    ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER, \
    TORCH_ADAM_PARAM, ADAM_W_MODE, ADAM_W_MODE_DEFAULT

from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.constants import \
    ROUTE_TRAIN, ROUTE_PREDICT, ROUTE_EVAL, \
    PLD_THETA, PLD_GAMMA, OPTIMIZER_STATE_DICT
from deepspeed.runtime.zero.constants import \
    ZERO_OPTIMIZATION_OPTIMIZER_STATES, ZERO_OPTIMIZATION_GRADIENTS, ZERO_OPTIMIZATION_WEIGHTS, \
    SINGLE_PARTITION_OF_FP32_GROUPS
from deepspeed.runtime.sparse_tensor import SparseTensor

import deepspeed.runtime.lr_schedules as lr_schedules
import deepspeed.utils.groups as groups
from deepspeed.runtime.utils import get_grad_norm
from deepspeed.utils import logger, log_dist, init_distributed
from deepspeed.utils.timer import ThroughputTimer, SynchronizedWallClockTimer
from deepspeed.utils.debug import debug_extract_module_and_param_names
from deepspeed.runtime.progressive_layer_drop import ProgressiveLayerDrop
from deepspeed.runtime.utils import clip_grad_norm_
from deepspeed.runtime.eigenvalue import Eigenvalue
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler

from .pipe.module import PipelineModule
from .utils import ensure_directory_exists, get_ma_status
from ..ops.op_builder import UtilsBuilder
from ..ops.adam import DeepSpeedCPUAdam
from ..ops.adam import FusedAdam
from ..moe.sharded_moe import TopKGate, MOELayer
from ..moe.layer import MoE
from ..git_version_info import version

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.utils.logging import print_json_dist

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
    pass


def split_half_float_double_sparse(tensors):
    supported_types = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
        "torch.cuda.BFloat16Tensor",
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


def print_configuration(args, name):
    logger.info("{}:".format(name))
    for arg in sorted(vars(args)):
        dots = "." * (29 - len(arg))
        logger.info("  {} {} {}".format(arg, dots, getattr(args, arg)))


FORWARD_MICRO_TIMER = 'forward_microstep'
FORWARD_GLOBAL_TIMER = 'forward'
BACKWARD_MICRO_TIMER = 'backward_microstep'
BACKWARD_GLOBAL_TIMER = 'backward'
BACKWARD_INNER_MICRO_TIMER = 'backward_inner_microstep'
BACKWARD_INNER_GLOBAL_TIMER = 'backward_inner'
BACKWARD_REDUCE_MICRO_TIMER = 'backward_allreduce_microstep'
BACKWARD_REDUCE_GLOBAL_TIMER = 'backward_allreduce'
STEP_MICRO_TIMER = 'step_microstep'
STEP_GLOBAL_TIMER = 'step'


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
                FORWARD_MICRO_TIMER,
                BACKWARD_MICRO_TIMER,
                BACKWARD_INNER_MICRO_TIMER,
                BACKWARD_REDUCE_MICRO_TIMER,
                STEP_MICRO_TIMER
            ]

        if enable_global_timers:
            self.forward_timers += [FORWARD_GLOBAL_TIMER]
            self.backward_timers += [BACKWARD_GLOBAL_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_GLOBAL_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_GLOBAL_TIMER]
            self.step_timers += [STEP_GLOBAL_TIMER]
            self.global_timers += [
                FORWARD_GLOBAL_TIMER,
                BACKWARD_GLOBAL_TIMER,
                BACKWARD_INNER_GLOBAL_TIMER,
                BACKWARD_REDUCE_GLOBAL_TIMER,
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
        config_params=None,
        dont_change_device=False,
    ):
        super(DeepSpeedEngine, self).__init__()
        self.dont_change_device = dont_change_device
        self.client_optimizer = optimizer
        self.client_model_parameters = model_parameters
        self.client_lr_scheduler = lr_scheduler
        self.training_data = training_data
        self.collate_fn = collate_fn
        self.mpu = mpu
        self.data_parallel_group = None
        self.global_steps = 0
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gradient_average = True
        self.warn_unscaled_loss = True
        self.config = config
        self.loaded_checkpoint_mp_world_size = None
        self.loaded_checkpoint_dp_world_size = None
        self.enable_backward_allreduce = True
        self.progressive_layer_drop = None
        self.eigenvalue = None
        self.block_eigenvalue = None
        self.gas_boundary_ctr = 0
        self.dist_backend = "nccl"
        self.has_moe_layers = False
        self.num_experts = None
        self.gate_modules = []
        self.moe_layers = []
        self._step_applied = False
        self._global_grad_norm = None
        self._is_gradient_accumulation_boundary = None

        # for debug purposes - can then debug print: debug_get_module_name(module)
        debug_extract_module_and_param_names(model)

        # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
        self.param_names = {param: name for name, param in model.named_parameters()}

        # Set config using config_params for backwards compat
        if self.config is None and config_params is not None:
            self.config = config_params

        if dist_init_required is None:
            dist_init_required = not dist.is_initialized()

        if dist_init_required is False:
            assert (
                dist.is_initialized() is True
            ), "Torch distributed not initialized. Please set dist_init_required to True or initialize before calling deepspeed.initialize()"
        else:
            # Initialize torch distributed if needed
            init_distributed(dist_backend=self.dist_backend)

        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)
        self._do_sanity_check()
        see_memory_usage(f"DeepSpeed Engine: After args sanity test",
                         force=self.memory_breakdown())
        if mpu is not None:
            assert not self.elasticity_enabled(), (
                "Elasticity is not currently supported" " with model parallelism."
            )

        self._set_distributed_vars(args)

        if self.tensorboard_enabled() and self.global_rank == 0:
            self.summary_writer = self.get_summary_writer()

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
            batch_size=self.train_micro_batch_size_per_gpu(),
            num_workers=self.dp_world_size,
            steps_per_output=self.steps_per_print(),
            monitor_memory=False,
        )

        if dist.get_rank() == 0:
            logger.info(
                f"DeepSpeed Flops Profiler Enabled: {self.flops_profiler_enabled()}")

        if self.flops_profiler_enabled():
            self.flops_profiler = FlopsProfiler(self.module, self)

        if training_data:
            self.training_dataloader = self.deepspeed_io(training_data)
        else:
            self.training_dataloader = None

        # Configure optimizer and scheduler
        self.optimizer = None
        self.basic_optimizer = None
        self.lr_scheduler = None
        if model_parameters or optimizer:
            self._configure_optimizer(optimizer, model_parameters)
            self._configure_lr_scheduler(lr_scheduler)
            self._report_progress(0)
        elif self.zero_optimization():
            # no optim selected but zero is enabled
            self.optimizer = self._configure_zero_optimizer(optimizer=None)

        self._get_model_parameters()

        # Bookkeeping for sparse support
        self.sparse_tensor_module_names = set()
        # if self.sparse_gradients_enabled():
        for name, module in self.module.named_modules():
            if isinstance(module,
                          (torch.nn.Embedding,
                           torch.nn.EmbeddingBag)) and self.sparse_gradients_enabled():
                self.sparse_tensor_module_names.add(name + ".weight")
                logger.info(
                    "Will convert {} to sparse tensor during training".format(name))

        self.save_non_zero_checkpoint = False
        self.save_zero_checkpoint = False
        self._configure_checkpointing(dist_init_required)

        if self.eigenvalue_enabled():
            self.eigenvalue = self._configure_eigenvalue()

        if self.pld_enabled():
            self.progressive_layer_drop = self._configure_progressive_layer_drop()

        if self.curriculum_enabled():
            self.curriculum_scheduler = self._configure_curriculum_scheduler()

        # Engine timers

        self.engine_timers = EngineTimers(
            enable_micro_timers=self.wall_clock_breakdown(),
            enable_global_timers=self.wall_clock_breakdown()
            or self.flops_profiler_enabled())

        if self.global_rank == 0:
            self._config.print("DeepSpeedEngine configuration")
            if self.dump_state():
                print_configuration(self, "DeepSpeedEngine")

        # Load pre-installed or JIT compile (un)flatten ops
        util_ops = UtilsBuilder().load()
        self.flatten = util_ops.flatten
        self.unflatten = util_ops.unflatten

    def _get_model_parameters(self):
        if self.autotuning_profile_model_info():
            self.autotuning_model_info = {}
            num_params = 0
            trainable_num_params = 0

            for p in self.module.parameters():
                # since user code might call deepspeed.zero.Init() before deepspeed.initialize(), need to check the attrbuite to check if the parameter is partitioned in zero 3 already or not
                n = 0
                if hasattr(p, "ds_tensor"):  # if the parameter is partitioned in zero 3
                    n += p.ds_numel
                else:  # if the parameter is not partitioned in zero 3 yet
                    n += p.numel()
                num_params += n
                if p.requires_grad:
                    trainable_num_params += n
            if self.global_rank == 0:
                self.autotuning_model_info[
                    "num_params"] = num_params * self.mp_world_size
                self.autotuning_model_info[
                    "trainable_num_params"] = trainable_num_params * self.mp_world_size

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
        if train_batch_size % (self.train_micro_batch_size_per_gpu() *
                               self.dp_world_size) != 0:
            #print(f'{train_batch_size=} {self.train_micro_batch_size_per_gpu()=} {self.dp_world_size=}')
            raise ValueError(
                f'Train batch size must be divisible by micro-batch data parallelism')
        new_gas = train_batch_size // (self.train_micro_batch_size_per_gpu() *
                                       self.dp_world_size)
        # overwrite config
        self._config.train_batch_size = train_batch_size
        self._config.gradient_accumulation_steps = new_gas

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
        if train_batch_size % (self.train_micro_batch_size_per_gpu() *
                               self.dp_world_size) != 0:
            #print(f'{train_batch_size=} {self.train_micro_batch_size_per_gpu()=} {self.dp_world_size=}')
            raise ValueError(
                f'Train batch size must be divisible by micro-batch data parallelism')
        new_gas = train_batch_size // (self.train_micro_batch_size_per_gpu() *
                                       self.dp_world_size)
        # overwrite config
        self._config.train_batch_size = train_batch_size
        self._config.gradient_accumulation_steps = new_gas

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

    def checkpoint_tag_validation_enabled(self):
        return self._config.checkpoint_tag_validation_enabled

    def checkpoint_tag_validation_fail(self):
        return self._config.checkpoint_tag_validation_fail

    def elasticity_enabled(self):
        return self._config.elasticity_enabled

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

    def curriculum_enabled(self):
        return self._config.curriculum_enabled

    def curriculum_params(self):
        return self._config.curriculum_params

    def tensorboard_enabled(self):
        return self._config.tensorboard_enabled

    def tensorboard_output_path(self):
        return self._config.tensorboard_output_path

    def tensorboard_job_name(self):
        return self._config.tensorboard_job_name

    def get_summary_writer(
            self,
            name="DeepSpeedJobName",
            base=os.path.join(os.path.expanduser("~"),
                              "tensorboard"),
    ):
        if self.tensorboard_output_path():
            base_dir = self.tensorboard_output_path()
            job_name = self.tensorboard_job_name()
            log_dir = os.path.join(base_dir, job_name)
        else:
            if self.tensorboard_job_name():
                name = self.tensorboard_job_name()

            # Infrastructure-specific job-id
            if "DLWS_JOB_ID" in os.environ:
                infra_job_id = os.environ["DLWS_JOB_ID"]
            elif "DLTS_JOB_ID" in os.environ:
                infra_job_id = os.environ["DLTS_JOB_ID"]
            else:
                infra_job_id = "unknown-job-id"

            summary_writer_dir_name = os.path.join(infra_job_id, "logs")
            log_dir = os.path.join(base, summary_writer_dir_name, name)

        os.makedirs(log_dir, exist_ok=True)
        try:
            # torch.utils.tensorboard will fail if `tensorboard` is not available,
            # see their docs for more details: https://pytorch.org/docs/1.8.0/tensorboard.html
            import tensorboard
        except ImportError:
            print(
                'If you want to use tensorboard logging please `pip install tensorboard`'
            )
            raise
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=log_dir)

    def wall_clock_breakdown(self):
        return self._config.wall_clock_breakdown

    def flops_profiler_enabled(self):
        return self._config.flops_profiler_config.enabled or self.autotuning_enabled()

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
            "profile",
            False)

    def sparse_gradients_enabled(self):
        return self._config.sparse_gradients_enabled

    def train_batch_size(self):
        return self._config.train_batch_size

    def train_micro_batch_size_per_gpu(self):
        return self._config.train_micro_batch_size_per_gpu

    def optimizer_name(self):
        return (self.client_optimizer.__class__.__name__
                if self.client_optimizer else self._config.optimizer_name)

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
            self._config.quantize_training_enabled,
            self._config.quantize_target_bits,
            self._config.quantize_start_bits,
            self._config.quantize_period,
            self._config.quantize_offset,
            self._config.quantize_groups,
            self._config.fp16_mixed_quantize,
            self._config.quantize_change_rate,
            self._config.quantize_type,
            self._config.quantize_rounding,
            self._config.quantize_verbose,
            self._config.use_quantizer_kernel,
        )

    def zero_optimization(self):
        return self._config.zero_enabled

    def zero_allow_untested_optimizer(self):
        return self._config.zero_allow_untested_optimizer

    def zero_reduce_scatter(self):
        return self._config.zero_config.reduce_scatter

    def zero_overlap_comm(self):
        return self._config.zero_config.overlap_comm

    def zero_offload_optimizer(self):
        return self._config.zero_config.offload_optimizer

    def zero_offload_param(self):
        return self._config.zero_config.offload_param

    def zero_cpu_offload(self):
        return self._config.zero_config.offload_optimizer is not None

    def zero_sub_group_size(self):
        return self._config.zero_config.sub_group_size

    def zero_optimization_stage(self):
        return self._config.zero_optimization_stage

    def zero_reduce_bucket_size(self):
        return self._config.zero_config.reduce_bucket_size

    def zero_allgather_bucket_size(self):
        return self._config.zero_config.allgather_bucket_size

    def zero_optimization_partition_gradients(self):
        return self.zero_optimization_stage() >= ZERO_OPTIMIZATION_GRADIENTS

    def zero_optimization_partition_weights(self):
        return self.zero_optimization_stage() >= ZERO_OPTIMIZATION_WEIGHTS

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

    def zero_gather_fp16_weights_on_model_save(self):
        return self._config.zero_config.gather_fp16_weights_on_model_save

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

    def loss_scale(self):
        return self._config.loss_scale

    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    @property
    def communication_data_type(self):
        res = self._config.communication_data_type
        if res is not None:
            return res
        elif self.fp16_enabled() or self.zero_optimization_stage():
            return torch.float16
        elif self.bfloat16_enabled():
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

    def _configure_lr_scheduler(self, client_lr_scheduler):
        # First check for scheduler in json configuration
        lr_scheduler = self._scheduler_from_config(self.optimizer)
        if lr_scheduler:
            if self.global_rank == 0:
                logger.info(
                    f"DeepSpeed using configured LR scheduler = {self.scheduler_name()}")
            self.lr_scheduler = lr_scheduler
        else:
            if isinstance(client_lr_scheduler, Callable):
                if self.global_rank == 0:
                    logger.info('DeepSpeed using client callable to create LR scheduler')
                self.lr_scheduler = client_lr_scheduler(self.basic_optimizer)
            else:
                if self.global_rank == 0:
                    logger.info('DeepSpeed using client LR scheduler')
                self.lr_scheduler = client_lr_scheduler

        log_dist(f'DeepSpeed LR Scheduler = {self.lr_scheduler}', ranks=[0])

    def _configure_checkpointing(self, dist_init_required):

        dp_rank = self.global_rank
        if self.mpu:
            dp_rank = self.mpu.get_data_parallel_rank()

        # only the first data parallel process needs to store the model checkpoint
        self.save_non_zero_checkpoint = (
            dp_rank == 0) or self.zero_optimization_partition_weights()

        if self.zero_optimization():
            param_rank = torch.distributed.get_rank(
                group=self.optimizer.dp_process_group)

            # Only the first parameter parallel process needs to store the
            # optimizer state checkpoints for zero
            self.save_zero_checkpoint = param_rank == dp_rank

    def _scheduler_from_config(self, optimizer):
        scheduler_name = self.scheduler_name()
        if scheduler_name is not None:
            if hasattr(lr_schedules, scheduler_name):
                scheduler = getattr(lr_schedules, scheduler_name)
            else:
                assert hasattr(
                    torch.optim.lr_scheduler, scheduler_name
                ), f"DeepSpeed does not recognize LR scheduler {scheduler_name}"

                scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)

            scheduler_params = self.scheduler_params()
            instantiated_scheduler = scheduler(optimizer, **scheduler_params)
            return instantiated_scheduler
        else:
            return None

    def _set_distributed_vars(self, args):
        device_rank = args.device_rank if args is not None and hasattr(
            args,
            'device_rank') else self.local_rank
        if device_rank >= 0:
            torch.cuda.set_device(device_rank)
            self.device = torch.device("cuda", device_rank)
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device("cuda")

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        # After the distributed backend is initialized we are guaranteed the LOCAL_RANK
        # environment variable is set. We must align args.local_rank to this value for
        # backwards compatability with scripts relying on [args|self].local_rank containing
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

        if self.config is None:
            self.config = (args.deepspeed_config
                           if hasattr(args,
                                      "deepspeed_config") else None)
        self._config = DeepSpeedConfig(self.config, mpu)

    # Validate command line arguments
    def _do_args_sanity_check(self, args):
        if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
            logger.warning(
                "************ --deepscale_config is deprecated, please use --deepspeed_config ************"
            )
            if hasattr(args, "deepspeed_config"):
                assert (
                    args.deepspeed_config is None
                ), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
            args.deepspeed_config = args.deepscale_config

        assert "LOCAL_RANK" in os.environ or "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ, "DeepSpeed requires the LOCAL_RANK environment " \
            "variable, it is set by the deepspeed launcher, deepspeed.init_distributed, or the torch.distributed launcher. If using a " \
            "different launcher please ensure LOCAL_RANK is set prior to initializing deepspeed."

        if hasattr(args, 'local_rank') and args.local_rank != None:
            assert isinstance(
                args.local_rank, int), f"args.local_rank of {args.local_rank} is an unknown type {type(args.local_rank)}"
            if args.local_rank >= 0:
                env_local_rank = int(os.environ.get("LOCAL_RANK"))
                assert (
                    env_local_rank == args.local_rank
                ), f"Mismatch in local rank setting, args.local_rank={args.local_rank} but env['LOCAL_RANK']={env_local_rank}."

        if self.config is None:
            assert (
                hasattr(
                    args, "deepspeed_config") and args.deepspeed_config is not None
            ), "DeepSpeed requires --deepspeed_config to specify configuration file"

            assert os.path.isfile(
                args.deepspeed_config
            ), "DeepSpeed configuration file: {} is not an existing file".format(
                args.deepspeed_config
            )

    def _is_supported_optimizer(self, optimizer_name):
        return (optimizer_name in DEEPSPEED_OPTIMIZERS
                or getattr(torch.optim,
                           optimizer_name,
                           None) is not None)

    # Validate configuration based on command line arguments
    def _do_sanity_check(self):
        assert isinstance(self.client_optimizer, (type(None), Optimizer, Callable)), \
            f'Client Optimizer is of unexpected type {type(self.client_optimizer)}'

        if not self.client_optimizer:
            if self.optimizer_name() is not None:
                assert self._is_supported_optimizer(
                    self.optimizer_name()
                ), "{} is not a supported DeepSpeed Optimizer".format(
                    self.optimizer_name()
                )

        if (self.optimizer_name() == LAMB_OPTIMIZER
                or self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER):
            assert (
                self.dynamic_loss_scale()
            ), "DeepSpeed {} optimizer requires dynamic loss scaling".format(
                self.optimizer_name()
            )

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
            if hasattr(p, 'allreduce') and not p.allreduce:
                if torch.is_tensor(p) and is_replicated(p):
                    dist.broadcast(p,
                                   self.expert_broadcast_src_rank,
                                   group=self.expert_data_parallel_group)
            else:
                if torch.is_tensor(p) and is_replicated(p):
                    dist.broadcast(p,
                                   self.broadcast_src_rank,
                                   group=self.data_parallel_group)

    def _configure_distributed_model(self, model):
        self.module = model
        if self.fp16_enabled():
            if self.zero_optimization_partition_weights() and any(
                [hasattr(param,
                         "ds_id") for param in self.module.parameters()]):
                if not all(
                    [param.dtype == torch.half for param in self.module.parameters()]):
                    names = [
                        n for n,
                        p in self.module.named_parameters() if p.dtype != torch.half
                    ]
                    raise ValueError(
                        f"fp16 is enabled but the following parameters have dtype that is not fp16: {', '.join(names)}"
                    )
            self.module.half()
        elif self.bfloat16_enabled():
            self.module.bfloat16()
        else:
            if not all(
                [param.dtype == torch.float for param in self.module.parameters()]):
                names = [
                    n for n,
                    p in self.module.named_parameters() if p.dtype != torch.float
                ]
                raise ValueError(
                    f"fp32 is enabled but the following parameters have dtype that is not fp32: {', '.join(names)}"
                )

        if not self.dont_change_device:
            self.module.to(self.device)

        # MoE related initialization
        for _, module in self.module.named_modules():
            if isinstance(module, MoE):
                self.has_moe_layers = True
                self.num_experts = module.num_experts
                break

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

        if not self.pipeline_parallelism:
            # PipeEngine's mpu object is different from Megatron's mpu object
            # so we handle them separately
            if self.mpu is not None:
                if groups.is_initialized():
                    # Scenario 4 - Case 1
                    assert self.mpu.get_data_parallel_world_size() == groups.get_data_parallel_world_size(
                    ), "mpu object provided must match mpu object provided to groups.initialize()"
                    assert self.mpu.get_model_parallel_world_size() == groups.get_model_parallel_world_size(
                    ), "mpu object provided must match mpu object provided to groups.initialize()"
                else:
                    # Scenario 3
                    groups.initialize(mpu=self.mpu)
            else:
                if not groups.is_initialized():
                    # Scenario 1
                    groups.initialize()
                # else:
                # Scenario 2
                # Scenario 4 - Case 2
                # pass

            self.data_parallel_group = groups.get_data_parallel_group()
            self.dp_world_size = groups.get_data_parallel_world_size()
            self.mp_world_size = groups.get_model_parallel_world_size()
            self.broadcast_src_rank = _get_global_rank(groups.get_data_parallel_group(),
                                                       0)
        else:
            self.data_parallel_group = self.mpu.get_data_parallel_group()
            self.dp_world_size = self.mpu.get_data_parallel_world_size()
            self.mp_world_size = self.mpu.get_model_parallel_world_size()
            self.broadcast_src_rank = _get_global_rank(
                self.mpu.get_data_parallel_group(),
                0)

        if self.has_moe_layers:
            # No assert needed because this will only be true if MoE Layer creation was successful
            self.expert_data_parallel_group = groups.get_expert_data_parallel_group()
            self.expert_parallel_group = groups.get_expert_parallel_group()
            self.ep_world_size = groups.get_expert_parallel_world_size()
            self.expert_broadcast_src_rank = _get_global_rank(
                groups.get_expert_data_parallel_group(),
                0)

        if not self.amp_enabled():
            self._broadcast_model()

    # check if parameters are duplicated in optimizer param_groups
    def _check_for_duplicates(self, optimizer):
        for name, param in self.module.named_parameters():
            param_id = id(param)

            def ids_list(group):
                return [id(param) for param in group]

            occurrence = sum([
                ids_list(group['params']).count(param_id)
                if param_id in ids_list(group['params']) else 0
                for group in optimizer.param_groups
            ])
            assert occurrence <= 1, f"Parameter with name: {name} occurs multiple times in optimizer.param_groups. Make sure it only appears once to prevent undefined behaviour."

    # Configure optimizer
    def _configure_optimizer(self, client_optimizer, model_parameters):
        if client_optimizer is not None:
            if isinstance(client_optimizer, Optimizer):
                client_optimizer.param_groups[:] = [
                    pg for pg in client_optimizer.param_groups if len(pg["params"]) != 0
                ]
                if self.global_rank == 0:
                    logger.info(
                        "Removing param_group that has no 'params' in the client Optimizer"
                    )

                basic_optimizer = client_optimizer
                if self.global_rank == 0:
                    logger.info('Using client Optimizer as basic optimizer')
            else:
                basic_optimizer = client_optimizer(model_parameters)
                if self.global_rank == 0:
                    logger.info('Using client callable to create basic optimizer')
        else:
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            if self.global_rank == 0:
                logger.info(
                    "Using DeepSpeed Optimizer param name {} as basic optimizer".format(
                        self.optimizer_name()))

        self._check_for_duplicates(basic_optimizer)

        self.basic_optimizer = basic_optimizer
        if self.global_rank == 0:
            logger.info("DeepSpeed Basic Optimizer = {}".format(
                basic_optimizer.__class__.__name__))

        if self.zero_optimization():
            assert (
                not self.amp_enabled()
            ), "Amp and ZeRO are not currently compatible, please use (legacy) fp16 mode which performs similar to amp opt_mode=O2"
            if not is_zero_supported_optimizer(basic_optimizer):
                assert (
                    self.zero_allow_untested_optimizer()
                ), 'You are using an untested ZeRO Optimizer. Please add <"zero_allow_untested_optimizer": true> in the configuration file to use it.'

                if self.global_rank == 0:
                    logger.warning(
                        "**** You are using ZeRO with an untested optimizer, proceed with caution *****"
                    )
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)
        elif self.amp_enabled():
            assert not (self.fp16_enabled() or self.bfloat16_enabled()), "Cannot enable both amp with (legacy) fp16 or bfloat16 mode"
            amp_params = self.amp_params()
            if self.global_rank == 0:
                logger.info(f"Initializing AMP with these params: {amp_params}")
            try:
                logger.info("Initializing Apex amp from: {}".format(amp.__path__))
            except NameError:
                # If apex/amp is available it will be imported above
                raise RuntimeError(
                    "Unable to import apex/amp, please make sure it is installed")
            self.module, self.optimizer = amp.initialize(
                self.module, basic_optimizer, **amp_params
            )
            self._broadcast_model()
            # TODO: maybe need to broadcast experts differently?
        elif self.fp16_enabled():
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer
        log_dist("DeepSpeed Final Optimizer = {}".format(self.optimizer_name()),
                 ranks=[0])

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

        if self.optimizer_name() in [ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER]:
            torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
            adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)

            # Optimizer name of Adam forces AdamW logic unless adam_w_mode is explicitly set
            effective_adam_w_mode = self.optimizer_name(
            ) == ADAMW_OPTIMIZER or adam_w_mode

            if torch_adam:
                if not effective_adam_w_mode:
                    optimizer = torch.optim.Adam(model_parameters,
                                                 **optimizer_parameters)
                else:
                    optimizer = torch.optim.AdamW(model_parameters,
                                                  **optimizer_parameters)
            else:
                if self.zero_cpu_offload():
                    if self.optimizer_name() == ADAGRAD_OPTIMIZER:
                        from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
                        optimizer = DeepSpeedCPUAdagrad(model_parameters,
                                                        **optimizer_parameters)
                    else:
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

        elif self.optimizer_name() == LAMB_OPTIMIZER:
            from deepspeed.ops.lamb import FusedLamb

            optimizer = FusedLamb(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.adam import OnebitAdam

            optimizer = OnebitAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(
                    f"Currently the convergence of 1-bit Adam is only verified under FP16"
                )
        elif self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Lamb is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb

            optimizer = OnebitLamb(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(
                    f"Currently the convergence of 1-bit Lamb is only verified under FP16"
                )
        else:
            torch_optimizer = getattr(torch.optim, self.optimizer_name())
            optimizer = torch_optimizer(model_parameters, **optimizer_parameters)
        return optimizer

    def _configure_quantization(self):
        (
            quantize_enabled,
            q_target_bits,
            q_start_bits,
            q_period,
            q_offset,
            q_groups,
            q_mixed_fp16,
            q_change_ratio,
            q_type,
            q_rounding,
            q_verbose,
            use_quantizer_kernel,
        ) = self.quantize_training()
        quantizer = None
        if quantize_enabled:
            from deepspeed.runtime.quantize import Quantizer

            quantizer = Quantizer(
                q_target_bits,
                q_start_bits,
                q_period,
                q_offset,
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
                or self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
            if self.dynamic_loss_scale():
                log_dist("Creating fp16 optimizer with dynamic loss scale", ranks=[0])
                timers = self.timers if self.wall_clock_breakdown() else None
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
                )
            else:
                log_dist(
                    "Creating fp16 optimizer with static loss scale: {}".format(
                        self.loss_scale()),
                    ranks=[0],
                )
                optimizer = FP16_Optimizer(
                    optimizer,
                    deepspeed=self,
                    static_loss_scale=self.loss_scale(),
                    mpu=self.mpu,
                    clip_grad=clip_grad,
                    fused_adam_legacy=self.optimizer_legacy_fusion(),
                )
        else:
            log_dist("Creating fp16 unfused optimizer with dynamic loss scale",
                     ranks=[0])
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

    def _configure_zero_optimizer(self, optimizer):
        zero_stage = self.zero_optimization_stage()
        log_dist('Creating fp16 ZeRO stage {} optimizer'.format(zero_stage), ranks=[0])
        assert self.communication_data_type in (torch.float16, torch.bfloat16), "ZeRO supports only 'communication_data_type': ['fp16', 'bfp16']"
        timers = self.timers if self.wall_clock_breakdown() else None

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        if self.zero_legacy_stage1():
            raise Exception(
                "The deprecated version of ZeRO Stage 1 is not supported in deepspeed >= 0.5.9. Please downgrade to a version less than 0.5.9 if you need to use this deprecated version of ZeRO."
            )

        if zero_stage <= ZERO_OPTIMIZATION_GRADIENTS:
            overlap_comm = self.zero_overlap_comm()
            contiguous_gradients = self.zero_contiguous_gradients()
            round_robin_gradients = self.zero_round_robin_gradients()
            assert not isinstance(optimizer, DummyOptim), "zero stage 2 requires an optimizer"

            # Overlap and contiguous grads are meaningless in stage 1 and are ignored
            if zero_stage == ZERO_OPTIMIZATION_OPTIMIZER_STATES:
                overlap_comm = False
                contiguous_gradients = False
                round_robin_gradients = False

            if isinstance(self.module, PipelineModule):
                if overlap_comm:
                    logger.warning(
                        "Pipeline parallelism does not support overlapped communication, will be disabled."
                    )
                    overlap_comm = False

            optimizer = DeepSpeedZeroOptimizer(
                optimizer,
                timers=timers,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                contiguous_gradients=contiguous_gradients,
                reduce_bucket_size=self.zero_reduce_bucket_size(),
                allgather_bucket_size=self.zero_allgather_bucket_size(),
                dp_process_group=self.data_parallel_group,
                expert_parallel_group=self.expert_parallel_group
                if self.has_moe_layers else None,
                expert_data_parallel_group=self.expert_data_parallel_group
                if self.has_moe_layers else None,
                reduce_scatter=self.zero_reduce_scatter(),
                overlap_comm=overlap_comm,
                cpu_offload=self.zero_cpu_offload(),
                mpu=self.mpu,
                postscale_gradients=self.postscale_gradients(),
                gradient_predivide_factor=self.gradient_predivide_factor(),
                gradient_accumulation_steps=self.gradient_accumulation_steps(),
                ignore_unused_parameters=self.zero_ignore_unused_parameters(),
                partition_grads=zero_stage == ZERO_OPTIMIZATION_GRADIENTS,
                round_robin_gradients=round_robin_gradients,
                has_moe_layers=self.has_moe_layers,
                fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(
                ),
                communication_data_type=self.communication_data_type,
                elastic_checkpoint=self.zero_elastic_checkpoint())

        elif zero_stage == ZERO_OPTIMIZATION_WEIGHTS:
            assert not self.has_moe_layers, "MoE not supported with Stage 3"
            logger.info("Initializing ZeRO Stage 3") if dist.get_rank() == 0 else None
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
                dp_process_group=self.data_parallel_group,
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
                communication_data_type=self.communication_data_type)

        else:
            raise NotImplementedError("ZeRO stage {} not implemented".format(zero_stage))

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

    def _configure_curriculum_scheduler(self):
        scheduler = CurriculumScheduler(self.curriculum_params())
        return scheduler

    @staticmethod
    def is_map_style_dataset(obj):
        return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")

    @staticmethod
    def is_iterable_style_dataset(obj):
        return isinstance(obj,
                          torch.utils.data.IterableDataset
                          )  # hasattr(obj, "__iter__") should work as well

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
        if not (self.is_map_style_dataset(dataset)
                or self.is_iterable_style_dataset(dataset)):
            raise ValueError("Training data must be a torch Dataset")

        if data_sampler is None and (route == ROUTE_PREDICT or route == ROUTE_EVAL):
            data_sampler = torch.utils.data.SequentialSampler(dataset)

        if batch_size is None:
            batch_size = self.train_micro_batch_size_per_gpu()

        if collate_fn is None:
            collate_fn = self.collate_fn

        # Currently we only use timer in train route
        deepspeed_io_timer = None
        if route == ROUTE_TRAIN:
            deepspeed_io_timer = self.tput_timer

        # If mpu is provided, forward world size and parallel rank to sampler.
        data_parallel_world_size = None
        data_parallel_rank = None
        if self.mpu is not None:
            data_parallel_world_size = self.mpu.get_data_parallel_world_size()
            data_parallel_rank = self.mpu.get_data_parallel_rank()

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
                                   dataloader_drop_last=self.dataloader_drop_last())

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
                logger.warning(
                    f"DeepSpeed unable to scale loss because of type: {type(prescaled_loss)}"
                )
                self.warn_unscaled_loss = False

        return scaled_loss

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

        flops_profiler_active = (self.flops_profiler_enabled() and self.global_steps
                                 == self.flops_profiler_profile_step()
                                 and self.global_rank == 0)

        if flops_profiler_active:
            self.flops_profiler.start_profile(ignore_list=None)

        if self.module.training and self.progressive_layer_drop:
            kwargs.update(self.progressive_layer_drop.get_state())

        if self.__class__.__name__ != "PipelineEngine":
            # TODO: The above if condition is a HACK since for PipelineEngine
            # it's difficult to inject argument in forward pass.
            if self.module.training and self.curriculum_enabled():
                self.curriculum_scheduler.update_difficulty(self.global_steps + 1)
                if self.curriculum_params()["curriculum_type"] == "seqlen":
                    kwargs.update({
                        "curriculum_seqlen":
                        self.curriculum_scheduler.get_current_difficulty()
                    })

        if self.zero_optimization_partition_weights():
            # Enable automated discovery of external parameters by indicating that
            # we are in a forward pass.
            for module in self.module.modules():
                module._parameters._in_forward = True
                pass

        self._start_timers(self.engine_timers.forward_timers)

        if self.training_dataloader is None:
            self.tput_timer.start()

        loss = self.module(*inputs, **kwargs)

        if self.zero_optimization_partition_weights():
            # Reset the ZeRO-3 state if we are only doing forward-passes (ie evaluation).
            if not torch._C.is_grad_enabled():
                self.optimizer.param_coordinator.reset_step()

            # Disable automated discovery of external parameters
            for module in self.module.modules():
                module._parameters._in_forward = False

        self._stop_timers(self.engine_timers.forward_timers)

        if flops_profiler_active:
            self.flops_profiler.stop_profile()

        if self.autotuning_profile_model_info():
            activation_mem = get_ma_status() - ma
            self.autotuning_model_info["activation_mem_per_gpu"] = activation_mem
            print_json_dist(self.autotuning_model_info,
                            [0],
                            path=self.autotuning_model_info_path())
            exit()
        else:
            see_memory_usage("Engine after forward", force=self.memory_breakdown())
        return loss

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

        # if torch.distributed.get_rank() == 0:
        log_dist(
            f"rank={torch.distributed.get_rank()} time (ms) | forward: {fwd_time:.2f} (forward_moe: {moe_time:.2f}, 1st alltoall: {falltoall:.2f}, 2nd alltoall: {salltoall:.2f}, top-k: {gate_time:.2f})",
            ranks=[0])

    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        # Pass (PP) gas boundary flag to optimizer (required for zero)
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary(
        )

        # ZeRO stage 2 communicates during non gradient accumulation boundaries as well
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        # Communicate only at gradient accumulation boundaries
        elif self.is_gradient_accumulation_boundary():
            if self.zero_optimization_stage() == ZERO_OPTIMIZATION_OPTIMIZER_STATES:
                self.optimizer.reduce_gradients(
                    pipeline_parallel=self.pipeline_parallelism)
            else:
                self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    def backward(self, loss, allreduce_gradients=True, release_loss=False):
        r"""Execute backward pass on the loss

        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: is deprecated, ignored, and will soon be removed'
        """

        see_memory_usage("Engine before backward", force=self.memory_breakdown())

        if not allreduce_gradients:
            logger.warning(
                f"Argument `allreduce_gradients` is deprecated, ignored, and will soon be removed"
            )

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps() > 1:
            loss = self._scale_loss_by_gas(loss.float())

        # Log training Loss
        if self.tensorboard_enabled():
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [(
                        f"Train/Samples/train_loss",
                        loss.mean().item() * self.gradient_accumulation_steps(),
                        self.global_samples,
                    )]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                    self.summary_writer.flush()

        self._start_timers(self.engine_timers.backward_timers)

        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            "must provide optimizer during init in order to use backward"

        self._start_timers(self.engine_timers.backward_inner_timers)

        if self.zero_optimization():
            self.optimizer.is_gradient_accumulation_boundary = (
                self.is_gradient_accumulation_boundary())
            self.optimizer.backward(loss)
        elif self.amp_enabled():
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not self.is_gradient_accumulation_boundary()
            with amp.scale_loss(loss,
                                self.optimizer,
                                delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward()
        elif self.fp16_enabled():
            if self.eigenvalue_enabled():
                self.optimizer.backward(loss, create_graph=True, retain_graph=True)
            else:
                self.optimizer.backward(loss)
        else:
            if self.eigenvalue_enabled():
                loss.backward(create_graph=True, retain_graph=True)
            else:
                loss.backward()

        self._stop_timers(self.engine_timers.backward_inner_timers)

        self._start_timers(self.engine_timers.backward_reduce_timers)

        if self.enable_backward_allreduce:
            self.allreduce_gradients()

        self._stop_timers(self.engine_timers.backward_reduce_timers)

        self._stop_timers(self.engine_timers.backward_timers)

        if release_loss:
            # loss.data = None
            pass

        see_memory_usage("Engine after backward", force=self.memory_breakdown())

        return loss

    def is_gradient_accumulation_boundary(self):
        """Query whether the current micro-batch is at the boundary of
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
        """Manually overrides the DeepSpeed engine's gradient accumulation boundary state, this is an optional
        feature and should be used with care. The state should be set before to the intended
        value before each forward/backward. The final fordward/backward should have the
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
        clip_grad_norm_(parameters=self.module.parameters(),
                        max_norm=self.gradient_clipping(),
                        mpu=self.mpu)

    def _take_model_step(self, lr_kwargs, block_eigenvalue={}):
        if self.gradient_clipping() > 0.0:
            if not (self.fp16_enabled() or self.amp_enabled()
                    or self.zero_optimization()):
                self.clip_fp32_gradients()
            elif self.amp_enabled():
                # AMP's recommended way of doing clipping
                # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                master_params = amp.master_params(self.optimizer)
                clip_grad_norm_(parameters=master_params,
                                max_norm=self.gradient_clipping(),
                                mpu=self.mpu)
        self.optimizer.step()

        if hasattr(self.optimizer, '_global_grad_norm'):
            self._global_grad_norm = self.optimizer._global_grad_norm

        # Quantize the updated parameter if there is no overflow
        if self.quantizer:
            if self.fp16_enabled():
                tensor_to_quantize = self.optimizer.bit16_groups if self.zero_optimization_stage(
                ) == 2 else self.optimizer.fp16_groups
            else:
                tensor_to_quantize = self.optimizer.param_groups
            self.quantizer.quantize(
                tensor_to_quantize,
                (self.optimizer.overflow if self.fp16_enabled() else False),
                self.eigenvalue_enabled(),
                block_eigenvalue,
            )
        # zero grad in basic optimizer could be unreliable and may not exhibit
        # the behaviour that we want
        if (not self.zero_optimization() and not self.fp16_enabled()
                and not self.amp_enabled()):
            self.zero_grad()
        else:
            self.optimizer.zero_grad()

        report_progress = self.global_rank == 0 if self.global_rank else True

        # Check overflow here since in DS fp16 optimizer, the overflow is updated in above step() function.
        overflow = False
        if hasattr(self.optimizer, "overflow"):
            overflow = self.optimizer.overflow
        self._step_applied = not overflow

        if overflow:
            self.skipped_steps += 1
        else:
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step(**(lr_kwargs or {}))
                except TypeError:
                    # XXX Hack to work with Megatron 2.0 and DeepSpeed pipelines.
                    # We don't currently have a way to specify lr_kwargs from
                    # pipe_engine.train_batch()
                    self.lr_scheduler.step(increment=self.train_batch_size())

        if report_progress and (self.global_steps + 1) % self.steps_per_print() == 0:
            self._report_progress(self.global_steps + 1)

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
        ) and self.global_steps == self.flops_profiler_profile_step(
        ) and self.global_rank == 0

        self._start_timers(self.engine_timers.step_timers)

        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            "must provide optimizer during init in order to use step"

        report_progress = self.global_rank == 0 if self.global_rank else True

        self._step_applied = False  # assume False, will flip to True

        # Update the model when we reach gradient accumulation boundaries
        if self.is_gradient_accumulation_boundary():
            self.gas_boundary_ctr += 1

            if (self.eigenvalue_enabled() and
                (self.gas_boundary_ctr % self.eigenvalue_gas_boundary_resolution() == 0)
                    and self.quantizer.any_precision_switch()):
                log_dist(f"computing eigenvalue...", ranks=[0])
                self.block_eigenvalue = self.eigenvalue.compute_eigenvalue(
                    self.module,
                    self.device,
                    self.optimizer.cur_scale)

            if self.progressive_layer_drop:
                self.progressive_layer_drop.update_state(self.global_steps)

            if (self.eigenvalue_enabled() and not self.gas_boundary_ctr %
                    self.eigenvalue_gas_boundary_resolution()
                    and self.quantizer.any_precision_switch()):
                self._take_model_step(lr_kwargs, self.block_eigenvalue)
            else:
                self._take_model_step(lr_kwargs)

        self.tput_timer.stop(report_progress)

        self._stop_timers(self.engine_timers.step_timers)

        # Log learning rate
        if self.tensorboard_enabled():
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [(f"Train/Samples/lr",
                                            self.get_lr()[0],
                                            self.global_samples)]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                    if self.fp16_enabled() and hasattr(self.optimizer, "cur_scale"):
                        self.summary_events.append((
                            f"Train/Samples/loss_scale",
                            self.optimizer.cur_scale,
                            self.global_samples,
                        ))

                    if (self.eigenvalue_enabled() and not self.gas_boundary_ctr %
                            self.eigenvalue_gas_boundary_resolution()):
                        ev_values = self.block_eigenvalue.values()
                        for i in range(len(ev_values)):
                            self.summary_writer.add_scalar(
                                f"Train/Eigenvalues/ModelBlockParam_{i}",
                                self.ev_values[i][0],
                                self.global_samples,
                            )
                            self.summary_writer.flush()

                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                    self.summary_writer.flush()

        # Check flops profiling
        if flops_profiler_active:
            if self.autotuning_enabled():
                self.flops = self.flops_profiler.get_total_flops() * 3
            else:
                self.flops_profiler.print_model_profile(
                    profile_step=self.global_steps,
                    module_depth=self.flops_profiler_module_depth(),
                    top_modules=self.flops_profiler_top_modules(),
                    detailed=self.flops_profiler_detailed(),
                    output_file=self.flops_profiler_output_file(),
                )
            self.flops_profiler.end_profile()

        if self.autotuning_enabled() and self.global_steps == (
                self.autotuning_end_profile_step() + 1):
            self._autotuning_exit()

        if self.wall_clock_breakdown():
            # Log micro timing and reset
            self.timers.log(names=self.engine_timers.micro_timers,
                            memory_breakdown=self.memory_breakdown())

        if self.wall_clock_breakdown() or self.flops_profiler_enabled():
            # Log global timing and reset
            if self.is_gradient_accumulation_boundary():
                if self.tensorboard_enabled():
                    self._write_tensorboard()

                if self.has_moe_layers:
                    fwd_time = self.timers(FORWARD_GLOBAL_TIMER).elapsed(
                        reset=False) * 1000
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
            ],
                                       reset=False)
            titer = msg[FORWARD_GLOBAL_TIMER] + msg[BACKWARD_GLOBAL_TIMER] + msg[
                STEP_GLOBAL_TIMER]
            msg["latency"] = titer
            msg["FLOPS_per_gpu"] = self.flops * self.gradient_accumulation_steps(
            ) / titer
            msg["throughput"] = self.train_batch_size() * 1000 / \
                msg["latency"]
            print_json_dist(msg, [0], path=self.autotuning_metric_path())
            import atexit
            atexit.register(print, "Autotuning: done with runing current ds config.")
        exit()

    def _write_tensorboard(self):
        if self.global_rank == 0:
            self.summary_events = [
                (
                    f"Train/Samples/elapsed_time_ms_forward",
                    self.timers(FORWARD_GLOBAL_TIMER).elapsed(reset=False) * 1000.0,
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward",
                    self.timers(BACKWARD_GLOBAL_TIMER).elapsed(reset=False) * 1000.0,
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward_inner",
                    self.timers(BACKWARD_INNER_GLOBAL_TIMER).elapsed(reset=False) *
                    1000.0,
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_backward_allreduce",
                    self.timers(BACKWARD_REDUCE_GLOBAL_TIMER).elapsed(reset=False) *
                    1000.0,
                    self.global_samples,
                ),
                (
                    f"Train/Samples/elapsed_time_ms_step",
                    self.timers(STEP_GLOBAL_TIMER).elapsed(reset=False) * 1000.0,
                    self.global_samples,
                ),
            ]
            for event in self.summary_events:  # write_summary_events
                self.summary_writer.add_scalar(event[0], event[1], event[2])
            self.summary_writer.flush()

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
        log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr}, mom={mom}",
                 ranks=[0])

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
                if self.gradient_predivide_factor() != dist.get_world_size(
                        group=dp_group):
                    tensor_to_allreduce.mul_(self.gradient_predivide_factor() /
                                             dist.get_world_size(group=dp_group))
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

    def buffered_allreduce_fallback(self, grads=None, elements_per_buffer=500000000):
        grads, expert_grads = [], []
        for param_name, param in self.module.named_parameters():
            if hasattr(param, 'allreduce') and not param.allreduce:
                is_moe_param = True
            else:
                is_moe_param = False
            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(param.size(),
                                         dtype=param.dtype,
                                         device=param.device)
                if is_moe_param:
                    expert_grads.append(param.grad.data)
                else:
                    grads.append(param.grad.data)
            else:
                grad_data = param.grad.data
                if param_name in self.sparse_tensor_module_names or grad_data.is_sparse:
                    if is_moe_param:
                        expert_grads.append(SparseTensor(grad_data))
                    else:
                        grads.append(SparseTensor(grad_data))
                else:
                    if is_moe_param:
                        expert_grads.append(grad_data)
                    else:
                        grads.append(grad_data)

        split_buckets = split_half_float_double_sparse(grads)
        for _, bucket_tuple in enumerate(split_buckets):
            bucket_type, bucket = bucket_tuple

            if self.pipeline_parallelism:
                dp_group = self.mpu.get_data_parallel_group()
            else:
                dp_group = groups.get_data_parallel_group()

            if bucket_type == SparseTensor.type():
                self.sparse_allreduce_no_retain(bucket, dp_group=dp_group)
            else:
                self.allreduce_no_retain(bucket,
                                         dp_group=dp_group,
                                         numel_per_bucket=elements_per_buffer)

        if self.has_moe_layers:
            expert_split_buckets = split_half_float_double_sparse(expert_grads)
            for i, bucket_tuple in enumerate(expert_split_buckets):
                bucket_type, bucket = bucket_tuple
                if bucket_type == SparseTensor.type():
                    self.sparse_allreduce_no_retain(
                        bucket,
                        groups.get_expert_data_parallel_group())
                else:
                    # Separate between diff groups
                    self.allreduce_no_retain(
                        bucket,
                        dp_group=groups.get_expert_data_parallel_group(),
                        numel_per_bucket=elements_per_buffer)

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
        # Pre-divide for fp16 stability
        sparse.values.mul_(1.0 / dist.get_world_size(group=dp_group))

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
            tensor_list = [
                value.new_empty(max_size)
                for _ in range(dist.get_world_size(group=dp_group))
            ]
        else:
            if fill_size > 0:
                value = torch.cat([value, value.new_empty(fill_size, value.size()[1])])
            tensor_list = [
                value.new_empty(max_size,
                                value.size()[1])
                for _ in range(dist.get_world_size(group=dp_group))
            ]

        dist.all_gather(tensor_list, value, group=dp_group)
        tensors = []
        for dev_idx, t in enumerate(tensor_list):
            size = all_sizes[dev_idx][0]
            tensors.append(
                t.index_select(0,
                               torch.arange(size,
                                            dtype=torch.long,
                                            device=self.device)))

        return tensors

    def all_gather_scalar(self, value, dp_group):
        tensor_list = [
            value.new_zeros(value.size())
            for _ in range(dist.get_world_size(group=dp_group))
        ]
        dist.all_gather(tensor_list, value, group=dp_group)
        return tensor_list

    def module_state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = self.module.state_dict(destination, prefix, keep_vars)
        return sd

    def load_moe_state_dict(self, checkpoint_path, tag, state_dict):
        expp_rank = groups.get_expert_parallel_rank()

        num_local_experts = self.num_experts // self.ep_world_size
        for local_expert_id in range(num_local_experts):
            global_expert_id = expp_rank * num_local_experts + local_expert_id
            expert_state_dict = torch.load(self._get_expert_ckpt_name(
                checkpoint_path,
                global_expert_id,
                tag),
                                           map_location=torch.device('cpu'))

            # Updating global -> local expert ids
            moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
            for key in list(expert_state_dict.keys()):
                local_key = key.replace(f'{moe_str_prefix}{global_expert_id}',
                                        f'{moe_str_prefix}{local_expert_id}')
                expert_state_dict[local_key] = expert_state_dict.pop(key)
            state_dict.update(expert_state_dict)

    def load_module_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

    def _get_rank_zero_ckpt_name(self, checkpoints_path, tag, mp_rank, dp_rank):
        filename = "zero_pp_rank_{}".format(dp_rank)
        zero_ckpt_name = os.path.join(
            checkpoints_path,
            str(tag),
            filename + "_mp_rank_{:02d}".format(mp_rank) + "_optim_states.pt",
        )
        return zero_ckpt_name

    def _get_zero_ckpt_name(self, checkpoints_path, tag):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        pp_rank = torch.distributed.get_rank(group=self.optimizer.dp_process_group)
        return self._get_rank_zero_ckpt_name(checkpoints_path, tag, mp_rank, pp_rank)

    def _get_ckpt_name(self, checkpoints_path, tag, mp_placeholder=None):
        if mp_placeholder is not None:
            mp_rank_str = mp_placeholder
        else:
            mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
            mp_rank_str = "{:02d}".format(mp_rank)

        if self.zero_optimization_partition_weights():
            filename = "zero_pp_rank_{}".format(
                torch.distributed.get_rank(group=self.optimizer.dp_process_group))
            ckpt_name = os.path.join(
                checkpoints_path,
                str(tag),
                filename + "_mp_rank_" + mp_rank_str + "_model_states.pt",
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
        ckpt_name = os.path.join(
            checkpoints_path,
            str(tag),
            f'expp_rank_{expp_rank}_mp_rank_{mp_rank:02d}_optim_states.pt')
        return ckpt_name

    def _get_expert_ckpt_name(self, checkpoints_path, expert_id, tag):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        ckpt_name = os.path.join(
            checkpoints_path,
            str(tag),
            f'expert_{expert_id}_mp_rank_{mp_rank:02d}_model_states.pt')
        return ckpt_name

    def _get_all_ckpt_names(self, checkpoints_path, tag):
        # It is required that (checkpoints_path, tag) are consistent among all ranks.
        ckpt_file_pattern = self._get_ckpt_name(checkpoints_path,
                                                tag,
                                                mp_placeholder="*")
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
                        load_module_only=False):
        """Load training checkpoint

        Arguments:
            load_dir: Required. Directory to load the checkpoint from
            tag: Checkpoint tag used as a unique identifier for checkpoint, if not provided will attempt to load tag in 'latest' file
            load_module_strict: Optional. Boolean to strictly enforce that the keys in state_dict of module and checkpoint match.
            load_optimizer_states: Optional. Boolean to load the training optimizer states from Checkpoint. Ex. ADAM's momentum and variance
            load_lr_scheduler_states: Optional. Boolean to add the learning rate scheduler states from Checkpoint.
            load_module_only: Optional. Boolean to load only the model weights from the checkpoint. Ex. warmstarting.
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
            latest_path = os.path.join(load_dir, "latest")
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    tag = fd.read().strip()
            else:
                logger.warning(
                    f"Unable to find latest file at {latest_path}, if trying to load latest "
                    "checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint."
                )
                return None, None

        load_path, client_states = self._load_checkpoint(load_dir,
                                                         tag,
                                                         load_module_strict=load_module_strict,
                                                         load_optimizer_states=load_optimizer_states,
                                                         load_lr_scheduler_states=load_lr_scheduler_states,
                                                         load_module_only=load_module_only)

        if self.zero_optimization() and load_path is not None:
            success = self._load_zero_checkpoint(
                load_dir,
                tag,
                load_optimizer_states=load_optimizer_states)
            if not success:
                self.optimizer._restore_from_fp16_weights()

        return load_path, client_states

    def _load_checkpoint(self,
                         load_dir,
                         tag,
                         load_module_strict=True,
                         load_optimizer_states=True,
                         load_lr_scheduler_states=True,
                         load_module_only=False):

        from deepspeed.runtime.state_dict_factory import SDLoaderFactory

        ckpt_list = self._get_all_ckpt_names(load_dir, tag)
        sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list)

        is_pipe_parallel = isinstance(self.module, PipelineModule)

        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        load_path, checkpoint, _ = sd_loader.load(
            self.mp_world_size, mp_rank, is_pipe_parallel=is_pipe_parallel
        )

        if checkpoint is None:
            return None, None

        if is_pipe_parallel:
            # Pipeline parallelism uses this to load its own checkpoint files.
            self._curr_ckpt_path = os.path.join(load_dir, tag)

        if self.has_moe_layers:
            self.load_moe_state_dict(load_dir, tag, state_dict=checkpoint['module'])

        self.load_module_state_dict(state_dict=checkpoint['module'],
                                    strict=load_module_strict)

        self.loaded_checkpoint_dp_world_size = checkpoint['dp_world_size']

        if load_module_only:
            deepspeed_states = ['module']
            if self.optimizer is not None and self.fp16_enabled():
                self.optimizer.refresh_fp32_params()
        else:
            if self.has_moe_layers:
                expp_rank = groups.get_expert_parallel_rank()
                optim_load_path = self._get_optimizer_ckpt_name(load_dir, tag, expp_rank)
                optim_checkpoint = torch.load(optim_load_path,
                                              map_location=torch.device('cpu'))
            else:
                optim_checkpoint = checkpoint

            if load_optimizer_states and self.optimizer is not None and not self.zero_optimization(
            ):
                if self.fp16_enabled():
                    self.optimizer.load_state_dict(
                        optim_checkpoint['optimizer'],
                        load_optimizer_states=load_optimizer_states)
                else:
                    self.optimizer.load_state_dict(optim_checkpoint['optimizer'])

            if load_lr_scheduler_states and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            def get_sparse_tensor_module_names(original_set,
                                               loaded_set,
                                               original_parameters,
                                               loaded_parameters):
                result = set()

                for name in original_set:
                    if name in loaded_parameters and name not in loaded_set:
                        continue  # parameter existed in previous model and was not sparse
                    result.add(name)

                for name in loaded_set:
                    if name in original_parameters:
                        result.add(
                            name)  # parameter exists in both configs and it was sparse

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
                        self.sparse_tensor_module_names,
                        sparse_tensor_module_names,
                        dict(self.module.named_parameters()),
                        checkpoint["module"])

            self.global_steps = checkpoint['global_steps']
            self.global_samples = checkpoint.get(
                'global_samples',
                self.global_steps * self.train_batch_size())
            self.skipped_steps = checkpoint['skipped_steps']
            self.loaded_checkpoint_mp_world_size = checkpoint['mp_world_size']
            deepspeed_states = [
                'module',
                'sparse_tensor_module_names',
                'skipped_steps',
                'global_steps',
                'dp_world_size',
                'mp_world_size'
            ]
        client_state = {}

        if load_lr_scheduler_states:
            deepspeed_states.append('lr_scheduler')
        if load_optimizer_states:
            deepspeed_states.append('optimizer')

        client_state = {
            key: value
            for key,
            value in checkpoint.items() if not key in deepspeed_states
        }

        if not load_optimizer_states and not load_module_only:
            client_state['optimizer'] = optim_checkpoint['optimizer']

        return load_path, client_state

    def _load_zero_checkpoint(self, load_dir, tag, load_optimizer_states=True):
        zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
        if zero_sd_list is None:
            return False

        if load_optimizer_states and self.dp_world_size != self.loaded_checkpoint_dp_world_size:
            raise ZeRORuntimeException("The checkpoint being loaded used a DP " \
                f"world size of {self.loaded_checkpoint_dp_world_size} but the " \
                f"current world size is {self.dp_world_size}. Automatic adjustment " \
                "of ZeRO's optimizer state partitioning with a new world size is not " \
                "currently supported.")

        self.optimizer.load_state_dict(
            state_dict_list=zero_sd_list,
            load_optimizer_states=load_optimizer_states,
            load_from_fp32_weights=self.zero_load_from_fp32_weights(),
        )
        logger.info(
            f"loading {len(zero_sd_list)} zero partition checkpoints for rank {self.global_rank}"
        )
        return True

    def _get_mp_rank_zero_checkpoint_names(self, load_dir, tag, mp_rank, dp_world_size):
        zero_ckpt_names = []
        for dp_rank in range(dp_world_size):
            ckpt_name = self._get_rank_zero_ckpt_name(checkpoints_path=load_dir,
                                                      tag=tag,
                                                      mp_rank=mp_rank,
                                                      dp_rank=dp_rank)
            zero_ckpt_names.append(ckpt_name)

        return zero_ckpt_names

    def _get_all_zero_checkpoint_names(self,
                                       load_dir,
                                       tag,
                                       mp_world_size,
                                       dp_world_size):
        zero_ckpt_names = []
        for mp_rank in range(mp_world_size):
            mp_rank_ckpt_names = self._get_mp_rank_zero_checkpoint_names(
                load_dir=load_dir,
                tag=tag,
                mp_rank=mp_rank,
                dp_world_size=dp_world_size)
            zero_ckpt_names += mp_rank_ckpt_names

        return zero_ckpt_names

    def _get_all_zero_checkpoints(self, load_dir, tag):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        zero_ckpt_names = self._get_mp_rank_zero_checkpoint_names(
            load_dir=load_dir,
            tag=tag,
            mp_rank=mp_rank,
            dp_world_size=self.loaded_checkpoint_dp_world_size,
        )
        invalid_zero_ckpt_paths = []
        for i, ckpt_name in enumerate(zero_ckpt_names):
            if not os.path.exists(ckpt_name):
                # transparently handle the old file pattern for optim_states
                if "optim_states.pt" in ckpt_name:
                    ckpt_name_try = ckpt_name.replace("_optim_states.pt",
                                                      "optim_states.pt")
                    if os.path.exists(ckpt_name_try):
                        zero_ckpt_names[i] = ckpt_name_try
                        continue
                invalid_zero_ckpt_paths.append(ckpt_name)

        if len(invalid_zero_ckpt_paths) > 0:
            logger.warn(
                f"The following zero checkpoints paths are missing: {invalid_zero_ckpt_paths}"
            )
            return None

        zero_sd_list = []
        for i, ckpt_name in enumerate(zero_ckpt_names):
            _state = None
            # Fully load state for current rank
            if self.zero_elastic_checkpoint() or dist.get_rank(
                    group=self.optimizer.dp_process_group) == i:
                _state = torch.load(ckpt_name, map_location='cpu')
            else:
                _state = {OPTIMIZER_STATE_DICT: None}
            zero_sd_list.append(_state)

        zero_optimizer_sd = [sd[OPTIMIZER_STATE_DICT] for sd in zero_sd_list]
        logger.info(
            f"successfully read {len(zero_optimizer_sd)} ZeRO state_dicts for rank {self.global_rank}"
        )
        return zero_optimizer_sd

    def _checkpoint_tag_validation(self, tag):
        if self.checkpoint_tag_validation_enabled():
            s_hash = hashlib.sha1(tag.encode())
            bhash = torch.ByteTensor([s_hash.digest()]).flatten().to(self.device)
            max_bhash = bhash.clone()
            min_bhash = bhash.clone()
            dist.all_reduce(max_bhash, op=torch.distributed.ReduceOp.MAX)
            dist.all_reduce(min_bhash, op=torch.distributed.ReduceOp.MIN)
            valid = all(min_bhash == bhash) and all(max_bhash == bhash)
            msg = (
                f"[rank={dist.get_rank()}] The checkpoint tag name '{tag}' is not consistent across "
                "all ranks. Including rank unique information in checkpoint tag could cause issues when "
                "restoring with different world sizes.")
            if self.checkpoint_tag_validation_fail():
                assert valid, msg
            elif not valid:
                logger.warning(msg)

    def save_checkpoint(self, save_dir, tag=None, client_state={}, save_latest=True):
        r"""Save training checkpoint

        Arguments:
            save_dir: Required. Directory for saving the checkpoint
            tag: Optional. Checkpoint tag used as a unique identifier for the checkpoint, global step is
                used if not provided. Tag name must be the same across all ranks.
            client_state: Optional. State dictionary used for saving required training states in the client code.
            save_latest: Optional. Save a file 'latest' pointing to the latest saved checkpoint.

        Important: all processes must call this method and not just the process with rank 0. It is
        because each process needs to save its master weights and scheduler+optimizer states. This
        method will hang waiting to synchronize with other processes if it's called just for the
        process with rank 0.
        """
        if self.zero_optimization_partition_weights():
            # Prepare for state_dict() by ensuring all parameters are partitioned
            self.optimizer.save_checkpoint_prologue()

        # This is to make sure the checkpoint names are created without collision
        # There seems to be issue creating them in parallel

        # Ensure save_dir directory exists
        os.makedirs(save_dir, exist_ok=True)
        torch.distributed.barrier()

        if tag is None:
            tag = f"global_step{self.global_steps}"

        # Ensure tag is a string
        tag = str(tag)

        # Ensure checkpoint tag is consistent across ranks
        self._checkpoint_tag_validation(tag)

        if self.has_moe_layers:
            self.save_non_zero_checkpoint = False
            self._create_checkpoint_file(save_dir, tag, False)
            self._save_moe_checkpoint(save_dir, tag, client_state=client_state)

        if self.save_non_zero_checkpoint:
            self._create_checkpoint_file(save_dir, tag, False)
            self._save_checkpoint(save_dir, tag, client_state=client_state)

        if self.save_zero_checkpoint:
            self._create_zero_checkpoint_files(save_dir, tag)
            self._save_zero_checkpoint(save_dir, tag)

        if self.zero_optimization_partition_weights():
            self.optimizer.save_checkpoint_epilogue()

        # Save latest checkpoint tag
        torch.distributed.barrier()
        if save_latest and self.global_rank == 0:
            with open(os.path.join(save_dir, 'latest'), 'w') as fd:
                fd.write(tag)

        return True

    def _get_moe_state_dict(self, full_state_dict, num_local_experts, expp_rank):
        """Compute moe and non moe state dict from complete local model state dict
            key : global_expert_id
            value : state_dict
            experts_state_dict =
            {
                '0': {
                    'models.seq2seq.encoder.layers.0.experts.moe.experts.experts.0.fc1.weight' <class 'torch.Tensor'>,
                    'models.seq2seq.encoder.layers.1.experts.moe.experts.experts.0.fc1.weight' <class 'torch.Tensor'>,
                    'models.seq2seq.encoder.layers.2.experts.moe.experts.experts.0.fc1.weight' <class 'torch.Tensor'>,
                    ...
                },
                '1' : {
                    ...
                }
            }

            returns experts_state_dict, model_state_dict
        """
        experts_state_dict, moe_state_dict = defaultdict(dict), {}
        for key in list(full_state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = full_state_dict.pop(key)
        non_moe_state_dict = full_state_dict

        moe_str_prefix = '.deepspeed_moe.experts.deepspeed_experts.'
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
            experts_state_dict[str(global_expert_id)][expert_key] = moe_state_dict.pop(
                key)

        return experts_state_dict, non_moe_state_dict

    def _save_moe_checkpoint(self, save_dir, tag, client_state={}):

        save_path = self._get_ckpt_name(save_dir, tag)
        # A hack to save the checkpointing directory. Pipeline parallelism overrides
        # module_state_dict() and uses this path to save the model. module_state_dict()
        # then instead just returns None.
        self._curr_ckpt_path = os.path.join(save_dir, tag)
        """"
        experts_state_dict = {
            'e_id' : state_dict_for_eid
        }
        """
        expp_rank = groups.get_expert_parallel_rank()
        exp_dp_rank = groups.get_expert_data_parallel_rank()

        # In the case of E + D parallelism, only the
        # first expert parallel group should save the expert weights
        # since each expert parallel group is a copy of the model's experts
        if exp_dp_rank != 0:
            return

        num_local_experts = self.num_experts // self.ep_world_size
        experts_state_dict, model_state_dict = self._get_moe_state_dict(
            self.module_state_dict(), num_local_experts, expp_rank)

        #  Each rank saves its local experts
        for global_expert_id, expert_state_dict in experts_state_dict.items():
            expert_save_dir = self._get_expert_ckpt_name(save_dir, global_expert_id, tag)
            logger.info(
                f'Saving model expert {global_expert_id} checkpoint: {expert_save_dir}')
            torch.save(expert_state_dict, expert_save_dir)

        # Save optimizer states. They are different across each exp parallel rank.
        optimizer_state = {
            'optimizer':
            self.optimizer.state_dict()
            if self.optimizer and not self.zero_optimization() else None
        }
        torch.save(optimizer_state,
                   self._get_optimizer_ckpt_name(save_dir,
                                                 tag,
                                                 expp_rank))

        if expp_rank == 0:
            # TODO: update num experts info,.. in checkpoint
            state = {
                'module':
                model_state_dict,
                'lr_scheduler':
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None else None,
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
            torch.save(state, save_path)
        self._curr_save_path = None

    def _create_checkpoint_file(self, save_dir, tag, zero_checkpoint):
        name_function = (self._get_zero_ckpt_name
                         if zero_checkpoint else self._get_ckpt_name)
        try:
            checkpoint_name = name_function(save_dir, tag)
            ensure_directory_exists(checkpoint_name)
        except:
            logger.error(f"Failed saving model checkpoint to {save_dir} with tag {tag}")
            return False

        return True

    def _create_zero_checkpoint_files(self, save_dir, tag):
        success = True
        # zero checkpoint files are created sequentially
        for rank in range(self.world_size):
            if rank == self.global_rank:
                success = self._create_checkpoint_file(save_dir, tag, True)

            dist.barrier()

        return success

    def _save_checkpoint(self, save_dir, tag, client_state={}):

        save_path = self._get_ckpt_name(save_dir, tag)
        # A hack to save the checkpointing directory. Pipeline parallelism overrides
        # module_state_dict() and uses this path to save the model. module_state_dict()
        # then instead just returns None.
        self._curr_ckpt_path = os.path.join(save_dir, tag)

        state = dict(module=self.module_state_dict(),
                     buffer_names=self._get_buffer_names(),
                     optimizer=self.optimizer.state_dict()
                     if self.optimizer and not self.zero_optimization() else None,
                     lr_scheduler=self.lr_scheduler.state_dict()
                     if self.lr_scheduler is not None else None,
                     sparse_tensor_module_names=self.sparse_tensor_module_names,
                     skipped_steps=self.skipped_steps,
                     global_steps=self.global_steps,
                     global_samples=self.global_samples,
                     dp_world_size=self.dp_world_size,
                     mp_world_size=self.mp_world_size,
                     ds_config=self.config,
                     ds_version=version)
        state.update(client_state)

        log_dist(message=f'Saving model checkpoint: {save_path}', ranks=[0, 1])
        #logger.info('Saving model checkpoint: {}'.format(save_path))
        torch.save(state, save_path)
        self._curr_save_path = None

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

    def _copy_recovery_script(self, save_path):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        script = "zero_to_fp32.py"
        src = os.path.join(base_dir, "utils", script)
        dst = os.path.join(save_path, script)
        #logger.info(f"creating recovery script {dst}")
        copyfile(src, dst)
        # make executable
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC)

    def _save_zero_checkpoint(self, save_path, tag):
        zero_checkpoint_name = self._get_zero_ckpt_name(save_path, tag)
        zero_sd = dict(optimizer_state_dict=self.optimizer.state_dict(),
                       param_shapes=self._get_zero_param_shapes(),
                       ds_config=self.config,
                       ds_version=version)
        torch.save(zero_sd, zero_checkpoint_name)
        if self.global_rank == 0:
            self._copy_recovery_script(save_path)
        logger.info('zero checkpoint saved {}'.format(zero_checkpoint_name))

    def _zero3_consolidated_fp16_state_dict(self):
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
        import deepspeed

        if not self.zero_optimization_partition_weights():
            raise ValueError("this function requires ZeRO-3 mode")

        state_dict = OrderedDict() if torch.distributed.get_rank() == 0 else None
        shared_params = {}

        def get_layer_state_dict(module, prefix=""):
            # gather one layer at a time to be memory-efficient
            # must use modifier_rank=0 to release GPU memory after each layer gathered
            #see_memory_usage("before GatheredParameters", force=True)
            with deepspeed.zero.GatheredParameters(list(
                    module.parameters(recurse=False)),
                                                   modifier_rank=0):
                if torch.distributed.get_rank() == 0:
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
                        if (buf is not None
                                and name not in module._non_persistent_buffers_set):
                            state_dict[prefix + name] = buf.detach().cpu()
            #see_memory_usage("after GatheredParameters", force=True)

            for name, child in module.named_children():
                if child is not None:
                    get_layer_state_dict(child, prefix + name + ".")

        see_memory_usage("before get_layer_state_dict", force=False)
        get_layer_state_dict(self.module, prefix="")
        see_memory_usage("after get_layer_state_dict", force=False)

        return state_dict

    def save_fp16_model(self, save_dir, save_filename="pytorch_model.bin"):
        r"""Save fp16 model weights

        This method saves the fp16 model weights at the desired destination.

        Arguments:
            save_dir: Required. Directory for saving the model
            save_filename: Optional. Filename to save to. Defaults to ``pytorch_model.bin``

        Returns:
            ``True`` when a model has been saved, ``False`` otherwise. It will not be saved if
            stage3_gather_fp16_weights_on_model_save is ``False``.

        Important: all processes must call this method and not just the process with rank 0. It is
        because the processes need to work in sync to gather the weights. This method will hang
        waiting to synchronize with other processes if it's called just for the process with rank 0.

        """

        path = os.path.join(save_dir, save_filename)

        if self.zero_optimization_partition_weights():
            if self.zero_gather_fp16_weights_on_model_save():
                # consolidation is expensive in time and memory and therefore isn't a default
                state_dict = self._zero3_consolidated_fp16_state_dict()
            else:
                # the model will be bogus if not consolidated so don't confuse the user by saving it
                logger.info(
                    f"Did not save the model {path} because `stage3_gather_fp16_weights_on_model_save` is False"
                )
                return False
        else:
            state_dict = self.module.state_dict()

        if torch.distributed.get_rank() == 0:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving model weights to {path}")
            torch.save(state_dict, path)

        return True
