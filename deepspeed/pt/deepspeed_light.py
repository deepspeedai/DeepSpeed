'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import torch
import os
import warnings
import torch.distributed as dist
from torch.nn.modules import Module
from torch.distributed.distributed_c10d import _get_global_rank
from apex import amp

from tensorboardX import SummaryWriter

from deepspeed.pt.deepspeed_timer import ThroughputTimer, SynchronizedWallClockTimer
from deepspeed.pt.deepspeed_zero_optimizer import FP16_DeepSpeedZeroOptimizer
from deepspeed.pt.zero_optimizer_stage1 import FP16_DeepSpeedZeroOptimizer_Stage1
from deepspeed.pt.log_utils import logger
import deepspeed.pt.deepspeed_checkpointing as deepspeed_activation_checkpointing

from deepspeed.pt.fp16_optimizer import FP16_Optimizer
from deepspeed.pt.fp16_unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.pt.deepspeed_fused_lamb import FusedLamb
from deepspeed.pt.deepspeed_config import DeepSpeedConfig, \
    ADAM_OPTIMIZER, LAMB_OPTIMIZER, DEEPSPEED_OPTIMIZERS

from deepspeed.pt.deepspeed_dataloader import DeepSpeedDataLoader
from deepspeed.pt.deepspeed_constants import \
    ROUTE_TRAIN, ROUTE_PREDICT, ROUTE_EVAL, \
    TORCH_DISTRIBUTED_DEFAULT_PORT, \
    ZERO_OPTIMIZATION_OPTIMIZER_STATES, ZERO_OPTIMIZATION_GRADIENTS

import deepspeed.pt.deepspeed_lr_schedules as lr_schedules
from deepspeed.pt.deepspeed_csr_tensor import CSRTensor

MEMORY_OPT_ALLREDUCE_SIZE = 500000000
SUMMARY_WRITER_DIR_NAME = "JobId"

try:
    from apex_C import flatten
    from apex_C import unflatten
except ImportError:
    try:
        _ = warned_flatten
    except NameError:
        logger.warning(
            "Warning:  apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten."
        )
        warned_flatten = True
    from torch._utils import _flatten_dense_tensors as flatten
    from torch._utils import _unflatten_dense_tensors as unflatten


def split_half_float_double_csr(tensors):
    dtypes = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
        CSRTensor.type()
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


def _initialize_parameter_parallel_groups(parameter_parallel_size=None):
    data_parallel_size = int(dist.get_world_size())
    if parameter_parallel_size is None:
        parameter_parallel_size = int(data_parallel_size)
    logger.info("data_parallel_size: %s, parameter_parallel_size: %s",
                data_parallel_size,
                parameter_parallel_size)
    assert data_parallel_size % parameter_parallel_size == 0, \
        'world size should be divisible by parameter parallel size'
    rank = dist.get_rank()
    my_group = None
    for i in range(dist.get_world_size() // parameter_parallel_size):
        ranks = range(i * parameter_parallel_size, (i + 1) * parameter_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            my_group = group
    return my_group


def print_configuration(args, name):
    logger.info('{}:'.format(name))
    for arg in sorted(vars(args)):
        dots = '.' * (29 - len(arg))
        logger.info('  {} {} {}'.format(arg, dots, getattr(args, arg)))


class DeepSpeedLight(Module):
    r"""DeepSpeed engine for training.
    """
    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 model_parameters=None,
                 training_data=None,
                 lr_scheduler=None,
                 mpu=None,
                 dist_init_required=None,
                 collate_fn=None,
                 config_params=None):
        super(DeepSpeedLight, self).__init__()

        self.client_optimizer = optimizer
        self.client_model_parameters = model_parameters
        self.client_lr_scheduler = lr_scheduler
        self.training_data = training_data
        self.collate_fn = collate_fn
        self.mpu = mpu
        self.data_parallel_group = None
        self.global_steps = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gradient_average = True
        self.warn_unscaled_loss = True
        self.config_params = config_params
        self.loaded_checkpoint_mp_world_size = None
        self.loaded_checkpoint_dp_world_size = None

        if dist_init_required is None:
            dist_init_required = not dist.is_initialized()

        self._mpi_check(args, dist_init_required)

        self.dist_backend = "nccl"
        if dist_init_required:
            if not dist.is_initialized():
                logger.info("Initializing torch distributed with backend: {}".format(
                    self.dist_backend))
                dist.init_process_group(backend=self.dist_backend)
            else:
                logger.warning(
                    "Was given dist_init_required=True but detected that torch"
                    "distributed was already initialized, cannot initialize twice.")

        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)
        self._do_sanity_check()

        self._init_distributed(dist_init_required)

        self.sample_count = 0
        if self.tensorboard_enabled() and self.global_rank == 0:
            self.summary_writer = self.get_summary_writer()

        # Configure distributed model
        self._configure_distributed_model(model)

        # Configure wall clock timer
        self.timers = SynchronizedWallClockTimer()

        # Throughput timer
        self.tput_timer = ThroughputTimer(
            batch_size=self.train_micro_batch_size_per_gpu(),
            num_workers=self.dp_world_size,
            steps_per_output=self.steps_per_print(),
            monitor_memory=False)

        self.training_dataloader = self.deepspeed_io(
            training_data) if training_data else None

        # Configure optimizer and scheduler
        self.optimizer = None
        self.lr_scheduler = None
        if model_parameters or optimizer:
            self._configure_optimizer(optimizer, model_parameters)
            self._configure_lr_scheduler(lr_scheduler)
            self._report_progress(0)

        # Bookkeeping for csr support
        self.csr_tensor_module_names = set()
        if self.sparse_gradients_enabled():
            for name, module in self.module.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    self.csr_tensor_module_names.add(name + ".weight")
                    logger.info("Will convert {} to sparse (csr) "
                                "tensor during training".format(name))

        self.save_non_zero_checkpoint = False
        self.save_zero_checkpoint = False
        self._configure_checkpointing(dist_init_required)

        if self.global_rank == 0:
            self._config.print('DeepSpeedLight configuration')
            if self.dump_state():
                print_configuration(self, 'DeepSpeedLight')

    def _mpi_check(self, args, dist_init_required):
        if hasattr(args, 'deepspeed_mpi') and args.deepspeed_mpi:
            from mpi4py import MPI
            import subprocess
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            world_size = comm.Get_size()

            master_addr = None
            if rank == 0:
                hostname_cmd = ["hostname -I"]
                result = subprocess.check_output(hostname_cmd, shell=True)
                master_addr = result.decode('utf-8').split()[0]
            master_addr = comm.bcast(master_addr, root=0)

            # Determine local rank by assuming hostnames are unique
            proc_name = MPI.Get_processor_name()
            all_procs = comm.allgather(proc_name)
            local_rank = sum([i == proc_name for i in all_procs[:rank]])

            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            args.local_rank = local_rank
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = TORCH_DISTRIBUTED_DEFAULT_PORT

            logger.info(
                "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
                .format(os.environ['RANK'],
                        args.local_rank,
                        os.environ['WORLD_SIZE'],
                        os.environ['MASTER_ADDR'],
                        os.environ['MASTER_PORT']))

            if not dist_init_required and dist.is_initialized():
                assert dist.get_rank() == rank, "MPI rank {} does not match torch rank {}".format(rank, dist.get_rank())
                assert dist.get_world_size() == world_size, "MPI world size {} does not match torch world size {}".format(
                    world_size, dist.get_world_size())

    def tensorboard_enabled(self):
        return self._config.tensorboard_enabled

    def tensorboard_output_path(self):
        return self._config.tensorboard_output_path

    def tensorboard_job_name(self):
        return self._config.tensorboard_job_name

    def get_summary_writer(self,
                           name="DeepSpeedJobName",
                           base=os.environ["HOME"] + "/tensorboard"):
        if self.tensorboard_job_name():
            name = self.tensorboard_job_name()
        if self.tensorboard_output_path():
            return SummaryWriter(log_dir=self.tensorboard_output_path())
        if 'DLWS_JOB_ID' in os.environ:
            SUMMARY_WRITER_DIR_NAME = os.environ['DLWS_JOB_ID'] + "/logs"
        return SummaryWriter(log_dir=os.path.join(base, SUMMARY_WRITER_DIR_NAME, name))

    def wall_clock_breakdown(self):
        return self._config.wall_clock_breakdown

    def memory_breakdown(self):
        return self._config.memory_breakdown

    def sparse_gradients_enabled(self):
        return self._config.sparse_gradients_enabled

    def train_batch_size(self):
        return self._config.train_batch_size

    def train_micro_batch_size_per_gpu(self):
        return self._config.train_micro_batch_size_per_gpu

    def optimizer_name(self):
        return self._config.optimizer_name

    def optimizer_params(self):
        return self._config.optimizer_params

    def optimizer_legacy_fusion(self):
        return self._config.optimizer_legacy_fusion

    def scheduler_name(self):
        return self._config.scheduler_name

    def scheduler_params(self):
        return self._config.scheduler_params

    def zero_optimization(self):
        return self._config.zero_enabled

    def zero_allow_untested_optimizer(self):
        return self._config.zero_allow_untested_optimizer

    def zero_reduce_scatter(self):
        return self._config.zero_config.reduce_scatter

    def zero_overlap_comm(self):
        return self._config.zero_config.overlap_comm

    def zero_optimization_stage(self):
        return self._config.zero_optimization_stage

    def zero_reduce_bucket_size(self):
        return self._config.zero_config.reduce_bucket_size

    def zero_allgather_bucket_size(self):
        return self._config.zero_config.allgather_bucket_size

    def zero_optimization_partition_gradients(self):
        return self.zero_optimization_stage() >= ZERO_OPTIMIZATION_GRADIENTS

    def zero_contiguous_gradients(self):
        return self._config.zero_config.contiguous_gradients

    def zero_load_from_fp32_weights(self):
        return self._config.zero_config.load_from_fp32_weights

    def allgather_size(self):
        return self._config.allgather_size

    def fp16_enabled(self):
        return self._config.fp16_enabled

    def amp_enabled(self):
        return self._config.amp_enabled

    def amp_params(self):
        return self._config.amp_params

    def loss_scale(self):
        return self._config.loss_scale

    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    def allreduce_always_fp32(self):
        return self._config.allreduce_always_fp32

    def postscale_gradients(self):
        return not self._config.prescale_gradients

    def gradient_predivide_factor(self):
        return self._config.gradient_predivide_factor

    def steps_per_print(self):
        return self._config.steps_per_print

    def zero_allgather_partitions(self):
        return self._config.zero_config.allgather_partitions

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

    def _configure_lr_scheduler(self, client_lr_scheduler):
        # First check for scheduler in json configuration
        lr_scheduler = self._scheduler_from_config(self.optimizer)
        if lr_scheduler:
            logger.info(
                f'DeepSpeed using configured LR scheduler = {self.scheduler_name()}')
            self.lr_scheduler = lr_scheduler
        else:
            logger.warning('DeepSpeed using client LR scheduler')
            self.lr_scheduler = client_lr_scheduler
        logger.info(f'DeepSpeed LR Scheduler = {self.lr_scheduler}')

    def _configure_checkpointing(self, dist_init_required):

        dp_rank = self.global_rank
        if self.mpu:
            dp_rank = self.mpu.get_data_parallel_rank()

        # only the first data parallel process needs to store the model checkpoint
        self.save_non_zero_checkpoint = (dp_rank == 0)

        if self.zero_optimization():
            pp_rank = torch.distributed.get_rank(group=self.optimizer.dp_process_group)

            # Only the first parameter parallel process needs to store the
            # optimizer state checkpoints for zero
            self.save_zero_checkpoint = (pp_rank == dp_rank)

    def _scheduler_from_config(self, optimizer):
        scheduler_name = self.scheduler_name()
        if scheduler_name is not None:
            if hasattr(lr_schedules, scheduler_name):
                scheduler = getattr(lr_schedules, scheduler_name)
            else:
                assert hasattr(torch.optim.lr_scheduler, scheduler_name), \
                    f"DeepSpeed does not recognize LR scheduler {scheduler_name}"

                scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)

            scheduler_params = self.scheduler_params()
            instantiated_scheduler = scheduler(optimizer, **scheduler_params)
            return instantiated_scheduler
        else:
            return None

    def _init_distributed(self, dist_init_required):
        if self.local_rank >= 0:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            logger.info("Set device to local rank {} within node.".format(
                self.local_rank))
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device("cuda")

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        self.local_rank = args.local_rank if hasattr(args, 'local_rank') else 0
        self._config = DeepSpeedConfig(args.deepspeed_config,
                                       mpu,
                                       param_dict=self.config_params)

    # Validate command line arguments
    def _do_args_sanity_check(self, args):
        if hasattr(args, 'deepscale_config') and args.deepscale_config is not None:
            logger.warning(
                "************ --deepscale_config is deprecated, please use --deepspeed_config ************"
            )
            if hasattr(args, 'deepspeed_config'):
                assert args.deepspeed_config is None, "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
            args.deepspeed_config = args.deepscale_config

        assert hasattr(args, 'local_rank') and type(args.local_rank) == int, \
            'DeepSpeed requires integer command line parameter --local_rank'

        if self.config_params is None:
            assert hasattr(args, 'deepspeed_config') and args.deepspeed_config is not None, \
                'DeepSpeed requires --deepspeed_config to specify configuration file'

            assert os.path.isfile(args.deepspeed_config), \
                'DeepSpeed configuration file: {} is not an existing file'.format(args.deepspeed_config)

    def _is_supported_optimizer(self, optimizer_name):
        return optimizer_name in DEEPSPEED_OPTIMIZERS or \
            getattr(torch.optim, optimizer_name, None) is not None

    # Validate configuration based on command line arguments
    def _do_sanity_check(self):
        if not self.client_optimizer:
            assert self._is_supported_optimizer(self.optimizer_name()), \
                '{} is not a supported DeepSpeed Optimizer'.format(self.optimizer_name())
            assert self.client_model_parameters, \
                'DeepSpeed {} optimizer requires parameters in initialize() call'.format(self.optimizer_name())

        if self.optimizer_name() == LAMB_OPTIMIZER:
            assert self.dynamic_loss_scale(), \
                'DeepSpeed {} optimizer requires dynamic loss scaling'.format(self.optimizer_name())

    def _broadcast_model(self):
        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p,
                               self.broadcast_src_rank,
                               group=self.data_parallel_group)

    def _configure_distributed_model(self, model):
        self.module = model
        if self.fp16_enabled():
            self.module.half()
        self.module.to(self.device)

        if self.mpu is None:
            self.data_parallel_group = _initialize_parameter_parallel_groups()
            self.dp_world_size = dist.get_world_size()
            self.mp_world_size = 1
            self.broadcast_src_rank = 0
        else:
            self.data_parallel_group = self.mpu.get_data_parallel_group()
            self.dp_world_size = self.mpu.get_data_parallel_world_size()
            self.mp_world_size = self.mpu.get_model_parallel_world_size()
            self.broadcast_src_rank = _get_global_rank(
                self.mpu.get_data_parallel_group(),
                0)
            logger.info(f"global src_rank={self.broadcast_src_rank}")

        if not self.amp_enabled():
            self._broadcast_model()

    # Configure optimizer
    def _configure_optimizer(self, client_optimizer, model_parameters):
        if client_optimizer is not None:
            basic_optimizer = client_optimizer
            logger.info('Using client Optimizer as basic optimizer')
        else:
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            logger.info(
                'Using DeepSpeed Optimizer param name {} as basic optimizer'.format(
                    self.optimizer_name()))

        logger.info('DeepSpeed Basic Optimizer = {}'.format(basic_optimizer))

        if self.zero_optimization():
            assert not self.amp_enabled(), "Amp and ZeRO are not currently compatible, please use (legacy) fp16 mode which performs similar to amp opt_mode=O2"
            if self.optimizer_name() != ADAM_OPTIMIZER:
                assert self.zero_allow_untested_optimizer(), \
                    'You are using an untested ZeRO Optimizer. Please add <"zero_allow_untested_optimizer": true> in the configuration file to use it.'

                logger.warning(
                    "**** You are using ZeRO with an untested optimizer, proceed with caution *****"
                )
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)
        elif self.amp_enabled():
            assert not self.fp16_enabled(), "Cannot enable both amp with (legacy) fp16 mode"
            amp_params = self.amp_params()
            logger.info(f"Initializing AMP with these params: {amp_params}")
            self.module, self.optimizer = amp.initialize(self.module, basic_optimizer, **amp_params)
            self._broadcast_model()
        elif self.fp16_enabled():
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer

        # logger.info('DeepSpeed Final Optimizer = {}'.format(self.optimizer.state_dict()))

    def _configure_basic_optimizer(self, model_parameters):
        optimizer_parameters = self.optimizer_params()
        if 'max_grad_norm' in optimizer_parameters.keys():
            raise ValueError(
                "'max_grad_norm' is not supported as an optimizer parameter, please switch to using the deepspeed parameter 'gradient_clipping' see: https://www.deepspeed.ai/docs/config-json/#gradient-clipping for more details"
            )
        if self.optimizer_name() == ADAM_OPTIMIZER:
            from apex.optimizers.fused_adam import FusedAdam
            optimizer = FusedAdam(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == LAMB_OPTIMIZER:
            optimizer = FusedLamb(model_parameters, **optimizer_parameters)
        else:
            torch_optimizer = getattr(torch.optim, self.optimizer_name())
            optimizer = torch_optimizer(model_parameters, **optimizer_parameters)
        return optimizer

    def _configure_fp16_optimizer(self, optimizer):
        initial_dynamic_scale = self.initial_dynamic_scale()
        dynamic_loss_args = self.dynamic_loss_scale_args()
        clip_grad = self.gradient_clipping()
        if self.optimizer_name() == ADAM_OPTIMIZER:
            if self.dynamic_loss_scale():
                logger.info('Creating fp16 optimizer with dynamic loss scale')
                timers = self.timers if self.wall_clock_breakdown() else None
                optimizer = FP16_Optimizer(
                    optimizer,
                    dynamic_loss_scale=True,
                    initial_dynamic_scale=initial_dynamic_scale,
                    dynamic_loss_args=dynamic_loss_args,
                    mpu=self.mpu,
                    clip_grad=clip_grad,
                    fused_adam_legacy=self.optimizer_legacy_fusion(),
                    timers=timers)
            else:
                logger.info('Creating fp16 optimizer with static loss scale: {}'.format(
                    self.loss_scale()))
                optimizer = FP16_Optimizer(
                    optimizer,
                    static_loss_scale=self.loss_scale(),
                    mpu=self.mpu,
                    clip_grad=clip_grad,
                    fused_adam_legacy=self.optimizer_legacy_fusion())
        else:
            logger.info('Creating fp16 unfused optimizer with dynamic loss scale')
            optimizer = FP16_UnfusedOptimizer(
                optimizer,
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_lamb_legacy=self.optimizer_name() == LAMB_OPTIMIZER)

        return optimizer

    def _configure_zero_optimizer(self, optimizer):
        zero_stage = self.zero_optimization_stage()
        logger.info('Creating fp16 ZeRO stage {} optimizer'.format(zero_stage))

        if zero_stage == ZERO_OPTIMIZATION_OPTIMIZER_STATES:
            assert self.zero_reduce_scatter(), 'Stage 1 only supports reduce scatter mode'
            optimizer = FP16_DeepSpeedZeroOptimizer_Stage1(
                optimizer,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                all_gather_partitions=self.zero_allgather_partitions(),
                allgather_size=self.zero_allgather_bucket_size(),
                max_elements_per_comm=self.zero_reduce_bucket_size(),
                dp_process_group=self.data_parallel_group,
                mpu=self.mpu)
        elif zero_stage == ZERO_OPTIMIZATION_GRADIENTS:
            assert self.gradient_accumulation_steps(
            ) == 1, "ZeRO stage 2 does not support gradient accumulation, if you need gradient accumulation please use stage 1"
            optimizer = FP16_DeepSpeedZeroOptimizer(
                optimizer,
                timers=self.timers,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                contiguous_gradients=self.zero_contiguous_gradients(),
                reduce_bucket_size=self.zero_reduce_bucket_size(),
                allgather_bucket_size=self.zero_allgather_bucket_size(),
                dp_process_group=self.data_parallel_group,
                reduce_scatter=self.zero_reduce_scatter(),
                overlap_comm=self.zero_overlap_comm(),
                mpu=self.mpu,
                postscale_gradients=self.postscale_gradients(),
                gradient_predivide_factor=self.gradient_predivide_factor())
        else:
            raise NotImplementedError("ZeRO stage {} not implemented".format(zero_stage))

        return optimizer

    def deepspeed_io(self,
                     dataset,
                     batch_size=None,
                     route=ROUTE_TRAIN,
                     pin_memory=True,
                     data_sampler=None,
                     collate_fn=None,
                     num_local_io_workers=None):
        if not isinstance(dataset, torch.utils.data.Dataset):
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

        # If mpu is provied, forward world size and parallel rank to sampler.
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
                                   data_parallel_rank=data_parallel_rank)

    def train(self):
        r"""
        """

        self.warn_unscaled_loss = True
        self.module.train()

    def eval(self):
        r"""
        """

        self.warn_unscaled_loss = True
        self.module.train(False)

    def _scale_loss(self, prescaled_loss):
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
                    f'DeepSpeed unable to scale loss because of type: {type(prescaled_loss)}'
                )
                self.warn_unscaled_loss = False

        return scaled_loss

    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """

        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward').start()

        if self.training_dataloader is None:
            self.tput_timer.start()
        loss = self.module(*inputs, **kwargs)

        if self.wall_clock_breakdown():
            self.timers('forward').stop()
            self.timers('forward_microstep').stop()

        return loss

    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        if self.is_gradient_accumulation_boundary():
            if self.zero_optimization_stage() == ZERO_OPTIMIZATION_OPTIMIZER_STATES:
                assert self.zero_reduce_scatter()
                self.optimizer.reduce_scatter_gradients(
                    postscale_gradients=self.postscale_gradients(),
                    gradient_predivide_factor=self.gradient_predivide_factor(),
                    gradient_average=self.gradient_average)
            elif self.zero_optimization_partition_gradients():
                self.optimizer.overlapping_partition_gradients_reduce_epilogue()
            else:
                self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    def backward(self, loss, allreduce_gradients=True):
        r"""Execute backward pass on the loss

        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: If this is False, then gradient averaging will be skipped. Default is True.
        """

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps() > 1:
            loss = self._scale_loss(loss.float())

        # Log training Loss
        if self.tensorboard_enabled():
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.sample_count += (self.train_micro_batch_size_per_gpu() *
                                          self.dp_world_size *
                                          self.gradient_accumulation_steps())
                    self.summary_events = [
                        (f'Train/Samples/train_loss',
                         loss.mean().item() * self.gradient_accumulation_steps(),
                         self.sample_count)
                    ]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                    self.summary_writer.flush()

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()

        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        if self.wall_clock_breakdown():
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        if self.zero_optimization():
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
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()

        if self.wall_clock_breakdown():
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce').start()

        if allreduce_gradients:
            self.allreduce_gradients()

        if self.wall_clock_breakdown():
            self.timers('backward_allreduce').stop()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        return loss

    def is_gradient_accumulation_boundary(self):
        return (self.micro_steps + 1) % \
            self.gradient_accumulation_steps() == 0

    def zero_grad(self):
        """
        Zero parameter grads.
        """
        for param_name, param in self.module.named_parameters():
            param.grad = None

    def clip_fp32_gradients(self):
        torch.nn.utils.clip_grad_norm_(parameters=self.module.parameters(),
                                       max_norm=self.gradient_clipping())

    def step(self):
        r"""Execute the weight update step after forward and backward propagation on effective_train_batch
        """
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()

        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use step"
        report_progress = self.global_rank == 0 if self.global_rank else True

        if self.is_gradient_accumulation_boundary():

            if self.gradient_clipping() > 0.0:
                if not self.fp16_enabled() and not self.amp_enabled():
                    self.clip_fp32_gradients()
                elif self.amp_enabled():
                    # AMP's recommended way of doing clipping
                    # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                    master_params = amp.master_params(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters=master_params,
                                                   max_norm=self.gradient_clipping())

            self.optimizer.step()

            #zero grad in basic optimizer could be unreliable and may not exhibit
            #the behaviour that we want
            if not self.zero_optimization() and not self.fp16_enabled(
            ) and not self.amp_enabled():
                self.zero_grad()
            else:
                self.optimizer.zero_grad()

            # Check overlow here since in DS fp16 optimizer, the overflow is updated in above step() function.
            overflow = False
            if hasattr(self.optimizer, 'overflow'):
                overflow = self.optimizer.overflow

            if overflow:
                self.skipped_steps += 1
            else:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                if report_progress and (self.global_steps +
                                        1) % self.steps_per_print() == 0:
                    self._report_progress(self.global_steps + 1)

            self.global_steps += 1

        self.tput_timer.stop(report_progress)

        # Log learning rate
        if self.tensorboard_enabled():
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [(f'Train/Samples/lr',
                                            self.get_lr()[0],
                                            self.sample_count)]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                    self.summary_writer.flush()

        if self.wall_clock_breakdown():
            self.timers('step').stop()
            self.timers('step_microstep').stop()
            timer_names = [
                'forward_microstep',
                'backward_microstep',
                'backward_inner_microstep',
                'backward_allreduce_microstep',
                'step_microstep'
            ]
            self.timers.log(names=timer_names, memory_breakdown=self.memory_breakdown())

            # Log timing
            if self.is_gradient_accumulation_boundary():
                if self.tensorboard_enabled():
                    if self.global_rank == 0:
                        self.summary_events = [
                            (f'Train/Samples/elapsed_time_ms_forward',
                             self.timers('forward').elapsed(reset=False) * 1000.0,
                             self.sample_count),
                            (f'Train/Samples/elapsed_time_ms_backward',
                             self.timers('backward').elapsed(reset=False) * 1000.0,
                             self.sample_count),
                            (f'Train/Samples/elapsed_time_ms_backward_inner',
                             self.timers('backward_inner').elapsed(reset=False) * 1000.0,
                             self.sample_count),
                            (f'Train/Samples/elapsed_time_ms_backward_allreduce',
                             self.timers('backward_allreduce').elapsed(reset=False) *
                             1000.0,
                             self.sample_count),
                            (f'Train/Samples/elapsed_time_ms_step',
                             self.timers('step').elapsed(reset=False) * 1000.0,
                             self.sample_count)
                        ]
                        for event in self.summary_events:  # write_summary_events
                            self.summary_writer.add_scalar(event[0], event[1], event[2])
                        self.summary_writer.flush()

            if self.wall_clock_breakdown():
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'step'
                ])

        self.micro_steps += 1

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
        return self._get_optimizer_param('lr')

    def get_mom(self):
        return self._get_optimizer_param('betas')

    def _report_progress(self, step):
        lr = self.get_lr()
        mom = self.get_mom()
        logger.info('rank:{} step={}, skipped={}, lr={}, mom={}'.format(
            self.global_rank,
            step,
            self.skipped_steps,
            lr,
            mom))

    def allreduce_bucket(self, bucket):
        tensor = flatten(bucket)

        tensor_to_allreduce = tensor

        if self.allreduce_always_fp32():
            tensor_to_allreduce = tensor.float()

        if self.postscale_gradients():
            if self.gradient_predivide_factor() != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor())

            dist.all_reduce(tensor_to_allreduce, group=self.data_parallel_group)

            if self.gradient_average:
                if self.gradient_predivide_factor() != self.dp_world_size:
                    tensor_to_allreduce.mul_(self.gradient_predivide_factor() /
                                             self.dp_world_size)
        else:
            tensor_to_allreduce.div_(self.dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=self.data_parallel_group)

        if self.allreduce_always_fp32() and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy(self, small_bucket):
        allreduced = self.allreduce_bucket(small_bucket)
        for buf, synced in zip(small_bucket, unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def allreduce_no_retain(self, bucket, numel_per_bucket=500000000):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket)
                small_bucket = []
                numel = 0
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket)

    def buffered_allreduce_fallback(self, grads=None, elements_per_buffer=500000000):
        grads = []
        for param_name, param in self.module.named_parameters():
            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(param.size(),
                                         dtype=param.dtype,
                                         device=param.device)
                grads.append(param.grad.data)
            else:
                grad_data = param.grad.data
                if self.sparse_gradients_enabled(
                ) and param_name in self.csr_tensor_module_names:
                    grads.append(CSRTensor(grad_data))
                else:
                    grads.append(grad_data)

        split_buckets = split_half_float_double_csr(grads)

        for i, bucket_tuple in enumerate(split_buckets):
            bucket_type, bucket = bucket_tuple
            if bucket_type == CSRTensor.type():
                self.csr_allreduce_no_retain(bucket)
            else:
                self.allreduce_no_retain(bucket, numel_per_bucket=elements_per_buffer)

    def csr_allreduce_no_retain(self, bucket):
        allreduced_csrs = self.csr_allreduce_bucket(bucket)
        # Densify csr tensor and copy back to original location
        for csr in allreduced_csrs:
            dense_tensor = csr.to_dense()
            csr.orig_dense_tensor.copy_(dense_tensor)

    def csr_allreduce_bucket(self, bucket):
        csr_list = []
        for csr in bucket:
            csr_list.append(self.csr_allreduce(csr))
        return csr_list

    def csr_allreduce(self, csr):
        # Pre-divide for fp16 stability
        csr.values.div_(self.dp_world_size)

        indices_device_list = self.csr_all_gather(csr.indices)
        values_device_list = self.csr_all_gather(csr.values)

        csr.indices = torch.cat(indices_device_list)
        csr.values = torch.cat(values_device_list)
        return csr

    def csr_all_gather(self, value):
        my_size = torch.LongTensor([value.size()[0]]).to(self.device)
        all_sizes = self.all_gather_scalar(my_size)
        max_size = torch.cat(all_sizes).max()
        fill_size = (max_size - my_size)

        assert value.dim() in [1, 2]
        if value.dim() == 1:
            if fill_size > 0:
                value = torch.cat([value, value.new_zeros(fill_size)])
            tensor_list = [value.new_zeros(max_size) for _ in range(self.dp_world_size)]
        else:
            if fill_size > 0:
                value = torch.cat([value, value.new_zeros(fill_size, value.size()[1])])
            tensor_list = [
                value.new_zeros(max_size,
                                value.size()[1]) for _ in range(self.dp_world_size)
            ]

        dist.all_gather(tensor_list, value, group=self.data_parallel_group)
        tensors = []
        for dev_idx, t in enumerate(tensor_list):
            size = all_sizes[dev_idx][0]
            tensors.append(
                t.index_select(0,
                               torch.LongTensor(range(size)).to(self.device)))

        return tensors

    def all_gather_scalar(self, value):
        tensor_list = [value.new_zeros(value.size()) for _ in range(self.dp_world_size)]
        dist.all_gather(tensor_list, value, group=self.data_parallel_group)
        return tensor_list

    def module_state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.module.state_dict(destination, prefix, keep_vars)
        return sd

    def load_module_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

    def _get_rank_zero_ckpt_name(self, checkpoints_path, tag, mp_rank, dp_rank):
        filename = 'zero_pp_rank_{}'.format(dp_rank)
        zero_ckpt_name = os.path.join(
            checkpoints_path,
            str(tag),
            filename + '_mp_rank_{:02d}'.format(mp_rank) + 'optim_states.pt')
        return zero_ckpt_name

    def _get_zero_ckpt_name(self, checkpoints_path, tag):
        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        pp_rank = torch.distributed.get_rank(group=self.optimizer.dp_process_group)
        return self._get_rank_zero_ckpt_name(checkpoints_path, tag, mp_rank, pp_rank)

    def _get_ckpt_name(self, checkpoints_path, tag):

        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        ckpt_name = os.path.join(checkpoints_path,
                                 str(tag),
                                 'mp_rank_{:02d}'.format(mp_rank) + '_model_states.pt')
        return ckpt_name

    def _ensure_directory_exists(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def load_checkpoint(self,
                        load_dir,
                        tag,
                        load_module_strict=True,
                        load_optimizer_states=True,
                        load_lr_scheduler_states=True):
        r"""Load training checkpoint

        Arguments:
            load_dir: Required. Directory to load the checkpoint from
            tag: Required. Checkpoint tag used as a unique identifier for the checkpoint. Ex. Global Step.
            load_module_strict: Optional. Boolean to strictly enforce that the keys in state_dict of module and checkpoint match.
            load_optimizer_states: Optional. Boolean to load the training optimizer states from Checkpoint. Ex. ADAM's momentum and variance
            load_lr_scheduler_states: Optional. Boolean to add the learning rate scheduler states from Checkpoint.
        Return:
            load_path: Path of the loaded checkpoint. None if loading the checkpoint failed
            client_state: State dictionary used for loading required training states in the client code.
        """

        load_path, client_states = self._load_checkpoint(load_dir,
                                                         tag,
                                                         load_module_strict=load_module_strict,
                                                         load_optimizer_states=load_optimizer_states,
                                                         load_lr_scheduler_states=load_lr_scheduler_states)

        if self.zero_optimization() and load_path is not None:
            self._load_zero_checkpoint(load_dir,
                                       tag,
                                       load_optimizer_states=load_optimizer_states)

        return load_path, client_states

    def _load_checkpoint(self,
                         load_dir,
                         tag,
                         load_module_strict=True,
                         load_optimizer_states=True,
                         load_lr_scheduler_states=True):

        load_path = self._get_ckpt_name(load_dir, tag)

        if not os.path.exists(load_path):
            logger.warn(
                'Client provided checkpoint load path: {} does not exist ... skip checkpoint load'
                .format(load_path))
            return None, None

        logger.info('Loading checkpoint: {}'.format(load_path))
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)

        self.load_module_state_dict(state_dict=checkpoint['module'],
                                    strict=load_module_strict)
        if not self.zero_optimization():
            if self.fp16_enabled():
                self.optimizer.load_state_dict(
                    checkpoint['optimizer'],
                    load_optimizer_states=load_optimizer_states)
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        if load_lr_scheduler_states and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.csr_tensor_module_names = checkpoint['csr_tensor_module_names']
        self.global_steps = checkpoint['global_steps']
        self.skipped_steps = checkpoint['skipped_steps']
        self.loaded_checkpoint_mp_world_size = checkpoint['mp_world_size']
        self.loaded_checkpoint_dp_world_size = checkpoint['dp_world_size']
        deepspeed_states = [
            'module',
            'optimizer',
            'lr_scheduler',
            'csr_tensor_module_names',
            'skipped_steps',
            'global_steps',
            'dp_world_size',
            'mp_world_size'
        ]
        client_state = {
            key: value
            for key,
            value in checkpoint.items() if not key in deepspeed_states
        }

        return load_path, client_state

    def _load_zero_checkpoint(self, load_dir, tag, load_optimizer_states=True):
        zero_sd_list = self._get_all_zero_checkpoints(load_dir, tag)
        if zero_sd_list is None:
            return

        self.optimizer.load_state_dict(
            state_dict_list=zero_sd_list,
            load_optimizer_states=load_optimizer_states,
            load_from_fp32_weights=self.zero_load_from_fp32_weights())
        print(
            f'loading {len(zero_sd_list)} zero partition checkpoints for rank {self.global_rank}'
        )

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
            dp_world_size=self.loaded_checkpoint_dp_world_size)
        invalid_zero_ckpt_paths = []
        for ckpt_name in zero_ckpt_names:
            if not os.path.exists(ckpt_name):
                invalid_zero_ckpt_paths.append(ckpt_name)

        if len(invalid_zero_ckpt_paths) > 0:
            logging.warn(
                f"Client provided zero checkpoint load paths: {invalid_zero_ckpt_paths} does not exist"
            )
            return None

        zero_sd_list = []
        for ckpt_name in zero_ckpt_names:
            zero_sd_list.append(torch.load(ckpt_name, map_location='cpu'))

        zero_optimizer_sd = [sd['optimizer_state_dict'] for sd in zero_sd_list]
        print(
            f"successfully loaded {len(zero_optimizer_sd)} ZeRO state_dicts for rank {self.global_rank}"
        )
        return zero_optimizer_sd

    def save_checkpoint(self, save_dir, tag, client_state={}):
        r"""Save training checkpoint

        Arguments:
            save_dir: Required. Directory for saving the checkpoint
            tag: Required. Checkpoint tag used as a unique identifier for the checkpoint. Ex. Global Step.
            client_state: Optional. State dictionary used for saving required training states in the client code.
        """

        # This is to make sure the checkpoint names are created without collision
        # There seems to be issue creating them in parallel

        if self.save_non_zero_checkpoint:
            self._create_checkpoint_file(save_dir, tag, False)
            self._save_checkpoint(save_dir, tag, client_state=client_state)

        if self.save_zero_checkpoint:
            self._create_zero_checkpoint_files(save_dir, tag)
            self._save_zero_checkpoint(save_dir, tag)

        return True

    def _create_checkpoint_file(self, save_dir, tag, zero_checkpoint):
        name_function = self._get_zero_ckpt_name if zero_checkpoint else self._get_ckpt_name
        try:
            checkpoint_name = name_function(save_dir, tag)
            self._ensure_directory_exists(checkpoint_name)
        except:
            logger.error(f'Failed Saving model checkpoint to {save_dir} with tag {tag}')
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
        # self._ensure_directory_exists(save_path)

        state = {
            'module':
            self.module_state_dict(),
            'optimizer':
            self.optimizer.state_dict()
            if self.optimizer and not self.zero_optimization() else None,
            'lr_scheduler':
            self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'csr_tensor_module_names':
            self.csr_tensor_module_names,
            'skipped_steps':
            self.skipped_steps,
            'global_steps':
            self.global_steps,
            'dp_world_size':
            self.dp_world_size,
            'mp_world_size':
            self.mp_world_size
        }
        state.update(client_state)

        logger.info('Saving model checkpoint: {}'.format(save_path))
        torch.save(state, save_path)

    def _save_zero_checkpoint(self, save_path, tag):
        zero_checkpoint_name = self._get_zero_ckpt_name(save_path, tag)
        # self._ensure_directory_exists(zero_checkpoint_name)
        zero_sd = {'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(zero_sd, zero_checkpoint_name)
        logger.info('zero checkpoint saved {}'.format(zero_checkpoint_name))
