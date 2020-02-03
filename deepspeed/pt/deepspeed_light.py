'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import logging
import torch
import os
import torch.distributed as dist
from torch.nn.modules import Module

from tensorboardX import SummaryWriter

from deepspeed.pt.deepspeed_timer import ThroughputTimer, SynchronizedWallClockTimer
from deepspeed.pt.deepspeed_zero_optimizer import FP16_DeepSpeedZeroOptimizer

from deepspeed.pt.fp16_optimizer import FP16_Optimizer
from deepspeed.pt.fp16_unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.pt.deepspeed_fused_lamb import FusedLamb
from deepspeed.pt.deepspeed_config import DeepSpeedConfig, \
    ADAM_OPTIMIZER, LAMB_OPTIMIZER, DEEPSPEED_OPTIMIZERS

from deepspeed.pt.deepspeed_dataloader import DeepSpeedDataLoader
from deepspeed.pt.deepspeed_constants import ROUTE_TRAIN, ROUTE_PREDICT, \
    ROUTE_EVAL

import deepspeed.pt.deepspeed_lr_schedules as lr_schedules
from deepspeed.pt.deepspeed_csr_tensor import CSRTensor

from apex import amp
from apex.optimizers.fused_adam import FusedAdam

MEMORY_OPT_ALLREDUCE_SIZE = 500000000
SUMMARY_WRITER_DIR_NAME = "JobId"

try:
    from apex_C import flatten
    from apex_C import unflatten
except ImportError:
    try:
        _ = warned_flatten
    except NameError:
        print(
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
    print(data_parallel_size, parameter_parallel_size)
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
    print('{}:'.format(name), flush=True)
    for arg in sorted(vars(args)):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


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
                 dist_init_required=True,
                 collate_fn=None):
        super(DeepSpeedLight, self).__init__()

        logging.basicConfig(level=logging.INFO,
                            format="[%(levelname)s %(asctime)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

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
        self.gradient_predivide_factor = 1.0
        self.gradient_average = True
        self.warn_unscaled_loss = True

        if dist_init_required:
            dist.init_process_group(backend="nccl")

        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)
        self._do_sanity_check()

        self.sample_count = 0
        if self.tensorboard_enabled():
            self.summary_writer = self.get_summary_writer()

        self._init_distributed(dist_init_required)

        # Throughput timer
        self.tput_timer = ThroughputTimer(
            batch_size=self.train_micro_batch_size_per_gpu(),
            num_workers=self.world_size,
            monitor_memory=False)

        self.training_dataloader = self.deepspeed_io(
            training_data) if training_data else None

        # Configure distributed model
        self._configure_distributed_model(model)

        # Configure optimizer and scheduler
        self.optimizer = None
        self.lr_scheduler = None
        if model_parameters or optimizer:
            self._configure_optimizer(optimizer, model_parameters)
            self._configure_lr_scheduler(lr_scheduler)
            self._report_progress(0)

        # Configure wall clock timer
        self.timers = SynchronizedWallClockTimer()

        # Bookkeeping for csr support
        self.csr_tensor_module_names = set()
        if self.sparse_gradients_enabled():
            for name, module in self.module.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    self.csr_tensor_module_names.add(name)
                    logging.info("Will convert {} to sparse (csr) "
                                 "tensor during training".format(name))

        self.save_non_zero_checkpoint = False
        self.save_zero_checkpoint = False
        self._configure_checkpointing(dist_init_required)

        if self.global_rank == 0:
            self._config.print('DeepSpeedLight configuration')
            if self.dump_state():
                print_configuration(self, 'DeepSpeedLight')

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

    def scheduler_name(self):
        return self._config.scheduler_name

    def scheduler_params(self):
        return self._config.scheduler_params

    def zero_optimization(self):
        return self._config.zero_enabled

    def allgather_size(self):
        return self._config.allgather_size

    def fp16_enabled(self):
        return self._config.fp16_enabled

    def loss_scale(self):
        return self._config.loss_scale

    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    def allreduce_always_fp32(self):
        return self._config.allreduce_always_fp32

    def postscale_gradients(self):
        return not self._config.prescale_gradients

    def steps_per_print(self):
        return self._config.steps_per_print

    def disable_allgather(self):
        return self._config.disable_allgather

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
            logging.info(
                f'DeepSpeed using configured LR scheduler = {self.scheduler_name()}')
            self.lr_scheduler = lr_scheduler
        else:
            logging.warning('DeepSpeed using client LR scheduler')
            self.lr_scheduler = client_lr_scheduler
        logging.info(f'DeepSpeed LR Scheduler = {self.lr_scheduler}')

    def _configure_checkpointing(self, dist_init_required):

        dp_rank = torch.distributed.get_rank(
        ) if self.mpu is None else self.mpu.get_data_parallel_rank()

        #only the first data parallel process needs to store the model checkpoint
        self.save_non_zero_checkpoint = True if dp_rank == 0 else False

        if self.zero_optimization():
            pp_rank = torch.distributed.get_rank(group=self.optimizer.dp_process_group)

            #only the first parameter parallel process needs to store the optimizer state checkpoints for zero
            self.save_zero_checkpoint = True if pp_rank == dp_rank else False

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
            logging.info("Set device to local rank {} within node.".format(
                self.local_rank))
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device("cuda")

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        self.local_rank = args.local_rank if hasattr(args, 'local_rank') else 0
        self._config = DeepSpeedConfig(args.deepspeed_config, mpu)

    # Validate command line arguments
    def _do_args_sanity_check(self, args):
        assert hasattr(args, 'local_rank') and type(args.local_rank) == int, \
            'DeepSpeed requires integer command line parameter --local_rank'

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

    def _configure_distributed_model(self, model):
        self.module = model
        if self.fp16_enabled():
            self.module.half()
        self.module.to(self.device)
        if self.mpu is None:
            self.data_parallel_group = _initialize_parameter_parallel_groups()
            self.dp_world_size = dist.get_world_size()
            src_rank = 0
        else:
            self.data_parallel_group = self.mpu.get_data_parallel_group()
            self.dp_world_size = self.mpu.get_data_parallel_world_size()
            src_rank = self.mpu.get_model_parallel_rank()
        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p, src_rank, group=self.data_parallel_group)

        # TODO: support new AMP optimizer
        # self.module.half()
        # self.module.to(self.local_rank)
        #self.module, self.optimizer = amp.initialize(self.module, self.optimizer, opt_level="O2")

    # Configure optimizer
    def _configure_optimizer(self, client_optimizer, model_parameters):
        if client_optimizer is not None:
            basic_optimizer = client_optimizer
            logging.info('Using client Optimizer as basic optimizer')
        else:
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            logging.info(
                'Using DeepSpeed Optimizer param name {} as basic optimizer'.format(
                    self.optimizer_name()))

        logging.info('DeepSpeed Basic Optimizer = {}'.format(basic_optimizer))

        if self.zero_optimization() and self.optimizer_name() == ADAM_OPTIMIZER:
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)
        elif self.fp16_enabled():
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer

        # logging.info('DeepSpeed Final Optimizer = {}'.format(self.optimizer.state_dict()))

    def _configure_basic_optimizer(self, model_parameters):
        optimizer_parameters = self.optimizer_params()
        if self.fp16_enabled() and 'max_grad_norm' in optimizer_parameters.keys():
            optimizer_parameters['max_grad_norm'] = 0.0
        if self.optimizer_name() == ADAM_OPTIMIZER:
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
                logging.info('Creating fp16 optimizer with dynamic loss scale')
                optimizer = FP16_Optimizer(optimizer,
                                           dynamic_loss_scale=True,
                                           initial_dynamic_scale=initial_dynamic_scale,
                                           dynamic_loss_args=dynamic_loss_args,
                                           mpu=self.mpu,
                                           clip_grad=clip_grad,
                                           fused_adam_legacy=True)
            else:
                logging.info('Creating fp16 optimizer with static loss scale: {}'.format(
                    self.loss_scale()))
                optimizer = FP16_Optimizer(optimizer,
                                           static_loss_scale=self.loss_scale(),
                                           mpu=self.mpu,
                                           clip_grad=clip_grad,
                                           fused_adam_legacy=True)
        else:
            logging.info('Creating fp16 unfused optimizer with dynamic loss scale')
            optimizer = FP16_UnfusedOptimizer(
                optimizer,
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_lamb_legacy=True
                if self.optimizer_name() == LAMB_OPTIMIZER else False)

        return optimizer

    def _configure_zero_optimizer(self, optimizer):
        logging.info('Creating fp16 zero optimizer')
        optimizer = FP16_DeepSpeedZeroOptimizer(
            optimizer,
            static_loss_scale=self.loss_scale(),
            dynamic_loss_scale=self.dynamic_loss_scale(),
            dynamic_loss_args=self.dynamic_loss_scale_args(),
            dp_process_group=self.data_parallel_group,
            clip_grad=self.gradient_clipping(),
            all_gather_partitions=not self.disable_allgather(),
            allgather_size=self.allgather_size(),
            mpu=self.mpu)

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

        return DeepSpeedDataLoader(dataset=dataset,
                                   batch_size=batch_size,
                                   pin_memory=pin_memory,
                                   collate_fn=collate_fn,
                                   local_rank=self.local_rank,
                                   tput_timer=deepspeed_io_timer,
                                   num_local_io_workers=num_local_io_workers,
                                   data_sampler=data_sampler)

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

    def _scale_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss / self.gradient_accumulation_steps()
        elif isinstance(loss, tuple) and isinstance(loss[0], torch.Tensor):
            loss = (l / self.gradient_accumulation_steps() for l in loss)
        elif isinstance(loss, list) and isinstance(loss[0], torch.Tensor):
            loss = [l / self.gradient_accumulation_steps() for l in loss]
        else:
            if self.warn_unscaled_loss:
                logging.warning(
                    f'DeepSpeed unable to scale loss because of type: {type(loss)}')
                self.warn_unscaled_loss = False

        return loss

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

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps() > 1:
            loss = self._scale_loss(loss)

        if self.wall_clock_breakdown():
            self.timers('forward').stop()
            self.timers('forward_microstep').stop()

        return loss

    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        if self.is_gradient_accumulation_boundary():
            self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    def backward(self, loss, allreduce_gradients=True):
        r"""Execute backward pass on the loss

        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: If this is False, then gradient averaging will be skipped. Default is True.
        """

        if self.is_gradient_accumulation_boundary() and self.tensorboard_enabled(
        ) and torch.distributed.get_rank(
        ) == 0:  # deepspeed tensorboard support for loss
            self.sample_count += (self.train_micro_batch_size_per_gpu() *
                                  torch.distributed.get_world_size() *
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
        elif self.fp16_enabled():
            self.optimizer.backward(loss)

            # TODO: Use new AMP semantics as below
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #    scaled_loss.backward()
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

    def is_gradient_accumulation_boundary(self):
        return (self.micro_steps + 1) % \
            self.gradient_accumulation_steps() == 0

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
            self.optimizer.step()
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

        if self.is_gradient_accumulation_boundary() and self.tensorboard_enabled(
        ) and torch.distributed.get_rank() == 0:  # deepspeed tensorboard support for lr
            self.summary_events = [(f'Train/Samples/lr',
                                    self.get_lr()[0],
                                    self.sample_count)]
            for event in self.summary_events:  # write_summary_events
                self.summary_writer.add_scalar(event[0], event[1], event[2])
            self.summary_writer.flush()

        if self.wall_clock_breakdown():
            self.timers('step').stop()
            self.timers('step_microstep').stop()
            self.timers.log([
                'forward_microstep',
                'backward_microstep',
                'backward_inner_microstep',
                'backward_allreduce_microstep',
                'step_microstep'
            ])
            if self.is_gradient_accumulation_boundary():
                if self.tensorboard_enabled() and torch.distributed.get_rank(
                ) == 0:  # this is done before the log because log resets timers
                    self.summary_events = [(f'Train/Samples/elapsed_time_ms_forward', self.timers('forward').elapsed(reset=False) * 1000.0, self.sample_count), \
                                            (f'Train/Samples/elapsed_time_ms_backward', self.timers('backward').elapsed(reset=False) * 1000.0, self.sample_count), \
                                            (f'Train/Samples/elapsed_time_ms_backward_inner', self.timers('backward_inner').elapsed(reset=False) * 1000.0, self.sample_count), \
                                            (f'Train/Samples/elapsed_time_ms_backward_allreduce', self.timers('backward_allreduce').elapsed(reset=False) * 1000.0, self.sample_count), \
                                            (f'Train/Samples/elapsed_time_ms_step', self.timers('step').elapsed(reset=False) * 1000.0, self.sample_count)
                                            ]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                    self.summary_writer.flush()
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
        logging.info('rank:{} step={}, skipped={}, lr={}, mom={}'.format(
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
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)

            dist.all_reduce(tensor_to_allreduce, group=self.data_parallel_group)

            if self.gradient_average:
                if self.gradient_predivide_factor != self.dp_world_size:
                    tensor_to_allreduce.mul_(self.gradient_predivide_factor /
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
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket)

    def buffered_allreduce_fallback(self, grads=None, elements_per_buffer=500000000):
        grads = []
        for param_name, param in self.module.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                param_name_root = param_name.split('.', 1)[0]
                if self.sparse_gradients_enabled(
                ) and param_name_root in self.csr_tensor_module_names:
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
        my_size = torch.LongTensor([value.size()[0]]).cuda()
        all_sizes = self.all_gather_scalar(my_size)
        max_size = torch.cat(all_sizes).max()
        fill_size = (max_size - my_size)

        assert value.dim() in [1, 2]
        if value.dim() == 1:
            if fill_size > 0:
                value = torch.cat([value, value.new_zeros(fill_size)])
            tensor_list = [
                value.new_zeros(max_size) for _ in range(dist.get_world_size())
            ]
        else:
            if fill_size > 0:
                value = torch.cat([value, value.new_zeros(fill_size, value.size()[1])])
            tensor_list = [
                value.new_zeros(max_size,
                                value.size()[1]) for _ in range(dist.get_world_size())
            ]

        dist.all_gather(tensor_list, value, group=self.data_parallel_group)
        tensors = []
        for dev_idx, t in enumerate(tensor_list):
            size = all_sizes[dev_idx][0]
            tensors.append(t.index_select(0, torch.LongTensor(range(size)).cuda()))

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

    def _get_zero_ckpt_name(self, checkpoints_path, tag):

        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        pp_rank = torch.distributed.get_rank(group=self.optimizer.dp_process_group)

        filename = 'zero_pp_rank_{}'.format(pp_rank)
        zero_ckpt_name = os.path.join(
            checkpoints_path,
            str(tag),
            filename + '_mp_rank_{:02d}'.format(mp_rank) + 'optim_states.pt')
        return zero_ckpt_name

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

    def load_checkpoint(self, load_dir, tag):
        r"""Load training checkpoint

        Arguments:
            load_dir: Required. Directory to load the checkpoint from
            tag: Required. Checkpoint tag used as a unique identifier for the checkpoint. Ex. Global Step.

        Return:
            load_path: Path of the loaded checkpoint. None if loading the checkpoint failed
            client_state: State dictionary used for loading required training states in the client code.
        """

        load_path, client_states = self._load_checkpoint(load_dir, tag)

        if self.zero_optimization() and load_path is not None:
            self._load_zero_checkpoint(load_dir, tag)

        return load_path, client_states

    def _load_checkpoint(self, load_dir, tag):

        load_path = self._get_ckpt_name(load_dir, tag)

        if not os.path.exists(load_path):
            logging.warn(
                'Client provided checkpoint load path: {} does not exist ... skip checkpoint load'
                .format(load_path))
            return None, None

        logging.info('Loading checkpoint: {}'.format(load_path))
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)

        self.load_module_state_dict(checkpoint['module'])
        if not self.zero_optimization():
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.csr_tensor_module_names = checkpoint['csr_tensor_module_names']
        self.global_steps = checkpoint['global_steps']
        self.skipped_steps = checkpoint['skipped_steps']
        deepspeed_states = [
            'module',
            'optimizer',
            'csr_tensor_module_names',
            'skipped_steps',
            'global_step'
        ]
        client_state = {
            key: value
            for key,
            value in checkpoint.items() if not key in deepspeed_states
        }

        return load_path, client_state

    def _load_zero_checkpoint(self, load_dir, tag):
        zero_checkpoint_name = self._get_zero_ckpt_name(load_dir, tag)

        if not os.path.exists(zero_checkpoint_name):
            logging.warn(
                'Client provided checkpoint load path: {} does not exist ... skip checkpoint load'
                .format(zero_checkpoint_name))
            return None

        zero_sd = torch.load(zero_checkpoint_name, map_location='cpu')
        self.optimizer.load_state_dict(zero_sd['optimizer_state_dict'])
        logging.info('loading zero checkpoint {}'.format(zero_checkpoint_name))

    def save_checkpoint(self, save_dir, tag, client_state={}):
        r"""Save training checkpoint

        Arguments:
            save_dir: Required. Directory for saving the checkpoint
            tag: Required. Checkpoint tag used as a unique identifier for the checkpoint. Ex. Global Step.
            client_state: Optional. State dictionary used for saving required training states in the client code.
        """

        #This is to make sure the checkpoint names are created without collision
        #There seems to be issue creating them in parallel
        self._create_checkpoint_files(save_dir, tag)

        try:
            if self.save_non_zero_checkpoint:
                self._save_checkpoint(save_dir, tag, client_state=client_state)

            if self.save_zero_checkpoint:
                self._save_zero_checkpoint(save_dir, tag)
        except:
            logging.error(f'Failed Saving model checkpoint to {save_dir} with tag {tag}')
            return False
        return True

    def _create_checkpoint_files(self, save_dir, tag):

        #checkpoint files are created sequentially
        for rank in range(dist.get_world_size()):
            if rank == dist.get_rank():
                try:
                    if self.save_non_zero_checkpoint:
                        checkpoint_name = self._get_ckpt_name(save_dir, tag)
                        self._ensure_directory_exists(checkpoint_name)

                    if self.save_zero_checkpoint:
                        checkpoint_name = self._get_zero_ckpt_name(save_dir, tag)
                        self._ensure_directory_exists(checkpoint_name)
                except:
                    logging.error(
                        f'Failed Saving model checkpoint to {save_dir} with tag {tag}')
                    return False
            dist.barrier()

    def _save_checkpoint(self, save_dir, tag, client_state={}):

        save_path = self._get_ckpt_name(save_dir, tag)
        #self._ensure_directory_exists(save_path)

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
        }
        state.update(client_state)

        logging.info('Saving model checkpoint: {}'.format(save_path))
        torch.save(state, save_path)

    def _save_zero_checkpoint(self, save_path, tag):
        try:
            zero_checkpoint_name = self._get_zero_ckpt_name(save_path, tag)
            #self._ensure_directory_exists(zero_checkpoint_name)

        except:
            logging.error(
                f'Failed Saving Zero model checkpoint to {save_path} with tag {tag}')

        zero_sd = {'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(zero_sd, zero_checkpoint_name)
        logging.info('zero checkpoint saved {}'.format(zero_checkpoint_name))
