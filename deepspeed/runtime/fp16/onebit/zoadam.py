# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types
import torch
import numpy as np
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import required_torch_version
from deepspeed import comm as dist


class ZeroOneAdam(torch.optim.Optimizer):
    """Implements the 0/1 Adam algorithm. Currently GPU-only.
    For usage example please see https://www.deepspeed.ai/tutorials/zero-one-adam/
    For technical details please read https://arxiv.org/abs/2202.06009
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        var_freeze_step (int, optional): The latest step to update the variance,
            using the notation from https://arxiv.org/abs/2202.06009, it denotes the
            max{i|i in T_v}. Note that this is different from the freeze step from the
            1-bit Adam. The var_freeze_step is usually the end of the learning rate warmup
            and thus does not require tuning. (default: 100000)
        var_update_scaler (int, optional): The interval to update the variance. Note that
            the update policy for variance follows an exponential rule, where var_update_scaler
            denotes the kappa in the 0/1 Adam paper. (default: 16)
        local_step_scaler (int, optional): The interval to scale the local steps interval
            according to the learning rate policy. (default: 32678)
        local_step_clipper (int, optional): The largest interval for local steps with
            learning rate policy. This corresponds to the variable H in the 0/1 Adam paper.
            (default: 16)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in 0/1 Adam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
        cuda_aware (boolean, required): Set True if the underlying MPI implementation
            supports CUDA-Aware communication. (default: False)
        comm_backend_name (string, optional): Set to 'mpi' if needed. (default: 'nccl')
    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 params,
                 deepspeed=None,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0.,
                 max_grad_norm=0.,
                 var_freeze_step=100000,
                 var_update_scaler=16,
                 local_step_scaler=32678,
                 local_step_clipper=16,
                 amsgrad=False,
                 cuda_aware=False,
                 comm_backend_name='nccl'):

        if amsgrad:
            raise RuntimeError('0/1 Adam does not support the AMSGrad variant.')

        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)

        super(ZeroOneAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        self.deepspeed = deepspeed
        self.initialize = False
        self.cuda_aware = cuda_aware
        self.using_pipeline = False

        self.var_freeze_step = var_freeze_step
        self.var_update_scaler = var_update_scaler
        self.local_step_scaler = local_step_scaler
        self.local_step_clipper = local_step_clipper
        self.freeze_key = False
        self.reinitial_error_buffer = False

        self.comm_backend_name = comm_backend_name

        assert dist.is_initialized(), "Please initialize the torch distributed backend."
        # Empty initializer. Set handle based on the comm backend as follows.
        self.comm_backend_handle = None
        if self.comm_backend_name == 'nccl':
            assert (
                required_torch_version(min_version=1.8)
            ), "Please use torch 1.8 or greater to enable NCCL backend in 0/1 Adam. Alternatively, please specify 'mpi' as the 'comm_backend_name' in config file to proceed with the MPI backend"
            from deepspeed.runtime.comm.nccl import NcclBackend
            self.using_pipeline = hasattr(self.deepspeed, 'pipeline_enable_backward_allreduce')
            self.comm_backend_handle = NcclBackend(self.deepspeed.mpu)
        elif self.comm_backend_name == 'mpi':
            from deepspeed.runtime.comm.mpi import MpiBackend
            self.comm_backend_handle = MpiBackend(cuda_aware)
        elif self.comm_backend_name == 'hccl':
            from deepspeed.runtime.comm.hccl import HcclBackend
            self.using_pipeline = hasattr(self.deepspeed, 'pipeline_enable_backward_allreduce')
            self.comm_backend_handle = HcclBackend(self.deepspeed.mpu)
        self.size = self.comm_backend_handle.size

        self.divider = int(self.size * 8 / np.gcd(self.size, 8))

    def step(self, closure=None, grads=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        for group, grads_this_group in zip(self.param_groups, grads_group):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad in zip(group['params'], grads_this_group):
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('0/1 Adam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if not self.initialize or 'worker_error' not in state.keys():
                    # Some scalars to help scale the variance update/local step policies
                    state['var_interval'] = 1
                    state['var_counter'] = 0
                    state['local_step_interval'] = 1
                    state['local_step_counter'] = 0
                    state['lrs'] = 0
                    state['tensor_size'] = torch.numel(p.data)
                    state['corrected_tensor_size'] = state['tensor_size']

                    if state['tensor_size'] % (self.size * self.divider) != 0:
                        state['corrected_tensor_size'] += ((self.size * self.divider) - (state['tensor_size'] %
                                                                                         (self.size * self.divider)))
                    state['server_chunk_size'] = state['corrected_tensor_size'] // self.size
                    get_accelerator().empty_cache()
                    state['worker_error'] = torch.zeros(state['corrected_tensor_size'], device=p.device)
                    state['server_error'] = torch.zeros(state['server_chunk_size'], device=p.device)
                    # Accumulation of momentum, i.e., the u variable in the 0/1 Adam paper
                    state['momentum_accumulator'] = torch.zeros_like(p.data)
                    get_accelerator().empty_cache()
                    # self.freeze_key = True
                    if not self.initialize and dist.get_rank() == 0:
                        print("Cupy Buffers Initialized Successfully.")

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                comm_buffer = state['momentum_accumulator']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if self.initialize:
                    if self.freeze_key is False:
                        if state['step'] % state['var_interval'] == 0:
                            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        else:
                            if self.size > 1:
                                with torch.no_grad():
                                    grad_onebit = self.comm_backend_handle.compressed_allreduce(
                                        grad, state['worker_error'], state['server_error'], self.deepspeed.local_rank)
                                    if 'exp_avg_mask' in group:
                                        if grad_onebit.device != group['exp_avg_mask'].device:
                                            group['exp_avg_mask'] = group['exp_avg_mask'].to(device=grad_onebit.device)
                                        grad_onebit.mul_(group['exp_avg_mask'])
                                    exp_avg.mul_(beta1).add_(1 - beta1, grad_onebit)
                    else:
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        state['lrs'] += group['lr']
                    grad = None

                if not self.initialize:
                    if self.size > 1:
                        comm_buffer.set_(
                            self.comm_backend_handle.compressed_allreduce(comm_buffer, state['worker_error'],
                                                                          state['server_error'],
                                                                          self.deepspeed.local_rank))
                        if 'exp_avg_mask' in group:
                            if comm_buffer.device != group['exp_avg_mask'].device:
                                group['exp_avg_mask'] = group['exp_avg_mask'].to(device=comm_buffer.device)
                            comm_buffer.mul_(group['exp_avg_mask'])

                if self.initialize:
                    update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data
                    with torch.no_grad():
                        p.data.add_(-group['lr'] * update)
                        if self.freeze_key is True:
                            comm_buffer.add_(-group['lr'] * update)
                    if state['step'] % state['local_step_interval'] == 0 and self.freeze_key:
                        with torch.no_grad():
                            p.data.add_(-1 * comm_buffer)
                            comm_buffer.mul_(exp_avg_sq.sqrt() + group['eps'])
                            if self.size > 1:
                                comm_buffer.copy_(
                                    self.comm_backend_handle.compressed_allreduce(comm_buffer, state['worker_error'],
                                                                                  state['server_error'],
                                                                                  self.deepspeed.local_rank))
                                if 'exp_avg_mask' in group:
                                    if comm_buffer.device != group['exp_avg_mask'].device:
                                        group['exp_avg_mask'] = group['exp_avg_mask'].to(device=comm_buffer.device)
                                    comm_buffer.mul_(group['exp_avg_mask'])
                            exp_avg.zero_().add_(comm_buffer / state['lrs'], alpha=-1)
                            p.data.add_(comm_buffer / (exp_avg_sq.sqrt() + group['eps']))
                            comm_buffer.zero_()

                            state['lrs'] = 0

                    # According to 0/1 Adam theory, a fixed variance would allow more accurate estimation of momentum
                    # However, in practice, we can also disable the manual freezing of variance, since the interval of
                    # updating variance will increase exponentially, so that it has negligible effect on the estimation.
                    if self.freeze_key is False:
                        if state['step'] % state['var_interval'] == 0:
                            state['var_counter'] += 1
                            if state['var_counter'] == self.var_update_scaler:
                                state['var_counter'] = 0
                                state['var_interval'] *= 2
                        if (state['step'] + 1) % state['var_interval'] == 0:
                            if self.using_pipeline:
                                self.deepspeed.pipeline_enable_backward_allreduce = True
                            else:
                                self.deepspeed.enable_backward_allreduce = True
                        else:
                            if self.using_pipeline:
                                self.deepspeed.pipeline_enable_backward_allreduce = False
                            else:
                                self.deepspeed.enable_backward_allreduce = False
                    else:
                        state['local_step_counter'] += 1
                        if state['local_step_counter'] == self.local_step_scaler:
                            state['local_step_counter'] = 0
                            state['local_step_interval'] = min(self.local_step_clipper,
                                                               state['local_step_interval'] * 2)

            if not self.initialize:
                print('Pop out errors', flush=True)
                self.freeze_key = False
                state.pop('worker_error')
                state.pop('server_error')

        if not self.initialize:
            self.initialize = True
            print(f"Finished the initialization step at rank {dist.get_rank()}")
            return loss

        if self.state[self.param_groups[0]['params'][0]]['step'] > self.var_freeze_step:
            self.freeze_key = True
            if self.using_pipeline:
                self.deepspeed.pipeline_enable_backward_allreduce = False
            else:
                self.deepspeed.enable_backward_allreduce = False

        if self.freeze_key is True and self.reinitial_error_buffer is False:
            # We need to reinitialize the error buffers when local step > 1 since
            # the errors will be logged for different metrics (gradient vs. accumulated momentum).
            for group in self.param_groups:
                for p in group['params']:
                    self.state[p]['worker_error'].zero_()
                    self.state[p]['server_error'].zero_()
            self.reinitial_error_buffer = True

        return loss

    def load_state_dict(self, state_dict):
        """
        Overrides load_state_dict() to add special handling when loading checkpoints
        """
        # Because at different stage exp_avg_mask may change (e.g.,
        # BERT pre-training seqlen 128 and 512 ), we don't use the exp_avg_mask
        # in checkpoints but always use the one user provided in training script.
        # (See example in DeepSpeedExamples/bing_bert/deepspeed_train.py.)
        # Thus here we keep the exp_avg_mask unchanged when loading checkpoint
        for i, group in enumerate(self.param_groups):
            if 'exp_avg_mask' in group:
                state_dict['param_groups'][i]['exp_avg_mask'] = group['exp_avg_mask']
            elif 'exp_avg_mask' not in group and 'exp_avg_mask' in state_dict['param_groups'][i]:
                state_dict['param_groups'][i].pop('exp_avg_mask')
        super().load_state_dict(state_dict)
        if self.state[self.param_groups[0]['params'][0]]['step'] < self.var_freeze_step:
            self.var_freeze_key = False
            if (self.state[self.param_groups[0]['params'][0]]['step'] +
                    1) % self.state[self.param_groups[0]['params'][0]]['var_interval'] == 0:
                if self.using_pipeline:
                    self.deepspeed.pipeline_enable_backward_allreduce = True
                else:
                    self.deepspeed.enable_backward_allreduce = True
            else:
                if self.using_pipeline:
                    self.deepspeed.pipeline_enable_backward_allreduce = False
                else:
                    self.deepspeed.enable_backward_allreduce = False
        else:
            self.var_freeze_key = True
            if self.using_pipeline:
                self.deepspeed.pipeline_enable_backward_allreduce = False
            else:
                self.deepspeed.enable_backward_allreduce = False
        self.reinitial_error_buffer = False
        for group in self.param_groups:
            for p in group['params']:
                if 'worker_error' in self.state[p]:
                    self.state[p].pop('worker_error')
                if 'server_error' in self.state[p]:
                    self.state[p].pop('server_error')
                if 'momentum_accumulator' in self.state[p]:
                    self.state[p].pop('momentum_accumulator')
