'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import types
import torch
import importlib
import numpy as np
import time
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from deepspeed.utils.logging import logger

from mpi4py import MPI
from deepspeed.runtime.custom_collectives import gather_cuda, gather_host, allgather_cuda, allgather_host


class OnebitAdam(torch.optim.Optimizer):
    """Implements the 1-bit Adam algorithm. Currently GPU-only.
    For usage example please see, TODO DeepSpeed Tutorial
    It has been proposed in APMSqueeze (https://arxiv.org/abs/2008.11343)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        freeze_step (int, optional): Number of steps for warmup (uncompressed)
            stage before we start using compressed communication. (default 100000)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_coeff(float, optional): maximum value of the lamb coefficient (default: 10.0)
        min_coeff(float, optional): minimum value of the lamb coefficient (default: 0.01)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in 1-bit Adam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
        cuda_aware (boolean, required): Set True if the underlying MPI implementation
            supports CUDA-Aware communication. (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params,
                 deepspeed=None,
                 lr=1e-3,
                 freeze_step=100000,
                 bias_correction=True,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0.,
                 max_grad_norm=0.,
                 amsgrad=False,
                 cuda_aware=False):

        if amsgrad:
            raise RuntimeError('1-bit Adam does not support the AMSGrad variant.')
        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)

        super(OnebitAdam, self).__init__(params, defaults)
        from mpi4py import MPI
        self.eps_mode = 0 if eps_inside_sqrt else 1

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.comm_time = 0.0
        self.step_time = 0.0
        self.ave_step = 1
        self.bk_time = 0.0
        self.divider = int(self.size * 8 / np.gcd(self.size, 8))
        self.deepspeed = deepspeed
        self.adam_freeze_key = False
        self.initialize = False
        self.freeze_step = freeze_step
        self.cuda_aware = cuda_aware

    def torch2cupy(self, tensor):
        return cupy.fromDlpack(to_dlpack(tensor))

    def cupy2torch(self, cupy_tensor):
        return from_dlpack(cupy_tensor.toDlpack())

    def compress_by_chunk(self, cupy_bool_tensor, num_chunks):
        packed_sign = cupy.packbits(cupy_bool_tensor)
        sign_list_packed = cupy.split(packed_sign, num_chunks)
        cupy.cuda.get_current_stream().synchronize()
        return sign_list_packed

    def Compressed_Allreduce(self,
                             buffer_m: torch.tensor,
                             worker_error,
                             server_error,
                             rank,
                             world_size,
                             comm,
                             local_rank):

        all_start_time = time.time()
        original_size = buffer_m.numel()
        cupy.cuda.Device(local_rank).use()

        if torch.numel(buffer_m) != torch.numel(worker_error):
            empty_tensor = torch.zeros(torch.numel(worker_error) - torch.numel(buffer_m),
                                       device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])

        buffer_m.add_(worker_error)
        worker_scale = torch.norm(buffer_m) / np.sqrt(torch.numel(buffer_m))
        sign_buffer_m = buffer_m.sign().add_(1).bool()
        sign_buffer_m = sign_buffer_m.float()
        sign_buffer_m.add_(-0.5).mul_(2.0)
        worker_error.set_((buffer_m - worker_scale * sign_buffer_m))
        sign_buffer_m = None

        compensated_buffer_m = buffer_m
        compensated_buffer_m.sign_()
        compensated_buffer_m = compensated_buffer_m.add_(1).bool()
        cupy_worker_scale = self.torch2cupy(worker_scale)
        cupy_compensated_buffer_m = self.torch2cupy(compensated_buffer_m)
        compensated_buffer_m = None

        cupy_sign_list_packed = self.compress_by_chunk(cupy_compensated_buffer_m,
                                                       world_size)
        cupy_compensated_buffer_m = None

        cupy_recvbuf_sign = cupy.zeros([world_size,
                                        cupy_sign_list_packed[rank].size],
                                       dtype=cupy_sign_list_packed[0].dtype)
        cupy_recvbuf_scale = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)

        # Communication Phase 1
        gather_start = time.time()
        if self.cuda_aware:
            gather_cuda(rank,
                        world_size,
                        comm,
                        cupy_sign_list_packed,
                        cupy_recvbuf_sign,
                        cupy_worker_scale,
                        cupy_recvbuf_scale)
        else:
            cupy_sign_list_packed, cupy_recvbuf_sign, cupy_worker_scale, cupy_recvbuf_scale = gather_host(rank,
               world_size,
               comm,
               cupy_sign_list_packed,
               cupy_recvbuf_sign,
               cupy_worker_scale,
               cupy_recvbuf_scale)
        gather_end = time.time()

        cupy_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
            world_size,
            -1)
        cupy_recvbuf_sign = None
        unpacked_sign = self.cupy2torch(cupy_unpacked_sign).float()
        cupy_unpacked_sign = None
        unpacked_sign = unpacked_sign.add_(-0.5).mul_(2.0)
        worker_scale = self.cupy2torch(cupy_recvbuf_scale).mul_(1 / world_size)
        compensated_server_m = unpacked_sign.mul_(worker_scale).sum(0)
        unpacked_sign = None

        compensated_server_m.add_(server_error)
        server_scale = torch.norm(compensated_server_m) / np.sqrt(
            compensated_server_m.numel())
        sign_server_m = compensated_server_m.sign().add_(1).bool()
        sign_server_m = sign_server_m.float()
        sign_server_m.add_(-0.5).mul_(2.0)
        server_error.set_(compensated_server_m - server_scale * sign_server_m)
        sign_server_m = None

        compensated_server_m.sign_()
        compensated_server_m = compensated_server_m.add_(1).bool()
        cupy_server_scale = self.torch2cupy(server_scale)
        cupy_compensated_server_m = self.torch2cupy(compensated_server_m)
        compensated_server_m = None

        cupy_server_sign_packed = self.compress_by_chunk(cupy_compensated_server_m, 1)

        cupy_recvbuf_sign_server = cupy.zeros(
            [world_size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_sign_list_packed[0].dtype)
        cupy_recvbuf_scale_server = cupy.zeros([world_size,
                                                1],
                                               dtype=cupy_worker_scale.dtype)

        # Communication Phase 2
        if self.cuda_aware:
            allgather_cuda(comm,
                           cupy_server_sign_packed[0],
                           cupy_recvbuf_sign_server,
                           cupy_server_scale,
                           cupy_recvbuf_scale_server)
        else:
            cupy_server_sign_packed[0], cupy_recvbuf_sign_server, cupy_server_scale, cupy_recvbuf_scale_server = allgather_host(comm,
                  cupy_server_sign_packed[0],
                  cupy_recvbuf_sign_server,
                  cupy_server_scale,
                  cupy_recvbuf_scale_server)

        cupy_server_unpacked_sign = (cupy.unpackbits(
            cupy_recvbuf_sign_server.flatten())).reshape(world_size,
                                                         -1)
        cupy_recvbuf_sign_server = None

        server_unpacked_sign = self.cupy2torch(cupy_server_unpacked_sign)
        cupy_server_unpacked_sign = None

        server_unpacked_sign = server_unpacked_sign.float().add_(-0.5).mul_(2.0)
        server_scale = self.cupy2torch(cupy_recvbuf_scale_server)
        buffer_m = server_unpacked_sign.mul_(server_scale).flatten()[0:original_size]

        return buffer_m

    def step(self, closure=None, grads=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced recision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        gather_time = 0
        allgather_time = 0
        all_time = 0

        if self.adam_freeze_key is False:
            v_diff_buffer = 0.0

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
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['tensor_size'] = torch.numel(p.data)
                    state['corrected_tensor_size'] = state['tensor_size']

                    if state['tensor_size'] % (self.size * self.divider) != 0:
                        state['corrected_tensor_size'] += ((self.size * self.divider) -
                                                           (state['tensor_size'] %
                                                            (self.size * self.divider)))
                    state['server_chunk_size'] = state[
                        'corrected_tensor_size'] // self.size

                if not self.initialize or (self.adam_freeze_key
                                           and 'worker_error' not in state.keys()):
                    torch.cuda.empty_cache()
                    state['worker_error'] = torch.zeros(state['corrected_tensor_size'],
                                                        device=p.device)
                    state['server_error'] = torch.zeros(state['server_chunk_size'],
                                                        device=p.device)
                    torch.cuda.empty_cache()
                    self.adam_freeze_key = True
                    if not self.initialize and torch.distributed.get_rank() == 0:
                        print("Cupy Buffers Initialized Successfully.")

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if self.adam_freeze_key is False:
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    grad = None
                    if self.initialize:
                        update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])

                else:
                    if 'non_freeze' in group.keys() and group['non_freeze'] is True:
                        dist.all_reduce(grad)
                        grad.mul_(1 / dist.get_world_size())
                        exp_avg.mul_(beta1).add(1 - beta1, grad)
                        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                        grad = None
                    else:
                        if self.initialize is True:
                            exp_avg.mul_(beta1).add_(1 - beta1, grad)
                        grad = None

                        if self.size > 1:
                            exp_avg.set_(
                                self.Compressed_Allreduce(exp_avg,
                                                          state['worker_error'],
                                                          state['server_error'],
                                                          self.rank,
                                                          self.size,
                                                          self.comm,
                                                          self.deepspeed.local_rank))
                    if self.initialize:
                        update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])

                if self.initialize:
                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data
                    with torch.no_grad():
                        p.add_(-group['lr'] * update)

            if not self.initialize:
                print('Pop out errors', flush=True)
                state.pop('worker_error')
                state.pop('server_error')

        if not self.initialize:
            self.adam_freeze_key = False
            self.initialize = True
            print(
                f"Finished the initialization step at rant {torch.distributed.get_rank()}"
            )
            return loss

        if self.adam_freeze_key is False:
            if state['step'] >= self.freeze_step:
                self.adam_freeze_key = True
                self.deepspeed.enable_backward_allreduce = False

        return loss
