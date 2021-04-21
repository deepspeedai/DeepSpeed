'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
import types
import torch
import numpy as np
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class OnebitLamb(torch.optim.Optimizer):
    """Implements the 1-bit Lamb algorithm. Currently GPU-only.
    For usage example please see https://www.deepspeed.ai/tutorials/onebit-lamb/
    For technical details please see our paper https://arxiv.org/abs/2104.06069.

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
            (default: False) NOT SUPPORTED in 1-bit Lamb!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
        cuda_aware (boolean, required): Set True if the underlying MPI implementation
            supports CUDA-Aware communication. (default: False)
        comm_backend_name (string, optional): Set to 'mpi' if needed. (default: 'nccl')
        coeff_beta (float, optional): coefficient used for computing
            running averages of lamb coefficient (default: 0.9) note that you may want to
            increase or decrease this beta depending on the freeze_step you choose, as
            1/(1 - coeff_beta) should be smaller than or equal to freeze_step
        factor_max (float, optional): maximum value of scaling factor to the frozen lamb
            coefficient during compression stage (default: 4.0)
        factor_min (float, optional): minimum value of scaling factor to the frozen lamb
            coefficient during compression stage (default: 0.5)
        factor_threshold (float, optional): threshold of how much the scaling factor can
            fluctuate between steps (default: 0.1)
    .. _Large Batch Optimization for Deep Learning\: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
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
                 max_coeff=10.0,
                 min_coeff=0.01,
                 amsgrad=False,
                 cuda_aware=False,
                 comm_backend_name='nccl',
                 coeff_beta=0.9,
                 factor_max=4.0,
                 factor_min=0.5,
                 factor_threshold=0.1):

        if amsgrad:
            raise RuntimeError('1-bit Lamb does not support the AMSGrad variant.')

        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm,
                        max_coeff=max_coeff,
                        min_coeff=min_coeff)

        super(OnebitLamb, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        assert (dist.is_initialized())

        self.deepspeed = deepspeed
        self.lamb_freeze_key = False
        self.initialize = False
        self.freeze_step = freeze_step
        self.cuda_aware = cuda_aware
        self.coeff_beta = coeff_beta
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.factor_threshold = factor_threshold
        self.using_pipeline = False

        self.comm_backend_name = comm_backend_name

        # Empty initializer. Set handle based on the comm backend as follows.
        self.comm_backend_handle = None

        if self.comm_backend_name == 'nccl':
            TORCH_MAJOR = int(torch.__version__.split('.')[0])
            TORCH_MINOR = int(torch.__version__.split('.')[1])
            assert TORCH_MAJOR >= 1 and TORCH_MINOR >= 8, "Please use torch 1.8 or greater to enable NCCL backend in 1-bit Adam. Alternatively, please specify 'mpi' as the 'comm_backend_name' in config file to proceed with the MPI backend"
            assert dist.is_initialized() == True, "Please initialize the torch distributed backend."
            from deepspeed.runtime.comm.nccl import NcclBackend
            self.using_pipeline = hasattr(self.deepspeed,
                                          'pipeline_enable_backward_allreduce')
            self.comm_backend_handle = NcclBackend(self.deepspeed.mpu)

        elif self.comm_backend_name == 'mpi':
            from deepspeed.runtime.comm.mpi import MpiBackend
            self.comm_backend_handle = MpiBackend(cuda_aware)

        self.size = self.comm_backend_handle.size

        self.divider = int(self.size * 8 / np.gcd(self.size, 8))

        self.exp_avg_flat = []
        self.dummy_exp_avg = {}
        self.corrected_tensor_sizes = []
        self.server_chunk_sizes = []
        self.worker_errors = []
        self.server_errors = []

        self.lamb_coeffs = []

    def step(self, closure=None, grads=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
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

        #remove the previous stats
        del self.lamb_coeffs[:]

        if self.lamb_freeze_key:
            exp_avg_last_step = []
            for group in self.param_groups:
                exp_avg_last_step.append(
                    [self.state[p]['exp_avg'].detach().clone() for p in group['params']])
            if 'scaling_coeff' not in self.state[self.param_groups[0]['params'][0]]:
                # Compute the scaling_coeff for each momentum at the end of warmup stage.
                # This is used to reduce compression error during compression stage.
                momentum_scales = []
                for group in self.param_groups:
                    momentum_scales.append([
                        (torch.norm(self.state[p]['exp_avg']) /
                         np.sqrt(torch.numel(self.state[p]['exp_avg']))).item()
                        for p in group['params']
                    ])
                united_scale = sum([sum(x) for x in momentum_scales]) / sum(
                    [len(x) for x in momentum_scales])
                for i, group in enumerate(self.param_groups):
                    for j, p in enumerate(group['params']):
                        self.state[p][
                            'scaling_coeff'] = united_scale / momentum_scales[i][j]

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
                    raise RuntimeError('1-bit Lamb does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0 or (len(state) == 1
                                       and 'scaling_coeff' in state.keys()):
                    state['step'] = 0
                    state['lamb_coeff_freeze'] = 0.0
                    state['last_factor'] = 1.0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq_fresh'] = torch.zeros_like(p.data)

                if not self.initialize:
                    self.lamb_freeze_key = True

                exp_avg, exp_avg_sq, exp_avg_sq_fresh = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_sq_fresh']
                beta1, beta2 = group['betas']
                max_coeff = group['max_coeff']
                min_coeff = group['min_coeff']

                state['step'] += 1

                if self.lamb_freeze_key is False:
                    # warmup stage, baseline Lamb optimization
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if state['step'] == self.freeze_step:
                        exp_avg_sq_fresh.data = exp_avg_sq.detach().clone()
                    grad = None
                    if self.initialize:
                        weight_norm = p.data.pow(2).sum().sqrt()
                        update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                        if group['weight_decay'] > 0.0:
                            update += group['weight_decay'] * p.data
                        update_norm = update.pow(2).sum().sqrt()
                        lamb_coeff = 1.0
                        if weight_norm != 0 and update_norm != 0:
                            lamb_coeff = (weight_norm / update_norm).item()
                            if lamb_coeff > max_coeff:
                                lamb_coeff = max_coeff
                            if lamb_coeff < min_coeff:
                                lamb_coeff = min_coeff
                        if lamb_coeff != 1.0:
                            state['lamb_coeff_freeze'] = self.coeff_beta * state[
                                'lamb_coeff_freeze'] + (1 - self.coeff_beta) * lamb_coeff
                        self.lamb_coeffs.append(lamb_coeff)
                        with torch.no_grad():
                            p.add_(-group['lr'] * lamb_coeff * update)
                else:
                    # compression stage, update each momentum locally, then
                    # communicate based on the compressed_allreduce below
                    if self.initialize:
                        exp_avg.mul_(beta1).add_(1 - beta1, grad)
                        exp_avg.mul_(self.state[p]['scaling_coeff'])
                    grad = None

        # init fused momentum
        if len(self.exp_avg_flat) == 0:
            momentum_groups = []
            tensor_size = 0
            for group in self.param_groups:
                for p in group['params']:
                    momentum_groups.append(self.state[p]['exp_avg'])
                    tensor_size += torch.numel(p.data)
            corrected_tensor_size = tensor_size
            if tensor_size % (self.size * self.divider) != 0:
                difference = ((self.size * self.divider) - (tensor_size %
                                                            (self.size * self.divider)))
                corrected_tensor_size += difference
                self.dummy_exp_avg[0] = torch.zeros(
                    difference,
                    device=momentum_groups[0].data.device)
                momentum_groups.append(self.dummy_exp_avg[0])
            self.corrected_tensor_sizes.append(corrected_tensor_size)
            self.server_chunk_sizes.append(corrected_tensor_size // self.size)

            self.exp_avg_flat.append(
                _flatten_dense_tensors([p.detach().clone() for p in momentum_groups]))
            updated_params = _unflatten_dense_tensors(self.exp_avg_flat[0],
                                                      momentum_groups)
            for p, q in zip(momentum_groups, updated_params):
                p.data = q.data

        if self.initialize and len(self.worker_errors) == 0:
            torch.cuda.empty_cache()
            for i in range(len(self.exp_avg_flat)):
                self.worker_errors.append(
                    torch.zeros(self.corrected_tensor_sizes[i],
                                device=self.exp_avg_flat[i].device))
                self.server_errors.append(
                    torch.zeros(self.server_chunk_sizes[i],
                                device=self.exp_avg_flat[i].device))
            torch.cuda.empty_cache()

        if self.lamb_freeze_key:
            if self.size > 1:
                for i in range(len(self.exp_avg_flat)):
                    if not self.initialize:
                        torch.cuda.empty_cache()
                        self.worker_errors.append(
                            torch.zeros(self.corrected_tensor_sizes[i],
                                        device=self.exp_avg_flat[i].device))
                        self.server_errors.append(
                            torch.zeros(self.server_chunk_sizes[i],
                                        device=self.exp_avg_flat[i].device))
                        torch.cuda.empty_cache()
                        if torch.distributed.get_rank() == 0:
                            print("Cupy Buffers Initialized Successfully.")

                        self.comm_backend_handle.compressed_allreduce(
                            self.exp_avg_flat[i],
                            self.worker_errors[0],
                            self.server_errors[0],
                            self.deepspeed.local_rank)

                        if torch.distributed.get_rank() == 0:
                            print('Pop out errors', flush=True)
                        del self.worker_errors[:]
                        del self.server_errors[:]
                    else:
                        self.comm_backend_handle.compressed_allreduce(
                            self.exp_avg_flat[i],
                            self.worker_errors[i],
                            self.server_errors[i],
                            self.deepspeed.local_rank)

        if self.lamb_freeze_key and self.initialize:
            for i, group in enumerate(self.param_groups):
                bias_correction = 1 if group['bias_correction'] else 0

                for j, p in enumerate(group['params']):
                    state = self.state[p]
                    exp_avg, exp_avg_sq, exp_avg_sq_fresh = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_sq_fresh']
                    beta1, beta2 = group['betas']
                    exp_avg.div_(self.state[p]['scaling_coeff'])
                    # Because 1-bit compression cannot represent exact zero, it is required to
                    # provide a momentum mask for those params that have constant exact zeros in their
                    # momentums, otherwise the compression error would keep accumulating.
                    # For example, for BERT pre-training seq 128, bert.embeddings.position_embeddings.weight
                    # always have exact zeros in its momentum for row 129 to 512, because it only
                    # learns up to seq length 128 while the model supports up to 512 seq length.
                    # (See example in DeepSpeedExamples/bing_bert/deepspeed_train.py about how
                    # to add this exp_avg_mask for BERT pre-training.)
                    if 'exp_avg_mask' in group:
                        if exp_avg.device != group['exp_avg_mask'].device:
                            group['exp_avg_mask'] = group['exp_avg_mask'].to(
                                device=exp_avg.device)
                        exp_avg.mul_(group['exp_avg_mask'])

                    grad_reconstruct = ((exp_avg - exp_avg_last_step[i][j] * beta1) /
                                        (1 - beta1))
                    exp_avg_sq_fresh.mul_(beta2).addcmul_(1 - beta2,
                                                          grad_reconstruct,
                                                          grad_reconstruct)
                    denom = exp_avg_sq.sqrt() + group['eps']
                    update_prelim = exp_avg / denom

                    if group['weight_decay'] > 0.0:
                        update = update_prelim + group['weight_decay'] * p.data
                    else:
                        update = update_prelim

                    lamb_coeff = 1.0
                    update_norm = update.pow(2).sum().sqrt()
                    denom_real = exp_avg_sq_fresh.sqrt() + group['eps']
                    factor = (denom / denom_real).max().item()
                    if group['weight_decay'] > 0.0:
                        update_ratio = min(1.0,
                                           (update_prelim.pow(2).sum().sqrt() /
                                            update_norm).item())
                        factor = factor * update_ratio + (1.0 - update_ratio)
                    if factor > self.factor_max:
                        factor = self.factor_max
                    if factor < self.factor_min:
                        factor = self.factor_min
                    if factor > state['last_factor'] * (1.0 + self.factor_threshold):
                        factor = state['last_factor'] * (1.0 + self.factor_threshold)
                    if factor < state['last_factor'] * (1.0 - self.factor_threshold):
                        factor = state['last_factor'] * (1.0 - self.factor_threshold)
                    state['last_factor'] = factor
                    lamb_coeff = state['lamb_coeff_freeze'] * factor
                    self.lamb_coeffs.append(lamb_coeff)
                    with torch.no_grad():
                        p.add_(-group['lr'] * lamb_coeff * update)
            del exp_avg_last_step[:]
            exp_avg_last_step = None

        if not self.initialize:
            self.lamb_freeze_key = False
            self.initialize = True
            print(
                f"Finished the initialization step at rank {torch.distributed.get_rank()}"
            )
            return loss

        if self.lamb_freeze_key is False:
            if state['step'] >= self.freeze_step:
                print('OnebitLamb - starting compressed communication')
                self.lamb_freeze_key = True
                if self.using_pipeline:
                    self.deepspeed.pipeline_enable_backward_allreduce = False
                else:
                    self.deepspeed.enable_backward_allreduce = False

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
            elif 'exp_avg_mask' not in group and 'exp_avg_mask' in state_dict[
                    'param_groups'][i]:
                state_dict['param_groups'][i].pop('exp_avg_mask')
        super().load_state_dict(state_dict)
        # need to reset the fused momentum since loading states will break the linking
        del self.exp_avg_flat[:]
        self.dummy_exp_avg.clear()
        del self.corrected_tensor_sizes[:]
        del self.server_chunk_sizes[:]
        if self.state[self.param_groups[0]['params'][0]]['step'] < self.freeze_step:
            if torch.distributed.get_rank() == 0:
                print("Checkpoint loaded and OnebitLamb warmup stage starts/continues.")
            if self.lamb_freeze_key is True:
                self.lamb_freeze_key = False
                if self.using_pipeline:
                    self.deepspeed.pipeline_enable_backward_allreduce = True
                else:
                    self.deepspeed.enable_backward_allreduce = True
            for group in self.param_groups:
                for p in group['params']:
                    self.state[p]['lamb_coeff_freeze'] = 0.0
                    self.state[p]['last_factor'] = 1.0
                    if 'scaling_coeff' in self.state[p]:
                        self.state[p].pop('scaling_coeff')
        else:
            if torch.distributed.get_rank() == 0:
                print(
                    "Checkpoint loaded and OnebitLamb compression stage starts/continues."
                )
            if self.lamb_freeze_key is False:
                self.lamb_freeze_key = True
                if self.using_pipeline:
                    self.deepspeed.pipeline_enable_backward_allreduce = False
                else:
                    self.deepspeed.enable_backward_allreduce = False
        # We reset the compression errors when loading checkpoints for 3 reasons:
        # 1) The worker and server error at each GPU are distinct, so in current implementation
        # only rank 0's errors are saved in the checkpoint. Thus we have to reset the errors.
        # If we want to save them correctly we need O(num_gpu*model_size) memory in order to
        # gather all the error, which is a very large memory requirement. It's possible to save
        # them in a distributed way, but it will make the checkpoint saving/loading much more complicated.
        # 2) Even if we are able to save the compression errors correctly, you need to have the
        # exact same number of GPUs in order to load them correctly.
        # 3) We verified on BERT pre-training that occasionally resetting the compression error
        # at checkpoint loading does not affect the convergence.
        # However, please avoid frequent checkpoint loading which could break the error
        # compensation mechanism thus affect the convergence.
        del self.worker_errors[:]
        del self.server_errors[:]

    def get_lamb_coeffs(self):
        return self.lamb_coeffs
