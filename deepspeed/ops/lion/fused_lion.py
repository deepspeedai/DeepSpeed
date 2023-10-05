# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
This file is modified from fused_adam.py
"""

import torch
from .multi_tensor_apply import MultiTensorApply

multi_tensor_applier = MultiTensorApply(2048 * 32)
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import FusedLionBuilder


class FusedLion(torch.optim.Optimizer):
    """Implements Lion algorithm.

    Currently GPU-only.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Symbolic Discovery of Optimization Algorithms:
        https://doi.org/10.48550/arXiv.2302.06675
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0., set_grad_none=True):

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(FusedLion, self).__init__(params, defaults)
        self.set_grad_none = set_grad_none

        fused_lion_cuda = FusedLionBuilder().load()
        # Skip buffer
        self._dummy_overflow_buf = get_accelerator().IntTensor([0])
        self.multi_tensor_lion = fused_lion_cuda.multi_tensor_lion

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedLion, self).zero_grad()

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError('FusedLion has been updated.')
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' not in group:
                group['step'] = 0

            # create lists for multi-tensor apply
            g_16, p_16, m_16 = [], [], []
            g_bf, p_bf, m_bf = [], [], []
            g_32, p_32, m_32 = [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise NotImplementedError('FusedLion does not support sparse gradients')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # DeepSpeed ZeRO 3 processes each subgroup a time, so we need to keep tracking step count for each tensor separately.
                    # While this is not an issue for ZeRO 1 & 2, since they apply a single optimization step to the whole param group at the same time.
                    # In order to keep backward compatibility for the existing checkpoints, we use group['state'] to initialize state['step'] if it exists.
                    state['step'] = group.get('step', 0)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state['exp_avg'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                else:
                    raise RuntimeError('FusedLion only support fp16, bf16 and fp32.')

            if len(g_16) > 0:
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_lion, self._dummy_overflow_buf, [g_16, p_16, m_16], group['lr'],
                                     beta1, beta2, state['step'], group['weight_decay'])

            if len(g_bf) > 0:
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_lion, self._dummy_overflow_buf, [g_bf, p_bf, m_bf], group['lr'],
                                     beta1, beta2, state['step'], group['weight_decay'])

            if len(g_32) > 0:
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_lion, self._dummy_overflow_buf, [g_32, p_32, m_32], group['lr'],
                                     beta1, beta2, state['step'], group['weight_decay'])

        return loss
