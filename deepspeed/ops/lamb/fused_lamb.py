# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex/optimizer/fused_adam and implements the LAMB optimizer
"""
import types
import torch
from deepspeed.ops.op_builder import FusedLambBuilder


class FusedLamb(torch.optim.Optimizer):
    """Implements the LAMB algorithm. Currently GPU-only.

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
    https://arxiv.org/abs/1904.00962

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        bias_correction (bool, optional): bias correction (default: True)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip global grad norm
            (default: 0.0)
        max_coeff(float, optional): maximum value of the lamb coefficient (default: 10.0)
        min_coeff(float, optional): minimum value of the lamb coefficient (default: 0.01)
        amsgrad (boolean, optional): NOT SUPPORTED in FusedLamb!
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0.,
                 max_grad_norm=0.,
                 max_coeff=10.0,
                 min_coeff=0.01,
                 amsgrad=False):
        self.fused_lamb_cuda = FusedLambBuilder().load()

        if amsgrad:
            raise RuntimeError('FusedLamb does not support the AMSGrad variant.')
        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm,
                        max_coeff=max_coeff,
                        min_coeff=min_coeff)
        super(FusedLamb, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        self.lamb_coeffs = []

    def step(self, closure=None, grads=None, output_params=None, scale=1., grad_norms=None):
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

        if output_params is None:
            output_params_group = [None] * len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0]) != list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        #remove the previous coeffs
        del self.lamb_coeffs[:]

        for group, grads_this_group, output_params_this_group, grad_norm_group in zip(
                self.param_groups, grads_group, output_params_group, grad_norms):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])
            if output_params_this_group is None:
                output_params_this_group = [None] * len(group['params'])

            if grad_norm_group is None:
                grad_norm_group = [None] * len(group['params'])
            elif not isinstance(grad_norm_group, list):
                grad_norm_group = [grad_norm_group]

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad, output_param, grad_norm in zip(group['params'], grads_this_group, output_params_this_group,
                                                        grad_norm_group):

                # compute combined scale factor for this group
                combined_scale = scale
                if group['max_grad_norm'] > 0:
                    # norm is in fact norm*scale
                    clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                    if clip > 1:
                        combined_scale = clip * scale

                #note: p.grad should not ever be set for correct operation of mixed precision optimizer that sometimes sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('FusedLamb does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                max_coeff = group['max_coeff']
                min_coeff = group['min_coeff']

                state['step'] += 1

                out_p = torch.tensor([], dtype=torch.float) if output_param is None else output_param
                lamb_coeff = self.fused_lamb_cuda.lamb(p.data, out_p, exp_avg, exp_avg_sq, grad, group['lr'], beta1,
                                                       beta2, max_coeff, min_coeff, group['eps'], combined_scale,
                                                       state['step'], self.eps_mode, bias_correction,
                                                       group['weight_decay'])
                self.lamb_coeffs.append(lamb_coeff)
        return loss

    def get_lamb_coeffs(self):
        lamb_coeffs = [lamb_coeff.item() for lamb_coeff in self.lamb_coeffs]
        return lamb_coeffs
