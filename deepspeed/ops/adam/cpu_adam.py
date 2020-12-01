'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import math
import torch
import time
from pathlib import Path
from ..op_builder import CPUAdamBuilder


class DeepSpeedCPUAdam(torch.optim.Optimizer):
    """Fast vectorized implementation of two variations of Adam optimizer on CPU:

        - Adam: A Method for Stochastic Optimization: (https://arxiv.org/abs/1412.6980);
        - AdamW: FIXING WEIGHT DECAY REGULARIZATION IN ADAM (https://arxiv.org/abs/1711.05101v1)

       DeepSpeed CPU Adam(W) provides between 5x to 7x speedu over torch.optim.adam(W).
       In order to apply this optimizer, the model requires to have its master parameter (in FP32)
       reside on the CPU memory.

       To train on a hetrogeneous system, such as coordinating CPU and GPU, DeepSpeed offers
       the ZeRO-Offload technology which efficiently offloads the optimizer states into CPU memory,
       with minimal impact on training througput. DeepSpeedCPUAdam plays an important role to minimize
       the overhead of the optimizer's latency on CPU. Please refer to ZeRO-Offload tutorial
       (https://www.deepspeed.ai/tutorials/zero-offload/) for more information on how to enable this technology.

       For calling step function, there are two options available: (1) update optimizer's states and (2) update
       optimizer's states and copy the parameters back to GPU at the same time. We have seen that the second
       option can bring 30% higher throughput than the doing the copy separately using option one.


    Arguments:
        model_params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in DeepSpeed CPUAdam!
        adamw_mode: select between Adam and AdamW implementations (default: AdamW)
    """

    optimizer_id = 0

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=True):

        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction,
                            amsgrad=amsgrad)
        super(DeepSpeedCPUAdam, self).__init__(model_params, default_args)

        self.opt_id = DeepSpeedCPUAdam.optimizer_id
        DeepSpeedCPUAdam.optimizer_id = DeepSpeedCPUAdam.optimizer_id + 1

        self.ds_opt_adam = CPUAdamBuilder().load()

        self.ds_opt_adam.create_adam(self.opt_id,
                                     lr,
                                     betas[0],
                                     betas[1],
                                     eps,
                                     weight_decay,
                                     adamw_mode)

    def __setstate__(self, state):
        super(DeepSpeedCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None, fp16_param_groups=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0
                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data,
                                                        dtype=p.dtype,
                                                        device='cpu')
                    #memory_format=torch.preserve_format)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data,
                                                           dtype=p.dtype,
                                                           device='cpu')
                    #memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                if fp16_param_groups is not None:
                    self.ds_opt_adam.adam_update_copy(
                        self.opt_id,
                        state['step'],
                        group['lr'],
                        beta1,
                        beta2,
                        group['eps'],
                        group['weight_decay'],
                        group['bias_correction'],
                        p.data,
                        p.grad.data,
                        state['exp_avg'],
                        state['exp_avg_sq'],
                        fp16_param_groups[group_id][param_id].data)
                else:
                    self.ds_opt_adam.adam_update(self.opt_id,
                                                 state['step'],
                                                 group['lr'],
                                                 beta1,
                                                 beta2,
                                                 group['eps'],
                                                 group['weight_decay'],
                                                 group['bias_correction'],
                                                 p.data,
                                                 p.grad.data,
                                                 state['exp_avg'],
                                                 state['exp_avg_sq'])
        return loss
