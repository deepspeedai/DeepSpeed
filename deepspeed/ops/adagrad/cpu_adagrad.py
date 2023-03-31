# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.ops.op_builder import CPUAdagradBuilder
from deepspeed.utils.logging import should_log_le


class DeepSpeedCPUAdagrad(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self, model_params, lr=1e-2, eps=1e-10, weight_decay=0, amsgrad=False, fp32_optimizer_states=True):

        default_args = dict(lr=lr, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(DeepSpeedCPUAdagrad, self).__init__(model_params, default_args)

        self.opt_id = DeepSpeedCPUAdagrad.optimizer_id
        DeepSpeedCPUAdagrad.optimizer_id = DeepSpeedCPUAdagrad.optimizer_id + 1
        self.fp32_optimizer_states = fp32_optimizer_states
        self.ds_opt_adagrad = CPUAdagradBuilder().load()

        self.ds_opt_adagrad.create_adagrad(self.opt_id, lr, eps, weight_decay, should_log_le("info"))

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        self.ds_opt_adagrad.destroy_adagrad(self.opt_id)

    def __setstate__(self, state):
        super(DeepSpeedCPUAdagrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None, fp16_param_groups=None):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.
            fp16_param_groups: FP16 GPU parameters to update. Performing the
                copy here reduces communication time. Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                assert p.device == device, f"CPUAdagrad param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    #memory_format=torch.preserve_format)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device='cpu')
                    #memory_format=torch.preserve_format)

                state['step'] += 1

                if p.grad.is_sparse == True:
                    sparse_param = p.sparse_mask(p.grad)
                    sparse_exp_avg_sq = state['exp_avg_sq'].sparse_mask(p.grad)
                    self.ds_opt_adagrad.adagrad_update(self.opt_id, state['step'], group['lr'], group['eps'],
                                                       group['weight_decay'], sparse_param.values(), p.grad.values(),
                                                       sparse_exp_avg_sq.values())
                    p[sparse_param.indices()] = sparse_param.values()
                    state['exp_avg_sq'][sparse_exp_avg_sq.indices()] = sparse_exp_avg_sq.values()
                    if fp16_param_groups is not None:
                        fp16_param_groups[group_id][param_id][sparse_param.indices()] = sparse_param.values()
                else:
                    if fp16_param_groups is not None:
                        self.ds_opt_adagrad.adagrad_update_copy(self.opt_id, state['step'], group['lr'], group['eps'],
                                                                group['weight_decay'], p.data, p.grad.data,
                                                                state['exp_avg_sq'],
                                                                fp16_param_groups[group_id][param_id].data)
                    else:
                        self.ds_opt_adagrad.adagrad_update(self.opt_id, state['step'], group['lr'], group['eps'],
                                                           group['weight_decay'], p.data, p.grad.data,
                                                           state['exp_avg_sq'])
        return loss
