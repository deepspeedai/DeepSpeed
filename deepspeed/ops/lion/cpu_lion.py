# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from cpuinfo import get_cpu_info
from deepspeed.utils import logger
from deepspeed.utils.logging import should_log_le
from deepspeed.ops.op_builder import CPULionBuilder


class DeepSpeedCPULion(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self, model_params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0, fp32_optimizer_states=True):
        """Fast vectorized implementation of Lion optimizer on CPU:

        See Symbolic Discovery of Optimization Algorithms (https://doi.org/10.48550/arXiv.2302.06675).

        .. note::
                We recommend using our `config
                <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
                to allow :meth:`deepspeed.initialize` to build this optimizer
                for you.


        Arguments:
            model_params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square. (default: (0.9, 0.999))
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            full_precision_optimizer_states: creates momentum and variance in full precision regardless of
                        the precision of the parameters (default: True)
        """

        default_args = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(DeepSpeedCPULion, self).__init__(model_params, default_args)

        cpu_info = get_cpu_info()
        self.cpu_vendor = cpu_info["vendor_id_raw"].lower() if "vendor_id_raw" in cpu_info else "unknown"
        if "amd" in self.cpu_vendor:
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group['params']):
                    if p.dtype == torch.half:
                        logger.warning("FP16 params for CPULion may not work on AMD CPUs")
                        break
                else:
                    continue
                break

        self.opt_id = DeepSpeedCPULion.optimizer_id
        DeepSpeedCPULion.optimizer_id = DeepSpeedCPULion.optimizer_id + 1
        self.fp32_optimizer_states = fp32_optimizer_states
        self.ds_opt_lion = CPULionBuilder().load()

        self.ds_opt_lion.create_lion(self.opt_id, lr, betas[0], betas[1], weight_decay, should_log_le("info"))

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        self.ds_opt_lion.destroy_lion(self.opt_id)

    def __setstate__(self, state):
        super(DeepSpeedCPULion, self).__setstate__(state)
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

        # converting the fp16 params to a group of parameter
        if type(fp16_param_groups) is list:
            if type(fp16_param_groups[0]) is not list:
                fp16_param_groups = [fp16_param_groups]
        elif fp16_param_groups is not None:
            fp16_param_groups = [[fp16_param_groups]]

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                assert p.device == device, f"CPULion param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                if fp16_param_groups is not None:
                    self.ds_opt_lion.lion_update_copy(self.opt_id, state['step'], group['lr'], beta1, beta2,
                                                      group['weight_decay'], p.data, p.grad.data, state['exp_avg'],
                                                      fp16_param_groups[group_id][param_id].data)
                else:
                    self.ds_opt_lion.lion_update(self.opt_id, state['step'], group['lr'], beta1, beta2,
                                                 group['weight_decay'], p.data, p.grad.data, state['exp_avg'])
        return loss
