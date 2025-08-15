# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch


class ZenFlowCPUAdam(DeepSpeedCPUAdam):

    def __init__(self, *args, overlap_step=False, **kwargs):
        super(ZenFlowCPUAdam, self).__init__(*args, **kwargs)
        self.overlap_step = overlap_step
        if not self.overlap_step:
            print("ZenFlowCPUAdam initialized with normal step.")
            self.step = self._sequential_step
        else:
            print("ZenFlowCPUAdam initialized with overlap step.")
            self.step = self._parallel_step

    @torch.no_grad()
    def _sequential_step(self, step_id, closure=None):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.

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

                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
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

                state['step'] = step_id
                beta1, beta2 = group['betas']
                self.ds_opt_adam.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                             group['weight_decay'], group['bias_correction'], p.data, p.grad.data,
                                             state['exp_avg'], state['exp_avg_sq'])
        return loss

    @torch.no_grad()
    def _parallel_step(self, step_id, now_state, group_info, closure=None):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        stale_param = None

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                assert p.data.is_shared(), "param.data must be in shared memory"
                if not hasattr(p, 'overlap_grad'):
                    continue

                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    # print("creating", flush=True)
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
                    exp_avg = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    exp_avg_sq = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    state['exp_avg'] = [exp_avg, exp_avg.clone()]
                    state['exp_avg_sq'] = [exp_avg_sq, exp_avg_sq.clone()]

                state['step'] = step_id
                beta1, beta2 = group_info['betas']
                self.ds_opt_adam.adam_update(self.opt_id, state['step'], group_info['lr'], beta1, beta2,
                                             group_info['eps'], group_info['weight_decay'],
                                             group_info['bias_correction'], p.data, p.overlap_grad[now_state].data,
                                             state['exp_avg'][now_state], state['exp_avg_sq'][now_state])
                p.stale_param.data.copy_(p.data.clone())
        return loss
