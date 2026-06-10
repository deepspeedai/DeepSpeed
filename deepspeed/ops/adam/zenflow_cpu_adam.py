# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.ops.adam import DeepSpeedCPUAdam
import torch


class ZenFlowCPUAdam(DeepSpeedCPUAdam):

    def __init__(self, *args, overlap_step=False, **kwargs):
        super(ZenFlowCPUAdam, self).__init__(*args, **kwargs)
        self.overlap_step = overlap_step
        # In the overlapped path the optimizer step is driven natively in the ZenFlow optimizer
        # process (see ZenFlowAdam / zenflow_utils.start_optimizer_process), so this object's own
        # step() is unused there. Only the sequential (non-overlap) offload path steps here.
        if not self.overlap_step:
            self.step = self._sequential_step

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
