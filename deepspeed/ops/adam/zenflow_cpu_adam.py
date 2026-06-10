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

        # Collect the per-group tensors and drive the whole group through a single fused
        # native call. This keeps the per-parameter loop in C++, avoiding one
        # Python<->C++ crossing per parameter, and lets the stale snapshot be written
        # natively (no Python-side clone()).
        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        stale_params = []

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
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
                    exp_avg = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    exp_avg_sq = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    state['exp_avg'] = [exp_avg, exp_avg.clone()]
                    state['exp_avg_sq'] = [exp_avg_sq, exp_avg_sq.clone()]

                state['step'] = step_id
                params.append(p.data)
                grads.append(p.overlap_grad[now_state].data)
                exp_avgs.append(state['exp_avg'][now_state])
                exp_avg_sqs.append(state['exp_avg_sq'][now_state])
                stale_params.append(p.stale_param.data)

        if not params:
            return loss

        beta1, beta2 = group_info['betas']
        self.ds_opt_adam.adam_update_multi(self.opt_id, step_id, group_info['lr'], beta1, beta2, group_info['eps'],
                                           group_info['weight_decay'], group_info['bias_correction'], params, grads,
                                           exp_avgs, exp_avg_sqs, stale_params)
        return loss

    @torch.no_grad()
    def init_native_overlap(self, zf_affinity):
        """Create the native ZenFlowAdam handle and register every parameter group with
        it. The optimizer state (double-buffered moments) is allocated eagerly here,
        since the in-process worker needs the tensors registered before the first step.
        Replaces the multiprocessing optimizer subprocess."""
        device = torch.device('cpu')
        self.zf_handle = self.ds_opt_adam.zenflow_adam_create(self.opt_id, list(zf_affinity))

        for group in self.param_groups:
            for p in group['params']:
                if not hasattr(p, 'overlap_grad'):
                    continue
                assert p.data.device == device, "ZenFlowCPUAdam params must be on CPU"

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
                    exp_avg = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    exp_avg_sq = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    state['exp_avg'] = [exp_avg, exp_avg.clone()]
                    state['exp_avg_sq'] = [exp_avg_sq, exp_avg_sq.clone()]

                self.ds_opt_adam.zenflow_adam_register_group(self.zf_handle, p.data, p.overlap_grad[0].data,
                                                             p.overlap_grad[1].data, state['exp_avg'][0],
                                                             state['exp_avg'][1], state['exp_avg_sq'][0],
                                                             state['exp_avg_sq'][1], p.stale_param.data)

    def submit_overlap_step(self, now_state, step_id, group_infos):
        """Hand one overlapped step to the native worker (non-blocking)."""
        for group_id, group in enumerate(self.param_groups):
            self.state[group['params'][0]]['step'] = step_id
        lr, beta1, beta2, eps, weight_decay, bias_correction = [], [], [], [], [], []
        for info in group_infos:
            lr.append(info['lr'])
            beta1.append(info['betas'][0])
            beta2.append(info['betas'][1])
            eps.append(info['eps'])
            weight_decay.append(info['weight_decay'])
            bias_correction.append(1 if info['bias_correction'] else 0)
        self.ds_opt_adam.zenflow_adam_submit(self.zf_handle, now_state, step_id, lr, beta1, beta2, eps, weight_decay,
                                             bias_correction)

    def wait_overlap_step(self):
        """Block (GIL released in C++) until the last submitted step finishes."""
        self.ds_opt_adam.zenflow_adam_wait(self.zf_handle)

    def __del__(self):
        if hasattr(self, 'zf_handle'):
            self.ds_opt_adam.zenflow_adam_destroy(self.zf_handle)
        super().__del__()
