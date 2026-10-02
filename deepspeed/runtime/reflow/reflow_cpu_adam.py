# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Reflow CPU-Adam optimizer.

ReflowCPUAdam extends :class:`DeepSpeedCPUAdam` with the additional CPU-Adam step
variants required by the Reflow async CPU-offload optimizer:

* cpu_conversion: gradients arrive on CPU in half precision and are promoted to
  FP32 inside the AVX kernel (the half-grad step methods).
* async_state: the parameter update and the optimizer-state (exp_avg/exp_avg_sq)
  update are split into separate calls so the state update can run on a background
  worker (``step_state_halfgrad``).
* cpu_bucketwise: per-element-range partial updates (the reflow_adam_update_params_halfgrad C++
  entry point) so the
  optimizer can run on gradient buckets as they arrive during backward.

It registers its native optimizer instance in a separate Reflow registry inside the
``cpu_adam`` C++ extension (the ``reflow_*`` entry points), with its own opt_id space.
This keeps its C++ state isolated from any :class:`DeepSpeedCPUAdam` instances running
in the same process.
"""

import torch
from cpuinfo import get_cpu_info

from deepspeed.utils import logger
from deepspeed.utils.logging import should_log_le
from deepspeed.ops.op_builder import CPUAdamBuilder
from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam

_CPU_DEVICE = torch.device('cpu')


class ReflowCPUAdam(DeepSpeedCPUAdam):
    optimizer_id = 0

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=True,
                 fp32_optimizer_states=True,
                 num_threads=None):
        """Reflow CPU Adam(W).

        Same arguments as :class:`DeepSpeedCPUAdam`, requiring ``fp32_optimizer_states=True``, plus:

        Arguments:
            num_threads (int): number of CPU threads for the Adam update. ``None`` uses
                all available cores.
        """
        if not fp32_optimizer_states:
            raise ValueError("ReflowCPUAdam requires FP32 optimizer states; fp32_optimizer_states=False "
                             "is not supported.")
        # Initialize torch.optim.Optimizer directly (rather than via DeepSpeedCPUAdam.__init__)
        # and register a Reflow C++ optimizer in its own registry. This avoids registering a
        # standard cpu_adam optimizer instance and keeps the native state isolated.
        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction,
                            amsgrad=amsgrad)
        super(DeepSpeedCPUAdam, self).__init__(model_params, default_args)

        cpu_info = get_cpu_info()
        self.cpu_vendor = cpu_info["vendor_id_raw"].lower() if "vendor_id_raw" in cpu_info else "unknown"
        if "amd" in self.cpu_vendor:
            for group in self.param_groups:
                for p in group['params']:
                    if p.dtype == torch.half:
                        logger.warning("FP16 params for ReflowCPUAdam may not work on AMD CPUs")
                        break
                else:
                    continue
                break

        self.opt_id = ReflowCPUAdam.optimizer_id
        ReflowCPUAdam.optimizer_id = ReflowCPUAdam.optimizer_id + 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        # None means "use all available cores"; converted to -1 at call time.
        self.num_threads = num_threads

        self.ds_opt_adam = CPUAdamBuilder().load()
        num_threads = int(self.num_threads) if self.num_threads is not None else -1
        self.ds_opt_adam.reflow_create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode,
                                            should_log_le("info"), num_threads)

    def step(self, closure=None):
        """Direct steps cannot use the separate Reflow native registry."""
        raise NotImplementedError("ReflowCPUAdam.step() is not supported; initialize Reflow with "
                                  "deepspeed.initialize() and use engine.step().")

    def memory_fence(self):
        """Flush the non-temporal (streaming) stores so updated params/state are globally visible
        before a worker future completes. Called by the Reflow ZeRO-3 async-state worker, which
        dispatches through this method to stay optimizer-agnostic (Adam vs Lion)."""
        self.ds_opt_adam.reflow_adam_memory_fence()

    @classmethod
    def bf16_accumulate(cls, dst, src):
        """Accumulate half-precision gradients using AVX: ``dst += src``.

        Uses AVX512 for BF16 and AVX256/512 for FP16. Both tensors must be contiguous
        and share the same dtype (BF16 or FP16). Used to accumulate gradients across micro-steps.
        """
        if not hasattr(cls, '_cpu_adam_module'):
            cls._cpu_adam_module = CPUAdamBuilder().load()
        cls._cpu_adam_module.reflow_bf16_accumulate(dst, src)

    def __del__(self):
        # Destroy the C++ object explicitly to avoid a leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker).
        module = getattr(self, 'ds_opt_adam', None)
        opt_id = getattr(self, 'opt_id', None)
        if module is not None and opt_id is not None:
            try:
                module.reflow_destroy_adam(opt_id)
            except Exception:
                pass

    def _ensure_param_state(self, p):
        """Lazily create the optimizer state (step, exp_avg, exp_avg_sq) for a CPU param."""
        state = self.state[p]
        if 'exp_avg' not in state or 'exp_avg_sq' not in state:
            state.setdefault('step', 0)
            state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=_CPU_DEVICE)
            if 'exp_avg_sq' not in state:
                state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=_CPU_DEVICE)
        return state

    @torch.no_grad()
    def step_state_halfgrad(self, half_grad_buffers, combined_scale=1.0, element_range=None, group_hyperparams=None):
        """Optimizer-state-only Adam step from half-precision gradients: update exp_avg/exp_avg_sq and
        skip the param store. Runs on a background worker so it overlaps with the next param update.

        The param update runs first and has already advanced ``state['step']``; the state
        optimizer must use the same step for bias correction, so we pass ``state['step']``.

        ``half_grad_buffers`` is indexed by param-group id. Only the group being committed has a buffer; the
        other entries are ``None`` and those groups are skipped.

        ``element_range=(start, numel)`` commits only that slice of each flat param, so the caller can
        change the worker's cores between slices. The kernel is element-wise, so slicing does not change
        the result.

        ``group_hyperparams`` maps a param-group index to the hyperparameters captured when the step was
        submitted. The commit runs after step() returns, when an LR scheduler may already have changed the
        live group, so the captured values keep it on the step's own lr/betas/weight_decay.
        """
        for group_id, group in enumerate(self.param_groups):
            grad_buffer = half_grad_buffers[group_id]
            if grad_buffer is None:
                continue
            hyperparams = group_hyperparams.get(group_id, group) if group_hyperparams else group
            for p in group['params']:
                assert p.device == _CPU_DEVICE, (
                    f"CPUAdam param is on {p.device} and must be 'cpu', "
                    f"make sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config.")
                # The param update already advanced state['step']; the state optimizer reuses it
                # for bias correction.
                state = self._ensure_param_state(p)

                params = p.data
                grad = grad_buffer
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                if element_range is not None:
                    start, numel = element_range
                    params = params.narrow(0, start, numel)
                    grad = grad.narrow(0, start, numel)
                    exp_avg = exp_avg.narrow(0, start, numel)
                    exp_avg_sq = exp_avg_sq.narrow(0, start, numel)

                beta1, beta2 = hyperparams['betas']
                self.ds_opt_adam.reflow_adam_update_state_halfgrad(self.opt_id, state['step'], hyperparams['lr'],
                                                                   beta1, beta2, hyperparams['eps'],
                                                                   hyperparams['weight_decay'],
                                                                   hyperparams['bias_correction'], params, grad,
                                                                   exp_avg, exp_avg_sq, combined_scale)
