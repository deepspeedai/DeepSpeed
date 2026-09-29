# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Reflow CPU-Lion optimizer.

ReflowCPULion extends :class:`DeepSpeedCPULion` with the additional CPU-Lion step
variants required by the Reflow async CPU-offload optimizer, mirroring
:class:`~deepspeed.runtime.reflow.reflow_cpu_adam.ReflowCPUAdam`:

* cpu_conversion: gradients arrive on CPU in half precision and are promoted to
  FP32 inside the AVX kernel (the half-grad step methods).
* async_state: the parameter update and the optimizer-state (exp_avg) update are
  split into separate calls so the state update can run on a background worker
  (``step_state_halfgrad``).
* cpu_bucketwise: per-element-range partial updates (the reflow_lion_update_params_halfgrad C++
  entry point) so the optimizer can run on gradient buckets as they arrive during backward.

Lion keeps a single momentum tensor (``exp_avg``); there is no second moment, no
eps, and no bias correction, so the optimizer state is half the size of Adam's.

It registers its native optimizer instance in a separate Reflow registry inside the
``cpu_lion`` C++ extension (the ``reflow_*`` entry points), with its own opt_id space.
This keeps its C++ state isolated from any :class:`DeepSpeedCPULion` instances running
in the same process.
"""

import torch
from cpuinfo import get_cpu_info

from deepspeed.utils import logger
from deepspeed.utils.logging import should_log_le
from deepspeed.ops.op_builder import CPULionBuilder
from deepspeed.ops.lion.cpu_lion import DeepSpeedCPULion

_CPU_DEVICE = torch.device('cpu')


class ReflowCPULion(DeepSpeedCPULion):
    optimizer_id = 0

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 weight_decay=0,
                 fp32_optimizer_states=True,
                 num_threads=None):
        """Reflow CPU Lion.

        Same arguments as :class:`DeepSpeedCPULion` plus:

        Arguments:
            num_threads (int): number of CPU threads for the Lion update. ``None`` uses
                all available cores.
        """
        # Initialize torch.optim.Optimizer directly (rather than via DeepSpeedCPULion.__init__)
        # and register a Reflow C++ optimizer in its own registry. This avoids registering a
        # standard cpu_lion optimizer instance and keeps the native state isolated.
        default_args = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(DeepSpeedCPULion, self).__init__(model_params, default_args)

        cpu_info = get_cpu_info()
        self.cpu_vendor = cpu_info["vendor_id_raw"].lower() if "vendor_id_raw" in cpu_info else "unknown"
        if "amd" in self.cpu_vendor:
            for group in self.param_groups:
                for p in group['params']:
                    if p.dtype == torch.half:
                        logger.warning("FP16 params for ReflowCPULion may not work on AMD CPUs")
                        break
                else:
                    continue
                break

        self.opt_id = ReflowCPULion.optimizer_id
        ReflowCPULion.optimizer_id = ReflowCPULion.optimizer_id + 1
        self.fp32_optimizer_states = fp32_optimizer_states
        # None means "use all available cores"; converted to -1 at call time.
        self.num_threads = num_threads

        self.ds_opt_lion = CPULionBuilder().load()
        num_threads = int(self.num_threads) if self.num_threads is not None else -1
        self.ds_opt_lion.reflow_create_lion(self.opt_id, lr, betas[0], betas[1], weight_decay, should_log_le("info"),
                                            num_threads)

    def __del__(self):
        # Destroy the C++ object explicitly to avoid a leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker).
        module = getattr(self, 'ds_opt_lion', None)
        opt_id = getattr(self, 'opt_id', None)
        if module is not None and opt_id is not None:
            try:
                module.reflow_destroy_lion(opt_id)
            except Exception:
                pass

    def _ensure_param_state(self, p):
        """Lazily create the optimizer state (step, exp_avg) for a CPU param. Lion keeps a single
        momentum tensor -- there is no exp_avg_sq."""
        state = self.state[p]
        if 'exp_avg' not in state:
            state.setdefault('step', 0)
            state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
            state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=_CPU_DEVICE)
        return state

    def memory_fence(self):
        """Flush the non-temporal (streaming) stores so updated params/state are globally visible
        before a worker future completes. Called by the Reflow ZeRO-3 async-state worker, which
        dispatches through this method to stay optimizer-agnostic (Adam vs Lion)."""
        self.ds_opt_lion.reflow_lion_memory_fence()

    @torch.no_grad()
    def step_state_halfgrad(self, half_grad_buffers, combined_scale=1.0, element_range=None, group_hyperparams=None):
        """Optimizer-state-only Lion step from half-precision gradients: commit the FP32 master +
        exp_avg. Runs on a background worker so it overlaps with the next param update.

        The param update runs first and has already advanced ``state['step']``; the state update
        reuses it for the shared step counter.

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
                    f"CPULion param is on {p.device} and must be 'cpu', "
                    f"make sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config.")
                # The param update already advanced state['step']; the state update reuses it.
                state = self._ensure_param_state(p)

                params = p.data
                grad = grad_buffer
                exp_avg = state['exp_avg']
                if element_range is not None:
                    start, numel = element_range
                    params = params.narrow(0, start, numel)
                    grad = grad.narrow(0, start, numel)
                    exp_avg = exp_avg.narrow(0, start, numel)

                beta1, beta2 = hyperparams['betas']
                self.ds_opt_lion.reflow_lion_update_state_halfgrad(self.opt_id, state['step'], hyperparams['lr'],
                                                                   beta1, beta2, hyperparams['weight_decay'], params,
                                                                   grad, exp_avg, combined_scale)
