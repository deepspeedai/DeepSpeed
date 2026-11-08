# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
SuperOffload's CPU Adam, which steps the CPU sub-groups in a thread of the training process.
"""

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List

import torch

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.op_builder import CPUAdamBuilder
from deepspeed.utils import logger
from deepspeed.utils.logging import should_log_le


def _bind_thread(cores: List[int]) -> None:
    """Run the calling thread on these cores; the threads it starts later, such as its OpenMP threads, inherit them."""
    if not cores:
        return
    try:
        os.sched_setaffinity(0, cores)
    except (AttributeError, OSError) as e:
        logger.debug(f"Could not set the CPU affinity of a SuperOffload thread: {e}")


def _kernel_args(hyperparams: Dict[str, Any]):
    beta1, beta2 = hyperparams["betas"]
    return (hyperparams["lr"], beta1, beta2, hyperparams["eps"], hyperparams["weight_decay"],
            hyperparams["bias_correction"])


class SuperOffloadCPUAdam:
    """DeepSpeed's CPU Adam for the CPU sub-groups, run by one thread of the training process on the cores given to
    CPU Adam.

    The native kernels release the GIL, so the training thread goes on queueing backward kernels while a sub-group
    steps. The thread runs what is submitted in order. The caller keeps the Adam state of each sub-group and passes it
    to each step.
    """

    def __init__(self, defaults: Dict[str, Any], adamw_mode: bool, fp32_optimizer_states: bool,
                 cpuadam_cores_perc: float) -> None:
        """defaults: lr, betas, eps and weight_decay to create the native optimizer with; each step passes its own."""
        if not 0 < cpuadam_cores_perc <= 1:
            raise ValueError("cpuadam_cores_perc must be between 0 and 1")
        try:
            cores = sorted(os.sched_getaffinity(0))
        except AttributeError:  # not Linux
            cores = []
        split = int((1 - cpuadam_cores_perc) * len(cores))
        # The training thread keeps the first cores and CPU Adam gets the others, so they do not compete.
        _bind_thread(cores[:split])
        self._thread = ThreadPoolExecutor(max_workers=1,
                                          thread_name_prefix="SuperOffloadCPUAdam",
                                          initializer=_bind_thread,
                                          initargs=(cores[split:], ))

        self.fp32_optimizer_states = fp32_optimizer_states
        self._ops = CPUAdamBuilder().load()
        self._opt_id = DeepSpeedCPUAdam.optimizer_id
        DeepSpeedCPUAdam.optimizer_id += 1
        self._ops.create_adam(self._opt_id, defaults["lr"], defaults["betas"][0], defaults["betas"][1],
                              defaults["eps"], defaults["weight_decay"], adamw_mode, should_log_le("info"))
        # Holds the gradient adam_rollback reads: in fp32, with the loss scale removed.
        self._grad_buffer = None

    def submit(self, fn: Callable, *args) -> Future:
        """Run fn(*args) in the CPU Adam thread, after what was submitted before it."""
        return self._thread.submit(fn, *args)

    def step(self,
             state: Dict[str, Any],
             weights: torch.Tensor,
             grad: torch.Tensor,
             hyperparams: Dict[str, Any],
             grad_scale: float = 1.0) -> None:
        """Step the weights on grad * grad_scale, as DeepSpeedCPUAdam does; state starts empty."""
        if not state:
            state["step"] = 0
            state_dtype = torch.float if self.fp32_optimizer_states else weights.dtype
            state["exp_avg"] = torch.zeros_like(weights, dtype=state_dtype)
            state["exp_avg_sq"] = torch.zeros_like(weights, dtype=state_dtype)
        state["step"] += 1
        # adam_update reads bf16 gradients of fp32 weights as they are, so they go to the CPU at half the bytes, and
        # multiplies them by grad_scale as it reads them.
        self._ops.adam_update(self._opt_id, state["step"], *_kernel_args(hyperparams), weights, grad, state["exp_avg"],
                              state["exp_avg_sq"], grad_scale)

    def rescale_step(self, state: Dict[str, Any], weights: torch.Tensor, grad: torch.Tensor,
                     hyperparams: Dict[str, Any], old_scale: float, new_scale: float) -> None:
        """Turn the last step of the weights, taken on grad * old_scale with these hyperparameters, into the step on
        grad * new_scale, in one pass over the weights and moments."""
        self._ops.adam_rescale_step(self._opt_id, state["step"], *_kernel_args(hyperparams), weights, grad,
                                    state["exp_avg"], state["exp_avg_sq"], old_scale, new_scale)

    def rollback(self,
                 state: Dict[str, Any],
                 weights: torch.Tensor,
                 grad: torch.Tensor,
                 hyperparams: Dict[str, Any],
                 grad_scale: float = 1.0) -> None:
        """Undo the last step of the weights, which used this gradient, grad_scale and these hyperparameters."""
        # adam_rollback reads gradients in the weights' dtype only.
        self._ops.adam_rollback(self._opt_id, state["step"], *_kernel_args(hyperparams), weights,
                                self._unscaled(grad, weights.dtype, grad_scale), state["exp_avg"], state["exp_avg_sq"])
        state["step"] -= 1

    def _unscaled(self, grad: torch.Tensor, dtype: torch.dtype, grad_scale: float) -> torch.Tensor:
        if grad_scale == 1.0 and grad.dtype == dtype:
            return grad
        buffer = self._grad_buffer
        if buffer is None or buffer.dtype != dtype or buffer.numel() < grad.numel():
            buffer = self._grad_buffer = torch.empty(grad.numel(), dtype=dtype)
        unscaled = buffer[:grad.numel()].view_as(grad)
        # The step of a sub-group submitted during backward read the gradient before the loss scale is removed.
        torch.mul(grad, grad_scale, out=unscaled)
        return unscaled

    def close(self) -> None:
        """Finish what was submitted, stop the thread and free the native optimizer."""
        self._thread.shutdown()
        self._ops.destroy_adam(self._opt_id)
