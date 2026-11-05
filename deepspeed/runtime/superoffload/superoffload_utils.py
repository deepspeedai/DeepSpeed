# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
SuperOffload utilities for 1) running CPU optimizers in separate processes.

"""

import time
from typing import Dict, Optional, Any
import torch
import torch.multiprocessing as mp
import psutil

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger


class TaskKeys:
    PARAM_DATA = "param_data"
    PARAM_GRAD = "param_grad"
    PARAM_GROUP_ID = "param_group_id"
    SUB_GROUP_ID = "sub_group_id"
    ROLLBACK = "rollback"
    HYPERPARAMS = "hyperparams"
    GRAD_SCALE = "grad_scale"
    COMMAND = "command"
    STATE = "state"


class ResultKeys:
    UPDATED_PARAM = "updated_param"
    EVENT_TYPE = "event_type"
    STATE = "state"


class EventTypes:
    ADAM_STEP = "adam_step"
    ROLLBACK = "rollback"
    GET_STATE = "get_state"
    SET_STATE = "set_state"


def _allocate_worker_grad_buffer(max_grad_numel: int,
                                 pin_memory: bool,
                                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Scratch destination for the GPU->CPU grad copy. Pinning is optional: it
    # speeds DMA but is not required for correctness, so honor
    # offload_optimizer.pin_memory rather than always page-locking.
    # dtype is the master weights dtype: the optimizer requires the gradient to match the parameter.
    if pin_memory:
        return get_accelerator().pin_empty(max_grad_numel, dtype=dtype, device='cpu')
    return torch.empty(max_grad_numel, dtype=dtype, device='cpu')


def superoffload_optimizer_worker(param_queue: mp.SimpleQueue,
                                  result_queue: mp.SimpleQueue,
                                  optimizer_config: Dict[str, Any],
                                  max_grad_numel: int,
                                  pin_memory: bool = True,
                                  adamw_mode: bool = True,
                                  fp32_optimizer_states: bool = True,
                                  grad_dtype: torch.dtype = torch.float32) -> None:
    """
    This function runs in a separate process and continuously processes optimization
    tasks from the parameter queue. It creates a DeepSpeedCPUAdam optimizer and
    applies optimization steps to parameters received from the main process.

    Args:
        param_queue: Queue for receiving optimization tasks
        result_queue: Queue for sending back optimization results
        optimizer_config: Configuration dictionary for the optimizer containing
                         lr, betas, eps, weight_decay, bias_correction, and amsgrad parameters
        max_grad_numel: Maximum number of elements expected in gradient tensors
        pin_memory: Whether to page-lock the reusable grad scratch buffer
        adamw_mode: Whether the optimizer is AdamW (decoupled weight decay) or Adam
        fp32_optimizer_states: Whether to keep the optimizer states in fp32 for low-precision parameters
        grad_dtype: dtype of the master weights and their gradients
    """
    cpu_tensor = torch.randn(1, device="cpu")
    cpu_param = torch.nn.Parameter(cpu_tensor)

    try:
        if isinstance(optimizer_config, list):
            pg_configs = optimizer_config
        else:
            pg_configs = [optimizer_config]

        first_cfg = pg_configs[0]
        optimizer = DeepSpeedCPUAdam([cpu_param],
                                     lr=first_cfg["lr"],
                                     bias_correction=first_cfg["bias_correction"],
                                     betas=first_cfg["betas"],
                                     eps=first_cfg["eps"],
                                     weight_decay=first_cfg["weight_decay"],
                                     amsgrad=first_cfg["amsgrad"],
                                     adamw_mode=adamw_mode,
                                     fp32_optimizer_states=fp32_optimizer_states)
        for cfg in pg_configs[1:]:
            dummy = torch.nn.Parameter(torch.randn(1, device="cpu"))
            optimizer.add_param_group({
                "params": [dummy],
                "lr": cfg["lr"],
                "betas": cfg["betas"],
                "eps": cfg["eps"],
                "weight_decay": cfg["weight_decay"],
                "bias_correction": cfg["bias_correction"],
                "amsgrad": cfg["amsgrad"],
            })
    except KeyError as e:
        error_msg = f"Missing required optimizer config key: {e}"
        logger.error(error_msg)
        result_queue.put({"error": error_msg})
        return

    # Pre-allocate reusable host buffer for gradients
    grad_buffer = _allocate_worker_grad_buffer(max_grad_numel, pin_memory, grad_dtype)

    while True:
        try:
            task = param_queue.get()

            if task is None:
                logger.debug("Received termination signal, shutting down worker")
                break

            # The Adam state lives in this process, so checkpoints get and set it here. Its tensors move to shared
            # memory when they are sent, without a copy that stays alive.
            command = task.get(TaskKeys.COMMAND)
            if command == EventTypes.GET_STATE:
                state = {sub_group_id: state for sub_group_id, state in optimizer.state.items() if state}
                result_queue.put({ResultKeys.EVENT_TYPE: EventTypes.GET_STATE, ResultKeys.STATE: state})
                continue
            if command == EventTypes.SET_STATE:
                optimizer.state.clear()
                optimizer.state.update(task[TaskKeys.STATE])
                result_queue.put({ResultKeys.EVENT_TYPE: EventTypes.SET_STATE})
                continue

            param_data = task[TaskKeys.PARAM_DATA]
            param_grad = task[TaskKeys.PARAM_GRAD]
            param_group_id = task[TaskKeys.PARAM_GROUP_ID]
            sub_group_id = task[TaskKeys.SUB_GROUP_ID]
            rollback = task.get(TaskKeys.ROLLBACK, False)
            hyperparams = task.get(TaskKeys.HYPERPARAMS, None)
            grad_scale = task.get(TaskKeys.GRAD_SCALE, 1.0)

            logger.debug(f"Processing param_group_id: {param_group_id}, sub_group_id: {sub_group_id}")

            del task[TaskKeys.PARAM_DATA]
            del task[TaskKeys.PARAM_GRAD]
            task.clear()

            if hyperparams is not None:
                optimizer.param_groups[param_group_id].update(hyperparams)

            grad_numel = param_grad.numel()
            if grad_numel > max_grad_numel:
                error_msg = (
                    f"Gradient size {grad_numel} exceeds pre-allocated buffer size {max_grad_numel}. "
                    f"This indicates insufficient buffer allocation. Please increase max_grad_numel parameter.")
                result_queue.put({"error": error_msg})
                break

            param_grad_cpu = grad_buffer[:grad_numel].view_as(param_grad)
            if grad_scale == 1.0:
                param_grad_cpu.copy_(param_grad, non_blocking=False)
            else:
                # A step submitted during backward gets the gradient before the loss scale is removed.
                torch.mul(param_grad, grad_scale, out=param_grad_cpu)

            fp32_param = torch.nn.Parameter(param_data)
            fp32_param.grad = param_grad_cpu

            optimizer.param_groups[param_group_id]['params'] = [fp32_param]

            if rollback:
                logger.debug(f"Rolling back optimizer state for sub_group_id: {sub_group_id}")
                optimizer.rollback_subgroup(sub_group_id)
            else:
                optimizer.step_subgroup(sub_group_id)

            # Send result back to main process
            event_type = EventTypes.ROLLBACK if rollback else EventTypes.ADAM_STEP
            result_queue.put({
                TaskKeys.PARAM_GROUP_ID: param_group_id,
                TaskKeys.SUB_GROUP_ID: sub_group_id,
                ResultKeys.UPDATED_PARAM: fp32_param.data,
                ResultKeys.EVENT_TYPE: event_type,
            })

            # Clean up references to free memory
            optimizer.param_groups[param_group_id]['params'] = []
            del param_grad_cpu, fp32_param.grad, fp32_param, param_grad, param_data

        except KeyError as e:
            error_msg = f"Missing required task key: {e}"
            logger.error(error_msg)
            result_queue.put({"error": error_msg})
            break
        except Exception as e:
            error_msg = f"Unexpected error in worker process: {e}"
            logger.error(error_msg)
            result_queue.put({"error": error_msg})
            break

    # Clean up the reusable host buffer. unpin_memory is a no-op on torch and
    # frees immediately on the native backend.
    if 'grad_buffer' in locals():
        if pin_memory:
            get_accelerator().unpin_memory(grad_buffer)
        del grad_buffer
        logger.debug("Cleaned up worker grad buffer")

    logger.debug("Worker process terminated")


class SuperOffloadCPUOptimizer:

    def __init__(self,
                 optimizer_config: Dict[str, Any],
                 cpuadam_cores_perc: float = 0.8,
                 max_grad_numel: int = 1000000,
                 pin_memory: bool = True,
                 adamw_mode: bool = True,
                 fp32_optimizer_states: bool = True,
                 grad_dtype: torch.dtype = torch.float32) -> None:
        if not 0 < cpuadam_cores_perc <= 1:
            raise ValueError("cpuadam_cores_perc must be between 0 and 1")

        self.max_grad_numel = max_grad_numel
        self.mp_context = mp.get_context('spawn')
        self.param_queue = self.mp_context.SimpleQueue()
        self.result_queue = self.mp_context.SimpleQueue()

        self.cpuadam_process = self.mp_context.Process(
            target=superoffload_optimizer_worker,
            args=(self.param_queue, self.result_queue, optimizer_config, max_grad_numel, pin_memory, adamw_mode,
                  fp32_optimizer_states, grad_dtype),
            daemon=True,
        )
        self.cpuadam_process.start()

        # Set CPU affinity for better performance isolation
        self._set_cpu_affinity(cpuadam_cores_perc)

    def _set_cpu_affinity(self, cpuadam_cores_perc: float) -> None:
        """
        Set CPU affinity for the main (Pytorch) process and worker (CPU Adam) process.

        Args:
            cpuadam_cores_perc: Percentage of cores to allocate to the worker (CPU Adam) process
        """
        try:
            current_process = psutil.Process()
            all_cores = current_process.cpu_affinity()
            num_cores = len(all_cores)

            split_idx = int((1 - cpuadam_cores_perc) * num_cores)
            pt_cores = all_cores[:split_idx]
            cpuadam_cores = all_cores[split_idx:]

            # Set affinity for main process (PyTorch)
            current_process.cpu_affinity(pt_cores)

            # Set affinity for optimizer process (CPU Adam)
            optimizer_process = psutil.Process(self.cpuadam_process.pid)
            optimizer_process.cpu_affinity(cpuadam_cores)

            logger.debug(f"Set CPU affinity - PyTorch cores: {pt_cores}, "
                         f"Optimizer cores: {cpuadam_cores}")

        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError) as e:
            logger.debug(f"Could not set CPU affinities for superoffload optimizer process: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error setting CPU affinity: {e}")

    def async_step(self,
                   param_group_id: int,
                   sub_group_id: int,
                   fp32_param: torch.Tensor,
                   fp32_grad: torch.Tensor,
                   rollback: bool = False,
                   hyperparams: Dict[str, Any] = None,
                   grad_scale: float = 1.0) -> None:
        """
        Queue parameter for optimization in the worker process.

        hyperparams (lr, betas, eps, weight_decay) replace those of the param group in the worker before the step.
        """
        if not self.cpuadam_process.is_alive():
            raise RuntimeError("Worker process is not alive")

        task = {
            TaskKeys.PARAM_DATA: fp32_param,
            TaskKeys.PARAM_GRAD: fp32_grad,
            TaskKeys.PARAM_GROUP_ID: param_group_id,
            TaskKeys.SUB_GROUP_ID: sub_group_id,
            TaskKeys.ROLLBACK: rollback,
        }
        if hyperparams is not None:
            task[TaskKeys.HYPERPARAMS] = hyperparams
        if grad_scale != 1.0:
            task[TaskKeys.GRAD_SCALE] = grad_scale
        self.param_queue.put(task)

    def _request(self, task: Dict[str, Any], event_type: str) -> Dict[str, Any]:
        """Send a command to the worker process and wait for its reply. Call it with no step in flight."""
        self.param_queue.put(task)
        while self.result_queue.empty():
            if not self.cpuadam_process.is_alive():
                raise RuntimeError("Worker process is not alive")
            time.sleep(0.001)
        return self.get_result(expected_event_type=event_type)

    def get_state(self) -> Dict[int, Dict[str, Any]]:
        """Return the Adam state of each sub-group that has taken a step, keyed by sub-group id."""
        result = self._request({TaskKeys.COMMAND: EventTypes.GET_STATE}, EventTypes.GET_STATE)
        return result[ResultKeys.STATE]

    def set_state(self, state: Dict[int, Dict[str, Any]]) -> None:
        """Replace the Adam state of the worker, for example with the state loaded from a checkpoint."""
        self._request({TaskKeys.COMMAND: EventTypes.SET_STATE, TaskKeys.STATE: state}, EventTypes.SET_STATE)

    def get_result(self, expected_event_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Get result from worker process with optional event type validation.

        Args:
            expected_event_type (str, optional): Expected event type ('adam_step' or 'rollback').
                                                If provided, validates that the result matches.
        """
        if self.result_queue.empty():
            return None

        result = self.result_queue.get()

        if "error" in result:
            raise RuntimeError(f"Error in worker process: {result['error']}")

        # Validate event type if expected_event_type is provided
        if expected_event_type is not None:
            result_event_type = result.get(ResultKeys.EVENT_TYPE)
            if result_event_type != expected_event_type:
                raise RuntimeError(f"Event type mismatch: expected '{expected_event_type}', got '{result_event_type}'")

        return result

    def close(self) -> None:
        """
        Shutdown the worker process gracefully.

        Sends termination signal to worker and waits for clean shutdown.
        If the process doesn't terminate within the timeout, it will be forcefully killed.
        """
        if not self.cpuadam_process.is_alive():
            logger.debug("Worker process already terminated")
            return

        # Send termination signal
        self.param_queue.put(None)

        # Wait for graceful shutdown
        self.cpuadam_process.join(timeout=5)

        if self.cpuadam_process.is_alive():
            logger.warning("Optimizer process did not terminate cleanly within timeout, "
                           "forcefully terminating")
            self.cpuadam_process.terminate()
            self.cpuadam_process.join(timeout=2)

            # Last resort: kill the process
            if self.cpuadam_process.is_alive():
                logger.error("Failed to terminate optimizer process, killing it")
                self.cpuadam_process.kill()
                self.cpuadam_process.join()

        logger.debug("SuperOffload CPU optimizer closed successfully")
