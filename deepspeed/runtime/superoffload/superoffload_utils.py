# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
SuperOffload utilities for 1) running CPU optimizers in separate processes.

"""

from typing import Dict, Optional, Any
import torch
import torch.multiprocessing as mp
import psutil

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils import logger


def superoffload_optimizer_worker(param_queue: mp.SimpleQueue, result_queue: mp.SimpleQueue,
                                  optimizer_config: Dict[str, Any]) -> None:
    """
    This function runs in a separate process and continuously processes optimization
    tasks from the parameter queue. It creates a DeepSpeedCPUAdam optimizer and
    applies optimization steps to parameters received from the main process.

    Args:
        param_queue: Queue for receiving optimization tasks
        result_queue: Queue for sending back optimization results
        optimizer_config: Configuration dictionary for the optimizer containing
                         lr, betas, eps, weight_decay, and amsgrad parameters
    """
    # Initialize dummy parameter for optimizer creation
    cpu_tensor = torch.randn(1, device="cpu")
    cpu_param = torch.nn.Parameter(cpu_tensor)

    try:
        optimizer = DeepSpeedCPUAdam([cpu_param],
                                     lr=optimizer_config["lr"],
                                     betas=optimizer_config["betas"],
                                     eps=optimizer_config["eps"],
                                     weight_decay=optimizer_config["weight_decay"],
                                     amsgrad=optimizer_config["amsgrad"])
    except KeyError as e:
        error_msg = f"Missing required optimizer config key: {e}"
        logger.error(error_msg)
        result_queue.put({'error': error_msg})
        return

    while True:
        try:
            task = param_queue.get()

            if task is None:
                logger.debug("Received termination signal, shutting down worker")
                break

            param_data = task['param_data']
            param_grad = task['param_grad']
            param_group_id = task['param_group_id']
            sub_group_id = task['sub_group_id']
            rollback = task.get('rollback', False)

            logger.debug(f"Processing param_group_id: {param_group_id}, sub_group_id: {sub_group_id}")

            del task['param_data']
            del task['param_grad']
            task.clear()

            # Transfer gradient to CPU with pinned memory for faster transfers
            param_grad_cpu = torch.empty_like(param_grad, device='cpu', pin_memory=True)
            param_grad_cpu.copy_(param_grad, non_blocking=True)

            fp32_param = torch.nn.Parameter(param_data)
            fp32_param.grad = param_grad_cpu

            optimizer.param_groups[param_group_id]['params'] = [fp32_param]

            if rollback:
                logger.debug(f"Rolling back optimizer state for sub_group_id: {sub_group_id}")
                optimizer.rollback_subgroup(sub_group_id)
            else:
                optimizer.step_subgroup(sub_group_id)

            # Send result back to main process
            event_type = 'rollback' if rollback else 'adam_step'
            result_queue.put({
                'param_group_id': param_group_id,
                'sub_group_id': sub_group_id,
                'updated_param': fp32_param.data,
                'event_type': event_type,
            })

            # Clean up references to free memory
            optimizer.param_groups[param_group_id]['params'] = []
            del param_grad_cpu, fp32_param.grad, fp32_param, param_grad, param_data

        except KeyError as e:
            error_msg = f"Missing required task key: {e}"
            logger.error(error_msg)
            result_queue.put({'error': error_msg})
            break
        except Exception as e:
            error_msg = f"Unexpected error in worker process: {e}"
            logger.error(error_msg)
            result_queue.put({'error': error_msg})
            break

    logger.debug("Worker process terminated")


class SuperOffloadCPUOptimizer:

    def __init__(self, optimizer_config: Dict[str, Any], cpuadam_cores_perc: float = 0.8) -> None:
        if not 0 < cpuadam_cores_perc <= 1:
            raise ValueError("cpuadam_cores_perc must be between 0 and 1")

        self.mp_context = mp.get_context('spawn')
        self.param_queue = self.mp_context.SimpleQueue()
        self.result_queue = self.mp_context.SimpleQueue()

        self.cpuadam_process = self.mp_context.Process(
            target=superoffload_optimizer_worker,
            args=(self.param_queue, self.result_queue, optimizer_config),
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
                   rollback: bool = False) -> None:
        """
        Queue parameter for optimization in the worker process.
        """
        if not self.cpuadam_process.is_alive():
            raise RuntimeError("Worker process is not alive")

        self.param_queue.put({
            'param_data': fp32_param,
            'param_grad': fp32_grad,
            'param_group_id': param_group_id,
            'sub_group_id': sub_group_id,
            'rollback': rollback,
        })

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

        if 'error' in result:
            raise RuntimeError(f"Error in worker process: {result['error']}")

        # Validate event type if expected_event_type is provided
        if expected_event_type is not None:
            result_event_type = result.get('event_type')
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
