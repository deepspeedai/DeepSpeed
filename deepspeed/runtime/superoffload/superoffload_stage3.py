# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import torch
from typing import List

from deepspeed.runtime.superoffload.superoffload_utils import SuperOffloadCPUOptimizer, TaskKeys, ResultKeys, EventTypes
from deepspeed.runtime.zero.partition_parameters import Parameter, Tensor
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils.nvtx import instrument_w_nvtx
from deepspeed.utils import logger
from deepspeed.accelerator import get_accelerator

OPTIMIZER_STEP_TIMER = 'optimizer_step'

def _validate_superoffload_accelerator():
    """Validate that the current accelerator is compatible with SuperOffload."""
    accelerator = get_accelerator()
    assert accelerator.device_name() == 'cuda', (
        f"SuperOffload only supports NVIDIA CUDA GPUs, but found accelerator '{accelerator.device_name()}'.")


class SuperOffloadOptimizer_Stage3(DeepSpeedZeroOptimizer_Stage3):

    def __init__(
        self,
        module,
        init_optimizer,
        param_names,
        timers,
        ds_config,
        **kwargs,
    ):
        _validate_superoffload_accelerator()

        self.sub_group_to_param_num = {}
        self.sub_group_grad_partition_counts = {}
        self.async_cpuadam_num = 0
        self.max_grad_numel = 0

        super().__init__(module, init_optimizer, param_names, timers, ds_config, **kwargs)

        optimizer_config = {
            "lr": self.optimizer.param_groups[0]["lr"],
            "betas": self.optimizer.param_groups[0]["betas"],
            "eps": self.optimizer.param_groups[0]["eps"],
            "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
            "amsgrad": self.optimizer.param_groups[0]["amsgrad"]
        }
        cpuadam_cores_perc = kwargs.get("cpuadam_cores_perc", 0.8)
        self.superoffload_cpu_optimizer = SuperOffloadCPUOptimizer(optimizer_config=optimizer_config,
                                                                   cpuadam_cores_perc=cpuadam_cores_perc,
                                                                   max_grad_numel=self.max_grad_numel)

    def _create_fp16_sub_groups(self, params_group):

        params_group_numel = sum([param.partition_numel() for param in params_group])
        sub_group_size = self.sub_group_size

        if sub_group_size is None or sub_group_size >= params_group_numel:
            return [params_group]

        sub_groups = []
        sub_group = []
        local_sub_group_size = 0

        for param in params_group:
            sub_group.append(param)
            local_sub_group_size += param.partition_numel()

            if local_sub_group_size >= sub_group_size or id(param) == id(params_group[-1]):
                self.max_grad_numel = max(self.max_grad_numel, local_sub_group_size)
                sub_groups.append(sub_group)
                self.sub_group_to_param_num[len(sub_groups) - 1] = len(sub_group)

                sub_group = []
                local_sub_group_size = 0

        return sub_groups

    def _optimizer_step(self, sub_group_id):
        param_group_id = self.sub_group_to_group_id[sub_group_id]
        fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]

        def step_with_gradscaler(optimizer):
            if self.torch_autocast_gradscaler:
                self.torch_autocast_gradscaler.step(optimizer)
                self.torch_autocast_gradscaler.update()
            else:
                optimizer.step()

        cur_device = self.subgroup_to_device[sub_group_id]
        if cur_device != 'cpu':
            self.backup_optimizer.param_groups[param_group_id]['params'] = [fp32_param]
            step_with_gradscaler(self.backup_optimizer)
            self.backup_optimizer.param_groups[param_group_id]['params'] = []

    @instrument_w_nvtx
    def independent_gradient_partition_epilogue(self):
        super().independent_gradient_partition_epilogue()
        self.sub_group_grad_partition_counts.clear()

    @instrument_w_nvtx
    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        if self.subgroup_to_device[sub_group_id] == 'cpu':
            self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                self.fp32_partitioned_groups_flat[sub_group_id].data)
            self._unflatten_partitioned_parameters(sub_group_id)
            return

        if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
            self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                self.fp32_partitioned_groups_flat[sub_group_id].data)
            self._unflatten_partitioned_parameters(sub_group_id)
        else:
            self._partitioned_params_swap_out(sub_group_id)

    @instrument_w_nvtx
    def _reassign_or_swap_out_partitioned_parameters_async(self, sub_group_id, updated_param):
        """Asynchronously update partitioned parameters with optimized values."""
        self.fp32_partitioned_groups_flat[sub_group_id].data.copy_(updated_param, non_blocking=True)

    @instrument_w_nvtx
    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        completed_sub_groups = []

        for param, grad_partition in zip(params_to_release, grad_partitions):
            i, dest_offset, _ = self.grad_position[self.get_param_id(param)]

            if self.is_gradient_accumulation_boundary:
                self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_partition)

                fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(
                    0, dest_offset, grad_partition.numel())
                fp32_grad_tensor.copy_(grad_partition.to(dtype=self.master_weights_and_grads_dtype), non_blocking=True)

            self.sub_group_grad_partition_counts[i] = self.sub_group_grad_partition_counts.get(i, 0) + 1
            if self.sub_group_grad_partition_counts[i] == self.sub_group_to_param_num[i]:
                completed_sub_groups.append(i)

        if self.is_gradient_accumulation_boundary and completed_sub_groups:
            get_accelerator().current_stream().synchronize()

            for i in completed_sub_groups:
                if self.subgroup_to_device[i] == 'cpu' and not self.clip_grad:
                    param_group_id = self.sub_group_to_group_id[i]
                    fp32_param = self.fp32_partitioned_groups_flat[i]

                    self.superoffload_cpu_optimizer.async_step(param_group_id, i, fp32_param.data,
                                                               fp32_param.grad.data)
                    self.async_cpuadam_num += 1

                    result = self.superoffload_cpu_optimizer.get_result()
                    if result is not None:
                        self._reassign_or_swap_out_partitioned_parameters_async(result[TaskKeys.SUB_GROUP_ID],
                                                                                result[ResultKeys.UPDATED_PARAM])
                        self.async_cpuadam_num -= 1

        for param in params_to_release:
            if not get_accelerator().is_synchronized_device():
                if param.grad is not None:
                    param.grad.record_stream(get_accelerator().current_stream())
            param.grad = None

    @instrument_w_nvtx
    def step(self, closure=None):
        """
            Not supporting closure.
        """
        self._wait_for_async_operations()

        self._pre_step()
        self._partition_all_parameters()

        if self._overflow_check_and_loss_scale_update():
            if not self.clip_grad:
                self._handle_overflow_rollback()
            return

        norm_groups = self._get_norm_groups()
        scaled_global_grad_norm = torch.linalg.vector_norm(torch.stack(norm_groups))
        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale

        timer_names = set()
        timer_names.add(OPTIMIZER_STEP_TIMER)
        self.timers(OPTIMIZER_STEP_TIMER).start()

        if self.clip_grad:
            self._step_with_clipping(scaled_global_grad_norm, timer_names)
        else:
            self._step_without_clipping(scaled_global_grad_norm, timer_names)

        self.timers(OPTIMIZER_STEP_TIMER).stop()
        self._post_step(timer_names)

    def _step_without_clipping(self, scaled_global_grad_norm, timer_names):
        """Fast path: async CPU steps already completed during backward."""
        for sub_group_id, group in enumerate(self.fp16_groups):
            self._prepare_sub_group(sub_group_id, timer_names)
            self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)
            self._optimizer_step(sub_group_id)
            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
            self._release_sub_group(sub_group_id, timer_names)

    def _step_with_clipping(self, scaled_global_grad_norm, timer_names):
        """Clipping path: no async steps were done during backward,
        so we unscale+clip first, then step all sub-groups."""
        for sub_group_id, group in enumerate(self.fp16_groups):
            self._prepare_sub_group(sub_group_id, timer_names)
            self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)

            if self.subgroup_to_device[sub_group_id] == 'cpu':
                param_group_id = self.sub_group_to_group_id[sub_group_id]
                fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
                self._sync_cpu_optimizer_step(param_group_id, sub_group_id,
                                              fp32_param.data, fp32_param.grad.data)
            else:
                self._optimizer_step(sub_group_id)

            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
            self._release_sub_group(sub_group_id, timer_names)

    def _wait_for_async_operations(self, timeout_seconds=60):
        """Wait for all pending asynchronous CPU optimizer operations to complete with timeout error.

        Args:
            timeout_seconds (int): Maximum time to wait before throwing an error. Default is 60 seconds.
        """
        if self.async_cpuadam_num > 0:
            logger.info(f"[INFO] {self.async_cpuadam_num} asynchronous CPU optimizer operations pending...")
        if self.async_cpuadam_num == 0:
            return

        start_time = time.time()
        initial_pending_ops = self.async_cpuadam_num

        while self.async_cpuadam_num > 0:
            result = self.superoffload_cpu_optimizer.get_result()
            if result is None:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Throw error if we've been waiting longer than the timeout
                if elapsed_time >= timeout_seconds:
                    raise RuntimeError(
                        f"SuperOffload CPU optimizer timeout after {elapsed_time:.1f} seconds. "
                        f"Still waiting for {self.async_cpuadam_num}/{initial_pending_ops} async operations to complete. "
                        f"This indicates a deadlock or critical performance issue in the CPU optimizer.")

                time.sleep(0.001)  # 1ms sleep
                continue

            self._reassign_or_swap_out_partitioned_parameters_async(result[TaskKeys.SUB_GROUP_ID],
                                                                    result[ResultKeys.UPDATED_PARAM])
            self.async_cpuadam_num -= 1

    def _wait_for_single_async_result(self, event_type: str, timeout_seconds=60):
        """Wait for a single asynchronous CPU-Adam optimizer operation with timeout.

        Args:
            event_type (str): Type of operation expected ('adam_step' or 'rollback').
            timeout_seconds (int): Maximum time to wait before throwing an error. Default is 60 seconds.
        """
        start_time = time.time()

        while True:
            result = self.superoffload_cpu_optimizer.get_result(expected_event_type=event_type)
            if result is not None:
                self._reassign_or_swap_out_partitioned_parameters_async(result[TaskKeys.SUB_GROUP_ID],
                                                                        result[ResultKeys.UPDATED_PARAM])
                break

            current_time = time.time()
            elapsed_time = current_time - start_time

            # Throw error if we've been waiting longer than the timeout
            if elapsed_time >= timeout_seconds:
                raise RuntimeError(f"SuperOffload CPU optimizer timeout after {elapsed_time:.1f} seconds. "
                                   f"This indicates a deadlock or critical performance issue in the CPU optimizer.")

            time.sleep(0.001)  # 1ms sleep

    def _sync_cpu_optimizer_step(self,
                                 param_group_id: int,
                                 sub_group_id: int,
                                 fp32_param_data,
                                 fp32_grad_data,
                                 rollback: bool = False,
                                 timeout_seconds: int = 60):
        event_type = EventTypes.ROLLBACK if rollback else EventTypes.ADAM_STEP
        self.superoffload_cpu_optimizer.async_step(param_group_id,
                                                   sub_group_id,
                                                   fp32_param_data,
                                                   fp32_grad_data,
                                                   rollback=rollback)
        # Wait for completion
        self._wait_for_single_async_result(event_type, timeout_seconds)

    def _handle_overflow_rollback(self):
        """Handle gradient overflow by rolling back CPU optimizer states."""
        for sub_group_id, _ in enumerate(self.fp16_groups):
            if self.subgroup_to_device[sub_group_id] == 'cpu':
                param_group_id = self.sub_group_to_group_id[sub_group_id]
                fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]

                # Trigger rollback
                self._sync_cpu_optimizer_step(param_group_id,
                                              sub_group_id,
                                              fp32_param.data,
                                              fp32_param.grad.data,
                                              rollback=True)

    def _handle_gradient_clipping(self, scaled_global_grad_norm):
        """Handle gradient clipping with CPU optimizer rollback and re-optimization."""
        for sub_group_id, _ in enumerate(self.fp16_groups):
            if self.subgroup_to_device[sub_group_id] == 'cpu':
                param_group_id = self.sub_group_to_group_id[sub_group_id]
                fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]

                # Rollback CPU optimizer states
                self._sync_cpu_optimizer_step(param_group_id,
                                              sub_group_id,
                                              fp32_param.data,
                                              fp32_param.grad.data,
                                              rollback=True)

                # Clip gradients and re-optimize
                self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)

                self._sync_cpu_optimizer_step(param_group_id,
                                              sub_group_id,
                                              fp32_param.data,
                                              fp32_param.grad.data,
                                              rollback=False)

    @instrument_w_nvtx
    def check_clip_grads(self, total_norm):
        """Check if gradients need to be clipped based on the global norm."""
        unscaled_norm = total_norm / self.loss_scale
        return self.clip_grad and unscaled_norm > self.clip_grad
