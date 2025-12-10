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

        self.sub_group_to_param_num = {}
        self.params_in_ipg_bucket_buffer = []
        self._cur_bucket_index = -1
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

    def reduce_independent_p_g_buckets_and_remove_grads(self, param):
        comm_dtype = self.get_param_comm_dtype(param)
        bucket = self.ipg_buckets[comm_dtype]
        i, _, _ = self.grad_position[self.get_param_id(param)]

        if len(bucket.params) == 0:
            self._cur_bucket_index = i
            if getattr(param, "ds_grad_is_ready", True):
                self._DeepSpeedZeroOptimizer_Stage3__add_grad_to_ipg_bucket(param)

            # If this is a single-parameter sub-group, reduce immediately
            if self.sub_group_to_param_num[self._cur_bucket_index] == 1:
                self._DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads(comm_dtype)

        elif i != self._cur_bucket_index:
            # Parameter belongs to different sub-group, buffer it
            self.params_in_ipg_bucket_buffer.append(param)
        else:
            # Parameter belongs to current bucket
            if getattr(param, "ds_grad_is_ready", True):
                self._DeepSpeedZeroOptimizer_Stage3__add_grad_to_ipg_bucket(param)

            # Check if bucket is complete
            if self.sub_group_to_param_num[self._cur_bucket_index] == len(bucket.params):
                self._DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads(comm_dtype)

                # Process buffered parameters
                while self.params_in_ipg_bucket_buffer:
                    buffered_param = self.params_in_ipg_bucket_buffer.pop(0)
                    ci, _, _ = self.grad_position[self.get_param_id(buffered_param)]
                    self._cur_bucket_index = ci
                    if getattr(buffered_param, "ds_grad_is_ready", True):
                        self._DeepSpeedZeroOptimizer_Stage3__add_grad_to_ipg_bucket(buffered_param)

    @instrument_w_nvtx
    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        if self.subgroup_to_device[sub_group_id] == 'cpu':
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
        # print("[DEBUG] partition_grads called")
        buffers = []
        device_buffers = {}
        buffer_numel_min = {}
        buffer_numel_max = {}

        for param, grad_partition in zip(params_to_release, grad_partitions):
            i, dest_offset, _ = self.grad_position[self.get_param_id(param)]

            if self.is_gradient_accumulation_boundary:
                self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_partition)

            buffer_numel = grad_partition.numel()
            buffers.append(grad_partition)

            if i not in device_buffers:
                device_buffers[i] = []
            device_buffers[i].append(grad_partition)

            if i not in buffer_numel_min:
                buffer_numel_min[i] = dest_offset
                buffer_numel_max[i] = dest_offset + buffer_numel
            else:
                buffer_numel_min[i] = min(buffer_numel_min[i], dest_offset)
                buffer_numel_max[i] = max(buffer_numel_max[i], dest_offset + buffer_numel)

        if self.is_gradient_accumulation_boundary:
            for i in buffer_numel_min.keys():
                fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(
                    0, buffer_numel_min[i], buffer_numel_max[i] - buffer_numel_min[i])
                concatenated_buffer = torch.cat(device_buffers[i], dim=0).to(dtype=self.master_weights_and_grads_dtype)

                if self.subgroup_to_device[i] == 'cpu':
                    # Trigger asynchronous CPU optimization
                    param_group_id = self.sub_group_to_group_id[i]
                    fp32_param = self.fp32_partitioned_groups_flat[i]

                    self.superoffload_cpu_optimizer.async_step(param_group_id, i, fp32_param.data,
                                                               concatenated_buffer.data)
                    self.async_cpuadam_num += 1

                    # Check for completed async operations
                    result = self.superoffload_cpu_optimizer.get_result()
                    if result is not None:
                        self._reassign_or_swap_out_partitioned_parameters_async(result[TaskKeys.SUB_GROUP_ID],
                                                                                result[ResultKeys.UPDATED_PARAM])
                        self.async_cpuadam_num -= 1

                    fp32_grad_tensor.copy_(concatenated_buffer, non_blocking=True)
                else:
                    fp32_grad_tensor.copy_(concatenated_buffer, non_blocking=True)

        # Clean up parameter gradients
        for param in params_to_release:
            if not get_accelerator().is_synchronized_device():
                param.grad.record_stream(get_accelerator().current_stream())
            param.grad = None

    @instrument_w_nvtx
    def step(self, closure=None):
        """
            Not supporting closure.
        """
        # Wait for any pending asynchronous CPU optimizer operations
        self._wait_for_async_operations()

        self._pre_step()
        self._partition_all_parameters()

        if self._overflow_check_and_loss_scale_update():
            self._handle_overflow_rollback()
            return

        norm_groups = self._get_norm_groups()
        scaled_global_grad_norm = torch.linalg.vector_norm(torch.stack(norm_groups))
        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale

        timer_names = set()
        timer_names.add(OPTIMIZER_STEP_TIMER)
        self.timers(OPTIMIZER_STEP_TIMER).start()

        if self.check_clip_grads(scaled_global_grad_norm):
            self._handle_gradient_clipping(scaled_global_grad_norm)

        for sub_group_id, group in enumerate(self.fp16_groups):
            # Prepare optimizer states, gradients and fp32 parameters for update
            self._prepare_sub_group(sub_group_id, timer_names)

            # Scale the fp32 gradients
            self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)

            # Apply the optimizer step on the sub group and copy fp32 parameters to fp16
            self._optimizer_step(sub_group_id)

            # Put fp16 parameters in appropriate location
            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)

            # Release memory or swap out optimizer states of fp32 parameters
            self._release_sub_group(sub_group_id, timer_names)

        self.timers(OPTIMIZER_STEP_TIMER).stop()
        self._post_step(timer_names)

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
