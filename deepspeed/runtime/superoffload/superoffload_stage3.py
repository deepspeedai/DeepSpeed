# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import torch
from collections import deque
from typing import List

import deepspeed.comm as dist
from deepspeed.runtime.superoffload.superoffload_utils import SuperOffloadCPUOptimizer, EventTypes
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

        self.sub_group_to_param_num = []
        self.params_in_ipg_bucket_buffer = deque()
        self.pending_cpu_grad_copies = deque()
        self._cur_bucket_index = -1
        self.async_cpuadam_num = 0
        self.max_grad_numel = 0

        super().__init__(module, init_optimizer, param_names, timers, ds_config, **kwargs)
        self._reset_sub_group_grad_partition_tracking()

        optimizer_config = self._get_superoffload_optimizer_config()
        cpuadam_cores_perc = kwargs.get("cpuadam_cores_perc", 0.8)
        self.superoffload_cpu_optimizer = SuperOffloadCPUOptimizer(optimizer_config=optimizer_config,
                                                                   cpuadam_cores_perc=cpuadam_cores_perc)

    def _get_superoffload_optimizer_config(self):
        optimizer_config = []
        for param_group in self.optimizer.param_groups:
            optimizer_config.append({
                "lr": param_group["lr"],
                "betas": param_group["betas"],
                "eps": param_group["eps"],
                "weight_decay": param_group["weight_decay"],
                "amsgrad": param_group.get("amsgrad", False)
            })
        return optimizer_config

    def _record_sub_group_metadata(self, sub_group, sub_group_numel):
        self.max_grad_numel = max(self.max_grad_numel, sub_group_numel)
        self.sub_group_to_param_num.append(len(sub_group))

    def _reset_sub_group_grad_partition_tracking(self):
        self.sub_group_partition_counts = [0] * len(self.sub_group_to_param_num)
        self.sub_group_grad_staging_buffers = {}

    def _create_fp16_sub_groups(self, params_group):

        params_group_numel = sum([param.partition_numel() for param in params_group])
        sub_group_size = self.sub_group_size

        if sub_group_size is None or sub_group_size >= params_group_numel:
            self._record_sub_group_metadata(params_group, params_group_numel)
            return [params_group]

        sub_groups = []
        sub_group = []
        local_sub_group_size = 0

        for param in params_group:
            sub_group.append(param)
            local_sub_group_size += param.partition_numel()

            if local_sub_group_size >= sub_group_size or id(param) == id(params_group[-1]):
                self._record_sub_group_metadata(sub_group, local_sub_group_size)
                sub_groups.append(sub_group)

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
        if bucket.elements + param.ds_numel > self.reduce_bucket_size and bucket.elements > 0:
            self._DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads(comm_dtype)

        if getattr(param, "ds_grad_is_ready", True) and param.grad is not None:
            self._DeepSpeedZeroOptimizer_Stage3__add_grad_to_ipg_bucket(param)

    @instrument_w_nvtx
    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
            self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                self.fp32_partitioned_groups_flat[sub_group_id].data)
            self._unflatten_partitioned_parameters(sub_group_id)
        else:
            self._partitioned_params_swap_out(sub_group_id)

    def _submit_async_cpu_optimizer_step(self,
                                         sub_group_id: int,
                                         fp32_grad_tensor: Tensor,
                                         rollback: bool = False) -> None:
        fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
        param_group_id = self.sub_group_to_group_id[sub_group_id]

        self.superoffload_cpu_optimizer.async_step(param_group_id,
                                                   sub_group_id,
                                                   fp32_param,
                                                   fp32_grad_tensor,
                                                   rollback=rollback)
        self.async_cpuadam_num += 1

    def _consume_completed_async_result(self, result) -> bool:
        if result is None:
            return False

        self.async_cpuadam_num -= 1
        return True

    def _submit_ready_cpu_grad_copies(self) -> None:
        while self.pending_cpu_grad_copies and self.pending_cpu_grad_copies[0]["event"].query():
            pending_copy = self.pending_cpu_grad_copies.popleft()
            self._submit_async_cpu_optimizer_step(pending_copy["sub_group_id"], pending_copy["fp32_grad_tensor"])

    def _wait_for_pending_grad_copies(self, timeout_seconds=60):
        if not self.pending_cpu_grad_copies:
            return

        start_time = time.time()
        initial_pending_copies = len(self.pending_cpu_grad_copies)

        while self.pending_cpu_grad_copies:
            self._submit_ready_cpu_grad_copies()
            if not self.pending_cpu_grad_copies:
                return

            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout_seconds:
                raise RuntimeError(f"SuperOffload grad copy timeout after {elapsed_time:.1f} seconds. "
                                   f"Still waiting for {len(self.pending_cpu_grad_copies)}/{initial_pending_copies} "
                                   f"CPU grad copies to complete.")

            time.sleep(0.001)

    @instrument_w_nvtx
    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        self._submit_ready_cpu_grad_copies()

        for param, grad_partition in zip(params_to_release, grad_partitions):
            i, dest_offset, _ = self.grad_position[self.get_param_id(param)]
            is_grad_boundary = self.is_gradient_accumulation_boundary
            contains_real_data = param.partition_numel() * dist.get_rank(self.dp_process_group) < param.ds_numel
            is_cpu_sub_group = self.subgroup_to_device[i] == 'cpu'

            if is_grad_boundary:
                if self.sub_group_partition_counts[i] == 0:
                    if is_cpu_sub_group:
                        expected_numel = self.fp32_partitioned_groups_flat[i].grad.numel()
                        self.sub_group_grad_staging_buffers[i] = torch.zeros(expected_numel,
                                                                            device=grad_partition.device,
                                                                            dtype=self.master_weights_and_grads_dtype)
                    else:
                        self.fp32_partitioned_groups_flat[i].grad.zero_()
                if contains_real_data:
                    self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_partition)

            if contains_real_data and is_grad_boundary:
                grad_buffer = grad_partition.to(dtype=self.master_weights_and_grads_dtype)
                if is_cpu_sub_group:
                    staging_buffer = self.sub_group_grad_staging_buffers[i]
                    staging_buffer.narrow(0, dest_offset, grad_buffer.numel()).copy_(grad_buffer, non_blocking=True)
                else:
                    fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad
                    fp32_grad_tensor.narrow(0, dest_offset, grad_buffer.numel()).copy_(grad_buffer, non_blocking=True)

            if is_grad_boundary:
                self.sub_group_partition_counts[i] += 1
                if is_cpu_sub_group and self.sub_group_partition_counts[i] == self.sub_group_to_param_num[i]:
                    fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad
                    staging_buffer = self.sub_group_grad_staging_buffers.pop(i)
                    fp32_grad_tensor.copy_(staging_buffer, non_blocking=True)
                    copy_event = get_accelerator().Event()
                    get_accelerator().current_stream().record_event(copy_event)
                    self.pending_cpu_grad_copies.append({
                        "sub_group_id": i,
                        "fp32_grad_tensor": fp32_grad_tensor,
                        "event": copy_event,
                    })
                    self._submit_ready_cpu_grad_copies()
                    self._consume_completed_async_result(self.superoffload_cpu_optimizer.get_result())

        # Clean up parameter gradients
        for param in params_to_release:
            if not get_accelerator().is_synchronized_device():
                param.grad.record_stream(get_accelerator().current_stream())
            param.grad = None

    def independent_gradient_partition_epilogue(self):
        super().independent_gradient_partition_epilogue()
        self._reset_sub_group_grad_partition_tracking()

    @instrument_w_nvtx
    def step(self, closure=None):
        """
            Not supporting closure.
        """
        self._wait_for_pending_grad_copies()
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
            if not self._consume_completed_async_result(result):
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

    def _wait_for_single_async_result(self, event_type: str, timeout_seconds=60):
        """Wait for a single asynchronous CPU-Adam optimizer operation with timeout.

        Args:
            event_type (str): Type of operation expected ('adam_step' or 'rollback').
            timeout_seconds (int): Maximum time to wait before throwing an error. Default is 60 seconds.
        """
        start_time = time.time()

        while True:
            result = self.superoffload_cpu_optimizer.get_result(expected_event_type=event_type)
            if self._consume_completed_async_result(result):
                break

            current_time = time.time()
            elapsed_time = current_time - start_time

            # Throw error if we've been waiting longer than the timeout
            if elapsed_time >= timeout_seconds:
                raise RuntimeError(f"SuperOffload CPU optimizer timeout after {elapsed_time:.1f} seconds. "
                                   f"This indicates a deadlock or critical performance issue in the CPU optimizer.")

            time.sleep(0.001)  # 1ms sleep

    def _sync_cpu_optimizer_step(self, sub_group_id: int, rollback: bool = False, timeout_seconds: int = 60):
        event_type = EventTypes.ROLLBACK if rollback else EventTypes.ADAM_STEP
        fp32_grad = self.fp32_partitioned_groups_flat[sub_group_id].grad
        self._submit_async_cpu_optimizer_step(sub_group_id, fp32_grad, rollback=rollback)
        self._wait_for_single_async_result(event_type, timeout_seconds)

    def _handle_overflow_rollback(self):
        """Handle gradient overflow by rolling back CPU optimizer states."""
        for sub_group_id, _ in enumerate(self.fp16_groups):
            if self.subgroup_to_device[sub_group_id] == 'cpu':
                # Trigger rollback
                self._sync_cpu_optimizer_step(sub_group_id, rollback=True)

    def _handle_gradient_clipping(self, scaled_global_grad_norm):
        """Handle gradient clipping with CPU optimizer rollback and re-optimization."""
        for sub_group_id, _ in enumerate(self.fp16_groups):
            if self.subgroup_to_device[sub_group_id] == 'cpu':
                # Rollback CPU optimizer states
                self._sync_cpu_optimizer_step(sub_group_id, rollback=True)

                # Clip gradients and re-optimize
                self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)

                self._sync_cpu_optimizer_step(sub_group_id, rollback=False)

    @instrument_w_nvtx
    def check_clip_grads(self, total_norm):
        """Check if gradients need to be clipped based on the global norm."""
        unscaled_norm = total_norm / self.loss_scale
        return self.clip_grad and unscaled_norm > self.clip_grad
