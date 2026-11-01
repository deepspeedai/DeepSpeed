# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import time
from collections import defaultdict
from typing import List

import torch

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.superoffload.superoffload_utils import SuperOffloadCPUOptimizer, TaskKeys, ResultKeys, EventTypes
from deepspeed.runtime.zero.partition_parameters import Parameter, Tensor
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.utils import combine_grad_norm_groups, is_invalid_grad_norm
from deepspeed.utils.nvtx import instrument_w_nvtx
from deepspeed.utils import logger
from deepspeed.accelerator import get_accelerator

OPTIMIZER_STEP_TIMER = 'optimizer_step'


def _validate_superoffload_accelerator():
    """Validate that the current accelerator is compatible with SuperOffload."""
    accelerator = get_accelerator()
    assert accelerator.device_name() == 'cuda', (
        f"SuperOffload only supports NVIDIA CUDA GPUs, but found accelerator '{accelerator.device_name()}'.")


def _cpu_adam_settings(optimizer):
    """Return (adamw_mode, fp32_optimizer_states) for the DeepSpeedCPUAdam that runs ``optimizer`` in the worker."""
    if isinstance(optimizer, DeepSpeedCPUAdam):
        return optimizer.adam_w_mode, optimizer.fp32_optimizer_states
    if isinstance(optimizer, torch.optim.AdamW):
        return True, True
    if isinstance(optimizer, torch.optim.Adam):
        return optimizer.defaults.get("decoupled_weight_decay", False), True
    raise ValueError(f"SuperOffload steps the CPU sub-groups with DeepSpeedCPUAdam, so it supports only Adam and "
                     f"AdamW, but the optimizer is {type(optimizer).__name__}.")


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
        adamw_mode, fp32_optimizer_states = _cpu_adam_settings(init_optimizer)

        self.sub_group_to_param_num = {}
        # Per sub-group, the ds_ids of the params whose gradient for this step is in the fp32 gradient buffer.
        self.sub_group_grads_ready = defaultdict(set)
        # Sub-groups stepped during backward, mapped to the (hyperparameters, grad_scale) of that step for a rollback.
        self._submitted_cpu_sub_groups = {}
        self.async_cpuadam_num = 0
        self.max_grad_numel = 0

        super().__init__(module, init_optimizer, param_names, timers, ds_config, **kwargs)

        optimizer_configs = []
        for pg in self.optimizer.param_groups:
            optimizer_configs.append({
                "lr": pg["lr"],
                "betas": pg["betas"],
                "eps": pg["eps"],
                "weight_decay": pg["weight_decay"],
                "bias_correction": pg.get("bias_correction", True),
                "amsgrad": pg["amsgrad"],
            })
        # A step taken during backward is undone with adam_rollback, which supports only fp32 AdamW with bias
        # correction. With other settings, step() runs the CPU sub-groups after the overflow and clipping checks.
        self.speculative_cpu_step = (adamw_mode and fp32_optimizer_states
                                     and self.master_weights_and_grads_dtype == torch.float32
                                     and all(config["bias_correction"] for config in optimizer_configs))
        cpuadam_cores_perc = kwargs.get("cpuadam_cores_perc", 0.8)
        self.superoffload_cpu_optimizer = SuperOffloadCPUOptimizer(optimizer_config=optimizer_configs,
                                                                   cpuadam_cores_perc=cpuadam_cores_perc,
                                                                   max_grad_numel=self.max_grad_numel,
                                                                   pin_memory=self.offload_optimizer_pin_memory,
                                                                   adamw_mode=adamw_mode,
                                                                   fp32_optimizer_states=fp32_optimizer_states,
                                                                   grad_dtype=self.master_weights_and_grads_dtype)
        self._check_shm_size(fp32_optimizer_states)

    def _check_shm_size(self, fp32_optimizer_states):
        """Warn when /dev/shm is too small for the tensors that SuperOffload shares with its worker process."""
        try:
            stat = os.statvfs("/dev/shm")
        except OSError:
            return
        numel = sum(self.fp32_partitioned_groups_flat[i].numel() for i, device in self.subgroup_to_device.items()
                    if device == 'cpu')
        weight_size = torch.empty(0, dtype=self.master_weights_and_grads_dtype).element_size()
        state_size = 4 if fp32_optimizer_states else weight_size
        # The fp32 weights and gradients move there at the first step, the Adam state at the first checkpoint save
        # or load.
        needed = numel * 2 * (weight_size + state_size)
        local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("LOCAL_SIZE", 1)))
        size = stat.f_blocks * stat.f_frsize
        if size < needed * local_ranks:
            logger.warning(
                f"SuperOffload shares the fp32 weights, gradients and Adam state of the CPU sub-groups with "
                f"its worker process through /dev/shm: {needed / 2**30:.1f} GiB on this rank, "
                f"{local_ranks} rank(s) on this node, but /dev/shm has {size / 2**30:.1f} GiB. A checkpoint "
                f"save or load may fail; increase the size of /dev/shm.")

    def _create_fp16_sub_groups(self, params_group):

        params_group_numel = sum([param.partition_numel() for param in params_group])
        sub_group_size = self.sub_group_size

        if sub_group_size is None or sub_group_size >= params_group_numel:
            global_idx = len(self.sub_group_to_param_num)
            self.sub_group_to_param_num[global_idx] = len(params_group)
            self.max_grad_numel = max(self.max_grad_numel, params_group_numel)
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
                global_idx = len(self.sub_group_to_param_num)
                self.sub_group_to_param_num[global_idx] = len(sub_group)

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

            # Accumulate gradient into the grad_buffer, mirroring base class logic
            grad_buffer = self._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[param.ds_id].narrow(
                0, 0, grad_partition.numel())
            # The buffer holds an older step's gradient until the param's first gradient of this step. Testing the
            # param rather than micro_step_id also keeps both gradients of a param that two reentrant checkpoint
            # segments use.
            first_grad = param.ds_id not in self._offload_boundary_param_ids
            self._offload_boundary_param_ids.add(param.ds_id)
            if first_grad:
                grad_buffer.copy_(grad_partition, non_blocking=True)
                grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
            elif get_accelerator().on_accelerator(grad_buffer):
                grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(grad_buffer.shape))
            else:
                cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)
                cuda_grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype).view(cuda_grad_buffer.shape))
                grad_buffer.copy_(cuda_grad_buffer, non_blocking=True)
                grad_buffer = cuda_grad_buffer

            if self.is_gradient_accumulation_boundary():
                if i in self._submitted_cpu_sub_groups:
                    # Another gradient for a sub-group that has already stepped: undo that step while the fp32
                    # buffer still holds the gradient it used.
                    self._wait_for_async_operations()
                    self._rollback_cpu_sub_group(i)
                self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_buffer)

                fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(
                    0, dest_offset, grad_buffer.numel())
                fp32_grad_tensor.copy_(grad_buffer.to(dtype=self.master_weights_and_grads_dtype), non_blocking=True)

                self.sub_group_grads_ready[i].add(param.ds_id)
                if len(self.sub_group_grads_ready[i]) == self.sub_group_to_param_num[i]:
                    completed_sub_groups.append(i)

        if self.is_gradient_accumulation_boundary() and completed_sub_groups:
            get_accelerator().current_stream().synchronize()
            # A param that gets a second gradient adds its sub-group again.
            for i in dict.fromkeys(completed_sub_groups):
                param_group_id = self.sub_group_to_group_id[i]
                if (self.subgroup_to_device[i] == 'cpu' and self.speculative_cpu_step
                        and self._betas_nonzero(param_group_id) and self._grads_finite(i)):
                    fp32_param = self.fp32_partitioned_groups_flat[i]
                    hyperparams = self._hyperparams(param_group_id)
                    # The gradients still carry the loss scale, which step() would remove.
                    grad_scale = 1.0 / self.loss_scale

                    self.superoffload_cpu_optimizer.async_step(param_group_id,
                                                               i,
                                                               fp32_param.data,
                                                               fp32_param.grad.data,
                                                               hyperparams=hyperparams,
                                                               grad_scale=grad_scale)
                    self._submitted_cpu_sub_groups[i] = (hyperparams, grad_scale)
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
        self._complete_grads()

        self._pre_step()
        self._partition_all_parameters()

        self._overflow_check_and_loss_scale_update(update_scale=False)
        if self.overflow:
            self._rollback_submitted_cpu_sub_groups()
            self._loss_scale_update_and_overflow_cleanup()
            return

        norm_groups = self._get_norm_groups()
        scaled_global_grad_norm = combine_grad_norm_groups(norm_groups)
        if is_invalid_grad_norm(scaled_global_grad_norm):
            self.overflow = True
            self._rollback_submitted_cpu_sub_groups()
            self._loss_scale_update_and_overflow_cleanup()
            return
        self._loss_scale_update_and_overflow_cleanup()
        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale

        timer_names = set()
        timer_names.add(OPTIMIZER_STEP_TIMER)
        self.timers(OPTIMIZER_STEP_TIMER).start()

        # The steps taken during backward used the unclipped gradients, and the loss scale of the backward, which
        # _loss_scale_update_and_overflow_cleanup may have raised before unscale_and_clip_grads removes it.
        if self.check_clip_grads(scaled_global_grad_norm) or self._loss_scale_changed():
            self._rollback_submitted_cpu_sub_groups()

        for sub_group_id, group in enumerate(self.fp16_groups):
            self._prepare_sub_group(sub_group_id, timer_names)
            self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)

            if self.subgroup_to_device[sub_group_id] != 'cpu':
                self._optimizer_step(sub_group_id)
            elif sub_group_id not in self._submitted_cpu_sub_groups:
                param_group_id = self.sub_group_to_group_id[sub_group_id]
                fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
                self._sync_cpu_optimizer_step(param_group_id,
                                              sub_group_id,
                                              fp32_param.data,
                                              fp32_param.grad.data,
                                              hyperparams=self._hyperparams(param_group_id))

            self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
            self._release_sub_group(sub_group_id, timer_names)

        self.timers(OPTIMIZER_STEP_TIMER).stop()
        self._post_step(timer_names)

    def _post_step(self, timer_names):
        super()._post_step(timer_names)
        self._submitted_cpu_sub_groups.clear()

    def _complete_grads(self):
        """Fill the fp32 gradient of the params that got no gradient in the boundary micro-step.

        A param used only in earlier micro-steps has its gradient in the reduce buffer. A param not used in this step
        gets a zero gradient, as set_none_gradients_to_zero gives it in ZeRO-1/2; zeroing its reduce buffer also keeps
        an older inf out of the fp16 overflow check. Their sub-groups did not step during backward, so step() runs
        them.
        """
        filled = False
        for i, group in enumerate(self.fp16_groups):
            ready = self.sub_group_grads_ready[i]
            if len(ready) == self.sub_group_to_param_num[i]:
                continue
            for param in group:
                if param.ds_id in ready:
                    continue
                _, dest_offset, num_elements = self.grad_position[self.get_param_id(param)]
                grad_buffer = self._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[param.ds_id].narrow(
                    0, 0, num_elements)
                if param.ds_id not in self._offload_boundary_param_ids:
                    grad_buffer.zero_()
                if not get_accelerator().on_accelerator(grad_buffer):
                    grad_buffer = grad_buffer.to(get_accelerator().current_device_name(), non_blocking=True)
                self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_buffer)
                fp32_grad_tensor = self.fp32_partitioned_groups_flat[i].grad.narrow(0, dest_offset, num_elements)
                fp32_grad_tensor.copy_(grad_buffer.to(dtype=self.master_weights_and_grads_dtype), non_blocking=True)
                filled = True
        if filled:
            get_accelerator().current_stream().synchronize()
        self.sub_group_grads_ready.clear()

    def _grads_finite(self, sub_group_id):
        """Whether the sub-group's gradients are finite. A step on inf or nan cannot be rolled back."""
        norms = [self.norm_for_param_grads[self.get_param_id(param)] for param in self.fp16_groups[sub_group_id]]
        return bool(torch.isfinite(torch.stack(norms)).all())

    def _betas_nonzero(self, param_group_id):
        """Whether a step of the param group can be rolled back: the rollback divides the moments by beta1 and beta2.
        Checked at each step, as momentum schedules change the betas."""
        return all(beta != 0 for beta in self.optimizer.param_groups[param_group_id]["betas"])

    def _hyperparams(self, param_group_id):
        """The hyperparameters of a param group, which LR, momentum or weight decay schedules change."""
        group = self.optimizer.param_groups[param_group_id]
        return {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}

    def _loss_scale_changed(self):
        grad_scale = 1.0 / self.loss_scale
        return any(scale != grad_scale for _, scale in self._submitted_cpu_sub_groups.values())

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
                                 hyperparams: dict = None,
                                 timeout_seconds: int = 60,
                                 grad_scale: float = 1.0):
        event_type = EventTypes.ROLLBACK if rollback else EventTypes.ADAM_STEP
        self.superoffload_cpu_optimizer.async_step(param_group_id,
                                                   sub_group_id,
                                                   fp32_param_data,
                                                   fp32_grad_data,
                                                   rollback=rollback,
                                                   hyperparams=hyperparams,
                                                   grad_scale=grad_scale)
        # Wait for completion
        self._wait_for_single_async_result(event_type, timeout_seconds)

    def _rollback_submitted_cpu_sub_groups(self):
        """Undo the CPU steps taken during backward, for a skipped step or before redoing them with clipping."""
        for sub_group_id in list(self._submitted_cpu_sub_groups):
            self._rollback_cpu_sub_group(sub_group_id)

    def _rollback_cpu_sub_group(self, sub_group_id):
        hyperparams, grad_scale = self._submitted_cpu_sub_groups.pop(sub_group_id)
        param_group_id = self.sub_group_to_group_id[sub_group_id]
        fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
        self._sync_cpu_optimizer_step(param_group_id,
                                      sub_group_id,
                                      fp32_param.data,
                                      fp32_param.grad.data,
                                      rollback=True,
                                      hyperparams=hyperparams,
                                      grad_scale=grad_scale)

    @instrument_w_nvtx
    def check_clip_grads(self, total_norm):
        """Check if unscale_and_clip_grads will scale the gradients down, with the same test."""
        if not self.clip_grad:
            return False
        clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
        return bool(clip > 1.0)

    def state_dict(self):
        # The worker process holds the Adam state of the CPU sub-groups. Save it where ZeRO-3 CPU offload keeps it.
        self._wait_for_async_operations()
        cpu_params = []
        for sub_group_id, state in self.superoffload_cpu_optimizer.get_state().items():
            fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
            self.optimizer.state[fp32_param] = state
            cpu_params.append(fp32_param)
        try:
            return super().state_dict()
        finally:
            for fp32_param in cpu_params:
                del self.optimizer.state[fp32_param]

    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None,
                        load_serial=None,
                        param_shapes=None):
        super().load_state_dict(state_dict_list,
                                load_optimizer_states=load_optimizer_states,
                                load_from_fp32_weights=load_from_fp32_weights,
                                checkpoint_folder=checkpoint_folder,
                                load_serial=load_serial,
                                param_shapes=param_shapes)
        if not load_optimizer_states:
            return
        # Move the loaded Adam state of the CPU sub-groups to the worker process, which steps them.
        cpu_state = {}
        for sub_group_id, fp32_param in enumerate(self.fp32_partitioned_groups_flat):
            if self.subgroup_to_device[sub_group_id] == 'cpu' and fp32_param in self.optimizer.state:
                cpu_state[sub_group_id] = self.optimizer.state.pop(fp32_param)
        self.superoffload_cpu_optimizer.set_state(cpu_state)
