# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from collections import defaultdict, deque
from concurrent.futures import Future
from typing import List, NamedTuple

import torch

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.superoffload.superoffload_utils import SuperOffloadCPUAdam
from deepspeed.runtime.zero.partition_parameters import Parameter, Tensor
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.utils import combine_grad_norm_groups, is_invalid_grad_norm
from deepspeed.utils.nvtx import instrument_w_nvtx
from deepspeed.utils import logger
from deepspeed.accelerator import get_accelerator

OPTIMIZER_STEP_TIMER = 'optimizer_step'
# Larger gradients go to the CPU in pieces of this size (see _copy_grad_to_cpu). On B200 over PCIe Gen5, 16 MiB pieces
# copy at 98% of the rate of whole gradients; smaller ones lose more to the switches between pieces.
_GRAD_COPY_PIECE_BYTES = 16 << 20


def _validate_superoffload_accelerator():
    """Validate that the current accelerator is compatible with SuperOffload."""
    accelerator = get_accelerator()
    assert accelerator.device_name() == 'cuda', (
        f"SuperOffload only supports NVIDIA CUDA GPUs, but found accelerator '{accelerator.device_name()}'.")


def _cpu_adam_settings(optimizer):
    """Return (adamw_mode, fp32_optimizer_states) for the CPU Adam that steps the CPU sub-groups of ``optimizer``."""
    if any(group.get("amsgrad", False) for group in optimizer.param_groups):
        raise ValueError(
            "SuperOffload steps the CPU sub-groups with DeepSpeedCPUAdam, which does not support AMSGrad.")
    if isinstance(optimizer, DeepSpeedCPUAdam):
        return optimizer.adam_w_mode, optimizer.fp32_optimizer_states
    if isinstance(optimizer, torch.optim.AdamW):
        return True, True
    if isinstance(optimizer, torch.optim.Adam):
        return optimizer.defaults.get("decoupled_weight_decay", False), True
    raise ValueError(f"SuperOffload steps the CPU sub-groups with DeepSpeedCPUAdam, so it supports only Adam and "
                     f"AdamW, but the optimizer is {type(optimizer).__name__}.")


class _CPUStep(NamedTuple):
    """A CPU step submitted during backward, and what a rollback of it needs."""
    # Its result tells whether the sub-group stepped: one whose gradients were not finite is left to step().
    stepped: Future
    hyperparams: dict
    grad_scale: float


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

        # Per sub-group, the ds_ids of the params whose gradient for this step is in the fp32 gradient buffer.
        self.sub_group_grads_ready = defaultdict(set)
        # Sub-groups whose CPU step was submitted during backward, until step() is done or the step is rolled back.
        self._submitted_cpu_sub_groups = {}
        # The submitted sub-groups whose step has not been collected yet, in the order the CPU Adam thread runs them.
        self._uncollected_cpu_steps = deque()
        # Host addresses of the CPU sub-groups' fp32 weights and gradients registered with CUDA.
        self._registered_host_addresses = []
        # Sub-groups stepped during backward whose new weights are converted into a GPU staging buffer, mapped to the
        # event of that conversion.
        self._staged_lp_params = {}
        self._lp_staging_buffers = {}
        # Per staging buffer, the event after which step() has copied it into the params.
        self._lp_staging_free_events = {}
        self._lp_staging_stream = None
        # Stream of the gradient copies to the CPU, so that they overlap the backward compute.
        self._grad_copy_stream = None

        super().__init__(module, init_optimizer, param_names, timers, ds_config, **kwargs)

        self.sub_group_to_param_num = [len(group) for group in self.fp16_groups]
        # A step taken during backward is undone with adam_rollback, which supports only fp32 AdamW with bias
        # correction. With other settings, step() runs the CPU sub-groups after the overflow and clipping checks.
        bias_correction = all(group.get("bias_correction", True) for group in self.optimizer.param_groups)
        self.speculative_cpu_step = (adamw_mode and fp32_optimizer_states
                                     and self.master_weights_and_grads_dtype == torch.float32 and bias_correction)
        self.cpu_adam = SuperOffloadCPUAdam(self.optimizer.param_groups[0], adamw_mode, fp32_optimizer_states,
                                            kwargs.get("cpuadam_cores_perc", 0.8))
        # With bf16 gradients accumulated into fp32 master weights in one micro-step, the gradients of the CPU
        # sub-groups go to the CPU in bf16, half the bytes of fp32, and the worker promotes them in its kernel.
        self._half_cpu_grads = {}
        if (self.dtype == torch.bfloat16 and self.gradient_accumulation_dtype == torch.bfloat16
                and self.master_weights_and_grads_dtype == torch.float32 and fp32_optimizer_states
                and self.gradient_accumulation_steps == 1):
            for sub_group_id, device in self.subgroup_to_device.items():
                if device == 'cpu' and not self._swappable_optimizer_subgroup(sub_group_id):
                    self._half_cpu_grads[sub_group_id] = torch.zeros(
                        self.fp32_partitioned_groups_flat[sub_group_id].numel(), dtype=torch.bfloat16)
        self._pin_cpu_buffers()
        self._grad_copy_stream = get_accelerator().Stream()
        # The pieces of a large gradient take turns on the copy stream and a second one, each with an event
        # (_copy_grad_to_cpu).
        self._grad_copy_streams = (self._grad_copy_stream, get_accelerator().Stream())
        self._grad_copy_events = (get_accelerator().Event(), get_accelerator().Event())
        if self.speculative_cpu_step and self._cpu_sub_group_weights_pinned():
            # Default priority is the lowest, so the conversions never delay the backward kernels.
            self._lp_staging_stream = get_accelerator().Stream()

    def _pin_cpu_buffers(self):
        """Keep the CPU sub-groups' fp32 weights and gradients (and bf16 gradients) page-locked.

        offload_optimizer.pin_memory pins the gradients but not the fp32 weights. The gradient copies of backward and
        the weight copies of step() would then run from pageable memory, at a fraction of the bandwidth and blocking
        the thread that issues them. Registering their memory with the device runtime keeps the copies asynchronous.
        """
        if not self.offload_optimizer_pin_memory:
            return
        accelerator = get_accelerator()
        for sub_group_id, device in self.subgroup_to_device.items():
            # Swapped sub-groups share one gradient buffer and do not keep their weights in memory.
            if device != 'cpu' or self._swappable_optimizer_subgroup(sub_group_id):
                continue
            fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
            for tensor in (fp32_param.data, fp32_param.grad, self._half_cpu_grads.get(sub_group_id)):
                if tensor is None or tensor.is_pinned():
                    continue
                storage = tensor.untyped_storage()
                if storage.data_ptr() in self._registered_host_addresses:
                    continue
                try:
                    registered = accelerator.register_host_memory(storage.data_ptr(), storage.nbytes())
                except Exception as e:
                    registered = False
                    logger.warning(f"SuperOffload could not page-lock its CPU buffers: {e}")
                if not registered:
                    return
                self._registered_host_addresses.append(storage.data_ptr())

    def _cpu_sub_group_weights_pinned(self):
        """Whether the fp32 weights of the CPU sub-groups are page-locked, which the GPU staging of step() needs."""
        weights = [
            self.fp32_partitioned_groups_flat[sub_group_id].data
            for sub_group_id, device in self.subgroup_to_device.items()
            if device == 'cpu' and not self._swappable_optimizer_subgroup(sub_group_id)
        ]
        return bool(weights) and all(weight.is_pinned() for weight in weights)

    def _worker_grad(self, sub_group_id):
        """The gradient of a sub-group that the CPU worker steps on: the bf16 one if it has one, else the fp32 one."""
        half_grad = self._half_cpu_grads.get(sub_group_id)
        return half_grad if half_grad is not None else self.fp32_partitioned_groups_flat[sub_group_id].grad.data

    def _cpu_step_grad(self, sub_group_id, total_norm):
        """The gradient for step() to step a CPU sub-group with that did not step during backward, unscaled and
        clipped. A bf16 gradient that needs neither goes as it is; otherwise the fp32 gradient takes the scaling."""
        half_grad = self._half_cpu_grads.get(sub_group_id)
        if half_grad is not None:
            if self.loss_scale == 1 and not self.check_clip_grads(total_norm):
                return half_grad
            self.fp32_partitioned_groups_flat[sub_group_id].grad.copy_(half_grad)
        self.unscale_and_clip_grads(sub_group_id, total_norm)
        return self.fp32_partitioned_groups_flat[sub_group_id].grad.data

    def destroy(self):
        self.cpu_adam.close()
        super().destroy()

    def _unpin_offload_buffers(self):
        for address in self._registered_host_addresses:
            get_accelerator().unregister_host_memory(address)
        self._registered_host_addresses = []
        super()._unpin_offload_buffers()

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
            self._sync_backup_optimizer_group(param_group_id)
            self.backup_optimizer.param_groups[param_group_id]['params'] = [fp32_param]
            step_with_gradscaler(self.backup_optimizer)
            self.backup_optimizer.param_groups[param_group_id]['params'] = []

    @instrument_w_nvtx
    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        if self.subgroup_to_device[sub_group_id] == 'cpu':
            lp_flat = self.fp16_partitioned_groups_flat[sub_group_id]
            fp32_flat = self.fp32_partitioned_groups_flat[sub_group_id].data
            if fp32_flat.is_pinned() and get_accelerator().on_accelerator(lp_flat):
                # copy_ from a CPU tensor of another dtype converts on the CPU into a pageable temporary and copies
                # that. Copying the pinned fp32 weights and converting on the GPU keeps the CPU free and the copy fast.
                fp32_flat = fp32_flat.to(lp_flat.device, non_blocking=True)
            lp_flat.data.copy_(fp32_flat)
            self._unflatten_partitioned_parameters(sub_group_id)
            return

        if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
            self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                self.fp32_partitioned_groups_flat[sub_group_id].data)
            self._unflatten_partitioned_parameters(sub_group_id)
        else:
            self._partitioned_params_swap_out(sub_group_id)

    @instrument_w_nvtx
    def _stage_lp_params(self, sub_group_id):
        """Convert the weights of a CPU step taken during backward to the low-precision dtype on the GPU.

        Backward may still gather the current params, and step() changes them only after the overflow and clipping
        checks, so the conversion goes to a staging buffer that step() copies in.
        """
        stream = self._lp_staging_stream
        lp_flat = self.fp16_partitioned_groups_flat[sub_group_id]
        fp32_flat = self.fp32_partitioned_groups_flat[sub_group_id].data
        if stream is None or lp_flat is None or not get_accelerator().on_accelerator(lp_flat):
            return
        if not fp32_flat.is_pinned():
            return
        staging = self._lp_staging_buffers.get(sub_group_id)
        if staging is None:
            # Allocate it on the stream that writes it. The caching allocator hands the current stream blocks freed
            # while kernels queued on that stream may still use them, which another stream does not wait for.
            with get_accelerator().stream(stream):
                staging = torch.empty_like(lp_flat)
            self._lp_staging_buffers[sub_group_id] = staging
        free_event = self._lp_staging_free_events.pop(sub_group_id, None)
        if free_event is not None:
            # The previous step() may still be copying this buffer into the params.
            stream.wait_event(free_event)
        with get_accelerator().stream(stream):
            staging.copy_(fp32_flat.to(staging.device, non_blocking=True))
            event = get_accelerator().Event()
            event.record(stream)
        self._staged_lp_params[sub_group_id] = event

    def _drop_staged_lp_params(self, sub_group_id):
        """Discard a staged conversion before the worker changes the sub-group's weights, once it has read them."""
        event = self._staged_lp_params.pop(sub_group_id, None)
        if event is not None:
            event.synchronize()

    @instrument_w_nvtx
    def _apply_staged_lp_params(self, sub_group_id):
        event = self._staged_lp_params.pop(sub_group_id)
        stream = get_accelerator().current_stream()
        stream.wait_event(event)
        self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(self._lp_staging_buffers[sub_group_id])
        free_event = get_accelerator().Event()
        free_event.record(stream)
        self._lp_staging_free_events[sub_group_id] = free_event
        self._unflatten_partitioned_parameters(sub_group_id)

    def _collect_cpu_steps(self, wait=False):
        """Stage the new weights of the CPU steps submitted during backward that have finished. With wait, wait for
        all of them, and raise what a step raised.

        The CPU Adam thread runs the steps in the order they were submitted, so they finish in that order.
        """
        while self._uncollected_cpu_steps:
            sub_group_id = self._uncollected_cpu_steps[0]
            stepped = self._submitted_cpu_sub_groups[sub_group_id].stepped
            if not wait and not stepped.done():
                return
            self._uncollected_cpu_steps.popleft()
            if stepped.result():
                self._stage_lp_params(sub_group_id)
            else:
                # Its gradients were not finite, so step() runs it after the overflow check.
                del self._submitted_cpu_sub_groups[sub_group_id]

    def _copies_grad_directly(self, sub_group_id):
        """Whether a gradient of the sub-group goes straight into the master-dtype gradient the worker reads.

        That needs a CPU sub-group and the only micro-step of the step, so that no earlier micro-step left a gradient
        in the reduce buffer. fp16 keeps the reduce buffer, which its overflow check scans.
        """
        return (self.subgroup_to_device[sub_group_id] == 'cpu' and self.dtype != torch.float16
                and self.is_gradient_accumulation_boundary() and self.micro_step_id == 0)

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
            if self._copies_grad_directly(i):
                if first_grad:
                    # The gradient goes to the CPU once, into the gradient the worker reads, instead of into the
                    # reduce buffer, back to the GPU for the norm, and out again. It is rounded to the accumulation
                    # dtype as the reduce buffer would round it, and goes in that dtype if the worker reads bf16.
                    grad_buffer = grad_partition.to(self.gradient_accumulation_dtype)
                    if i not in self._half_cpu_grads:
                        grad_buffer = grad_buffer.to(self.master_weights_and_grads_dtype)
                else:
                    # A second gradient in this step: the first one is only in the gradient the worker reads, which
                    # the copy stream may still be writing. The sum is taken in the accumulation dtype, as in the
                    # reduce buffer; the first gradient was rounded to it, so converting it back is exact.
                    cpu_grad_tensor = self._worker_grad(i).narrow(0, dest_offset, grad_partition.numel())
                    get_accelerator().current_stream().wait_stream(self._grad_copy_stream)
                    grad_buffer = cpu_grad_tensor.to(grad_partition.device, non_blocking=True)
                    grad_buffer = grad_buffer.to(self.gradient_accumulation_dtype)
                    grad_buffer.add_(grad_partition.to(grad_buffer.dtype).view(grad_buffer.shape))
            elif first_grad:
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
                    # buffer still holds the gradient it used. One whose gradients were not finite did not step.
                    self._collect_cpu_steps(wait=True)
                    if i in self._submitted_cpu_sub_groups:
                        self._rollback_cpu_sub_group(i)
                self.norm_for_param_grads[self.get_param_id(param)] = self._constant_buffered_norm2(grad_buffer)

                cpu_grad_tensor = self._worker_grad(i).narrow(0, dest_offset, grad_buffer.numel())
                grad_to_copy = grad_buffer.to(dtype=cpu_grad_tensor.dtype)
                self._copy_grad_to_cpu(cpu_grad_tensor, grad_to_copy)

                self.sub_group_grads_ready[i].add(param.ds_id)
                if len(self.sub_group_grads_ready[i]) == self.sub_group_to_param_num[i]:
                    completed_sub_groups.append(i)

        if self.is_gradient_accumulation_boundary() and completed_sub_groups:
            steps = []
            finite_on_device = []
            # A param that gets a second gradient adds its sub-group again.
            for i in dict.fromkeys(completed_sub_groups):
                param_group_id = self.sub_group_to_group_id[i]
                if (self.subgroup_to_device[i] == 'cpu' and self.speculative_cpu_step
                        and self._betas_nonzero(param_group_id)):
                    hyperparams = self._hyperparams(param_group_id)
                    # The gradients still carry the loss scale, which step() would remove.
                    grad_scale = 1.0 / self.loss_scale

                    self._drop_staged_lp_params(i)
                    steps.append((i, hyperparams, grad_scale))
                    finite_on_device.append(self._grads_finite(i))
            if steps:
                # The CPU Adam thread reads the gradients that the copy stream writes, and waits for them itself, so
                # that this thread goes on queueing the backward kernels.
                copy_stream = self._grad_copy_stream
                copy_stream.wait_stream(get_accelerator().current_stream())
                with get_accelerator().stream(copy_stream):
                    # Behind the gradient copies: a copy to the CPU on the compute stream would wait for them in the
                    # copy engine, and hold up the kernels queued after it.
                    finite = [flag.to("cpu", non_blocking=True) for flag in finite_on_device]
                    copies_done = get_accelerator().Event()
                    copies_done.record(copy_stream)
                for flag in finite_on_device:
                    flag.record_stream(copy_stream)
                for (i, hyperparams, grad_scale), grads_finite in zip(steps, finite):
                    stepped = self.cpu_adam.submit(self._step_when_copied, i, copies_done, grads_finite, hyperparams,
                                                   grad_scale)
                    self._submitted_cpu_sub_groups[i] = _CPUStep(stepped, hyperparams, grad_scale)
                    self._uncollected_cpu_steps.append(i)

        if self.is_gradient_accumulation_boundary():
            self._collect_cpu_steps()

        for param in params_to_release:
            if not get_accelerator().is_synchronized_device():
                if param.grad is not None:
                    param.grad.record_stream(get_accelerator().current_stream())
            param.grad = None

    def _copy_grad_to_cpu(self, cpu_grad_tensor, grad_to_copy):
        """Queue the copy of a gradient to the CPU on the gradient copy stream, behind the kernels that computed it.

        The copy engine keeps serving a stream while it has a copy ready, so a copy to the CPU on another stream, such
        as the one .item() makes in backward, waits behind all the gradient copies queued before it: up to 28 ms with
        OPT-30B. A large gradient goes in pieces that take turns on two streams, each waiting for the one before, so
        that neither stream has a copy ready while a piece runs and the copies of other streams go in between.
        """
        copy_stream = self._grad_copy_stream
        copy_stream.wait_stream(get_accelerator().current_stream())
        piece = _GRAD_COPY_PIECE_BYTES // grad_to_copy.element_size()
        if grad_to_copy.numel() <= piece:
            with get_accelerator().stream(copy_stream):
                cpu_grad_tensor.copy_(grad_to_copy, non_blocking=True)
            # The copy reads the gradient later, so keep the allocator from handing its memory out before then.
            grad_to_copy.record_stream(copy_stream)
            return
        streams = self._grad_copy_streams
        events = self._grad_copy_events
        previous = None
        for turn, start in enumerate(range(0, grad_to_copy.numel(), piece)):
            stream = streams[turn % 2]
            if previous is not None:
                stream.wait_event(previous)
            with get_accelerator().stream(stream):
                cpu_grad_tensor[start:start + piece].copy_(grad_to_copy[start:start + piece], non_blocking=True)
            previous = events[turn % 2]
            previous.record(stream)
        # What waits for the copy stream, as step() and the CPU steps do, then waits for the last piece too.
        copy_stream.wait_event(previous)
        for stream in streams:
            grad_to_copy.record_stream(stream)

    @instrument_w_nvtx
    def step(self, closure=None):
        """
            Not supporting closure.
        """
        # Sub-groups that did not complete in backward may still have gradient copies in flight.
        self._grad_copy_stream.synchronize()
        self._collect_cpu_steps(wait=True)
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
            # A sub-group stepped during backward had the loss scale removed in the worker, and one that needed
            # clipping was rolled back above, so its gradients are not read again.
            if self.subgroup_to_device[sub_group_id] != 'cpu':
                self.unscale_and_clip_grads(sub_group_id, scaled_global_grad_norm)
                self._optimizer_step(sub_group_id)
            elif sub_group_id not in self._submitted_cpu_sub_groups:
                param_group_id = self.sub_group_to_group_id[sub_group_id]
                self._cpu_step_and_wait(sub_group_id, self._cpu_step_grad(sub_group_id, scaled_global_grad_norm),
                                        self._hyperparams(param_group_id))

            if sub_group_id in self._staged_lp_params:
                self._apply_staged_lp_params(sub_group_id)
            else:
                self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
            self._release_sub_group(sub_group_id, timer_names)

        self.timers(OPTIMIZER_STEP_TIMER).stop()
        self._post_step(timer_names)

    def _post_step(self, timer_names):
        super()._post_step(timer_names)
        for sub_group_id in list(self._staged_lp_params):
            self._drop_staged_lp_params(sub_group_id)
        self._submitted_cpu_sub_groups.clear()

    def _complete_grads(self):
        """Fill the gradient the worker reads for the params that got no gradient in the boundary micro-step.

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
                cpu_grad_tensor = self._worker_grad(i).narrow(0, dest_offset, num_elements)
                cpu_grad_tensor.copy_(grad_buffer.to(dtype=cpu_grad_tensor.dtype), non_blocking=True)
                filled = True
        if filled:
            get_accelerator().current_stream().synchronize()
        self.sub_group_grads_ready.clear()

    def _grads_finite(self, sub_group_id):
        """Whether the sub-group's gradients are finite, as a device tensor, so that this thread does not wait for the
        norms. A step on inf or nan cannot be rolled back."""
        norms = [self.norm_for_param_grads[self.get_param_id(param)] for param in self.fp16_groups[sub_group_id]]
        return torch.isfinite(torch.stack(norms)).all()

    def _betas_nonzero(self, param_group_id):
        """Whether a step of the param group can be rolled back: the rollback divides the moments by beta1 and beta2.
        Checked at each step, as momentum schedules change the betas."""
        return all(beta != 0 for beta in self.optimizer.param_groups[param_group_id]["betas"])

    def _hyperparams(self, param_group_id):
        """The hyperparameters of a param group, which LR, momentum or weight decay schedules change."""
        group = self.optimizer.param_groups[param_group_id]
        hyperparams = {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
        hyperparams["bias_correction"] = group.get("bias_correction", True)
        return hyperparams

    def _loss_scale_changed(self):
        grad_scale = 1.0 / self.loss_scale
        return any(step.grad_scale != grad_scale for step in self._submitted_cpu_sub_groups.values())

    def _step_when_copied(self, sub_group_id, copies_done, grads_finite, hyperparams, grad_scale):
        """In the CPU Adam thread: step a sub-group once its gradients are on the CPU, and tell whether it did.

        A step on inf or nan cannot be rolled back, so step() runs those sub-groups.
        """
        copies_done.synchronize()
        if not grads_finite.item():
            return False
        self._cpu_adam_step(sub_group_id, self._worker_grad(sub_group_id), hyperparams, grad_scale)
        return True

    def _cpu_adam_step(self, sub_group_id, grad, hyperparams, grad_scale=1.0, rollback=False):
        """In the CPU Adam thread: step a CPU sub-group, or undo its last step."""
        fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
        update = self.cpu_adam.rollback if rollback else self.cpu_adam.step
        # The Adam state stays where ZeRO-3 CPU offload keeps it, so that checkpoints save and load it as they are.
        update(self.optimizer.state[fp32_param], fp32_param.data, grad, hyperparams, grad_scale)

    def _cpu_step_and_wait(self, sub_group_id, grad, hyperparams, grad_scale=1.0, rollback=False):
        self._drop_staged_lp_params(sub_group_id)
        self.cpu_adam.submit(self._cpu_adam_step, sub_group_id, grad, hyperparams, grad_scale, rollback).result()

    def _rollback_submitted_cpu_sub_groups(self):
        """Undo the CPU steps taken during backward, for a skipped step or before redoing them with clipping."""
        for sub_group_id in list(self._submitted_cpu_sub_groups):
            self._rollback_cpu_sub_group(sub_group_id)

    def _rollback_cpu_sub_group(self, sub_group_id):
        step = self._submitted_cpu_sub_groups.pop(sub_group_id)
        # The rollback reads the gradient that the step read.
        self._cpu_step_and_wait(sub_group_id,
                                self._worker_grad(sub_group_id),
                                step.hyperparams,
                                step.grad_scale,
                                rollback=True)

    @instrument_w_nvtx
    def check_clip_grads(self, total_norm):
        """Check if unscale_and_clip_grads will scale the gradients down, with the same test."""
        if not self.clip_grad:
            return False
        clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
        return bool(clip > 1.0)

    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None,
                        load_serial=None,
                        param_shapes=None):
        get_accelerator().current_stream().synchronize()
        super().load_state_dict(state_dict_list,
                                load_optimizer_states=load_optimizer_states,
                                load_from_fp32_weights=load_from_fp32_weights,
                                checkpoint_folder=checkpoint_folder,
                                load_serial=load_serial,
                                param_shapes=param_shapes)
