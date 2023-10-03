# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from FP16_Optimizer in NVIDIA/apex
"""

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.runtime import DeepSpeedOptimizer
from deepspeed.runtime.utils import get_global_norm, get_grad_norm, CheckOverflow, get_weight_norm, required_torch_version
from deepspeed.runtime.fp16.loss_scaler import INITIAL_LOSS_SCALE, SCALE_WINDOW, MIN_LOSS_SCALE
from deepspeed.utils import groups, logger, log_dist
from deepspeed import comm as dist
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, CLIP_GRAD
from deepspeed.accelerator import get_accelerator

OVERFLOW_CHECK_TIMER = 'overflow_check'
COMPUTE_NORM_TIMER = 'compute_norm'
UNSCALE_AND_CLIP_TIMER = 'unscale_and_clip'
BASIC_STEP_TIMER = 'basic_step'
UPDATE_FP16_TIMER = 'update_fp16'

OVERFLOW_TIMERS = [COMPUTE_NORM_TIMER, OVERFLOW_CHECK_TIMER]
STEP_TIMERS = OVERFLOW_TIMERS + [UNSCALE_AND_CLIP_TIMER, BASIC_STEP_TIMER, UPDATE_FP16_TIMER]


class FP16_Optimizer(DeepSpeedOptimizer):
    """
   FP16 Optimizer for training fp16 models. Handles loss scaling.

   For usage example please see, TODO:  DeepSpeed V2 Tutorial
    """

    def __init__(self,
                 init_optimizer,
                 deepspeed=None,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 initial_dynamic_scale=2**32,
                 dynamic_loss_args=None,
                 verbose=True,
                 mpu=None,
                 clip_grad=0.0,
                 fused_adam_legacy=False,
                 has_moe_layers=False,
                 timers=None):

        self.fused_adam_legacy = fused_adam_legacy
        self.timers = timers
        self.deepspeed = deepspeed
        self.has_moe_layers = has_moe_layers
        self.using_pipeline = self.deepspeed.pipeline_parallelism
        if not get_accelerator().is_available():
            raise SystemError("Cannot use fp16 without accelerator.")
        self.optimizer = init_optimizer

        # param flattened by groups
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.fp32_groups_flat = []

        self._global_grad_norm = 0.

        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            # push this group to list before modify
            self.fp16_groups.append(param_group['params'])
            # init fp16 weight buffer, flattened
            self.fp16_groups_flat.append(_flatten_dense_tensors([p.clone().detach() for p in self.fp16_groups[i]]))
            # set model fp16 weight to slices of flattened buffer
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
            # init master weight, flattened
            self.fp32_groups_flat.append(self.fp16_groups_flat[i].clone().float().detach())
            # modify optimizer of have flat master weight
            self.fp32_groups_flat[i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [self.fp32_groups_flat[i]]

        # we may have a way of fusing dynamic scale. Do not support for now
        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2

            if dynamic_loss_args is None:
                self.cur_scale = initial_dynamic_scale
                self.scale_window = 1000
                self.min_loss_scale = 1
            else:
                self.cur_scale = dynamic_loss_args[INITIAL_LOSS_SCALE]
                self.scale_window = dynamic_loss_args[SCALE_WINDOW]
                self.min_loss_scale = dynamic_loss_args[MIN_LOSS_SCALE]
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.verbose = verbose

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        self.clip_grad = clip_grad
        self.norm_type = 2
        self.step_count = 0

        if required_torch_version(max_version=0.4):
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

        #model parallel object
        self.mpu = mpu

        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu, deepspeed=deepspeed)
        self.initialize_optimizer_states()

    def initialize_optimizer_states(self):
        for i, group in enumerate(self.fp16_groups):
            self.fp32_groups_flat[i].grad = torch.zeros(self.fp32_groups_flat[i].size(),
                                                        device=self.fp32_groups_flat[i].device)

        self.optimizer.step()

        for i, group in enumerate(self.fp16_groups):
            self.fp32_groups_flat[i].grad = None

        return

    def zero_grad(self, set_to_none=True):
        """
        Zero FP16 parameter grads.
        """
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def step_fused_adam(self, closure=None):
        """
        Not supporting closure.
        """

        # First compute norm for all group so we know if there is overflow
        grads_groups_flat = []
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            grads_groups_flat.append(
                _flatten_dense_tensors([
                    torch.zeros(p.size(), dtype=p.dtype, device=p.device) if p.grad is None else p.grad for p in group
                ]))
            norm_groups.append(get_weight_norm(grads_groups_flat[i], mpu=self.mpu))

        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)

        if self.overflow:
            if self.verbose:
                logger.info("[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                            "scale: {}, reducing to {}".format(prev_scale, self.cur_scale))
            return self.overflow

        scaled_grad_norm = get_global_norm(norm_list=norm_groups)

        combined_scale = self.unscale_and_clip_grads(grads_groups_flat, scaled_grad_norm, apply_scale=False)

        # Stash unscaled gradient norm
        self._global_grad_norm = scaled_grad_norm / self.cur_scale

        # norm is in fact norm*cur_scale
        self.optimizer.step(grads=[[g] for g in grads_groups_flat],
                            output_params=[[p] for p in self.fp16_groups_flat],
                            scale=combined_scale,
                            grad_norms=norm_groups)
        # TODO: we probably don't need this? just to be safe
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
        return self.overflow

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def override_loss_scale(self, loss_scale):
        if loss_scale != self.external_loss_scale:
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def step(self, closure=None):
        """
        Not supporting closure.
        """

        if self.fused_adam_legacy:
            return self.step_fused_adam()

        # First determine if there is overflow.
        self.timers(OVERFLOW_CHECK_TIMER).start()
        fp16_params = []
        for i, group in enumerate(self.fp16_groups):
            fp16_params.extend([p for p in group if p.grad is not None])
        self.overflow = self.overflow_checker.has_overflow(fp16_params)
        self.timers(OVERFLOW_CHECK_TIMER).stop()
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                log_dist(
                    "Overflow detected. Skipping step. Attempted loss "
                    f"scale: {prev_scale}, reducing to {self.cur_scale}",
                    ranks=[0])
            # Clear gradients
            for i, group in enumerate(self.fp16_groups):
                for p in group:
                    p.grad = None

            self.timers.log(OVERFLOW_TIMERS)
            return self.overflow

        grads_groups_flat = []
        for i, group in enumerate(self.fp16_groups):
            data_type = self.fp32_groups_flat[i].dtype

            grads_groups_flat.append(
                _flatten_dense_tensors([
                    torch.zeros(p.size(), dtype=data_type, device=p.device) if p.grad is None else p.grad.to(data_type)
                    for p in group
                ]))

            for p in group:
                p.grad = None

            self.fp32_groups_flat[i].grad = grads_groups_flat[i]

        self.timers(COMPUTE_NORM_TIMER).start()

        all_groups_norm = get_grad_norm(self.fp32_groups_flat, mpu=self.mpu)

        self.timers(COMPUTE_NORM_TIMER).stop()

        if self.has_moe_layers:
            all_groups_norm = self._get_norm_with_moe_layers(all_groups_norm)

        scaled_global_grad_norm = get_global_norm(norm_list=[all_groups_norm])

        # Stash unscaled gradient norm
        self._global_grad_norm = scaled_global_grad_norm / self.cur_scale

        self.timers(UNSCALE_AND_CLIP_TIMER).start()
        self.unscale_and_clip_grads(grads_groups_flat, scaled_global_grad_norm)
        self.timers(UNSCALE_AND_CLIP_TIMER).stop()

        self.timers(BASIC_STEP_TIMER).start()
        self.optimizer.step()
        self.timers(BASIC_STEP_TIMER).stop()

        #get rid of the fp32 gradients. Not needed anymore
        for group in self.fp32_groups_flat:
            group.grad = None

        self.timers(UPDATE_FP16_TIMER).start()

        for i in range(len(self.fp16_groups)):
            updated_params = _unflatten_dense_tensors(self.fp32_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data.copy_(q.data)

        self.timers(UPDATE_FP16_TIMER).stop()

        self.timers.log(STEP_TIMERS)

        self.step_count += 1

        return self.overflow

    def _get_norm_with_moe_layers(self, all_groups_norm):
        #all_groups_norm_old = all_groups_norm
        # Need to allreduce (avg) the norms across different ranks because moe params will not be synced during allreduce
        if self.using_pipeline:
            pg = self.deepspeed.mpu.get_data_parallel_group()
        else:
            pg = groups._get_data_parallel_group()
        scaled_norm = all_groups_norm * 1.0 / float(dist.get_world_size(group=pg))
        scaled_norm_tensor = torch.tensor(scaled_norm, device=self.fp32_groups_flat[0].device, dtype=torch.float)
        dist.all_reduce(scaled_norm_tensor, group=pg)
        all_groups_norm = scaled_norm_tensor.item()
        #print(f"old = {all_groups_norm_old} and new = {all_groups_norm} at rank: {deepspeed.comm.get_rank()}")
        return all_groups_norm

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm, apply_scale=True):
        # compute combined scale factor for this group
        combined_scale = self.cur_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.cur_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.cur_scale

        if apply_scale:
            for grad in grad_groups_flat:
                grad.data.mul_(1. / combined_scale)

        return combined_scale

    def backward(self, loss, create_graph=False, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            scaled_loss = (loss.float()) * self.cur_scale
            scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            prev_scale = self.cur_scale
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
                if self.verbose:
                    logger.info(f"\nGrad overflow on iteration {self.cur_iter}")
                    logger.info(f"Reducing dynamic loss scale from {prev_scale} to {self.cur_scale}")
            else:
                # Ensure self.scale_window updates since last overflow
                stable_interval = (self.cur_iter - self.last_overflow_iter) - 1
                if (stable_interval > 0) and (stable_interval % self.scale_window == 0):
                    self.cur_scale *= self.scale_factor
                    if self.verbose:
                        logger.info(f"No Grad overflow for {self.scale_window} iterations")
                        logger.info(f"Increasing dynamic loss scale from {prev_scale} to {self.cur_scale}")
        else:
            if skip:
                logger.info("Grad overflow on iteration: %s", self.cur_iter)
                logger.info("Using static loss scale of: %s", self.cur_scale)
        self.cur_iter += 1
        return

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict[OPTIMIZER_STATE_DICT] = self.optimizer.state_dict()
        state_dict['fp32_groups_flat'] = self.fp32_groups_flat
        state_dict[CLIP_GRAD] = self.clip_grad
        return state_dict

    # Refresh fp32 master params from fp16 copies
    def refresh_fp32_params(self):
        for current, saved in zip(self.fp32_groups_flat, self.fp16_groups_flat):
            current.data.copy_(saved.data)

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).to(get_accelerator().device_name()).half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # I think it should actually be ok to reload the optimizer before the model.
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_DICT])
        self.clip_grad = state_dict[CLIP_GRAD]
        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 2.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.
        for current, saved in zip(self.fp32_groups_flat, state_dict['fp32_groups_flat']):
            current.data.copy_(saved.data)

    def __repr__(self):
        return repr(self.optimizer)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        if self.custom_loss_scaler:
            return self.external_loss_scale
        else:
            return self.cur_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
