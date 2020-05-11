'''
Copyright 2019 The Microsoft DeepSpeed Team

Copyright NVIDIA/apex
This file is adapted from FP16_Optimizer in NVIDIA/apex
'''

import torch
import logging
import math
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.pt.deepspeed_utils import get_grad_norm, CheckOverflow, get_weight_norm
from deepspeed.pt.loss_scaler import INITIAL_LOSS_SCALE, SCALE_WINDOW, MIN_LOSS_SCALE


class FP16_Optimizer(object):
    """
   FP16 Optimizer for training fp16 models. Handles loss scaling.

   For usage example please see, TODO:  DeepSpeed V2 Tutorial
    """
    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 initial_dynamic_scale=2**32,
                 dynamic_loss_args=None,
                 verbose=True,
                 mpu=None,
                 clip_grad=0.0,
                 fused_adam_legacy=False):

        self.fused_adam_legacy = fused_adam_legacy

        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        # param flattened by groups
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.fp32_groups_flat = []

        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            # push this group to list before modify
            self.fp16_groups.append(param_group['params'])
            # init fp16 weight buffer, flattened
            self.fp16_groups_flat.append(
                _flatten_dense_tensors([p.clone().detach()
                                        for p in self.fp16_groups[i]]))
            # set model fp16 weight to slices of flattened buffer
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
                                                      self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
            # init master weight, flattened
            self.fp32_groups_flat.append(
                self.fp16_groups_flat[i].clone().float().detach())
            # modify optimizer of have flat master weight
            self.fp32_groups_flat[
                i].requires_grad = True  # keep this in case internal optimizer uses it
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

        self.clip_grad = clip_grad
        self.norm_type = 2

        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        if TORCH_MAJOR == 0 and TORCH_MINOR <= 4:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

        #model parallel object
        self.mpu = None

        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu)

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
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
                    torch.zeros(p.size(),
                                dtype=p.dtype,
                                device=p.device) if p.grad is None else p.grad
                    for p in group
                ]))
            norm_groups.append(get_weight_norm(grads_groups_flat[i], mpu=self.mpu))

        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)

        if self.overflow:
            if self.verbose:
                print("[deepspeed] OVERFLOW! Skipping step. Attempted loss "
                      "scale: {}, reducing to {}".format(prev_scale,
                                                         self.cur_scale))
            return self.overflow
        combined_scale = self.unscale_and_clip_grads(grads_groups_flat,
                                                     norm_groups,
                                                     apply_scale=False)
        # norm is in fact norm*cur_scale
        self.optimizer.step(grads=[[g] for g in grads_groups_flat],
                            output_params=[[p] for p in self.fp16_groups_flat],
                            scale=combined_scale,
                            grad_norms=norm_groups)
        # TODO: we probably don't need this? just to be safe
        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
                                                      self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
        return self.overflow

    def step(self, closure=None):
        """
        Not supporting closure.
        """

        if self.fused_adam_legacy:
            return self.step_fused_adam()

        # First compute norm for all group so we know if there is overflow
        grads_groups_flat = []
        norm_groups = []

        for i, group in enumerate(self.fp16_groups):
            data_type = self.fp32_groups_flat[i].dtype

            grads_groups_flat.append(
                _flatten_dense_tensors([
                    torch.zeros(p.size(),
                                dtype=data_type,
                                device=p.device)
                    if p.grad is None else p.grad.to(data_type) for p in group
                ]))

            self.fp32_groups_flat[i].grad = grads_groups_flat[i]

            norm_groups.append(get_grad_norm(self.fp32_groups_flat, mpu=self.mpu))

        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)

        if self.overflow:
            if self.verbose:
                print("[deepspeed] OVERFLOW! Skipping step. Attempted loss "
                      "scale: {}, reducing to {}".format(prev_scale,
                                                         self.cur_scale))
            return self.overflow

        self.unscale_and_clip_grads(grads_groups_flat, norm_groups)

        self.optimizer.step()

        #get rid of the fp32 gradients. Not needed anymore
        for group in self.fp32_groups_flat:
            group.grad = None

        for i in range(len(norm_groups)):
            updated_params = _unflatten_dense_tensors(self.fp32_groups_flat[i],
                                                      self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data.copy_(q.data)

        return self.overflow

    def unscale_and_clip_grads(self, grad_groups_flat, norm_groups, apply_scale=True):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm**2.0
        total_norm = math.sqrt(total_norm)

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

    def backward(self, loss):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        scaled_loss = (loss.float()) * self.cur_scale
        scaled_loss.backward()

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            prev_scale = self.cur_scale
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor,
                                     self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
                if self.verbose:
                    print(f"\nGrad overflow on iteration {self.cur_iter}")
                    print(
                        f"Reducing dynamic loss scale from {prev_scale} to {self.cur_scale}"
                    )
            else:
                # Ensure self.scale_window updates since last overflow
                stable_interval = (self.cur_iter - self.last_overflow_iter) - 1
                if (stable_interval > 0) and (stable_interval % self.scale_window == 0):
                    self.cur_scale *= self.scale_factor
                    if self.verbose:
                        print(f"\nNo Grad overflow for {self.scale_window} iterations")
                        print(
                            f"Increasing dynamic loss scale from {prev_scale} to {self.cur_scale}"
                        )
        else:
            if skip:
                print("\nGrad overflow on iteration", self.cur_iter)
                print("Using static loss scale of", self.cur_scale)
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
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_groups_flat'] = self.fp32_groups_flat
        state_dict['clip_grad'] = self.clip_grad
        return state_dict

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
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
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.clip_grad = state_dict['clip_grad']
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
