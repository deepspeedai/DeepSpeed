'''
Copyright 2019 The Microsoft DeepSpeed Team

Copyright NVIDIA/apex
This file is adapted from FP16_Optimizer in NVIDIA/apex
'''

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import math

from deepspeed.runtime.utils import get_grad_norm, CheckOverflow, get_weight_norm
from deepspeed.runtime.fp16.loss_scaler import INITIAL_LOSS_SCALE, SCALE_WINDOW, MIN_LOSS_SCALE
from deepspeed.utils import logger


class FP16_UnfusedOptimizer(object):
    """
    FP16 Optimizer without weight fusion to support LAMB optimizer

    For usage example please see, TODO:  DeepSpeed V2 Tutorial
    """
    def __init__(self,
                 init_optimizer,
                 deepspeed=None,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 mpu=None,
                 clip_grad=0.0,
                 fused_lamb_legacy=False):

        self.fused_lamb_legacy = fused_lamb_legacy

        if torch.distributed.get_rank() == 0:
            logger.info(f'Fused Lamb Legacy : {self.fused_lamb_legacy} ')

        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        # param groups
        self.fp16_groups = []
        self.fp32_groups = []

        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            #fp16 weights that represents the actual model weights
            self.fp16_groups.append(param_group['params'])

            #creating a fp32 copy of the weights that will be updated first then
            #copied to fp16 weights
            fp32_group = [p.clone().float().detach() for p in param_group['params']]

            #incase the internal optimizer needs it
            for p in fp32_group:
                p.requires_grad = True

            #setting the param groups in the optimizer to point to fp32
            #note these are not the weights used by the model
            #the model uses the fp16 version that we added to fp16_group
            self.fp32_groups.append(fp32_group)
            param_group['params'] = self.fp32_groups[i]

        # we may have a way of fusing dynamic scale. Do not support for now
        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2.0
            if dynamic_loss_args is None:
                self.cur_scale = 1.0 * 2**16
                self.scale_window = 1000
                self.min_loss_scale = 0.25
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

        self.mpu = mpu

        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups,
                                              mpu=self.mpu,
                                              deepspeed=deepspeed)

        self.initialize_optimizer_states()

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist outside of the step function
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def step_fused_lamb(self, closure=None):
        """
        Not supporting closure.
        """
        # First compute norm for all group so we know if there is overflow
        grads_groups_flat = []
        grads_groups = []
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            grads = [
                torch.zeros(p.size(),
                            dtype=p.dtype,
                            device=p.device) if p.grad is None else p.grad for p in group
            ]
            grads_groups.append(grads)
            grads_groups_flat.append(_flatten_dense_tensors(grads))
            norm_groups.append(get_weight_norm(grads_groups_flat[i], mpu=self.mpu))

        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        prev_scale = self.cur_scale

        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                logger.info(
                    "[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                    "scale: {}, reducing to {}".format(prev_scale,
                                                       self.cur_scale))
            return self.overflow

        combined_scale = self.unscale_and_clip_grads(norm_groups, apply_scale=False)
        self.optimizer.step(grads=grads_groups,
                            output_params=self.fp16_groups,
                            scale=combined_scale)

        for fp32_group, fp16_group in zip(self.fp32_groups, self.fp16_groups):
            for idx, (fp32_param, fp16_param) in enumerate(zip(fp32_group, fp16_group)):

                #remove the fp32 grad
                fp32_param.grad = None

                #copy data from fp32 to fp16
                fp16_param.data.copy_(fp32_param.data)

        return self.overflow

    def step(self, closure=None):
        """
        Not supporting closure.
        """

        if self.fused_lamb_legacy:
            return self.step_fused_lamb()

        self.overflow = self.overflow_checker.check()
        prev_scale = self.cur_scale

        self._update_scale(self.overflow)
        if self.overflow:
            if self.verbose:
                logger.info(
                    "[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                    "scale: {}, reducing to {}".format(prev_scale,
                                                       self.cur_scale))
            return self.overflow

        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
            norm_groups.append(get_grad_norm(group, mpu=self.mpu))

            # copying gradients to fp32 to wor  k with fp32 parameters
            for fp32_param, fp16_param in zip(self.fp32_groups[i], self.fp16_groups[i]):
                if fp16_param.grad is None:
                    fp32_param.grad = torch.zeros(fp16_param.size(),
                                                  dtype=fp32_param.dtype,
                                                  device=fp32_param.device)
                else:
                    fp32_param.grad = fp16_param.grad.to(fp32_param.dtype)

        self.unscale_and_clip_grads(norm_groups)

        self.optimizer.step()

        for fp32_group, fp16_group in zip(self.fp32_groups, self.fp16_groups):
            for idx, (fp32_param, fp16_param) in enumerate(zip(fp32_group, fp16_group)):

                #remove the fp32 grad
                fp32_param.grad = None

                #copy data from fp32 to fp16
                fp16_param.data.copy_(fp32_param.data)

        return self.overflow

    def unscale_and_clip_grads(self, norm_groups, apply_scale=True):
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
            for group in self.fp32_groups:
                for param in group:
                    if param.grad is not None:
                        param.grad.data.mul_(1. / combined_scale)

        return combined_scale

    def backward(self, loss, create_graph=False, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        scaled_loss = (loss.float()) * self.cur_scale

        scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            prev_scale = self.cur_scale
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor,
                                     self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
                if self.verbose:
                    logger.info("Grad overflow on iteration: %s", self.cur_iter)
                    logger.info(
                        f"Reducing dynamic loss scale from {prev_scale} to {self.cur_scale}"
                    )
            else:
                # Ensure self.scale_window updates since last overflow
                stable_interval = (self.cur_iter - self.last_overflow_iter) - 1
                if (stable_interval > 0) and (stable_interval % self.scale_window == 0):
                    self.cur_scale *= self.scale_factor
                    if self.verbose:
                        logger.info(
                            f"No Grad overflow for {self.scale_window} iterations")
                        logger.info(
                            f"Increasing dynamic loss scale from {prev_scale} to {self.cur_scale}"
                        )
        else:
            if skip:
                logger.info("Grad overflow on iteration %s", self.cur_iter)
                logger.info("Using static loss scale of %s", self.cur_scale)
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
        state_dict['fp32_groups'] = self.fp32_groups
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
        for current_group, saved_group in zip(self.fp32_groups, state_dict['fp32_groups']):
            for current, saved in zip(current_group, saved_group):
                current.data.copy_(saved.data)

    def __repr__(self):
        return repr(self.optimizer)

    def initialize_optimizer_states(self):
        for i, group in enumerate(self.fp16_groups):
            for param in group:
                param.grad = torch.zeros(param.size(),
                                         dtype=param.dtype,
                                         device=torch.cuda.current_device())

        for i, group in enumerate(self.fp32_groups):
            for param in group:
                param.grad = torch.zeros(param.size(),
                                         dtype=param.dtype,
                                         device=torch.cuda.current_device())

        self.optimizer.step()

        for i, group in enumerate(self.fp16_groups):
            for param in group:
                param.grad = None

        for i, group in enumerate(self.fp32_groups):
            for param in group:
                param.grad = None
