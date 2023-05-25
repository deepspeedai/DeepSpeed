# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
Taken and modified for DeepSpeed from:
    https://github.com/NVIDIA/Megatron-LM/blob/master/fp16/loss_scaler.py
Commit: 93ab4bea59dc5cbf97c079d313741866af4deac9
"""

import torch
from deepspeed import comm as dist
from deepspeed.utils import logger

INITIAL_LOSS_SCALE = 'init_scale'
SCALE_WINDOW = 'scale_window'
DELAYED_SHIFT = 'delayed_shift'
CONSECUTIVE_HYSTERESIS = 'consecutive_hysteresis'
MIN_LOSS_SCALE = 'min_scale'


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    return t[0]


class LossScalerBase:
    """LossScalarBase
    Base class for a loss scaler
    """

    def __init__(self, cur_scale):
        self.cur_scale = cur_scale
        self.dynamic = False

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def update_scale(self, overflow):
        pass

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)
        # print(f'LossScalerBackward: {scaled_loss=}')


class LossScaler(LossScalerBase):
    """
    Class that manages a static loss scale.  This class is intended to interact with
    :class:`FP16_Optimizer`, and should not be directly manipulated by the user.

    Use of :class:`LossScaler` is enabled via the ``static_loss_scale`` argument to
    :class:`FP16_Optimizer`'s constructor.

    Args:
        scale (float, optional, default=1.0):  The loss scale.
    """

    def __init__(self, scale=1):
        super(LossScaler, self).__init__(scale)

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        return False


class DynamicLossScaler(LossScalerBase):
    """
    Class that manages dynamic loss scaling.  It is recommended to use :class:`DynamicLossScaler`
    indirectly, by supplying ``dynamic_loss_scale=True`` to the constructor of
    :class:`FP16_Optimizer`.  However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the
    the ``dynamic_loss_args`` argument to :class:`FP16_Optimizer`'s constructor.

    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.  If overflowing gradients are
    encountered, :class:`DynamicLossScaler` informs :class:`FP16_Optimizer` that an overflow has
    occurred.
    :class:`FP16_Optimizer` then skips the update step for this particular iteration/minibatch,
    and :class:`DynamicLossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.

    Args:
        init_scale (float, optional, default=2**32):  Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float, optional, default=2.0):  Factor used when adjusting the loss scale. If an overflow is encountered, the loss scale is readjusted to loss scale/``scale_factor``.  If ``scale_window`` consecutive iterations take place without an overflow, the loss scale is readjusted to loss_scale*``scale_factor``.
        scale_window (int, optional, default=1000):  Number of consecutive iterations without an overflow to wait before increasing the loss scale.
        consecutive_hysteresis (bool, optional, default=False): Whether to refill hysteresis if we reach an iteration that doesn't overflow
    """

    def __init__(self,
                 init_scale=2**32,
                 scale_factor=2.,
                 scale_window=1000,
                 min_scale=1,
                 delayed_shift=1,
                 consecutive_hysteresis=False,
                 raise_error_at_min_scale=True,
                 dtype=torch.half):
        super(DynamicLossScaler, self).__init__(init_scale)
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.dynamic = True
        self.dtype = dtype

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float('inf'), -float('inf')] or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
        if overflow:
            # self.cur_scale /= self.scale_factor
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                if (self.cur_scale == self.min_scale) and self.raise_error_at_min_scale:
                    raise Exception(
                        "Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.")
                else:
                    next_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
                    if dist.get_rank() == 0:
                        overflow_msg = f"[deepspeed] OVERFLOW! Rank {dist.get_rank()} Skipping step."
                        if self.dtype == torch.half:
                            overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, reducing to {int(next_scale)}"
                        logger.info(overflow_msg)
                    self.cur_scale = next_scale
            else:
                if dist.get_rank() == 0:
                    overflow_msg = f"[deepspeed] OVERFLOW! Rank {dist.get_rank()} Skipping step."
                    if self.dtype == torch.half:
                        overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, but hysteresis is {self.cur_hysteresis}. Reducing hysteresis to {self.cur_hysteresis-1}"
                    logger.info(overflow_msg)
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                if dist.get_rank() == 0:
                    hysteresis_msg = f"Consecutive hysteresis is enabled. Restoring hysteresis to {self.delayed_shift}"
                    logger.info(hysteresis_msg)
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1


# Although loss scaling is only defined for fp16, yet for backwards compatibility
# we still create a scaler for other dtypes (fp32, bf16) which does not perform any scaling.
def CreateLossScaler(dtype, static_loss_scale, dynamic_scaling, dynamic_loss_args):
    if dtype == torch.half and dynamic_scaling:
        if dynamic_loss_args is None:
            return DynamicLossScaler(dtype=dtype)
        return DynamicLossScaler(dtype=dtype, **dynamic_loss_args)

    loss_scale_value = static_loss_scale if dtype == torch.half else 1.0
    return LossScaler(scale=loss_scale_value)


##############################################################
# Example usage below here -- assuming it's in a separate file
##############################################################
"""
TO-DO separate out into an example.
if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    from dynamic_loss_scaler import DynamicLossScaler

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
    x = Variable(torch.randn(N, D_in), requires_grad=False)
    y = Variable(torch.randn(N, D_out), requires_grad=False)

    w1 = Variable(torch.randn(D_in, H), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out), requires_grad=True)
    parameters = [w1, w2]

    learning_rate = 1e-6
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    loss_scaler = DynamicLossScaler()

    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum() * loss_scaler.loss_scale
        print('Iter {} loss scale: {}'.format(t, loss_scaler.loss_scale))
        print('Iter {} scaled loss: {}'.format(t, loss.data[0]))
        print('Iter {} unscaled loss: {}'.format(t, loss.data[0] / loss_scaler.loss_scale))

        # Run backprop
        optimizer.zero_grad()
        loss.backward()

        # Check for overflow
        has_overflow = DynamicLossScaler.has_overflow(parameters)

        # If no overflow, unscale grad and update as usual
        if not has_overflow:
            for param in parameters:
                param.grad.data.mul_(1. / loss_scaler.loss_scale)
            optimizer.step()
        # Otherwise, don't do anything -- ie, skip iteration
        else:
            print('fp16 dynamic loss scale overflow!')

        # Update loss scale for next iteration
        loss_scaler.update_scale(has_overflow)

"""
