# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

#Linear Module to use with ZeRO Stage 3 to allow for parameter memory release
#after the module execution during forward
#Instead of saving variables using save_for_backward, we save variable ids
#Allowing us to retrieve the variable without creating pointer to it
#Which allows for underlying tensor to be garbage collected
#When partitioned as needed by the Zero Stage 3 optimizer
#TODO instead of patching Linear module, we could patch the ctx.save_for_backward
#ctx.saved_tensors so that this approach works for all nn modules that are built upon
#torch.nn.function. However the issue is that many modules uses C++ implementations
#which does not have pytorch implementation. Eg torch.addmm which acts as a functional
#when implemented outside of torch.autograd.Function

import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.module import Module
from deepspeed.runtime.utils import noop_decorator
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator


def print_rank_0(message, debug=False, force=False):
    if dist.get_rank() == 0 and (debug or force):
        print(message)


try:
    autocast_custom_fwd = get_accelerator().amp().custom_fwd
    autocast_custom_bwd = get_accelerator().amp().custom_bwd
except (ImportError, AttributeError) as exp:
    autocast_custom_fwd = noop_decorator
    autocast_custom_bwd = noop_decorator


class LinearFunctionForZeroStage3(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    @autocast_custom_fwd
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):

        ctx.save_for_backward(input, weight, bias)

        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            ret = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
            ret = output

        return ret

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        #print(f"backward shaped grad_output {grad_output.shape}, input {input.shape}, weight {weight.shape} and bias {bias.shape if bias is not None else None}")
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            #print(f"Computing grad input weight {weight.shape} grad_output {grad_output.shape}")
            grad_input = grad_output.matmul(weight)
            #print(f"Computed grad input {grad_input.shape}")
        if ctx.needs_input_grad[1]:
            #print("Computing grad weight")
            dim = grad_output.dim()
            if dim > 2:
                grad_weight = grad_output.reshape(-1,
                                                  grad_output.shape[-1]).t().matmul(input.reshape(-1, input.shape[-1]))
            else:
                grad_weight = grad_output.t().matmul(input)
            #print(f"Computed grad weight grad_weight {grad_weight.shape}")
        if bias is not None and ctx.needs_input_grad[2]:
            #print("Computing grad bias")
            grad_bias = grad_output.sum(0)
            #print("Done computing grad bias")
            #print("needs bias")
        #print(f"backward shaped grad_input {grad_input.shape}, grad_weight {grad_weight.shape}, grad_bias {grad_bias.shape if grad_bias is not None else None}")
        return grad_input, grad_weight, grad_bias


def zero3_linear_wrap(input, weight, bias=None):
    if bias is None:
        return LinearFunctionForZeroStage3.apply(input, weight)
    else:
        return LinearFunctionForZeroStage3.apply(input, weight, bias)


class LinearModuleForZeroStage3(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    The weights are pre-transposed and stored as A^T instead of transposing during each
    forward. Memory savings proportional to the parameter size.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LinearModuleForZeroStage3, self).__init__()
        print("Building ZeRO module")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return LinearFunctionForZeroStage3.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias
                                                                 is not None)
