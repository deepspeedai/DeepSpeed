# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/23.08/megatron/core/tensor_parallel/layers.py

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from typing import Callable

TP_group = None


class DominoAsyncColumnParallelLinearImpl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, weight, bias, handle_dic, h_id):  # inp: (b, s, k), weight: (m, k), bias (m)
        ctx.save_for_backward(inp, weight, bias)
        ctx.handle_dic = handle_dic
        ctx.h_id = h_id
        output = torch.matmul(inp, weight.t())  # (b, s, k) @ (k, m) -> (b, s, m)
        if bias is not None:  # bias (m)
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input = torch.matmul(grad_output, weight)  # (b, s, m) @ (m, k) -> (b, s, k)
        handle = dist.all_reduce(grad_input, group=TP_group, async_op=True)
        ctx.handle_dic[ctx.h_id] = handle
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])  # (b*s, m)

        inp = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])  # (b*s, k)
        grad_weight = torch.matmul(grad_output.t(), inp)  # (m, b*s) @ (b*s, k) -> (m, k)

        if bias is not None:
            grad_bias = grad_output.sum(dim=0)  # (b*s, m) -> (m)
        return grad_input, grad_weight, grad_bias, None, None


class DominoAsyncColumnParallelLinear(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 _tp_group,
                 config,
                 init_method: Callable,
                 bias=True,
                 skip_bias_add=False):
        super(DominoAsyncColumnParallelLinear, self).__init__()

        self.skip_bias_add = skip_bias_add

        global TP_group
        if TP_group == None:
            TP_group = _tp_group

        self.weight = Parameter(
            torch.empty(
                output_size,
                input_size,
                device=get_accelerator().current_device_name(),
                dtype=config.params_dtype,
            ))
        if config.perform_initialization:
            init_method(self.weight)

        if bias:
            self.bias = Parameter(
                torch.empty(output_size, device=get_accelerator().current_device_name(), dtype=config.params_dtype))

            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_: torch.Tensor, handle_dic, h_id):

        bias = self.bias if not self.skip_bias_add else None

        output = DominoAsyncColumnParallelLinearImpl.apply(input_, self.weight, bias, handle_dic, h_id)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinearNoComm(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config,
        init_method: Callable,
        bias: bool = True,
        stride: int = 1,
        skip_bias_add: bool = False,
    ):
        super(RowParallelLinearNoComm, self).__init__()

        self.skip_bias_add = skip_bias_add

        self.weight = Parameter(
            torch.empty(
                output_size,
                input_size,
                device=get_accelerator().current_device_name(),
                dtype=config.params_dtype,
            ))
        if config.perform_initialization:
            init_method(self.weight)
        if bias:
            self.bias = Parameter(
                torch.empty(
                    output_size,
                    device=get_accelerator().current_device_name(),
                    dtype=config.params_dtype,
                ))

            if config.perform_initialization:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        output = F.linear(input_, self.weight, bias)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
