# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch import autograd
import math


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.
    Implementation is inspired from:
        https://github.com/yaozhewei/MLPruning
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float, sigmoid: bool):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
            sigmoid (`bool`)
                Whether to apply a sigmoid on the threshold
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold
        if sigmoid:
            threshold = torch.sigmoid(threshold).item()
        ctx.sigmoid = sigmoid
        mask = inputs.clone()

        _, idx = inputs.flatten().sort(descending=True)
        j = math.ceil(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0.
        flat_out[idx[:j]] = 1.
        ctx.save_for_backward(mask)

        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        mask, = ctx.saved_tensors
        if ctx.sigmoid:
            return gradOutput.clone(), ((gradOutput * mask).sum()).view(-1), None
        else:
            return gradOutput.clone(), None, None


class SymQuantizer(torch.autograd.Function):
    """
    Symmetric quantization
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, num_groups=1):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int, >=4)
                Number of bits to use for quantization
            min_value/max_value (torch.FloatTensor)
                Used for static activation quantization
            num_groups (int)
                How many groups to partition the quantization into
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """
        assert (min_value is None and max_value is None) or (min_value is not None and max_value is not None
                                                             and num_groups == 1)
        q_range = 2**num_bits
        input_shape = input.shape
        if min_value is None:
            input = input.reshape(num_groups, -1)
            max_input = torch.amax(torch.abs(input), dim=-1).view(num_groups, -1)
        else:
            max_input = torch.max(min_value.abs(), max_value).view(-1)

        scale = 2 * max_input / q_range
        output = (input / scale).round().clamp(-q_range // 2, q_range // 2 - 1) * scale
        output = output.reshape(input_shape).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
    Asymmetric quantization
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, num_groups=1):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int, >=4)
                Number of bits to use for quantization
            min_value/max_value (torch.FloatTensor)
                Used for static activation quantization
            num_groups (int)
                How many groups to partition the quantization into
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """

        assert (min_value is None and max_value is None) or (min_value is not None and max_value is not None
                                                             and num_groups == 1)
        q_range = 2**num_bits
        input_shape = input.shape
        if min_value is None:
            input = input.reshape(num_groups, -1)
            min_value = input.amin(dim=-1, keepdim=True)
            max_value = input.amax(dim=-1, keepdim=True)

        scale = (max_value - min_value) / q_range
        zero_point = (min_value / scale).round() * scale

        output = ((input - zero_point) / scale).round().clamp(0, q_range - 1) * scale + zero_point
        output = output.reshape(input_shape).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class TernaryQuantizer(torch.autograd.Function):
    """
    Ternary quantization
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, num_groups=1):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int)
                Dummy variable
            min_value/max_value (torch.FloatTensor)
                Used for static activation quantization; for now they are dummy variable
            num_groups (int)
                How many groups to partition the quantization into
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """

        assert (min_value is None and max_value is None)
        input_flat = input.reshape(num_groups, -1)
        n = input_flat.shape[1]
        m = input_flat.norm(p=1, dim=1).div(n)
        thres = (0.7 * m).view(-1, 1)
        pos = (input_flat > thres).type(input.type())
        neg = (input_flat < -thres).type(input.type())
        mask = (input_flat.abs() > thres).type(input.type())
        alpha = ((mask * input_flat).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
        output = alpha * pos - alpha * neg
        output = output.reshape(input.shape).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class BinaryQuantizer(torch.autograd.Function):
    """
    Binary quantization
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, num_groups=1):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input which needs to be quantized
            num_bits (int)
                Dummy variable
            min_value/max_value (torch.FloatTensor)
                Used for static activation quantization; for now they are dummy variable
            num_groups (int)
                How many groups to partition the quantization into
        Returns:
            quantized_input (`torch.FloatTensor`)
                Quantized input
        """

        assert (min_value is None and max_value is None)
        input_flat = input.reshape(num_groups, -1)
        n = input_flat.shape[1]
        m = input_flat.norm(p=1, dim=1, keepdim=True).div(n)
        output = input_flat.sign().mul(m)
        output = output.reshape(input.shape).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None
