# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch

from ... import DSKernelBase
from ....inference_utils import ActivationType, elem_size
from deepspeed.ops.op_builder import InferenceCoreBuilder


class CUDAGatedActivation(DSKernelBase):
    """
    CUDA implementation of gated activation kernel. This kernel assumes that the input
    tensor has gate and activation values in adjacent channels. The output tensor should
    have half the dimensionality of the input tensor.
    """

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    supported_act_fns = [ActivationType.GEGLU, ActivationType.ReGLU, ActivationType.SiGLU]

    def __init__(self, channels: int, fp_dtype: torch.dtype, act_fn: ActivationType) -> None:
        """
        Compile and validate for the gated activation function.

        Args:
            channels (int): Number of columns in the output tensor. Must be divisible to align
                to 8 bytes.
            fp_dtype (torch.dtype): Data type for the input/output/gamma. Supported values
                are torch.float16, torch.bfloat16, and torch.float32.
            act_fn (ActivationType): Activation function to use. Only GEGLU is supported.
        """
        if fp_dtype not in CUDAGatedActivation.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                fp_dtype, CUDAGatedActivation.supported_dtypes))

        act_fn = ActivationType(act_fn)
        if act_fn not in CUDAGatedActivation.supported_act_fns:
            raise ValueError("Unsupported activation function: {}, supported_act_fns are {}".format(
                act_fn, CUDAGatedActivation.supported_act_fns))

        if elem_size(fp_dtype) * channels % 8 != 0:
            raise ValueError("Channels must be divisible by 16 bytes")

        if elem_size(fp_dtype) * channels > 98304:
            raise ValueError(
                "Kernel only compiled to support 98304 bytes per row, please file an issue if your model requires more."
            )

        self.inf_module = InferenceCoreBuilder().load()
        self.act_fn = act_fn
        self.kernel = self.inf_module.gated_activation

    def __call__(self, output: torch.Tensor, input: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        """
        Performs gated activation on the input tensor, writing the result to the output tensor.

        Args:
            output (torch.Tensor): Output tensor. Can be of [T, C // 2] or [B, S, C // 2]
            input (torch.Tensor): Input tensor. Can be of [T, C] or [B, S, C]
        """
        self.kernel(output, input, bias, self.act_fn.value)
