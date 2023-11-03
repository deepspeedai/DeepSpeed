# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch

from ....inference_utils import ActivationType, DtypeEnum
from deepspeed.ops.op_builder import InferenceCoreBuilder
from ... import DSKernelBase


class CUDABiasActivation(DSKernelBase):
    """
    CUDA implementation of bias activation kernel. This kernel should be deprecated once
    we are fusing the bias activation into the linear kernel in all scenarios.
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]
    supported_act_fns = [ActivationType.IDENTITY, ActivationType.GELU, ActivationType.RELU, ActivationType.SILU]

    def __init__(self, channels: int, dtype: DtypeEnum, act_fn: ActivationType) -> None:
        """
        Compile and validate for the fused bias-activation kernel.

        Parameters:
            channels (int): Number of channels to expect in the activation.
            dtype (torch.dtype): Data type for the input/output. Supported values
                are DtypeEnum.fp16 and DtypeEnum.bf16.
            act_fn (ActivationType): Activation function to use. Only IDENTITY, GELU, RELU, and SILU are supported.
        """

        if channels % 8 != 0:
            raise ValueError("channels must be divisible by 8")

        if DtypeEnum(dtype) not in CUDABiasActivation.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                dtype, CUDABiasActivation.supported_dtypes))

        act_fn = ActivationType(act_fn)
        if act_fn not in CUDABiasActivation.supported_act_fns:
            raise ValueError("Unsupported activation function: {}, supported_act_fns are {}".format(
                act_fn, CUDABiasActivation.supported_act_fns))

        inf_module = InferenceCoreBuilder().load()
        self.kernel = inf_module.bias_activation
        self.act_fn = act_fn

    def __call__(self, activation: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add an optional bias and perform the non-linear activation function.

        Parameters:
            activation (torch.Tensor): Input tensor of shape [tokens, channels]
            bias (torch.Tensor): Optional bias tensor of shape [channels]

        Returns:
            activation that has been updated in-place
        """
        self.kernel(activation, bias, self.act_fn.value)
