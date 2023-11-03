# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ....inference_utils import DtypeEnum
from deepspeed.ops.op_builder import InferenceCoreBuilder
from ... import DSKernelBase


class BlasLibLinear(DSKernelBase):
    """
    Wrapper around the BLAS matmul kernel for FP16/BF16/FP32 for CUDA/RoCM.

    Performs z = x @ y
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16, DtypeEnum.fp32]

    def __init__(self, fp_dtype: DtypeEnum):
        """
        Parameters:
            fp_dtype (torch.dtype): Data type for the input/output. Supported values
                are torch.float16, torch.bfloat16, and torch.float32.
        """
        fp_dtype = DtypeEnum(fp_dtype)
        if fp_dtype not in BlasLibLinear.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                fp_dtype, BlasLibLinear.supported_dtypes))

        self.inf_module = InferenceCoreBuilder().load()
        self.inf_module.create_handle()
        self.kernel = self.inf_module.blas_linear

    def __call__(self, output: torch.Tensor, hidden_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Matmul kernel as implemented by platform BLAS library. The input must be 2D or larger. If
        n-dimensional, the leading dimensions are folded into each other:
            2D: m = x.size(0)
            3D: m = x.size(0) * x.size(1)
            4D: m = x.size(0) * x.size(1) * x.size(2) (etc...)
        All inputs should be contiguous.

        Parameters:
            output (torch.Tensor): Output tensor. Shape is of [*, out_features]
            hidden_states (torch.Tensor): Input tensor. Shape is of [*, in_features]
            weights (torch.Tensor): Input tensor. Shape is of [out_features, in_features]

        Returns:
            z (torch.Tensor): Output tensor. Shape is of [m, n]
        """
        self.kernel(output, hidden_states, weights)
        return output
