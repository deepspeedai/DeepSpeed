# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ... import DSKernelBase
from ....inference_utils import elem_size
from deepspeed.ops.op_builder import InferenceCoreBuilder


class CUDAFPLNBase(DSKernelBase):
    """
    Base class for CUDA LN kernels. They all same the same validation logic,
    so we can share it here.
    """

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(self, channels: int, fp_dtype: torch.dtype, epsilon: float = 1e-5):
        """
        Parameters:
            channels (int): Number of channels in the input tensor. Must be divisible to align
                to 16 bytes.
            fp_dtype (torch.dtype): Data type for the input/output/gamma. Supported values
                are torch.float16, torch.bfloat16, and torch.float32.
        """
        if fp_dtype not in CUDAFPLNBase.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                fp_dtype, CUDAFPLNBase.supported_dtypes))

        if elem_size(fp_dtype) * channels % 16 != 0:
            raise ValueError("channels must be divisible by 16 bytes")

        self.inf_module = InferenceCoreBuilder().load()
        self.epsilon = epsilon
