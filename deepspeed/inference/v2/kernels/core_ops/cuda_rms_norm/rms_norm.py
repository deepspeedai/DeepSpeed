# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .rms_norm_base import CUDARMSNormBase


class CUDARMSNorm(CUDARMSNormBase):
    """
    Floating point layer norm kernel for CUDA/RoCM.

    Performs: z = ln(x)
    """

    def __call__(self, output_z: torch.Tensor, input_x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        output_z may alias input_x directly. All Tensors should have the same shape.

        Parameters:
            output_z (torch.Tensor): Output tensor.
            input_x (torch.Tensor): Input tensor.
            gamma (torch.Tensor): Gamma tensor.
        """
        self.inf_module.rms_norm(output_z, input_x, gamma, self.epsilon)
        return output_z
