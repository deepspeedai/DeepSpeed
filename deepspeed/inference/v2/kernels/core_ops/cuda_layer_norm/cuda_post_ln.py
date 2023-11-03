# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .cuda_fp_ln_base import CUDAFPLNBase


class CUDAFPPostLN(CUDAFPLNBase):
    """
    Floating point post-LayerNorm kernel for CUDA/RoCM.

    Performs: z = ln(x + y)
    """

    def __call__(self, output_z: torch.Tensor, input_x: torch.Tensor, input_y: torch.Tensor, gamma: torch.Tensor,
                 beta: torch.Tensor) -> torch.Tensor:
        """
        Either input_x or input_y can alias output_z.

        Parameters:
            output_z (torch.Tensor): Output tensor.
            input_x (torch.Tensor): Input tensor.
            input_y (torch.Tensor): Input tensor.
            gamma (torch.Tensor): Gamma tensor.
            beta (torch.Tensor): Beta tensor.

        Returns:
            output (torch.Tensor): Output tensor.
        """
        self.inf_module.post_layer_norm(output_z, input_x, input_y, gamma, beta, self.epsilon)
        return output_z
