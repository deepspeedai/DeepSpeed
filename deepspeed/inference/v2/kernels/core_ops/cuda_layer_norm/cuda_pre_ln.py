# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple

import torch

from .cuda_fp_ln_base import CUDAFPLNBase


class CUDAFPPreLN(CUDAFPLNBase):
    """
    Floating point pre-LayerNorm kernel for CUDA/RoCM.

    Performs: z_res = x_res + y_hid
              z_hid = ln(z_hid)
    """

    def __call__(self, z_res: torch.Tensor, z_hid: torch.Tensor, x_res: torch.Tensor, y_hid: torch.Tensor,
                 gamma: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_res can alias x_res. All non-parameter input/output tensors
        must have the same shape. z_hid can alias y_hid.

        Parameters:
            z_res (torch.Tensor): Output residual.
            z_hid (torch.Tensor): Output hidden states.
            x_res (torch.Tensor): Input residual.
            y_hid (torch.Tensor): Input hidden states.
            gamma (torch.Tensor): Gamma tensor.
            beta (torch.Tensor): Beta tensor.

        Returns:
            output (torch.Tensor): Output tensor.
        """
        self.inf_module.pre_layer_norm(z_res, z_hid, x_res, y_hid, gamma, beta, self.epsilon)
        return z_res, z_hid
