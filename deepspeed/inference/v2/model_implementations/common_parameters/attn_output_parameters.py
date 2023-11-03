# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ...model_implementations.parameter_base import ParameterBase
"""
Common Attention Output Parameter Patterns
"""


class AttentionOutputParameter(ParameterBase):
    """
    Attention output parameter container.

    Note: The differentiation for something like GQA for this matrix is primarily
    encompassed in the sharding logic, which is currently expected to be performed by
    the model implementation.
    """

    params: torch.Tensor
    """
    Unsharded attention output parameter of shape [model_dim, model_dim]
    """

    def finalize(self) -> torch.Tensor:
        return self.inference_model.transform_attn_out_param(self.params)
