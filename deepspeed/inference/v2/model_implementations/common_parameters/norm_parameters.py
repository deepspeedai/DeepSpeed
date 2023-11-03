# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ...model_implementations.parameter_base import ParameterBase
"""
Common Attention Output Parameter Patterns
"""


class NormParameter(ParameterBase):
    """
    Simple normalization container.
    """

    params: torch.Tensor

    def finalize(self) -> torch.Tensor:
        return self.inference_model.transform_norm_param(self.params)
