# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ...model_implementations.parameter_base import ParameterBase
"""
Common InvFreq Parameter Patterns
"""


class InvFreqParameter(ParameterBase):

    params: torch.Tensor

    def finalize(self) -> torch.Tensor:
        return self.params.to(self.inference_model.activation_dtype.value)
