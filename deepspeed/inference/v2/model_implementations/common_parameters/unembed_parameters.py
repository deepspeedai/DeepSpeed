# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ...model_implementations.parameter_base import ParameterBase
"""
Unembedding containers.
"""


class UnembedParameter(ParameterBase):
    """
    Unembedding parameter. This will likely be mapped to the same original weight in the model as the
    embedding, but we have a different preferred sharding approach.
    """

    params: torch.Tensor
    """
    Unembedding parameter of shape [vocab_size, model_dim].
    """

    def finalize(self) -> torch.Tensor:
        return self.inference_model.transform_unembed_param(self.params)
