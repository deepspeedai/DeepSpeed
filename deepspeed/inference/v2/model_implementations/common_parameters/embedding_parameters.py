# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ...model_implementations.parameter_base import ParameterBase
from ...allocator import on_device
"""
Embedding containers.
"""


class EmbeddingParameter(ParameterBase):
    """
    Embedding container. This should be safe to use for all types of embeddings (i.e. word, position,
    and token type).
    """

    params: torch.Tensor
    """
    Vocabulary parameter of shape [vocab_size, model_dim].
    """

    @on_device
    def finalize(self) -> torch.Tensor:
        return self.params
        #return self.inference_model.transform_embed_param(self.params)
