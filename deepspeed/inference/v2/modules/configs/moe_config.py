# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ...inference_utils import ActivationType, DtypeEnum
from ...modules.ds_module import DSModuleConfig


class DSMoEConfig(DSModuleConfig):
    """
    Config class for DSMoEBase
    """

    model_dim: int
    """
    Size of input activation.
    """

    intermediate_features: int
    """
    Size of intermediate activation. Specifically, this is the number of input features
    in the second linear layer. Depending on the activation function, the output of the first
    linear layer may have increased dimensionality.
    """

    n_experts: int
    """
    Number of experts.
    """

    top_k: int = 1
    """
    top-k gating function (like top-1 or top-2)
    """

    input_dtype: DtypeEnum = DtypeEnum.fp16
    """
    Data type for the input activations.
    """

    output_dtype: DtypeEnum = DtypeEnum.fp16
    """
    Data type for the output activations.
    """

    activation: ActivationType = ActivationType.IDENTITY
    """
    Activation function of the first MLP1
    """

    normalize_scores: bool = False
    """
    Whether normalization is applied to the selected scores. If true, the module
    should rescale the scores such that their sum is 1.0.
    """
