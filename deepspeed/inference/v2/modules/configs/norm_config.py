# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ...inference_utils import DtypeEnum, NormTypeEnum
from ...modules.ds_module import DSModuleConfig


class DSNormConfig(DSModuleConfig):
    """
    Config class for both DSPreLN and DSPostLN.
    """

    # Type of normalization
    type: NormTypeEnum

    # Number of channels in the model embedding
    channels: int

    # Data type of the residual input/outputs (we assume the residual must
    # be the same data type for the entire model).
    residual_dtype: DtypeEnum = DtypeEnum.fp16

    # Data type of the hidden states input
    input_dtype: DtypeEnum = DtypeEnum.fp16

    # Data type of the hidden states output
    output_dtype: DtypeEnum = DtypeEnum.fp16

    # Epsilon value for numerical stability
    eps: float = 1e-5
