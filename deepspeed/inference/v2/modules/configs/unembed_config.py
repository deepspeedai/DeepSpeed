# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ...inference_utils import DtypeEnum, NormTypeEnum
from ...modules.ds_module import DSModuleConfig
from typing import Optional


class DSUnembedConfig(DSModuleConfig):
    """
    Config class for DSUnembed
    """

    dtype: DtypeEnum = DtypeEnum.fp16
    """
    Expected data type.
    """

    norm_type: Optional[NormTypeEnum] = None
    """
    Whether the input to the unembed is normalized prior to the unembedding projection.
    """

    model_dim: int
    """
    Model embedding size.
    """

    max_sequences: int
    """
    Max sequences composing the ragged batch.
    """

    vocab_size: int
    """
    Local vocab size (the full vocab size may have been sharded across model parallel ranks)
    """
