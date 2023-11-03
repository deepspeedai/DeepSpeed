# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

from ...inference_utils import DtypeEnum, NormTypeEnum
from ...modules.ds_module import DSModuleConfig
"""
Trying to define the space we need to support here right now:

Types of embeddings I've found so far:
    1. Token embedding
    2. Position embedding
    3. Token type embedding
    4. LN

GPTNeo: 1, 2, 3 (shared with 1)
GPTNeoX: 1
GPTJ: 1, 3
LLaMA: 1
BERT: 1, 2, 3, 4
GPT2: 1, 2, 3 (shared with 1)

Sidebar for OPT:
OPT: 1, 2
1 may not actually project to the actual hidden dimension according to the raw
code, but for the model configs we care about it does.
2 has a weird offset associated with it that the others do not.
"""


class DSEmbeddingsConfig(DSModuleConfig):
    """
    Config class for DSEmbeddings.
    """

    residual_dtype: DtypeEnum = DtypeEnum.fp16
    """
    Data type the module should use for its output.
    """

    embedding_dim: int
    """
    Dimensionality of the embedding projections.
    """

    positional_embedding: bool = False
    """
    Whether the module should expect a positional embedding matrix. The shape of this
    matrix should be of shape [max_seq_len + positional_offset, embedding_dim]
    """

    positional_offset: int = 0
    """
    Whether the linearized token IDs should be offset by a certain amount. For an example
    of this, see the OPT model implementation.
    """

    use_token_type: bool = False
    """
    Whether the module should expect a token type embedding matrix.
    """

    output_normalization: Optional[NormTypeEnum] = None
    """
    If a the output of the embedding module should be normalized, specify here. See
    ``inference.inference_utils.NormTypeEnum`` for supported values.
    """
