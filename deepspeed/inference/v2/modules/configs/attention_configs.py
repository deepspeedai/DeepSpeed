# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
from typing import Dict

from ...inference_utils import DtypeEnum
from ...modules.ds_module import DSModuleConfig


class PositionalEmbeddingType(Enum):

    # No positional embeddings
    none = "none"

    # Rotary positional embeddings - every half
    rotate_half = "rotate_half"

    # Rotary positional embeddings - every other
    rotate_every_other = "rotate_every_other"

    # Alibi
    alibi = "alibi"


class MaskingType(Enum):

    # No masking
    none = "none"

    # Causal masking
    causal = "causal"

    # Local masking
    local = "local"

    # Symmetric masking (this is a 1D tensor mask)
    symmetric = "symmetric"

    # Arbitrary masking (this would correspond to a 2D tensor mask)
    asymmetric = "asymmetric"


class DSSelfAttentionConfig(DSModuleConfig):
    """
    Config class for attention.
    """

    # Number of query attention heads on this shard
    n_heads_q: int

    # Number of KV attention heads on this shard
    n_heads_kv: int

    # Size of each attention head
    head_size: int

    # Max number of sequences that may compose a ragged batch
    max_sequences: int

    # Scale factor for attention scores
    scale_factor: float = 1.0

    # Input data type
    input_dtype: DtypeEnum = DtypeEnum.fp16

    # Output data type
    output_dtype: DtypeEnum = DtypeEnum.fp16

    # Masking type
    masking_type: MaskingType = MaskingType.causal

    # Masking args
    masking_args: Dict = {}

    # Positional embedding type
    positional_embedding_type: PositionalEmbeddingType = PositionalEmbeddingType.none

    # Positional embedding args
    positional_embedding_args: Dict = {}
