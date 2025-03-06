# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
from typing import Dict, Optional

from ...inference_utils import DtypeEnum
from ...modules.ds_module import DSModuleConfig
from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class PositionalEmbeddingType(Enum):

    # No positional embeddings
    none = "none"

    # Rotary positional embeddings - every half
    rotate_half = "rotate_half"

    # Rotary positional embeddings - every other
    rotate_every_other = "rotate_every_other"

    # Alibi
    alibi = "alibi"


class RotateHalfConfig(DeepSpeedConfigModel):

    use_trained_freqs: bool = False
    """
    Whether to use a passed `trained_freqs` tensor for the attention implementation
    or to use default synthesized frequencies.
    """

    theta_base: float = 10_000.0
    """
    Base for theta. This will only be used if `use_trained_freqs` is False.
    """

    rotate_dim: Optional[int] = None
    """
    How many neurons to rotate. If None, then all neurons will be rotated. Many external configs
    will set this number to half the head dimension and then internally multiply by 2. To make it
    more clear to understand what is happening (rotate_dim < head_dim -> then only partial rotation),
    we do not do this multiplication internally.
    """


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
    positional_embedding_config: Optional[RotateHalfConfig] = None
    """
    To extend this for the other positional embedding types, we would need to add
    new configs for each type (as necessary) and annotate this with the
    Union[RotateHalfConfig, OtherConfig, ...] type.
    """
