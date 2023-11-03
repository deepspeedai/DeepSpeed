# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch

from ... import DSKernelBase
from deepspeed.ops.op_builder import RaggedOpsBuilder
from ....inference_utils import elem_size
from ....ragged import RaggedBatchWrapper


class RaggedEmbeddingKernel(DSKernelBase):
    """
    Ragged-aware CUDA kernel implementation for an embedding lookup. This will only lookup
    the necessary tokens for a padded batch (i.e. if we are CGed and running with a slightly
    larger batch size than the actual tokens).
    """

    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    supported_token_dtypes = [torch.int32, torch.int64]

    def __init__(self, embed_dtype: torch.dtype, token_dtype: torch.dtype, embed_dim: int) -> None:
        """
        Args:
            fp_dtype (torch.dtype): Data type of the embedding table and output dtype.
                Supported values are torch.float16, torch.bfloat16, and torch.float32.
            token_dtype (torch.dtype): Data type of the token ids. Supported values are
                torch.int32 and torch.int64.
            embed_dim (int): Embedding dimension. Must be aligned to 16 bytes.
        """
        if embed_dtype not in RaggedEmbeddingKernel.supported_dtypes:
            raise ValueError("Unsupported embedding data type: {}, supported_dtypes are {}".format(
                embed_dtype, RaggedEmbeddingKernel.supported_dtypes))

        if token_dtype not in RaggedEmbeddingKernel.supported_token_dtypes:
            raise ValueError("Unsupported token data type: {}, supported_dtypes are {}".format(
                token_dtype, RaggedEmbeddingKernel.supported_token_dtypes))

        if elem_size(embed_dtype) * embed_dim % 16 != 0:
            raise ValueError("Embedding dimension must be aligned to 16 bytes, got {}".format(embed_dim))

        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.ragged_embed

    def __call__(self,
                 embedded_tokens: torch.Tensor,
                 ragged_wrapper: RaggedBatchWrapper,
                 embedding_weight: torch.Tensor,
                 position_embed_weight: Optional[torch.Tensor] = None,
                 position_embed_offset: int = 0) -> torch.Tensor:
        """
        Ragged aware embedding lookup.

        Args:
            embedded_tokens (torch.Tensor): Output tensor of shape [num_tokens, embed_dim]
            ragged_wrapper (RaggedBatchWrapper): Wrapper for the ragged batch.
            embedding_weight (torch.Tensor): Embedding table of shape [vocab_size, embed_dim]
        """
        self.kernel(embedded_tokens, ragged_wrapper.input_ids(),
                    embedding_weight, position_embed_weight, position_embed_offset,
                    ragged_wrapper.batch_metadata_buffer(), ragged_wrapper.inflight_seq_descriptors(),
                    ragged_wrapper.tokens_to_seq(), ragged_wrapper.kv_ptrs())
        return embedded_tokens
