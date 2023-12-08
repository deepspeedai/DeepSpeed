# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional

import torch

from deepspeed.accelerator import get_accelerator
from ....allocator import empty_from
from ....inference_utils import DtypeEnum
from ....kernels.ragged_ops import RaggedEmbeddingKernel
from ....ragged import RaggedBatchWrapper
from ...interfaces import DSEmbeddingBase, DSEmbeddingRegistry
from ...configs import DSEmbeddingsConfig


@DSEmbeddingRegistry.register_module
class DSRaggedEmbedding(DSEmbeddingBase):

    @staticmethod
    def name():
        return 'ragged_embedding'

    @staticmethod
    def supports_config(config: DSEmbeddingsConfig) -> bool:

        if DtypeEnum(config.residual_dtype) not in [DtypeEnum.fp16, DtypeEnum.bf16, DtypeEnum.fp32]:
            return False

        if config.use_token_type:
            return False

        if config.output_normalization is not None:
            return False

        try:
            _ = RaggedEmbeddingKernel(config.residual_dtype, torch.int32, config.embedding_dim)
        except ValueError:
            return False

        return True

    def __init__(self, config: DSEmbeddingsConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

        self.embed_offset = self._config.positional_offset

        # TODO(cmikeh2): How do we want to avoid the int32 vs int64 issue?
        self._ragged_embed = RaggedEmbeddingKernel(self._config.residual_dtype, torch.int32,
                                                   self._config.embedding_dim)

        self._output = torch.empty((self._config.max_tokens, self._config.embedding_dim),
                                   dtype=self._config.residual_dtype,
                                   device=get_accelerator().current_device())

    @property
    def output(self) -> torch.Tensor:
        return self._output

    def forward(self,
                ragged_batch: RaggedBatchWrapper,
                word_embeddings: torch.Tensor,
                position_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters:
            ragged_batch (RaggedBatchWrapper): The input ids and associated ragged batch metadata.
            word_embeddings (torch.Tensor): The word embedding table
        """
        output = empty_from(self._output, (ragged_batch.tensor_toks, self._config.embedding_dim))
        self._ragged_embed(output,
                           ragged_batch,
                           word_embeddings,
                           position_embed_weight=position_embeddings,
                           position_embed_offset=self.embed_offset)
        return output
