# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ....inference_utils import DtypeEnum
from deepspeed.ops.op_builder import RaggedOpsBuilder
from ....ragged import RaggedBatchWrapper
from ... import DSKernelBase


class BlockedRotaryEmbeddings(DSKernelBase):
    """
    CUDA Kernel implementation that will perform rotary position embeddings on the queries and keys
    before copying into a blocked KV cache.
    """

    supported_dtypes = [DtypeEnum.fp16, DtypeEnum.bf16]
    supported_head_sizes = [64, 80, 128]
    supported_q_ratios = [1, 2, 4, 5, 8, 16, 29, 35, 36, 71]

    def __init__(self, head_size: int, n_q_heads: int, n_kv_heads: int, dtype: torch.dtype, rotary_dim: int,
                 theta_base: float) -> None:
        """
        Args:
            head_size: The size of the attention head.
            q_ratio: Ratio of q heads to kv heads (for GQA)
            dtype: Data type for the input/output. Supported values are torch.float16 and torch.bfloat16.
        """

        q_ratio = n_q_heads // n_kv_heads

        if head_size not in BlockedRotaryEmbeddings.supported_head_sizes:
            raise ValueError("Unsupported head size: {}, supported_head_sizes are {}".format(
                head_size, BlockedRotaryEmbeddings.supported_head_sizes))

        if q_ratio not in BlockedRotaryEmbeddings.supported_q_ratios:
            raise ValueError("Unsupported q_ratio: {}, supported_q_ratios are {}".format(
                q_ratio, BlockedRotaryEmbeddings.supported_q_ratios))

        if not isinstance(dtype, DtypeEnum):
            dtype = DtypeEnum(dtype)

        if dtype not in BlockedRotaryEmbeddings.supported_dtypes:
            raise ValueError("Unsupported data type: {}, supported_dtypes are {}".format(
                dtype, BlockedRotaryEmbeddings.supported_dtypes))

        inf_module = RaggedOpsBuilder().load()
        self.kernel = inf_module.kv_rotary_embeddings
        self.head_size = head_size
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.rotary_dim = rotary_dim
        self.theta_base = theta_base

    def __call__(self, kv_cache: torch.Tensor, qkv: torch.Tensor, ragged_batch: RaggedBatchWrapper) -> None:
        """
        Perform rotary embeddings on the queries and keys before copying into a blocked KV cache.

        Args:
            kv_cache (torch.Tensor): Pre-allocated KV cache of [num_blocks, block_size, 2, n_kv_heads, head_size]
            qkv: Input tensor of shape [num_tokens, head_size * (n_q_heads + 2 * n_kv_heads)]
            ragged_batch: Wrapper for the ragged batch.
        """

        q = qkv[:, :self.head_size * self.n_q_heads]
        k = qkv[:, self.head_size * self.n_q_heads:self.head_size * (self.n_q_heads + self.n_kv_heads)]
        v = qkv[:, self.head_size * (self.n_q_heads + self.n_kv_heads):]

        self.kernel(kv_cache, q, k, v, self.rotary_dim, self.theta_base, ragged_batch.batch_metadata_buffer(),
                    ragged_batch.inflight_seq_descriptors(), ragged_batch.tokens_to_seq(), ragged_batch.kv_ptrs())
