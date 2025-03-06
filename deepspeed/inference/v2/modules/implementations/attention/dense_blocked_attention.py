# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional

import torch

from deepspeed.accelerator import get_accelerator
from ....allocator import empty_from
from ....inference_utils import DtypeEnum
from ....kernels.ragged_ops import (
    AtomBuilder,
    BlockedFlashAttn,
    BlockedRotaryEmbeddings,
    BlockedTrainedRotaryEmbeddings,
    get_q_block_size,
    get_kv_block_size,
    LinearBlockedKVCopy,
)
from ....ragged import RaggedBatchWrapper, split_kv
from deepspeed.ops.op_builder import RaggedUtilsBuilder

from ...interfaces import DSSelfAttentionBase, DSSelfAttentionRegistry
from ...configs import DSSelfAttentionConfig, PositionalEmbeddingType, MaskingType

try:
    from functools import cached_property
except ImportError:

    def cached_property(func):
        return property(func)


@DSSelfAttentionRegistry.register_module
class DSDenseBlockedAttention(DSSelfAttentionBase):
    """
    Self attention implementation for dense, blocked self attention.
    """

    @staticmethod
    def name() -> str:
        return 'dense_blocked_attention'

    @staticmethod
    def supports_config(config: DSSelfAttentionConfig) -> bool:

        if config.input_dtype != config.output_dtype:
            return False

        if DtypeEnum(config.input_dtype) not in (DtypeEnum.fp16, DtypeEnum.bf16):
            return False

        if PositionalEmbeddingType(config.positional_embedding_type) not in [
                PositionalEmbeddingType.none, PositionalEmbeddingType.rotate_half
        ]:
            return False

        if MaskingType(config.masking_type) != MaskingType.causal:
            return False

        return True

    def __init__(self, config: DSSelfAttentionConfig, implementation_config: Dict[str, Any]) -> None:
        """
        Create the Attention DSModule.

        Args:
            config (DSSelfAttentionConfig): The self attention config for all attention DSModules.
            implementation_config (Dict[str, Any]):
                There are two (dependent) potential components in the implementtion config.

                1. `trained_freqs` - If the embedding weights for RoPE are trained, the implementation
                config should contain {'trained_freqs': True}. This will mean the implementation will
                expect a `trained_freqs` tensor in the `forward` method and will not synthesize the
                values internally.

                2. `theta_base` - The base value for synthesized frequencies in the rotary embeddings.
                This will only be used if `trained_freqs` is False or not present in the `implementation_config`. If this is not included, the default value of 10000.0 will be used.
        """
        super().__init__(config, implementation_config)

        embed_type = PositionalEmbeddingType(config.positional_embedding_type)
        if embed_type == PositionalEmbeddingType.none:
            self._kv_copy = LinearBlockedKVCopy(self._config.head_size, self._config.n_heads_q,
                                                self._config.n_heads_kv, self._config.input_dtype)
        elif embed_type == PositionalEmbeddingType.rotate_half:
            rotary_config = config.positional_embedding_config
            assert rotary_config is not None, "Rotary config must be provided if using rotate_half as Positional Embedding Type."

            if rotary_config.use_trained_freqs:
                # Theta and rotary dim are effectively embedded into either the values (theta) or the shape (rotary_dim)
                # of the trained_freqs tensor.
                self._kv_copy = BlockedTrainedRotaryEmbeddings(self._config.head_size, self._config.n_heads_q,
                                                               self._config.n_heads_kv, self._config.input_dtype)
            else:
                theta_base = rotary_config.theta_base
                rotary_dim = rotary_config.rotate_dim if rotary_config.rotate_dim is not None else self._config.head_size
                self._kv_copy = BlockedRotaryEmbeddings(self._config.head_size, self._config.n_heads_q,
                                                        self._config.n_heads_kv, self._config.input_dtype, rotary_dim,
                                                        theta_base)

        self._softmax_scale = self._config.scale_factor

        # TODO(cmikeh2): Attention kernel gets created here.
        self._attn_kernel = BlockedFlashAttn(self._config.head_size, self._config.input_dtype)
        self._atom_builder = AtomBuilder()

        self.model_dim = self._config.head_size * self._config.n_heads_q
        self._output = torch.empty((self._config.max_tokens, self._config.head_size * self._config.n_heads_q),
                                   dtype=self._config.output_dtype,
                                   device=get_accelerator().current_device())

        # TODO(cmikeh2): Pre-allocate storage buffer for the attention atoms.
        self._max_atoms = self._config.max_sequences
        self._atoms = torch.empty((self._max_atoms, 8), dtype=torch.int32, device=get_accelerator().current_device())

        alloc_func = RaggedUtilsBuilder().load().allocate_fast_host_buffer
        self._atoms_shadow = alloc_func(self._atoms)
        self._cur_atoms = 0

    @cached_property
    def kv_block_size(self) -> int:
        """
        Return preferred granulatity for blocked KV-cache implementation.
        """
        return get_kv_block_size(self._config.head_size)

    @cached_property
    def q_block_size(self) -> int:
        """
        Property to calculate blocking granularity for the query dimension.
        This has no impact on the KV-cache structure, but will  affect the
        number of attention atoms associated with a batch.
        """
        return get_q_block_size(self._config.head_size)

    def build_atoms(self, ragged_batch: RaggedBatchWrapper) -> None:
        """
        Build the atoms for the attention kernel.

        Args:
            ragged_batch (RaggedBatchWrapper): The input ids and associated ragged batch metadata.
        """
        host_atoms, n_atoms = self._atom_builder(self._atoms_shadow, ragged_batch, self.q_block_size,
                                                 self.kv_block_size)

        self._cur_atoms = n_atoms
        self._atoms[:n_atoms].copy_(host_atoms[:n_atoms], non_blocking=True)

    def forward(self,
                q_k_v: torch.Tensor,
                kv_cache: torch.Tensor,
                batch: RaggedBatchWrapper,
                inv_freqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward implementation.

        Args:
            q_k_v (torch.Tensor): Query/Key/Value projection Tensor of shape
                [n_heads, (n_heads_q + 2 * n_heads_kv) * head_size].
            kv_cache (torch.Tensor): Blocked persistent cache of shape
                [2, batch, block_size, n_heads_kv, head_size].
            batch (RaggedBatchWrapper): The input ids and associated ragged batch metadata.
            inv_freqs (Optional[torch.Tensor]): The inverse frequencies for the rotary embeddings if they
                have been modified from synthesizable values.
        """
        if inv_freqs is not None:
            self._kv_copy(kv_cache, q_k_v, batch, inv_freqs)
        else:
            self._kv_copy(kv_cache, q_k_v, batch)

        q = q_k_v[:, :self._config.head_size * self._config.n_heads_q]
        output = empty_from(self._output, q.shape)
        k_cache, v_cache = split_kv(kv_cache)

        self._attn_kernel(output, q, k_cache, v_cache, self._atoms[:self._cur_atoms], self._softmax_scale)

        return output
