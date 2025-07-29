# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Optional, Tuple

import torch

import deepspeed.comm as dist

from ...allocator import empty_from
from ...inference_utils import ActivationType, DtypeEnum
from .. import *
from ...modules.configs import *
from ...modules.interfaces import *
from ...ragged import RaggedBatchWrapper

from .container import ExaoneNonTransformerContainer, ExaoneTransformerContainer


class ExaoneInferenceModel(DSTransformerModelBase):
    """
    Inference model implementation for ragged batching for EXAONE 4.0 models.

    Key features:
    - Hybrid attention: sliding_attention (local) vs full_attention (global) layers
    - QK-Reorder-Norm: RMSNorm applied after Q/K projections
    - Conditional RoPE: Skip RoPE for full_attention layers
    - Grouped Query Attention: 40 query heads, 8 key-value heads
    """

    _non_transformer: Optional[ExaoneNonTransformerContainer]
    """
    Embed + unembed container. Specializing the type annotation.
    """

    _transformer: Optional[Iterable[ExaoneTransformerContainer]]
    """
    Per-layer transformer container. Specializing the type annotation.
    """

    # EXAONE 4.0 specific attributes
    _layer_types: Optional[list] = None
    """
    Layer types for hybrid attention: 'sliding_attention' or 'full_attention'
    """

    def __init__(self, config, engine_config, base_mp_group):
        super().__init__(config, engine_config, base_mp_group)

        # Store layer types for hybrid attention handling
        if hasattr(self._config, 'layer_types'):
            self._layer_types = self._config.layer_types
        else:
            # Fallback: infer from sliding_window_pattern (LLLG = 3 local, 1 global)
            pattern = getattr(self._config, 'sliding_window_pattern', 'LLLG')
            layer_types = []
            for i in range(self.num_layers):
                if pattern[i % len(pattern)] == 'G':
                    layer_types.append('full_attention')
                else:
                    layer_types.append('sliding_attention')
            self._layer_types = layer_types

    """
    Properties inherited from `DSInferenceModelBase`
    """

    @property
    def max_sequence_length(self) -> int:
        return self._config.max_position_embeddings

    """
    Properties inherited from `DSTransformerModelBase`
    """

    @property
    def num_layers(self) -> int:
        return self._config.num_hidden_layers

    @property
    def model_dim(self) -> int:
        return self._config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size

    @property
    def head_size(self) -> int:
        return getattr(self._config, 'head_dim', self.model_dim // self.n_heads)

    @property
    def n_heads(self) -> int:
        return self._config.num_attention_heads

    @property
    def intermediate_dim(self) -> int:
        return self._config.intermediate_size

    @property
    def n_heads_kv(self) -> int:
        return self._config.num_key_value_heads

    @property
    def activation_dtype(self) -> DtypeEnum:
        if self._config.torch_dtype == torch.float16:
            return DtypeEnum.fp16
        elif self._config.torch_dtype == torch.bfloat16:
            return DtypeEnum.bf16
        else:
            raise NotImplementedError("Only fp16 and bf16 are supported")

    @property
    def mlp_activation_fn(self) -> ActivationType:
        activation = self._config.hidden_act.lower()
        # EXAONE 4.0 uses gated SiLU activation like LLaMA
        if activation == "silu":
            return ActivationType.SiGLU
        elif activation == "gelu":
            return ActivationType.GEGLU
        elif activation == "relu":
            return ActivationType.ReGLU
        else:
            raise NotImplementedError(f"Activation {activation} not supported")

    @property
    def norm_type(self) -> NormTypeEnum:
        return NormTypeEnum.RMSNorm

    @property
    def positional_embedding_type(self) -> PositionalEmbeddingType:
        return PositionalEmbeddingType.rotate_half

    @property
    def positional_embedding_config(self) -> Optional[RotateHalfConfig]:
        return RotateHalfConfig(theta_base=self._config.rope_theta)

    """
    Helper methods for EXAONE 4.0 specific features
    """

    def is_global_attention_layer(self, layer_idx: int) -> bool:
        """Check if layer uses global (full) attention vs local (sliding) attention"""
        if self._layer_types and layer_idx < len(self._layer_types):
            return self._layer_types[layer_idx] == 'full_attention'
        return False

    def should_apply_rope(self, layer_idx: int) -> bool:
        """EXAONE 4.0 skips RoPE for global attention layers"""
        return not self.is_global_attention_layer(layer_idx)

    """
    Forward implementations
    """

    def _forward_embed(self, ragged_batch: RaggedBatchWrapper) -> torch.Tensor:
        """
        Performs the embedding lookup prior to running the transformer of the model.

        Arguments:
            ragged_batch (RaggedBatchWrapper): The batch to embed.

        Returns:
            torch.Tensor: The embedded batch.
        """
        embed = self.embed(ragged_batch, self._non_transformer.word_emb)

        if embed.shape[-1] != self.model_dim:
            raise ValueError(f"Embedding output shape {embed.shape} does not match model_dim {self.model_dim}")

        return embed

    def _forward_transformer_layer(self, layer_idx: int, residual: torch.Tensor, hidden_states: torch.Tensor,
                                   ragged_batch_info: RaggedBatchWrapper) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one transformer layer with EXAONE 4.0 specific features:
        - Hybrid attention (sliding vs full)
        - QK-Reorder-Norm (RMSNorm after Q/K projections)
        - Conditional RoPE (skip for global layers)

        Arguments:
            layer_idx (int): The index of the layer to execute.
            residual (torch.Tensor): The residual tensor from the previous layer.
            hidden_states (torch.Tensor): The hidden states from the previous layer.
            ragged_batch_info (RaggedBatchWrapper): The batch metadata.
        """
        cur_params = self._transformer[layer_idx]
        kv_cache = self.state_manager.get_cache(layer_idx)

        # QKV projection
        hidden_states = self.qkv(hidden_states, cur_params.qkv_w, b=None)

        # EXAONE 4.0 attention with hybrid pattern and conditional RoPE
        # NOTE: The attention module should handle QK-Reorder-Norm internally
        # and respect the RoPE configuration based on layer type
        if self.is_global_attention_layer(layer_idx):
            # Global attention: full attention, no RoPE
            hidden_states = self.attn(hidden_states, kv_cache, ragged_batch_info, apply_rotary_pos_emb=False)
        else:
            # Local attention: sliding window, with RoPE
            hidden_states = self.attn(hidden_states,
                                      kv_cache,
                                      ragged_batch_info,
                                      apply_rotary_pos_emb=True,
                                      sliding_window=getattr(self._config, 'sliding_window', 4096))

        # Attention output projection
        hidden_states = self.attn_out(hidden_states, cur_params.attn_out_w, b=None)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        # Post-attention normalization
        residual, hidden_states = self.norm(residual, hidden_states, cur_params.mlp_norm_gamma, beta=None)

        # MLP forward pass (gated SiLU)
        hidden_states = self.mlp_1(hidden_states, cur_params.mlp_1_w, b=None)
        hidden_states = self.mlp_2(hidden_states, cur_params.mlp_2_w, b=None)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        # Prepare for next layer normalization
        if layer_idx != self.num_layers - 1:
            next_params = self._transformer[layer_idx + 1]
            residual, hidden_states = self.norm(residual, hidden_states, next_params.attn_norm_gamma, beta=None)
        else:
            # On last layer, just perform the residual add
            residual.add_(hidden_states)

        return residual, hidden_states

    def _forward_unembed(self, hidden_states: torch.Tensor, ragged_batch_info: RaggedBatchWrapper) -> torch.Tensor:
        """
        Performs unembedding of the hidden states to logits. This will only sample the final
        token of each sequence.
        """
        logits = self.unembed(hidden_states,
                              self._non_transformer.word_unembed,
                              ragged_batch_info,
                              gamma=self._non_transformer.final_norm)

        if self.tp_size > 1:
            comm_buffer = empty_from(self._comm_logits, (self.tp_size, logits.shape[0], logits.shape[1]))
            full_logits = empty_from(self._return_logits, (logits.shape[0], self.vocab_size))

            dist.all_gather_into_tensor(comm_buffer, logits, group=self._base_mp_group)

            full_logits.copy_(comm_buffer.permute(1, 0, 2).reshape(logits.shape[0], self.vocab_size))

            return full_logits
        else:
            return logits

    def forward(self, wrapped_batch: RaggedBatchWrapper) -> torch.Tensor:
        """
        Forward pass for EXAONE 4.0 model with hybrid attention support.
        """
        residual = self._forward_embed(wrapped_batch)

        # Initial normalization
        residual, hidden_states = self.norm(residual, None, self._transformer[0].attn_norm_gamma, beta=None)

        # Forward through all transformer layers
        for layer_idx in range(self.num_layers):
            residual, hidden_states = self._forward_transformer_layer(layer_idx, residual, hidden_states,
                                                                      wrapped_batch)

        return self._forward_unembed(residual, wrapped_batch)
