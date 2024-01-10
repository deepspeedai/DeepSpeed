# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Optional, Tuple

import torch

import deepspeed.comm as dist

from ...allocator import empty_from
from ...config_v2 import RaggedInferenceEngineConfig
from ...inference_utils import ActivationType, DtypeEnum
from ...model_implementations import *
from ...modules.configs import *
from ...modules.interfaces import *
from ...ragged import RaggedBatchWrapper
from ..inference_model_base import (
    DSModelImplementationConfig,
    MPType,
)

from .container import MixtralNonTransformerContainer, MixtralTransformerContainer


class MixtralInferenceModel(DSMoETransformerModelBase):
    """
    Inference model implementation for Mixtral models.
    """

    _non_transformer: Optional[MixtralNonTransformerContainer]
    """
    Embed + unembed container. Specializing the type annotation.
    """

    _transformer: Optional[Iterable[MixtralTransformerContainer]]
    """
    Per-layer transformer container. Specializing the type annotation.
    """
    """
    Properties ineherited from `DSInferenceModelBase`
    """

    @property
    def max_sequence_length(self) -> int:
        return self._config.max_position_embeddings

    """
    Properties ineherited from `DSTransformerModelBase`
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
        return self.model_dim // self.n_heads

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
        if activation == "gelu":
            return ActivationType.GEGLU
        elif activation == "relu":
            return ActivationType.ReGLU
        elif activation == "gegelu":
            return ActivationType.GEGLU
        elif activation == "silu":
            return ActivationType.SiGLU
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
        """
        The positional embedding configuration for the model.
        """
        return RotateHalfConfig(theta_base=self._config.rope_theta)

    """
    Inherited from `DSMoETransformerModelBase`
    """

    @property
    def n_experts(self) -> int:
        return self._config.num_local_experts

    @property
    def n_top_k(self) -> int:
        return self._config.num_experts_per_tok

    @property
    def normalize_expert_scores(self) -> bool:
        return True

    """
    Model implementation
    """

    def __init__(self, config: DSModelImplementationConfig, engine_config: RaggedInferenceEngineConfig,
                 base_mp_group: MPType) -> None:
        """
        Base implementation for initialization. By default, this will initialize
        the traditional components of a transformer model:
            - Embedding
            - QKV projection
            - Self attention
            - Attention output projection
            - Feed forward network
            - Normalization
            - Unembedding

        Arguments:
            config (DSModelImplementationConfig): Model-specific configuration. No assumptions
                should be made about this config that are not closely tied to the specific
                model implementation.
            engine_config (RaggedInferenceEngineConfig): Engine configuration.
            base_mp_group (MPType): Base communication group for Tensor-parallel inference.
        """
        super().__init__(config, engine_config, base_mp_group)

        self.make_norm_layer()
        self.make_qkv_layer()
        self.make_attn_layer()
        self.make_attn_out_layer()
        self.make_moe_layer()
        self.make_embedding_layer()
        self.make_unembedding_layer()
        self._kv_cache_config = None

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

    def _forward_transformer(self, layer_idx: int, residual: torch.Tensor, hidden_states: torch.Tensor,
                             ragged_batch_info: RaggedBatchWrapper) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one (slightly offset) layer of the transformer. This implementation does a peak-ahead
        optimization to fuse the layer norm of the next layer into the current layer.

        Arguments:
            layer_idx (int): The index of the layer to execute.
            residual (torch.Tensor): The residual tensor from the previous layer.
            hidden_states (torch.Tensor): The hidden states from the previous layer. This is the
                hidden states after pre normalization.
            ragged_batch_info (RaggedBatchWrapper): The batch metadata.
        """
        # TODO(cmikeh2): Distribute ragged_batch_info to all modules

        cur_params = self._transformer[layer_idx]
        kv_cache = self.state_manager.get_cache(layer_idx)

        hidden_states = self.qkv(hidden_states, cur_params.qkv_w)
        hidden_states = self.attn(hidden_states, kv_cache, ragged_batch_info)
        hidden_states = self.attn_out(hidden_states, cur_params.attn_out_w)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        residual, hidden_states = self.norm(residual, hidden_states, cur_params.mlp_norm_gamma)

        hidden_states = self.moe(hidden_states, ragged_batch_info, cur_params.moe_gate, cur_params.moe_mlp_1,
                                 cur_params.moe_mlp_2)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        if layer_idx != self.num_layers - 1:
            next_params = self._transformer[layer_idx + 1]
            residual, hidden_states = self.norm(residual, hidden_states, next_params.attn_norm_gamma)
        else:
            # On last layer, we just need to perform the residual add. Adding into the residual
            # here is safe.
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

        residual = self._forward_embed(wrapped_batch)

        residual, hidden_states = self.norm(residual, None, self._transformer[0].attn_norm_gamma, beta=None)

        for layer_idx in range(self.num_layers):
            residual, hidden_states = self._forward_transformer(layer_idx, residual, hidden_states, wrapped_batch)

        return self._forward_unembed(residual, wrapped_batch)
