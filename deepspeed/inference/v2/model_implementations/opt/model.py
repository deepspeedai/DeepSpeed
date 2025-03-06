# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Optional, Tuple

import torch

import deepspeed.comm as dist

from ...allocator import empty_from
from ...inference_utils import ActivationType, DtypeEnum
from ...model_implementations import *
from ...modules.configs import *
from ...ragged import RaggedBatchWrapper
from .container import OPTNonTransformerContainer, OPTTransformerContainer

from ...modules.heuristics import instantiate_embed


class OPTInferenceModel(DSTransformerModelBase):
    """
    Inference model implementation for ragged batching for OPT models.
    """

    _non_transformer: Optional[OPTNonTransformerContainer]
    """
    Embed + unembed container. Specializing the type annotation.
    """

    _transformer: Optional[Iterable[OPTTransformerContainer]]
    """
    Per-layer transformer container. Specializing the type annotation.
    """
    """
    Properties ineherited from `DSInferenceModelBase`
    """

    @property
    def max_sequence_length(self) -> int:
        return self._config.max_seq_length

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
        return self._config.ffn_dim

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
        return ActivationType.RELU

    @property
    def norm_type(self) -> NormTypeEnum:
        return NormTypeEnum.LayerNorm

    @property
    def positional_embedding_type(self) -> PositionalEmbeddingType:
        return PositionalEmbeddingType.none

    @property
    def positional_embedding_config(self) -> Optional[RotateHalfConfig]:
        return None

    """
    Overrides of ``DSTransformerModelBase`` methods
    """

    def make_embedding_layer(self) -> None:
        """
        Performs setup and creates embedding DSModule. Since OPT includes trained
        positional embeddings, we will override the base model implementation.
        """

        embed_config = DSEmbeddingsConfig(max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
                                          residual_dtype=self.activation_dtype,
                                          embedding_dim=self.model_dim,
                                          positional_embedding=True,
                                          positional_offset=2)

        self.embed = instantiate_embed(embed_config, self._engine_config)

    """
    Forward implementations
    """

    def _forward_embed(self, ragged_batch: RaggedBatchWrapper) -> torch.Tensor:
        embed = self.embed(ragged_batch, self._non_transformer.word_emb, self._non_transformer.word_emb_pos)
        if embed.shape[-1] != self.model_dim:
            raise ValueError(f"Embedding output shape {embed.shape} does not match model_dim {self.model_dim}")

        return embed

    def _forward_transformer_layer(self, layer_idx: int, residual: torch.Tensor, hidden_states: torch.Tensor,
                                   ragged_batch_info: RaggedBatchWrapper) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO(cmikeh2): Distribute ragged_batch_info to all modules

        cur_params = self._transformer[layer_idx]
        kv_cache = self.state_manager.get_cache(layer_idx)

        hidden_states = self.qkv(hidden_states, cur_params.qkv_w, b=cur_params.qkv_b)
        hidden_states = self.attn(hidden_states, kv_cache, ragged_batch_info)
        hidden_states = self.attn_out(hidden_states, cur_params.attn_out_w, b=cur_params.attn_out_b)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        residual, hidden_states = self.norm(residual,
                                            hidden_states,
                                            cur_params.mlp_norm_gamma,
                                            beta=cur_params.mlp_norm_beta)

        # Should be configurable in the future
        hidden_states = self.mlp_1(hidden_states, cur_params.mlp_1_w, b=cur_params.mlp_1_b)
        hidden_states = self.mlp_2(hidden_states, cur_params.mlp_2_w, b=cur_params.mlp_2_b)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        if layer_idx != self.num_layers - 1:
            next_params = self._transformer[layer_idx + 1]
            residual, hidden_states = self.norm(residual,
                                                hidden_states,
                                                next_params.attn_norm_gamma,
                                                beta=next_params.attn_norm_beta)
        else:
            # On last layer, we just need to perform the residual add. Adding into the residual
            # here is safe.
            residual.add_(hidden_states)

        return residual, hidden_states

    def _forward_unembed(self, hidden_states: torch.Tensor, ragged_batch_info: RaggedBatchWrapper) -> torch.Tensor:
        logits = self.unembed(hidden_states,
                              self._non_transformer.word_unembed,
                              ragged_batch_info,
                              gamma=self._non_transformer.final_norm_w,
                              beta=self._non_transformer.final_norm_b)

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

        residual, hidden_states = self.norm(residual,
                                            None,
                                            self._transformer[0].attn_norm_gamma,
                                            beta=self._transformer[0].attn_norm_beta)

        for layer_idx in range(self.num_layers):
            residual, hidden_states = self._forward_transformer_layer(layer_idx, residual, hidden_states,
                                                                      wrapped_batch)

        return self._forward_unembed(residual, wrapped_batch)
