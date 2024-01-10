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

from .container import FalconNonTransformerContainer, FalconTransformerContainer


class FalconInferenceModel(DSTransformerModelBase):
    """
    Inference model implementation for ragged batching for Llama-2 models.
    """

    _non_transformer: Optional[FalconNonTransformerContainer]
    """
    Embed + unembed container. Specializing the type annotation.
    """

    _transformer: Optional[Iterable[FalconTransformerContainer]]
    """
    Per-layer transformer container. Specializing the type annotation.
    """
    """
    Properties inherited from `DSInferenceModelBase`
    """

    @property
    def max_sequence_length(self) -> int:
        return self._config.max_seq_length

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
        return self.model_dim // self.n_heads

    @property
    def n_heads(self) -> int:
        return self._config.num_attention_heads

    @property
    def intermediate_dim(self) -> int:
        return 4 * self._config.hidden_size

    @property
    def n_heads_kv(self) -> int:
        return self._config.num_kv_heads if (self._config.new_decoder_architecture
                                             or not self._config.multi_query) else 1

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
        return ActivationType.GELU

    @property
    def norm_type(self) -> NormTypeEnum:
        return NormTypeEnum.LayerNorm

    @property
    def positional_embedding_type(self) -> PositionalEmbeddingType:
        return PositionalEmbeddingType.rotate_half

    @property
    def positional_embedding_config(self) -> RotateHalfConfig:
        """
        The positional embedding configuration for the model.
        """
        return RotateHalfConfig()

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
        Executes one (slightly offset) layer of the transformer. This implementation does a peak-ahead
        optimization to fuse the layer norm of the next layer into the current layer.

        Arguments:
            layer_idx (int): The index of the layer to execute.
            residual (torch.Tensor): The residual tensor from the previous layer.
            hidden_states (torch.Tensor): The hidden states from the previous layer. This is the
                hidden states after pre normalization.
            ragged_batch_info (RaggedBatchWrapper): The batch metadata.
        """
        assert self.config.parallel_attn, "Only parallel attention implementation is supported"

        cur_params = self._transformer[layer_idx]
        kv_cache = self.state_manager.get_cache(layer_idx)

        attn_ln_out = hidden_states
        attn_hidden_state = self.qkv(attn_ln_out, cur_params.qkv_w, b=None)
        attn_hidden_state = self.attn(attn_hidden_state, kv_cache, ragged_batch_info)
        attention_output = self.attn_out(attn_hidden_state, cur_params.attn_out_w, b=None)

        if self.config.new_decoder_architecture:
            residual, mlp_ln_out = self.norm(residual,
                                             None,
                                             gamma=cur_params.ln_mlp_gamma,
                                             beta=cur_params.ln_mlp_beta)
        else:
            mlp_ln_out = hidden_states

        mlp_hidden_state = self.mlp_1(mlp_ln_out, cur_params.mlp_1_w, b=None)
        mlp_output = self.mlp_2(mlp_hidden_state, cur_params.mlp_2_w, b=None)

        mlp_output.add_(attention_output)

        if self.tp_size > 1:
            dist.all_reduce(mlp_output, group=self._base_mp_group)

        if layer_idx != self.num_layers - 1:
            next_params = self._transformer[layer_idx + 1]
            residual, mlp_output = self.norm(residual,
                                             mlp_output,
                                             next_params.ln_attn_gamma,
                                             beta=next_params.ln_attn_beta)
        else:
            # On last layer, we just need to perform the residual add. Adding into the residual
            # here is safe.
            residual.add_(mlp_output)

        return residual, mlp_output

    def _forward_unembed(self, hidden_states: torch.Tensor, ragged_batch_info: RaggedBatchWrapper) -> torch.Tensor:
        """
        Performs unembedding of the hidden states to logits. This will only sample the final
        token of each sequence.
        """
        logits = self.unembed(hidden_states,
                              self._non_transformer.word_unembed,
                              ragged_batch_info,
                              gamma=self._non_transformer.final_norm_gamma,
                              beta=self._non_transformer.final_norm_beta)

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
                                            gamma=self._transformer[0].ln_attn_gamma,
                                            beta=self._transformer[0].ln_attn_beta)

        for layer_idx in range(self.num_layers):
            residual, hidden_states = self._forward_transformer_layer(layer_idx, residual, hidden_states,
                                                                      wrapped_batch)

        return self._forward_unembed(residual, wrapped_batch)
