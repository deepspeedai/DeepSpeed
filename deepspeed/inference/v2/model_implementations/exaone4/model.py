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
from ...modules.interfaces import *
from ...ragged import RaggedBatchWrapper
from ...kernels.core_ops.cuda_rms_norm.rms_norm import CUDARMSNorm

from .container import Exaone4NonTransformerContainer, Exaone4TransformerContainer


class Exaone4InferenceModel(DSTransformerModelBase):
    """
    Inference model implementation for ragged batching for EXAONE 4.0 models.

    Key differences from Mistral/Llama:
    - Post-norm architecture (norm after attn/mlp, not before)
    - QK-Norm (RMSNorm on Q and K projections per head)
    """

    _non_transformer: Optional[Exaone4NonTransformerContainer]
    _transformer: Optional[Iterable[Exaone4TransformerContainer]]

    @property
    def max_sequence_length(self) -> int:
        return self._config.max_position_embeddings

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
        return getattr(self._config, "head_dim", self.model_dim // self.n_heads)

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
        rope_theta = getattr(self._config, "rope_theta", 1000000.0)
        return RotateHalfConfig(theta_base=rope_theta)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qk_norm = CUDARMSNorm(
            channels=self.head_size,
            fp_dtype=torch.float16 if self.activation_dtype == DtypeEnum.fp16 else torch.bfloat16,
            epsilon=getattr(self._config, "rms_norm_eps", 1e-5),
        )

    def _apply_qk_norm(self, hidden_states: torch.Tensor,
                       q_norm_gamma: torch.Tensor,
                       k_norm_gamma: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to Q and K projections independently per head.
        hidden_states shape: [tokens, (n_q + n_kv + n_kv) * head_size]
        """
        tokens = hidden_states.shape[0]
        local_n_heads = self.n_heads // max(self.tp_size, 1)
        local_n_heads_kv = self.n_heads_kv // max(self.tp_size, 1)
        q_len = local_n_heads * self.head_size
        kv_len = local_n_heads_kv * self.head_size

        q = hidden_states[:, :q_len].contiguous()
        k = hidden_states[:, q_len:q_len + kv_len].contiguous()
        v = hidden_states[:, q_len + kv_len:]

        # Reshape to [tokens * n_heads, head_size] for per-head RMSNorm
        q = q.view(-1, self.head_size)
        self._qk_norm(q, q, q_norm_gamma)
        q = q.view(tokens, q_len)

        k = k.view(-1, self.head_size)
        self._qk_norm(k, k, k_norm_gamma)
        k = k.view(tokens, kv_len)

        hidden_states[:, :q_len] = q
        hidden_states[:, q_len:q_len + kv_len] = k

        return hidden_states

    def _forward_embed(self, ragged_batch: RaggedBatchWrapper) -> torch.Tensor:
        embed = self.embed(ragged_batch, self._non_transformer.word_emb)
        if embed.shape[-1] != self.model_dim:
            raise ValueError(f"Embedding output shape {embed.shape} does not match model_dim {self.model_dim}")
        return embed

    def _forward_transformer(self, layer_idx: int, residual: torch.Tensor, hidden_states: torch.Tensor,
                             ragged_batch_info: RaggedBatchWrapper) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EXAONE 4.0 uses post-norm architecture:
            hidden = attn(hidden)
            hidden = post_attn_norm(hidden)
            residual = residual + hidden
            hidden = mlp(residual)
            hidden = post_ff_norm(hidden)
            residual = residual + hidden
        """
        cur_params = self._transformer[layer_idx]
        kv_cache = self.state_manager.get_cache(layer_idx)

        # Attention block
        hidden_states = self.qkv(hidden_states, cur_params.qkv_w, b=None)
        hidden_states = self._apply_qk_norm(hidden_states, cur_params.q_norm_gamma, cur_params.k_norm_gamma)
        hidden_states = self.attn(hidden_states, kv_cache, ragged_batch_info)
        hidden_states = self.attn_out(hidden_states, cur_params.attn_out_w, b=None)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        # Post-attn norm + residual add
        _, hidden_states = self.norm(hidden_states, None, cur_params.post_attn_norm_gamma, beta=None)
        residual.add_(hidden_states)

        # MLP block
        hidden_states = self.mlp_1(residual, cur_params.mlp_1_w, b=None)
        hidden_states = self.mlp_2(hidden_states, cur_params.mlp_2_w, b=None)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        # Post-ff norm + residual add
        _, hidden_states = self.norm(hidden_states, None, cur_params.post_ff_norm_gamma, beta=None)
        residual.add_(hidden_states)

        return residual, residual

    def _forward_unembed(self, hidden_states: torch.Tensor, ragged_batch_info: RaggedBatchWrapper) -> torch.Tensor:
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

        for layer_idx in range(self.num_layers):
            residual, hidden_states = self._forward_transformer(layer_idx, residual, residual, wrapped_batch)

        return self._forward_unembed(residual, wrapped_batch)
