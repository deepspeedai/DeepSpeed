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
from ...modules import heuristics
from ...ragged import RaggedBatchWrapper
from ..inference_model_base import (
    DSModelImplementationConfig,
    MPType,
)

from .container import Qwen2MoeNonTransformerContainer, Qwen2MoeTransformerContainer


class Qwen2MoeInferenceModel(DSMoETransformerModelBase):
    """
    Inference model implementation for Qwen2MoE models.
    """

    _non_transformer: Optional[Qwen2MoeNonTransformerContainer]
    """
    Embed + unembed container. Specializing the type annotation.
    """

    _transformer: Optional[Iterable[Qwen2MoeTransformerContainer]]
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
        return self._config.shared_expert_intermediate_size

    @property
    def n_heads_kv(self) -> int:
        return self._config.num_key_value_heads

    @property
    def activation_dtype(self) -> DtypeEnum:
        # TODO(ZonePG): bf16 inference results may be different from huggingface bf16,
        # because in rms_norm, Qwen still use float() instead of bf16
        # if self._config.torch_dtype == torch.float16:
        #     return DtypeEnum.fp16
        # elif self._config.torch_dtype == torch.bfloat16:
        #     return DtypeEnum.bf16
        # else:
        #     raise NotImplementedError("Only fp16 and bf16 are supported")
        return DtypeEnum.fp16

    @property
    def mlp_activation_fn(self) -> ActivationType:
        return ActivationType.SiGLU

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
    Inherited from `DSMoETransformerModelBase`
    """

    @property
    def n_experts(self) -> int:
        return self._config.num_experts

    @property
    def n_top_k(self) -> int:
        return self._config.num_experts_per_tok

    @property
    def normalize_expert_scores(self) -> bool:
        return self._config.norm_topk_prob

    def make_moe_layer(self) -> None:
        """
        Instantiates the MoE layer for the model. This sets the `self.moe` attribute.
        """
        sharded_dim = sharded_intermediate_dim(self.intermediate_dim // self.n_top_k, self.tp_size, self.tp_rank)

        moe_config = DSMoEConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            model_dim=self.model_dim,
            intermediate_features=sharded_dim,
            activation=self.mlp_activation_fn,
            n_experts=self.n_experts,
            top_k=self.n_top_k,
            input_dtype=self.activation_dtype,
            output_dtype=self.activation_dtype,
            normalize_scores=self.normalize_expert_scores,
        )

        self.moe = heuristics.instantiate_moe(moe_config, self._engine_config)

    ######### MLP 1 #########
    def make_shared_expert_mlp_1_layer(self) -> None:
        """
        Instantiates the linear projection layer for the first MLP in the feedforward network.
        This sets the `self.mlp_1` attribute.
        """
        shard_size = sharded_intermediate_dim(self.intermediate_dim, self.tp_size, self.tp_rank)

        linear_config = DSLinearConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            in_channels=self.model_dim,
            out_channels=shard_size,
            activation=self.mlp_activation_fn,
            input_dtype=self.activation_dtype,
            output_dtype=self.activation_dtype,
        )

        self.shared_expert_mlp_1 = heuristics.instantiate_linear(linear_config, self._engine_config)

    ######### MLP 2 #########
    def make_shared_expert_mlp_2_layer(self) -> None:
        """
        Instantiates the linear projection layer for the second MLP in the feedforward network.
        This sets the `self.mlp_2` attribute.
        """
        shard_size = sharded_intermediate_dim(self.intermediate_dim, self.tp_size, self.tp_rank)

        linear_config = DSLinearConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            in_channels=shard_size,
            out_channels=self.model_dim,
            input_dtype=self.activation_dtype,
            output_dtype=self.activation_dtype,
        )

        self.shared_expert_mlp_2 = heuristics.instantiate_linear(linear_config, self._engine_config)

    ######### MLP 2 #########
    def make_shared_expert_gate_layer(self) -> None:
        """
        Instantiates the linear projection layer for the second MLP in the feedforward network.
        This sets the `self.mlp_2` attribute.
        """
        shard_size = sharded_intermediate_dim(self.model_dim, self.tp_size, self.tp_rank)

        linear_config = DSLinearConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            in_channels=shard_size,
            out_channels=8,
            input_dtype=self.activation_dtype,
            output_dtype=self.activation_dtype,
        )

        self.shared_expert_gate = heuristics.instantiate_linear(linear_config, self._engine_config)

    def make_norm_layer(self) -> None:
        """
        Instantiates the normalization layer for the model. This sets the `self.norm` attribute.

        TODO(cmikeh2): In the future we'll distinguish between the different norm objects,
        but for now we'll just use the same one for all of them.
        """
        norm_config = DSNormConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            type=self.norm_type,
            channels=self.model_dim,
            residual_dtype=self.activation_dtype,
            input_dtype=self.activation_dtype,
            output_dtype=self.activation_dtype,
            eps=self._config.rms_norm_eps,
        )

        self.norm = heuristics.instantiate_pre_norm(norm_config, self._engine_config)

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
        self.make_shared_expert_mlp_1_layer()
        self.make_shared_expert_mlp_2_layer()
        self.make_shared_expert_gate_layer()
        self.make_embedding_layer()
        self.make_unembedding_layer()
        self._kv_cache_config = None

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

        hidden_states = self.qkv(hidden_states, cur_params.qkv_w, b=cur_params.qkv_b)
        hidden_states = self.attn(hidden_states, kv_cache, ragged_batch_info)
        hidden_states = self.attn_out(hidden_states, cur_params.attn_out_w, b=None)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        residual, hidden_states = self.norm(residual, hidden_states, cur_params.mlp_norm_gamma, beta=None)

        shared_expert_output = self.shared_expert_mlp_1(hidden_states, cur_params.shared_moe_mlp_1, b=None)
        shared_expert_output = self.shared_expert_mlp_2(shared_expert_output, cur_params.shared_moe_mlp_2, b=None)
        shared_expert_gate_output = self.shared_expert_gate(hidden_states, cur_params.shared_moe_gate, b=None)[..., :1]
        # shared_expert_gate_output shape[-1] is 1
        shared_expert_output.mul_(torch.sigmoid(shared_expert_gate_output))
        hidden_states = self.moe(hidden_states, ragged_batch_info, cur_params.moe_gate, cur_params.moe_mlp_1,
                                 cur_params.moe_mlp_2)
        hidden_states.add_(shared_expert_output)

        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self._base_mp_group)

        if layer_idx != self.num_layers - 1:
            next_params = self._transformer[layer_idx + 1]
            residual, hidden_states = self.norm(residual, hidden_states, next_params.attn_norm_gamma, beta=None)
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
