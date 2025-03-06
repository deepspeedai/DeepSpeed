# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import abstractmethod
from typing import Optional

import torch

from deepspeed.accelerator import get_accelerator
from ..config_v2 import RaggedInferenceEngineConfig
from ..inference_utils import ActivationType, ceil_div, is_gated
from ..model_implementations import *
from ..model_implementations.sharding import *
from ..modules.configs import (
    DSEmbeddingsConfig,
    DSLinearConfig,
    DSMoEConfig,
    DSNormConfig,
    DSSelfAttentionConfig,
    DSUnembedConfig,
    NormTypeEnum,
    PositionalEmbeddingType,
    RotateHalfConfig,
)
from ..modules import heuristics
from ..ragged import (
    DSSequenceDescriptor,
    KVCacheConfig,
    RaggedBatchWrapper,
)
from .inference_model_base import (
    DSInferenceModelBase,
    DSModelImplementationConfig,
    MPType,
)
from ..inference_parameter import InferenceParameter

try:
    from functools import cached_property
except ImportError:

    def cached_property(func):
        return property(func)


class DSTransformerModelBase(DSInferenceModelBase):
    """
    Dimensioning properties
    """

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """
        Number of the layers in the model
        """
        ...

    @property
    @abstractmethod
    def model_dim(self) -> int:
        """
        Size of embedding projection and residuals.
        """
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Size of the vocabulary (including padding).
        """
        ...

    @property
    @abstractmethod
    def head_size(self) -> int:
        """
        Size of each attention head.
        """
        ...

    @property
    @abstractmethod
    def n_heads(self) -> int:
        """
        The number of query heads on the model. This should not take into account
        any dimension reductions from model sharding.
        """
        ...

    @property
    def n_heads_q(self) -> int:
        """
        Alias to n_heads.
        """
        return self.n_heads

    @property
    def n_heads_kv(self) -> int:
        """
        The number of key and value heads on the model. For GQA or MQA, overload this attribute.
        Otherwise it adopts MHA formulations and uses n_heads. This should not take into account
        any dimension reductions from model sharding.
        """
        return self.n_heads

    @property
    @abstractmethod
    def intermediate_dim(self) -> int:
        """
        The size of the (unsharded) intermediate projection dim. For a gated activation function
        this is the size of the input to the second MLP layer. This should not take into account
        any dimension reductions from model sharding.
        """
        ...

    @property
    @abstractmethod
    def positional_embedding_type(self) -> PositionalEmbeddingType:
        """
        The type of positional embedding used by the model.
        """
        ...

    """
    Architectural properties
    """

    @property
    @abstractmethod
    def activation_dtype(self) -> torch.dtype:
        """
        The activation dtype of the model.
        """
        ...

    @property
    @abstractmethod
    def mlp_activation_fn(self) -> ActivationType:
        """
        The activation function used in the MLP.
        """
        ...

    @property
    @abstractmethod
    def norm_type(self) -> NormTypeEnum:
        """
        The type of normalization used in the model.
        """
        ...

    @property
    @abstractmethod
    def positional_embedding_config(self) -> Optional[RotateHalfConfig]:
        """
        The positional embedding configuration for the model.
        """
        ...

    """
    Derived helpers
    """

    @cached_property
    def n_heads_q_local(self) -> int:
        """
        Number of local heads post sharding.
        """
        return get_local_heads(self.tp_rank, self.tp_size, self.n_heads_q, self.n_heads_kv)[0]

    @cached_property
    def n_heads_kv_local(self) -> int:
        """
        Number of local heads post sharding.
        """
        return get_local_heads(self.tp_rank, self.tp_size, self.n_heads_q, self.n_heads_kv)[1]

    @property
    def gated_mlp(self) -> bool:
        """
        Return a boolean to determine whether the model uses a gated activation function.
        """
        return is_gated(self.mlp_activation_fn)

    """
    Method implementations
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
        self.make_mlp_1_layer()
        self.make_mlp_2_layer()
        self.make_embedding_layer()
        self.make_unembedding_layer()
        self._kv_cache_config = None

    ######### Embedding #########
    def make_embedding_layer(self) -> None:
        """
        Performs setup and creates embedding DSModule. This will set the `self.embed` attribute.
        """

        embed_config = DSEmbeddingsConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            residual_dtype=self.activation_dtype,
            embedding_dim=self.model_dim,
        )

        self.embed = heuristics.instantiate_embed(embed_config, self._engine_config)

    def transform_embedding_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Performs embedding sharding along the channels dimension.
        """
        # Until we can do non-contiguous all-gather, we won't shard the embedding parameters.
        param = param.to(self.activation_dtype.value)
        return InferenceParameter.initialize(param)

    ######### Unembedding #########
    def make_unembedding_layer(self) -> None:
        """
        Performs setup and creates an unembedding layer. This implementation assumes
        normalization prior to the LM head projection. If this does not match the model's
        implementation, override this method. This will set the ``self.unembed`` attribute.
        """
        unembed_dim = sharded_unembed_dim(self.vocab_size, self.tp_rank, self.tp_size)

        unembed_config = DSUnembedConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            max_sequences=self._engine_config.state_manager.max_ragged_sequence_count,
            dtype=self.activation_dtype,
            model_dim=self.model_dim,
            vocab_size=unembed_dim,
            norm_type=self.norm_type,
        )

        self.unembed = heuristics.instantiate_unembed(unembed_config, self._engine_config)

        if self.tp_size > 1:
            self._comm_logits = torch.empty(self.tp_size,
                                            self._engine_config.state_manager.max_ragged_sequence_count,
                                            unembed_dim,
                                            device=get_accelerator().current_device(),
                                            dtype=self.activation_dtype.value)
            self._return_logits = torch.empty(self._engine_config.state_manager.max_ragged_sequence_count,
                                              self.vocab_size,
                                              device=get_accelerator().current_device(),
                                              dtype=self.activation_dtype.value)

    def transform_unembed_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Performs sharding along the vocab dimension.
        """
        param = shard_unembed_param(param, self.tp_rank, self.tp_size).to(self.activation_dtype.value)
        return InferenceParameter.initialize(param)

    ######### QKV #########
    def make_qkv_layer(self) -> None:
        """
        Instantiates the linear projection layer for the QKV linear layer. This sets the
        `self.qkv` attribute.
        """
        out_features = qkv_out_features(self.model_dim, self.tp_rank, self.tp_size, self.head_size, self.n_heads_q,
                                        self.n_heads_kv)

        linear_config = DSLinearConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            in_channels=self.model_dim,
            out_channels=out_features,
            input_dtype=self.activation_dtype,
            output_dtype=self.activation_dtype,
        )

        self.qkv = heuristics.instantiate_linear(linear_config, self._engine_config)

    def transform_qkv_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Passes a QKV parameter to the underlying implementation for any necessary
        transformations.

        Args:
            param (torch.Tensor): The parameter to transform. This may be either a bias or weight and should have
                the shape (out_neurons, in_neurons)
        """
        param = shard_qkv_param(param, self.tp_rank, self.tp_size, self.head_size, self.n_heads_q, self.n_heads_kv)
        return self.qkv.transform_param(param)

    ######### Attention #########
    def make_attn_layer(self) -> None:
        """
        Builds the attention layer for the model. This sets the `self.attn` attribute.
        """
        softmax_scale = 1.0 / (self.head_size**0.5)

        attn_config = DSSelfAttentionConfig(max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
                                            n_heads_q=self.n_heads_q_local,
                                            n_heads_kv=self.n_heads_kv_local,
                                            head_size=self.head_size,
                                            max_sequences=self._engine_config.state_manager.max_ragged_sequence_count,
                                            scale_factor=softmax_scale,
                                            input_dtype=self.activation_dtype,
                                            output_dtype=self.activation_dtype,
                                            positional_embedding_type=self.positional_embedding_type,
                                            positional_embedding_config=self.positional_embedding_config)

        self.attn = heuristics.instantiate_attention(attn_config, self._engine_config)

    def get_kv_requirements(self, sequence: DSSequenceDescriptor, max_new_tokens: int,
                            max_new_blocks: int) -> Tuple[int, int]:
        """
        See ``DSInferenceModelBase.get_kv_requirements`` for documentation.

        This method assumes an autoregressive dense attention pattern. Override this method
        if this does not match the model's attention pattern.
        """
        total_tokens = sequence.seen_tokens + max_new_tokens
        req_blocks = ceil_div(total_tokens, self.attn.kv_block_size)
        block_lim = req_blocks - sequence.cur_allocated_blocks

        if block_lim <= max_new_blocks:
            return max_new_tokens, block_lim

        token_capacity = (max_new_blocks +
                          sequence.cur_allocated_blocks) * self.attn.kv_block_size - sequence.seen_tokens

        return token_capacity, max_new_blocks

    def get_remaining_block_capacity(self, sequence: DSSequenceDescriptor) -> int:
        return sequence.seen_tokens % self.attn.kv_block_size

    def maybe_allocate_kv(self, sequence: DSSequenceDescriptor, n_new_tokens: int) -> None:
        """
        See ``DSInferenceModelBase.maybe_allocate_kv`` for documentation.

        This method assumes an autoregressive dense attention pattern. Override this method
        if this does not match the model's attention pattern.
        """
        free_block = self.state_manager.free_blocks[0]
        _, n_needed_blocks = self.get_kv_requirements(sequence, n_new_tokens, free_block)

        if n_needed_blocks > 0:
            new_blocks = self.state_manager.allocate_blocks(n_needed_blocks)
            sequence.extend_kv_cache(new_blocks)

    def kv_cache_config(self) -> Tuple[KVCacheConfig, ...]:
        """
        See ``DSInferenceModelBase.kv_cache_config`` for documentation.

        This method assumes an autoregressive dense attention pattern. Override this method
        if this does not match the model's attention pattern.
        """
        if self._kv_cache_config is None:
            cache_shape = (self.num_layers, self.n_heads_kv_local, self.head_size)
            max_blocks = ceil_div(self.max_sequence_length, self.attn.kv_block_size)
            self._kv_cache_config = KVCacheConfig(block_size=self.attn.kv_block_size,
                                                  cache_shape=cache_shape,
                                                  cache_dtype=self.activation_dtype,
                                                  max_blocks_per_allocation_group=max_blocks)
        return (self._kv_cache_config, )

    def prepare_batch(self, wrapped_batch: RaggedBatchWrapper) -> None:
        """
        See ``DSInferenceModelBase.prepare_batch`` for documentation.

        This method assumes an autoregressive dense attention pattern. Override this method
        if this does not match the model's attention pattern.
        """
        self.attn.build_atoms(wrapped_batch)

    ######### Attention output #########
    def make_attn_out_layer(self) -> None:
        """
        Instantiates the linear projection layer for the attention output linear layer. This sets the
        `self.attn_out` attribute.
        """
        in_features = attn_out_in_features(self.model_dim, self.tp_rank, self.tp_size, self.head_size, self.n_heads_q,
                                           self.n_heads_kv)

        linear_config = DSLinearConfig(
            max_tokens=self._engine_config.state_manager.max_ragged_batch_size,
            in_channels=in_features,
            out_channels=self.model_dim,
            input_dtype=self.activation_dtype,
            output_dtype=self.activation_dtype,
        )

        self.attn_out = heuristics.instantiate_linear(linear_config, self._engine_config)

    def transform_attn_out_param(self, param: torch.Tensor) -> Optional[InferenceParameter]:
        """
        Shards an attention output projection parameter and passes it to the underlying
        implementation for any necessary transformations. This will return `None` for bias parameters
        if they are not on TP rank 0.

        Args:
            param (torch.Tensor): The parameter to transform. This may be either a bias or weight and should have
                the shape (out_neurons, in_neurons).
        """
        param = shard_attn_out_param(param, self.tp_rank, self.tp_size, self.head_size, self.n_heads_q,
                                     self.n_heads_kv)

        if param is not None:
            param = self.attn_out.transform_param(param)

        return param

    ######### MLP #########
    def make_mlp_1_layer(self) -> None:
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

        self.mlp_1 = heuristics.instantiate_linear(linear_config, self._engine_config)

    def transform_mlp_1_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Shards the first MLP parameter and passes it to the underlying implementation
        for any necessary transformations.

        Args:
            param (torch.Tensor): The parameter to transform. This may be either a bias or weight and should have
                the shape (out_neurons, in_neurons).
        """
        param = shard_mlp_1_param(param, self.tp_rank, self.tp_size, gated=self.gated_mlp)

        return self.mlp_1.transform_param(param)

    def make_mlp_2_layer(self) -> None:
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

        self.mlp_2 = heuristics.instantiate_linear(linear_config, self._engine_config)

    def transform_mlp_2_param(self, param: torch.Tensor) -> Optional[InferenceParameter]:
        """
        Shards the second MLP parameter and passes it to the underlying implementation
        for any necessary transformations. This will return `None` for bias parameters
        if they are not on TP rank 0.

        Args:
            param (torch.Tensor): The parameter to transform. This may be either a bias or weight and should have
                the shape (out_neurons, in_neurons).
        """
        param = shard_mlp_2_param(param, self.tp_rank, self.tp_size)

        if param is not None:
            param = self.mlp_2.transform_param(param)

        return param

    ######### Norm #########
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
        )

        self.norm = heuristics.instantiate_pre_norm(norm_config, self._engine_config)

    def transform_norm_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Passes a normalization parameter to the underlying implementation for any
        necessary transformations.

        TODO(cmikeh2): In the future we'll distinguish between the different norm objects,
        but for now we'll just use the same one for all of them.

        Args:
            param (torch.Tensor): The parameter to transform. This may be either a bias or weight and should have
                shape (model_dim,)
        """
        return self.norm.transform_param(param)


class DSMoETransformerModelBase(DSTransformerModelBase):

    @property
    def n_experts(self) -> int:
        """
        Return the number of experts in the model.
        """
        raise NotImplementedError("Attempted to access an unimplemented number of experts")

    @property
    def n_top_k(self) -> int:
        """
        Number of experts per token.
        """
        raise NotImplementedError("Attempted to access an unimplemented number of experts per token")

    @property
    def normalize_expert_scores(self) -> bool:
        """
        Whether to normalize expert scores. If true, sum(expert_scores) = 1.
        """
        raise NotImplementedError("Attempted to access an unimplemented normalization flag")

    def make_moe_layer(self) -> None:
        """
        Instantiates the MoE layer for the model. This sets the `self.moe` attribute.
        """
        sharded_dim = sharded_intermediate_dim(self.intermediate_dim, self.tp_size, self.tp_rank)

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

    def transform_moe_gate_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Passes a MoE gate parameter to the underlying implementation for any necessary transformations.

        TODO(cmikeh2): This will need to be updated/overridden for expert parallelism.
        """
        return self.moe.transform_gate_param(param)

    def transform_moe_mlp_1_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Shards the first MoE param and passes it to the underlying implementation. Since it's possible for an architecture
        to have both MoE and non-MoE layers, this can't be overloaded on the MLP1 transform. Furthermore, since both
        the MoE DSModule owns both MLP1 and MLP2, under certain sharding conditions it's not possible for the model implementation
        to infer from the shape whether to perform a different transformation based on MLP1 or MLP2. This (and the below)
        separations are intended to solve both these issues.

        Args:
            param (torch.Tensor): The parameter to transform. This should have shape (n_experts, out_neurons, in_neurons).
        """
        param = shard_mlp_1_param(param, self.tp_rank, self.tp_size, gated=self.gated_mlp, is_moe=True)

        return self.moe.transform_moe_mlp_1_param(param)

    def transform_moe_mlp_2_param(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Shards the second MoE param and passes it to the underlying implementation. See the above for context on why this API
        exists.

        This will return `None` for expert bias params not on TP rank 0. NOTE(cmikeh2): Does it make sense to round-robin assign?
        My intuition is that this will make debugging much more difficult for minimal memory reduction.

        Args:
            param (torch.Tensor): The parameter to transform. This should have shape (n_experts, out_neurons, in_neurons).
        """
        param = shard_mlp_2_param(param, self.tp_rank, self.tp_size, is_moe=True)

        if param is not None:
            param = self.moe.transform_moe_mlp_2_param(param)

        return param
