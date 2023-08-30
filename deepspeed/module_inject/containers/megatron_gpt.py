# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features.megatron import MegatronContainer
from deepspeed.model_implementations.transformers.ds_megatron_gpt import DeepSpeedMegatronGPTInference
import torch
from ..policy import TransformerPolicy
from packaging import version as pkg_version


class DS_MegatronGPTContainer(MegatronContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedMegatronGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention

        if self.megatron_v2:
            self.module.config.rotate_half = True
            self.module.config.rotate_every_two = False

        return self.module


# TODO: Megatron GPT MoE inherits from Megatron policy and replaces mlp
# TODO: Generalize MoE overall goal, expand beyond Megatron
class MegatronLayerPolicy(TransformerPolicy):
    _orig_layer_class = None
    version = 0
    moe_type = 'standard'
    megatron_v2 = True
    use_mup = False

    def __init__(self, client_module, inference=True):
        super().__init__(inference, megatron_v2=MegatronLayerPolicy.megatron_v2, use_mup=MegatronLayerPolicy.use_mup)
        self.client_module = client_module
        # we use megatron version to differentiate between the old and new
        # megatron-lm source code
        if MegatronLayerPolicy._orig_layer_class is None:
            if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
                MegatronLayerPolicy._orig_layer_class = None
            else:
                try:
                    from megatron.model.transformer import ParallelTransformerLayer
                    MegatronLayerPolicy._orig_layer_class = ParallelTransformerLayer
                    MegatronLayerPolicy.version = 1
                except ImportError:
                    MegatronLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        if MegatronLayerPolicy.version == 0:
            return self.client_module.attention.query_key_value.weight.shape[1], \
                    self.client_module.attention.num_attention_heads, \
                    self.client_module.input_layernorm.eps, \
                    DEFAULT_INTERMEDIATE_SIZE
        else:
            return self.client_module.self_attention.query_key_value.weight.shape[1], \
                    self.client_module.self_attention.num_attention_heads, \
                    self.client_module.input_layernorm.eps, \
                    DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        if self.inference:
            if MegatronLayerPolicy.version == 0:
                attention = self.client_module.attention
            else:
                attention = self.client_module.self_attention

        return attention.query_key_value.weight, \
               attention.query_key_value.bias, \
               attention.dense.weight, \
               attention.dense.bias

    def mlp(self, moe_type='standard', enable_training=False):
        from deepspeed.moe.utils import has_moe_layers
        moe, _ = has_moe_layers(self.client_module)

        if moe:
            moe_experts = self.client_module.mlp.deepspeed_moe.experts.deepspeed_experts if moe_type == 'standard' else \
                            self.client_module.mlp.moe.deepspeed_moe.experts.deepspeed_experts
            num_experts = len(moe_experts)
            if moe_type == 'standard':
                return [moe_experts[i].dense_h_to_4h.weight for i in range(num_experts)], \
                       [moe_experts[i].dense_h_to_4h.bias for i in range(num_experts)], \
                       [moe_experts[i].dense_4h_to_h.weight for i in range(num_experts)], \
                       [moe_experts[i].dense_4h_to_h.bias for i in range(num_experts)]
            else:

                return [moe_experts[i].dense_h_to_4h.weight for i in range(num_experts)], \
                       [moe_experts[i].dense_h_to_4h.bias for i in range(num_experts)], \
                       [moe_experts[i].dense_4h_to_h.weight for i in range(num_experts)], \
                       [moe_experts[i].dense_4h_to_h.bias for i in range(num_experts)], \
                       self.client_module.mlp.mlp.dense_h_to_4h.weight, \
                       self.client_module.mlp.mlp.dense_h_to_4h.bias, \
                       self.client_module.mlp.mlp.dense_4h_to_h.weight, \
                       self.client_module.mlp.mlp.dense_4h_to_h.bias, \
                       self.client_module.mlp.coefficient.weight

        else:
            return self.client_module.mlp.dense_h_to_4h.weight, \
                   self.client_module.mlp.dense_h_to_4h.bias, \
                   self.client_module.mlp.dense_4h_to_h.weight, \
                   self.client_module.mlp.dense_4h_to_h.bias

    def layernorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias
