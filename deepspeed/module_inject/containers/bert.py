# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from deepspeed.model_implementations.transformers.ds_bert import DeepSpeedBERTInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy


class DS_BERTContainer(BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.
        self.return_tuple = True
        self.triangular_masking = False
        self.use_triton = kwargs['config'].use_triton and deepspeed.HAS_TRITON

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedBERTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module


class HFBertLayerPolicy(TransformerPolicy):

    def __init__(self, client_module, inference=False):
        super().__init__(inference, pre_attn_norm=False)
        self.client_module = client_module
        self.cuda_graph_supported = True

        if HFBertLayerPolicy._orig_layer_class is None:
            try:
                import transformers
                HFBertLayerPolicy._orig_layer_class = [
                    transformers.models.bert.modeling_bert.BertLayer,
                    transformers.models.roberta.modeling_roberta.RobertaLayer
                ]
            except:
                HFBertLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        if self.pre_attn_norm:
            attention_layernorm = self.client_module.PostAttentionLayerNorm
        else:
            attention_layernorm = self.client_module.attention.output.LayerNorm
        return self.client_module.attention.self.query.weight.shape[1], \
                self.client_module.attention.self.num_attention_heads, \
                attention_layernorm.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        qw = self.client_module.attention.self.query.weight
        qb = self.client_module.attention.self.query.bias
        kw = self.client_module.attention.self.key.weight
        kb = self.client_module.attention.self.key.bias
        vw = self.client_module.attention.self.value.weight
        vb = self.client_module.attention.self.value.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=enable_training)

        return qkvw, \
               qkvb, \
               self.client_module.attention.output.dense.weight, \
               self.client_module.attention.output.dense.bias, \

    def mlp(self, enable_training=False):
        if self.pre_attn_norm:
            intermediate_ff = self.client_module.intermediate.dense_act
        else:
            intermediate_ff = self.client_module.intermediate.dense

        return intermediate_ff.weight, intermediate_ff.bias, \
            self.client_module.output.dense.weight, \
            self.client_module.output.dense.bias

    def layernorm(self):
        if self.pre_attn_norm:
            attention_layernorm = self.client_module.PostAttentionLayerNorm
            transformer_layernorm = self.client_module.PreAttentionLayerNorm
        else:
            attention_layernorm = self.client_module.attention.output.LayerNorm
            transformer_layernorm = self.client_module.output.LayerNorm
        return attention_layernorm.weight, \
               attention_layernorm.bias, \
               transformer_layernorm.weight, \
               transformer_layernorm.bias
