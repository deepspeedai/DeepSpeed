# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from deepspeed.model_implementations.transformers.ds_bert import DeepSpeedBERTInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy


class DS_DistilBERTContainer(BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.
        self.triangular_masking = False
        self.return_single_tuple = True
        self.use_triton = kwargs['config'].use_triton and deepspeed.HAS_TRITON

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedBERTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module


class HFDistilBertLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=False, preln=False):
        super().__init__(inference)
        self.client_module = client_module
        self.preln = preln
        self.cuda_graph_supported = True
        if HFDistilBertLayerPolicy._orig_layer_class is None:
            try:
                import transformers
                HFDistilBertLayerPolicy._orig_layer_class = [
                    transformers.models.distilbert.modeling_distilbert.TransformerBlock,
                ]
            except:
                HFDistilBertLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attention.q_lin.weight.shape[1], \
                self.client_module.attention.n_heads, \
                self.client_module.sa_layer_norm.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        qw = self.client_module.attention.q_lin.weight
        qb = self.client_module.attention.q_lin.bias
        kw = self.client_module.attention.k_lin.weight
        kb = self.client_module.attention.k_lin.bias
        vw = self.client_module.attention.v_lin.weight
        vb = self.client_module.attention.v_lin.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=enable_training)

        return qkvw, \
               qkvb, \
               self.client_module.attention.out_lin.weight, \
               self.client_module.attention.out_lin.bias

    def mlp(self, enable_training=False):
        intermediate_ff = self.client_module.ffn.lin1

        return intermediate_ff.weight, intermediate_ff.bias, \
            self.client_module.ffn.lin2.weight, \
            self.client_module.ffn.lin2.bias

    def layernorm(self):
        attention_layernorm = self.client_module.sa_layer_norm
        transformer_layernorm = self.client_module.output_layer_norm
        return attention_layernorm.weight, \
               attention_layernorm.bias, \
               transformer_layernorm.weight, \
               transformer_layernorm.bias
