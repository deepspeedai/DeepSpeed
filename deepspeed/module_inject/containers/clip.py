# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy


class DS_CLIPContainer(BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module


class HFCLIPLayerPolicy(TransformerPolicy):

    def __init__(self, client_module, inference=False):
        super().__init__(inference, pre_attn_norm=True, scale_attention=True)
        self.client_module = client_module
        self.cuda_graph_supported = True

        if HFCLIPLayerPolicy._orig_layer_class is None:
            try:
                import transformers
                HFCLIPLayerPolicy._orig_layer_class = transformers.models.clip.modeling_clip.CLIPEncoderLayer
            except:
                HFCLIPLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.self_attn.q_proj.weight.shape[1], \
                self.client_module.self_attn.num_heads, \
                self.client_module.layer_norm1.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        qw = self.client_module.self_attn.q_proj.weight
        qb = self.client_module.self_attn.q_proj.bias
        kw = self.client_module.self_attn.k_proj.weight
        kb = self.client_module.self_attn.k_proj.bias
        vw = self.client_module.self_attn.v_proj.weight
        vb = self.client_module.self_attn.v_proj.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=enable_training)

        return qkvw, \
               qkvb, \
               self.client_module.self_attn.out_proj.weight, \
               self.client_module.self_attn.out_proj.bias

    def mlp(self, enable_training=False):
        return self.client_module.mlp.fc1.weight, \
               self.client_module.mlp.fc1.bias, \
               self.client_module.mlp.fc2.weight, \
               self.client_module.mlp.fc2.bias

    def layernorm(self):
        return self.client_module.layer_norm2.weight, \
               self.client_module.layer_norm2.bias, \
               self.client_module.layer_norm1.weight, \
               self.client_module.layer_norm1.bias
