# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features.meta_tensor import MetaTensorContainer
from .features.split_qkv import HybridSplitQKVContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy
from ..policy import transformer_param_names
from ..policy import maybe_copy
from ..policy import maybe_copy_qkv

from ..policy import maybe_get_lora


class DS_GPTNEOContainer(MetaTensorContainer, HybridSplitQKVContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.mlp.c_fc, self.policy.client_module.mlp.c_proj,
                self.policy.client_module.attn.attention.q_proj, self.policy.client_module.attn.attention.k_proj,
                self.policy.client_module.attn.attention.v_proj, self.policy.client_module.attn.attention.out_proj
            ]
        ]

    def set_q_k_v(self):
        """
        Necessary to implement for `HybridSplitQKVContainer`
        """
        self.qw = self.policy.client_module.attn.attention.q_proj.weight
        self.qb = None
        self.kw = self.policy.client_module.attn.attention.k_proj.weight
        self.kb = None
        self.vw = self.policy.client_module.attn.attention.v_proj.weight
        self.vb = None

    def get_lora_matched_pair(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        fc1_lora, fc2_lora, q_lora, k_lora, v_lora, out_lora = self.get_lora_params()
        ret = [(fc1_lora, self._h4h_w), (fc2_lora, self._4hh_w), (out_lora, self.dense_w), (q_lora, self.qw),
               (k_lora, self.kw), (v_lora, self.vw)]
        return ret

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'attn.attention.q_proj.weight', \
            'attn.attention.k_proj.weight', \
            'attn.attention.v_proj.weight', \
            'attn.attention.out_proj.weight', \
            'attn.attention.out_proj.bias', \
            'mlp.c_fc.weight', \
            'mlp.c_fc.bias', \
            'mlp.c_proj.weight', \
            'mlp.c_proj.bias', \
            'ln_2.weight', \
            'ln_2.bias', \
            'ln_1.weight', \
            'ln_1.bias'
        )
        maybe_copy_qkv(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       'attn_qkvw', [prefix + param_names[0], prefix + param_names[1], prefix + param_names[2]],
                       split_qkv=self.policy.split_qkv)
        for i in range(3, 5):
            maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[i - 1],
                       prefix + param_names[i])
        for i in range(5, 11):
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[i - 1],
                       prefix + param_names[i])
        for i in range(11, 13):
            maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[i - 1],
                       prefix + param_names[i])


class HFGPTNEOLayerPolicy(TransformerPolicy):

    def __init__(self, client_module, inference=True):
        super().__init__(inference, scale_attention=False)
        self.client_module = client_module
        try:
            import transformers
            HFGPTNEOLayerPolicy._orig_layer_class = transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock
        except:
            HFGPTNEOLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.attention.embed_dim, \
                self.client_module.attn.attention.num_heads, \
                self.client_module.ln_1.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def get_q_k_v(self):
        return self.client_module.attn.attention.q_proj.weight, \
               None, \
               self.client_module.attn.attention.k_proj.weight, \
               None, \
               self.client_module.attn.attention.v_proj.weight, \
               None

    def attention(self, enable_training=False):
        qw = self.client_module.attn.attention.q_proj.weight
        kw = self.client_module.attn.attention.k_proj.weight
        vw = self.client_module.attn.attention.v_proj.weight

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)

        return qkvw, \
               None, \
               self.client_module.attn.attention.out_proj.weight, \
               self.client_module.attn.attention.out_proj.bias

    def mlp(self, enable_training=False):
        return self.client_module.mlp.c_fc.weight, \
               self.client_module.mlp.c_fc.bias, \
               self.client_module.mlp.c_proj.weight, \
               self.client_module.mlp.c_proj.bias

    def layernorm(self):
        return self.client_module.ln_2.weight, \
               self.client_module.ln_2.bias, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias
