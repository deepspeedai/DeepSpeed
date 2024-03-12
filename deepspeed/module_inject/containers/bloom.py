# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features.meta_tensor import MetaTensorContainer
from .features.hybrid_engine import HybridEngineContainer
from deepspeed.model_implementations.transformers.ds_bloom import DeepSpeedBloomInference
from ..policy import TransformerPolicy
from ..policy import transformer_param_names
from ..policy import maybe_copy

from ..policy import maybe_get_lora

supported_models = {None}


class DS_BloomContainer(MetaTensorContainer, HybridEngineContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.
        self.bigscience_bloom = True
        self.triangular_masking = False

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config

        self.module = DeepSpeedBloomInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        self.module.config.invert_mask = False
        return self.module

    def attention_qkv_mp(self, mp_replace, reversed_dim=False):
        self.module.attention.attn_qkvw = mp_replace.copy(self.module.attention.attn_qkvw, self.qkvw)
        self.module.attention.attn_qkvb = mp_replace.copy(self.module.attention.attn_qkvb, self.qkvb)

    def get_lora_matched_pair(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        fc1_lora, fc2_lora, qkv_lora, out_lora = self.get_lora_params()
        ret = [(fc1_lora, self._h4h_w), (fc2_lora, self._4hh_w), (qkv_lora, self.qkvw), (out_lora, self.dense_w)]
        return ret

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.mlp.dense_h_to_4h, self.policy.client_module.mlp.dense_4h_to_h, self.policy.
                client_module.self_attention.query_key_value, self.policy.client_module.self_attention.dense
            ]
        ]

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'self_attention.query_key_value.weight', \
            'self_attention.query_key_value.bias', \
            'self_attention.dense.weight', \
            'self_attention.dense.bias', \
            'mlp.dense_h_to_4h.weight', \
            'mlp.dense_h_to_4h.bias', \
            'mlp.dense_4h_to_h.weight', \
            'mlp.dense_4h_to_h.bias', \
            'post_attention_layernorm.weight', \
            'post_attention_layernorm.bias', \
            'input_layernorm.weight', \
            'input_layernorm.bias'
        )
        for i in range(0, 2):
            maybe_copy(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i],
                       prefix + param_names[i],
                       qkv=True,
                       megatron_v2=self.policy.is_megatron_v2,
                       split_qkv=self.policy.split_qkv)
        for i in range(2, 4):
            maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(4, 10):
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(10, 12):
            maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[i], prefix + param_names[i])


class BLOOMLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True, use_load_prefix=True, split_qkv=False):
        super().__init__(inference, linear_layer=True, use_load_prefix=use_load_prefix, split_qkv=split_qkv)
        self.client_module = client_module
        try:
            import transformers
            BLOOMLayerPolicy._orig_layer_class = transformers.models.bloom.modeling_bloom.BloomBlock
            global supported_models
            supported_models.update({transformers.models.bloom.modeling_bloom.BloomModel})
        except Exception as e:
            print(f"WARNING! Setting BLOOMLayerPolicy._orig_layer_class to None due to Exception: {e}")
            BLOOMLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.self_attention.hidden_size, \
                self.client_module.self_attention.num_heads, \
                self.client_module.input_layernorm.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        return self.client_module.self_attention.query_key_value.weight, \
                self.client_module.self_attention.query_key_value.bias, \
                self.client_module.self_attention.dense.weight, \
                self.client_module.self_attention.dense.bias,

    def mlp(self, enable_training=False):
        return self.client_module.mlp.dense_h_to_4h.weight, \
               self.client_module.mlp.dense_h_to_4h.bias, \
               self.client_module.mlp.dense_4h_to_h.weight, \
               self.client_module.mlp.dense_4h_to_h.bias

    def layernorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias
