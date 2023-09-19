# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features import HybridSplitQKVContainer, HybridGatedMLPContainer, MetaTensorContainer
from deepspeed.utils.types import ActivationFuncType, NormType
from deepspeed.model_implementations.transformers.ds_llama2 import DeepSpeedLlama2Inference
import torch
from torch.nn.parameter import Parameter

from ..policy import (
    TransformerPolicy,
    transformer_param_names,
    maybe_copy,
    maybe_copy_qkv,
    maybe_copy_geglu,
    maybe_get_lora,
)


class DS_LLAMA2Container(MetaTensorContainer, HybridGatedMLPContainer, HybridSplitQKVContainer,
                         BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config

        _config.rotate_half = False
        _config.rotate_every_two = True
        _config.rotary_dim = self.hidden_size // self.num_attention_heads
        _config.num_kv = self.policy.client_module.attention.n_kv_heads
        self.module = DeepSpeedLlama2Inference(_config, mp_group=self.mp_group)

        return self.module

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.feed_forward.w3.weight, self.policy.client_module.feed_forward.w1.weight,
                self.policy.client_module.feed_forward.w2.weight, self.policy.client_module.attention.wq.weight,
                self.policy.client_module.attention.wk.weight, self.policy.client_module.attention.wv.weight,
                self.policy.client_module.attention.wo.weight
            ]
        ]

    def get_lora_matched_pair(self):
        up_proj_lora, gate_proj_lora, down_proj_lora, q_lora, k_lora, v_lora, out_lora = self.get_lora_params()
        ret = [(up_proj_lora, self.inter_up_w), (gate_proj_lora, self.inter_gate_w), (down_proj_lora, self._4hh_w),
               (out_lora, self.dense_w), (q_lora, self.qw), (k_lora, self.kw), (v_lora, self.vw)]
        return ret

    def set_q_k_v(self):
        """
        Necessary to implement for `HybridSplitQKVContainer`
        """
        self.qw = self.policy.client_module.attention.wq.weight
        self.qb = None
        self.kw = self.policy.client_module.attention.wk.weight
        self.kb = None
        self.vw = self.policy.client_module.attention.wv.weight
        self.vb = None

    def set_mlp_gate(self):
        """
        Necessary to implement for `HybridGatedMLPContainer`
        """
        self.inter_up_w = self.policy.client_module.feed_forward.w2.weight
        self.inter_up_b = None
        self.inter_gate_w = self.policy.client_module.feed_forward.w1.weight
        self.inter_gate_b = None

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'attention.wq.weight', \
            'attention.wk.weight', \
            'attention.wv.weight', \
            'attention.wo.weight', \
            'feed_forward.w3.weight', \
            'feed_forward.w1.weight', \
            'feed_forward.w2.weight', \
            'ffn_norm.weight', \
            'attention_norm.weight'
        )

        maybe_copy_qkv(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       'attn_qkvw', [prefix + param_names[0], prefix + param_names[1], prefix + param_names[2]],
                       split_qkv=self.policy.split_qkv)
        for i in range(3, 4):
            maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[i - 1],
                       prefix + param_names[i])
        maybe_copy_geglu(module.mlp, sd, weight_quantizer, mp_replace, 'inter_w',
                         [prefix + param_names[4], prefix + param_names[5]])
        maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, 'output_w', prefix + param_names[6])

        maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[8], prefix + param_names[7])
        maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[10], prefix + param_names[8])


class LLAMA2LayerPolicy(TransformerPolicy):

    def __init__(self, client_module, inference=True):
        super().__init__(
            inference,
            mlp_act_func_type=ActivationFuncType.GATED_SILU,
            norm_type=NormType.RMSNorm,
        )
        self.client_module = client_module
        try:
            import llama
            LLAMA2LayerPolicy._orig_layer_class = llama.model.TransformerBlock  # type: ignore
        except:
            LLAMA2LayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attention.wq.weight.shape[1], \
                self.client_module.n_heads, \
                self.client_module.ffn_norm.eps, \
                (self.client_module.feed_forward.w1.weight.shape[0] * \
                    deepspeed.comm.get_world_size() if deepspeed.comm.is_initialized() else 1) # this is a hack to inject when model is already partitioned!

    def attention(self, enable_training=False):
        qw = self.client_module.attention.wq.weight
        kw = self.client_module.attention.wk.weight
        vw = self.client_module.attention.wv.weight

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)

        return qkvw, \
                None, \
                self.client_module.attention.wo.weight, \
                None

    def mlp(self, enable_training=False):
        mlp1_up = self.client_module.feed_forward.w3.weight
        mlp1_gate = self.client_module.feed_forward.w1.weight
        mlp2 = self.client_module.feed_forward.w2.weight

        mlp1 = Parameter(torch.cat((mlp1_up, mlp1_gate), dim=0), requires_grad=enable_training)

        return mlp1, None, mlp2, None

    def layernorm(self):
        return self.client_module.ffn_norm.weight, \
               None, \
               self.client_module.attention_norm.weight, \
               None
