# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features import HybridSplitQKVContainer, HybridGatedMLPContainer, MetaTensorContainer
from deepspeed.utils.types import ActivationFuncType, NormType
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import os
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


class DS_QWEN3_5Container(MetaTensorContainer, HybridGatedMLPContainer, HybridSplitQKVContainer,
                          BaseTransformerContainer):
    """Container for the full_attention blocks of the Qwen3.5/3.6/3.8 family.

    This is the structural rung of the Qwen injection prototype: it proves the
    plumbing (policy match, layer filtering, parameter mapping) but the fused
    kernels do not yet compute this block exactly. Known kernel gaps, tracked
    in experiments/qwen-he-kernel-inject-proto.md:

    1. head_dim != hidden_size // num_attention_heads (27B: 5120/24 is not even
       an integer), which breaks the fused kernel's qkv layout math.
    2. q_proj packs [query | output_gate] (attn_output_gate=True); the gate
       half has no kernel input and is dropped by the mapping below.
    3. Per-head q_norm/k_norm (RMSNorm over head_dim between projection and
       rope) have no place in the fused attention path.
    4. Partial rotary (partial_rotary_factor) and interleaved mrope sections
       are not expressed by the rotate_half/rotary_dim config knobs alone.

    Injection is therefore opt-in via DS_QWEN35_INJECTION=1 until forward
    parity is demonstrated on GPU.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config

        _config.rotate_half = True
        _config.rotate_every_two = False
        # Qwen3.5 decouples head_dim from hidden_size/heads, so take the real
        # head_dim from the client module and only fall back to the quotient.
        attn = self.policy.client_module.self_attn
        head_dim = getattr(attn, 'head_dim', None) or (self.hidden_size // self.num_attention_heads)
        partial = getattr(self._text_config(), 'partial_rotary_factor', None)
        _config.rotary_dim = int(head_dim * partial) if partial else head_dim
        _config.num_kv = getattr(self._text_config(), 'num_key_value_heads', -1)
        _config.rope_theta = self._rope_theta()
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)

        return self.module

    def _text_config(self):
        # Multimodal checkpoints nest the text settings under text_config.
        model_config = getattr(self, 'model_config', None)
        return getattr(model_config, 'text_config', model_config)

    def _rope_theta(self):
        text_config = self._text_config()
        theta = getattr(text_config, 'rope_theta', None)
        if theta is None:
            rope_parameters = getattr(text_config, 'rope_parameters', None) or {}
            theta = rope_parameters.get('rope_theta', 1000000.0)
        return theta

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.mlp.up_proj.weight, self.policy.client_module.mlp.gate_proj.weight,
                self.policy.client_module.mlp.down_proj.weight, self.policy.client_module.self_attn.q_proj.weight,
                self.policy.client_module.self_attn.k_proj.weight, self.policy.client_module.self_attn.v_proj.weight,
                self.policy.client_module.self_attn.o_proj.weight
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
        self.qw = self.policy.client_module.self_attn.q_proj.weight
        self.qb = None
        self.kw = self.policy.client_module.self_attn.k_proj.weight
        self.kb = None
        self.vw = self.policy.client_module.self_attn.v_proj.weight
        self.vb = None

    def set_mlp_gate(self):
        """
        Necessary to implement for `HybridGatedMLPContainer`
        """
        self.inter_up_w = self.policy.client_module.mlp.up_proj.weight
        self.inter_up_b = None
        self.inter_gate_w = self.policy.client_module.mlp.gate_proj.weight
        self.inter_gate_b = None

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'self_attn.q_proj.weight', \
            'self_attn.k_proj.weight', \
            'self_attn.v_proj.weight', \
            'self_attn.o_proj.weight', \
            'mlp.up_proj.weight', \
            'mlp.gate_proj.weight', \
            'mlp.down_proj.weight', \
            'post_attention_layernorm.weight', \
            'input_layernorm.weight',
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

        module.mlp.output_b = None


class Qwen3_5LayerPolicy(TransformerPolicy):
    """Injection policy for Qwen3.5/3.6/3.8 decoder layers (full_attention blocks only)."""

    @staticmethod
    def should_replace(child):
        # The family exposes one decoder-layer class for hybrid blocks; only
        # full_attention blocks map onto the fused attention kernels. The
        # GatedDeltaNet (linear_attention) blocks keep their native forward
        # until linear-attention kernels land.
        return getattr(child, 'block_type', 'full_attention') == 'full_attention'

    def __init__(self, client_module, inference=True):
        super().__init__(
            inference,
            mlp_act_func_type=ActivationFuncType.GATED_SILU,
            norm_type=NormType.RMSNorm,
        )
        self.client_module = client_module
        try:
            import transformers
            # Opt-in gate: the container maps parameters structurally but the
            # fused kernels do not yet reach forward parity (see the container
            # docstring), so the family stays on native generate by default.
            if os.environ.get('DS_QWEN35_INJECTION', '0') == '1':
                Qwen3_5LayerPolicy._orig_layer_class = \
                    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer  # type: ignore
            else:
                Qwen3_5LayerPolicy._orig_layer_class = None
        except (ImportError, AttributeError):
            Qwen3_5LayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        if hasattr(self.client_module.self_attn, 'config'):
            num_heads = self.client_module.self_attn.config.num_attention_heads
        else:
            num_heads = self.client_module.self_attn.num_heads
        # transformers 5.x RMSNorm exposes `eps`; older releases used `variance_epsilon`.
        norm = self.client_module.input_layernorm
        epsilon = getattr(norm, 'eps', None)
        if epsilon is None:
            epsilon = norm.variance_epsilon
        hidden_heads = (
            self.client_module.self_attn.q_proj.in_features,
            num_heads,
            epsilon,
            self.client_module.mlp.gate_proj.out_features,
        )
        return hidden_heads

    def attention(self, enable_training=False):
        attn = self.client_module.self_attn
        # q_proj is 2x-wide: [query | output_gate] along dim0 (attn_output_gate).
        # The fused kernel has no gate input, so only the query half is fused;
        # the dropped gate half is kernel-gap #2 in the container docstring.
        q_len = attn.config.num_attention_heads * attn.head_dim
        qw = attn.q_proj.weight[:q_len]
        kw = attn.k_proj.weight
        vw = attn.v_proj.weight

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)

        return qkvw, \
                None, \
                attn.o_proj.weight, \
                None

    def mlp(self, enable_training=False):
        mlp1_up = self.client_module.mlp.up_proj.weight
        mlp1_gate = self.client_module.mlp.gate_proj.weight
        mlp2 = self.client_module.mlp.down_proj.weight

        mlp1 = Parameter(torch.cat((mlp1_up, mlp1_gate), dim=0), requires_grad=enable_training)

        return mlp1, None, mlp2, None

    def layernorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               None, \
               self.client_module.input_layernorm.weight, \
               None
