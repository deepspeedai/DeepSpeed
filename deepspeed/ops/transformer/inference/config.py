# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import torch
from deepspeed.utils.types import ActivationFuncType, NormType


class TransformerConfig():

    def __init__(self, hidden_size, intermediate_size, heads, num_hidden_layers):
        self.layer_id = -1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.heads = heads
        self.num_hidden_layers = num_hidden_layers


class DeepSpeedInferenceConfig(TransformerConfig):
    """Initialize the DeepSpeed Transformer Config.
        Arguments:
            hidden_size: The hidden size of the transformer layer
            intermediate_size: The intermediate size of the feed-forward part of transformer layer
            heads: The number of heads in the self-attention of the transformer layer
            num_hidden_layers: The number of transformer layers
            layer_norm_eps: The epsilon value for the layer norm
            local_rank: Optional: The rank of GPU running the transformer kernel, it is not required
                to use if the model already set the current device, otherwise need to set it
                so that the transformer kernel can work on the right device
            mp_size (optional): This argument is mainly used to create the parameters on the kernel side
                using model-parallel architecture. If the client model already takes care of this, there is no
                need to pass this argument.
            pre_layer_norm: Select between Pre-LN or Post-LN transformer architecture
            stochastic_mode:  Enable for high performance, please note that this flag has some level of
                non-determinism and can produce different results on different runs.  However, we have seen
                that by enabling it, the pretraining tasks such as BERT are not affected and can obtain
                a high accuracy level. On the other hand, for the downstream tasks, such as fine-tuning, we recommend
                to turn it off in order to be able to reproduce the same result through the regular kernel execution.

            scale_attention: If true, both q and k are scaled by 1/sqrt(attention_heads) before attention computation.
            return_tuple: if True, returns the transformer output as a tuple, otherwise returns as a tensor
            bigscience_bloom: This flag is added temporarily for supporting the BLOOM-176B model architecture.
            use_triton: This flag is to enable triton kernels in inference or not.
            invert_mask: If True, the attention mask is inverted when passed to attention block.
    """

    def __init__(self,
                 hidden_size=-1,
                 intermediate_size=-1,
                 heads=-1,
                 num_hidden_layers=-1,
                 layer_norm_eps=1e-12,
                 local_rank=-1,
                 mp_size=1,
                 dtype=torch.float16,
                 pre_layer_norm=True,
                 norm_type=NormType.LayerNorm,
                 stochastic_mode=False,
                 scale_attention=True,
                 triangular_masking=True,
                 local_attention=False,
                 window_size=256,
                 rotary_dim=-1,
                 rotate_half=False,
                 rotate_every_two=True,
                 return_tuple=True,
                 mlp_after_attn=True,
                 mlp_act_func_type=ActivationFuncType.GELU,
                 training_mp_size=1,
                 bigscience_bloom=False,
                 max_out_tokens=1024,
                 min_out_tokens=1,
                 enable_qkv_quantization=False,
                 use_mup=False,
                 scale_attn_by_inverse_layer_idx=False,
                 return_single_tuple=False,
                 set_empty_params=False,
                 transposed_mode=False,
                 use_triton=False,
                 triton_autotune=False,
                 num_kv=-1,
                 rope_theta=10000,
                 invert_mask=True):
        super(DeepSpeedInferenceConfig,
              self).__init__(hidden_size, (intermediate_size if intermediate_size > 0 else 4 * hidden_size), heads,
                             num_hidden_layers)
        self.dtype = dtype
        self.pre_layer_norm = pre_layer_norm
        self.norm_type = norm_type
        self.local_rank = local_rank
        self.stochastic_mode = stochastic_mode
        self.epsilon = layer_norm_eps
        self.mp_size = mp_size
        self.scale_attention = scale_attention
        self.triangular_masking = triangular_masking
        self.local_attention = local_attention
        self.window_size = window_size
        self.rotary_dim = rotary_dim
        self.rotate_half = rotate_half
        self.rotate_every_two = rotate_every_two
        self.return_tuple = return_tuple
        self.mlp_after_attn = mlp_after_attn
        self.mlp_act_func_type = mlp_act_func_type
        self.training_mp_size = training_mp_size
        self.bigscience_bloom = bigscience_bloom
        self.max_out_tokens = max_out_tokens
        self.min_out_tokens = min_out_tokens
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.enable_qkv_quantization = enable_qkv_quantization
        self.use_mup = use_mup
        self.return_single_tuple = return_single_tuple
        self.set_empty_params = set_empty_params
        self.transposed_mode = transposed_mode
        self.use_triton = use_triton
        self.triton_autotune = triton_autotune
        self.num_kv = num_kv
        self.rope_theta = rope_theta
        self.invert_mask = invert_mask

    @classmethod
    def from_dict(cls, json_object):
        config = DeepSpeedInferenceConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))
