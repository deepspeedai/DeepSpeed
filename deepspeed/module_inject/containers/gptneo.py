from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy


class DS_GPTNEOContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module


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
        return self.client_module.attn.attention.q_proj.weight.shape[1], \
                self.client_module.attn.attention.num_heads

    def attention(self):
        qw = self.client_module.attn.attention.q_proj.weight
        kw = self.client_module.attn.attention.k_proj.weight
        vw = self.client_module.attn.attention.v_proj.weight

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

        return qkvw, \
               None, \
               self.client_module.attn.attention.out_proj.weight, \
               self.client_module.attn.attention.out_proj.bias

    def mlp(self):
        return self.client_module.mlp.c_fc.weight, \
               self.client_module.mlp.c_fc.bias, \
               self.client_module.mlp.c_proj.weight, \
               self.client_module.mlp.c_proj.bias

    def layernorm(self):
        return self.client_module.ln_2.weight, \
               self.client_module.ln_2.bias, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias

    def get_param_names(self):
        return 'attention.query_key_value.weight', \
               'attention.query_key_value.bias', \
               'attention.dense.weight', \
               'attention.dense.bias', \
               'mlp.dense_h_to_4h.weight', \
               'mlp.dense_h_to_4h.bias', \
               'mlp.dense_4h_to_h.weight', \
               'mlp.dense_4h_to_h.bias', \
               'input_layernorm.weight', \
               'input_layernorm.bias', \
               'post_attention_layernorm.weight', \
               'post_attention_layernorm.bias', \
               self.use_load_prefix, \
               self.split_qkv
