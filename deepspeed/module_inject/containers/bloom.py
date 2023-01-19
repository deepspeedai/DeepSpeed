from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_bloom import DeepSpeedBloomInference
from ..policy import TransformerPolicy

supported_models = {None}


class DS_BloomContainer(MetaTensorContainer, BaseTransformerContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.
        self.bigscience_bloom = True

    def create_module(self, config=None):
        _config = config if config is not None else self.config

        self.module = DeepSpeedBloomInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def attention_qkv_mp(self, mp_replace):
        self.module.attention.attn_qkvw = mp_replace.copy(
            self.module.attention.attn_qkvw,
            self.qkvw)
        self.module.attention.attn_qkvb = mp_replace.copy(
            self.module.attention.attn_qkvb,
            self.qkvb)


class BLOOMLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self,
                 client_module,
                 inference=True,
                 use_load_prefix=True,
                 split_qkv=False):
        super().__init__(inference, linear_layer=True)
        self.client_module = client_module
        try:
            import transformers
            BLOOMLayerPolicy._orig_layer_class = transformers.models.bloom.modeling_bloom.BloomBlock
            global supported_models
            supported_models.update(
                {transformers.models.bloom.modeling_bloom.BloomModel})
        except Exception as e:
            print(
                f"WARNING! Setting BLOOMLayerPolicy._orig_layer_class to None due to Exception: {e}"
            )
            BLOOMLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.self_attention.hidden_size, \
                self.client_module.self_attention.num_heads

    def attention(self):
        return self.client_module.self_attention.query_key_value.weight, \
                self.client_module.self_attention.query_key_value.bias, \
                self.client_module.self_attention.dense.weight, \
                self.client_module.self_attention.dense.bias,

    def mlp(self):
        return self.client_module.mlp.dense_h_to_4h.weight, \
            self.client_module.mlp.dense_h_to_4h.bias, \
            self.client_module.mlp.dense_4h_to_h.weight, \
            self.client_module.mlp.dense_4h_to_h.bias

    def layernorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias

    def get_param_names(self):
        return 'self_attention.query_key_value.weight', \
               'self_attention.query_key_value.bias', \
               'self_attention.dense.weight', \
               'self_attention.dense.bias', \
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
