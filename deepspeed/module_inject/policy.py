'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from abc import ABC, abstractmethod
from deepspeed.utils.types import ActivationFuncType


class DSPolicy(ABC):
    _orig_layer_class = None

    def __init__(self):
        self.cuda_graph_supported = False

    @abstractmethod
    def attention(self):
        """
        Returns attention qkv and dense parameters
        weight: (3*hidden, hidden) and (hidden, hidden)
        bias: (3*hidden) and (hidden)
        """
        raise NotImplementedError


class TransformerPolicy(DSPolicy):
    # a static class variable containing the HuggingFace model configuration.
    # see e.g., transformers.models.opt.configuration_opt.OPTConfig
    hf_model_config = None

    def __init__(
            self,
            inference=True,
            linear_layer=True,
            scale_attention=True,
            megatron_v2=False,
            use_mup=False,
            # the type of activation function used in MLP
            mlp_act_func_type=ActivationFuncType.GELU,
            # applies layer norm before attention if `pre_attn_norm` is set to True
            pre_attn_norm=True,
            # this flag shows whether or not using prefix in loading the checkpoint
            use_load_prefix=False,
            # whether or not the qkv is stored in the split-format
            split_qkv=True):
        super().__init__()
        self.cuda_graph_supported = False
        self.inference = inference
        self.linear_layer = linear_layer
        self.scale_attention = scale_attention
        self.is_megatron_v2 = megatron_v2
        self.use_mup = use_mup
        self.mlp_act_func_type = mlp_act_func_type
        self.pre_attn_norm = pre_attn_norm
        self.use_load_prefix = use_load_prefix
        self.split_qkv = split_qkv

    @abstractmethod
    def attention(self):
        """
        Returns attention qkv and dense parameters
        weight: (3*hidden, hidden) and (hidden, hidden)
        bias: (3*hidden) and (hidden)
        """
        raise NotImplementedError

    @abstractmethod
    def get_hidden_heads(self):
        """
        return hidden_size and number of heads
        """
        raise NotImplementedError

    @abstractmethod
    def mlp(self):
        """
        Returns mlp intermediate and output
        weight: (intermediate, hidden) and (hidden, intermediate)
        bias: (intermediate) and (hidden)
        """
        raise NotImplementedError

    @abstractmethod
    def layernorm(self):
        """
        Returns LayerNorms used in transformer layer
        Post-Attention and pre/post layer norm
        gamma and beta with shape: (hidden)
        """
        raise NotImplementedError
