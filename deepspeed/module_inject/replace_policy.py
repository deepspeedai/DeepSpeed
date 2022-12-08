'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
from abc import ABC

import torch
from torch.nn.parameter import Parameter
from packaging import version as pkg_version

from deepspeed.utils.types import ActivationFuncType

supported_models = {None}


class DSPolicy(ABC):
    _orig_layer_class = None

    def __init__(self):
        self.cuda_graph_supported = False

    def attention(self):
        """
        Returns attention qkv and dense parameters
        weight: (3*hidden, hidden) and (hidden, hidden)
        bias: (3*hidden) and (hidden)
        """
        raise NotImplementedError


class UNetPolicy(DSPolicy):
    def __init__(self):
        super().__init__()
        try:
            import diffusers
            self._orig_layer_class = diffusers.models.unet_2d_condition.UNet2DConditionModel
        except ImportError:
            self._orig_layer_class = None

    def match(self, module):
        return isinstance(module, self._orig_layer_class)

    def apply(self, module, enable_cuda_graph=True):
        # TODO(cmikeh2): Enable cuda graph should be an inference configuration
        from ..model_implementations.diffusers.unet import DSUNet
        return DSUNet(module, enable_cuda_graph=enable_cuda_graph)

    def attention(self, client_module):
        qw = client_module.to_q.weight
        kw = client_module.to_k.weight
        vw = client_module.to_v.weight

        if qw.shape[1] == kw.shape[1]:
            qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

            return qkvw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
        else:
            #return None
            #kvw = Parameter(torch.cat((kw, vw), dim=0), requires_grad=False)
            return qw, \
                   kw, vw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads


class VAEPolicy(DSPolicy):
    def __init__(self):
        super().__init__()
        try:
            import diffusers
            self._orig_layer_class = diffusers.models.vae.AutoencoderKL
        except ImportError:
            self._orig_layer_class = None

    def match(self, module):
        return isinstance(module, self._orig_layer_class)

    def apply(self, module, enable_cuda_graph=True):
        # TODO(cmikeh2): Enable cuda graph should be an inference configuration
        from ..model_implementations.diffusers.vae import DSVAE
        return DSVAE(module, enable_cuda_graph=enable_cuda_graph)


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
            # the type of activation function used in MLP
            mlp_act_func_type=ActivationFuncType.GELU,
            # applies layer norm before attention if `pre_attn_norm` is set to True
            pre_attn_norm=True,
            # this flag shows whether or not using prefix in loading the checkpoint
            use_load_prefix=False,
            # whether or not the qkv is stored in the split-format
            split_qkv=True):
        super().__init__()
        self.inference = inference
        self.linear_layer = linear_layer
        self.scale_attention = scale_attention
        self.is_megatron_v2 = megatron_v2
        self.mlp_act_func_type = mlp_act_func_type
        self.pre_attn_norm = pre_attn_norm
        self.use_load_prefix = use_load_prefix
        self.split_qkv = split_qkv

    def attention(self):
        """
        Returns attention qkv and dense parameters
        weight: (3*hidden, hidden) and (hidden, hidden)
        bias: (3*hidden) and (hidden)
        """
        raise NotImplementedError

    def get_hidden_heads(self):
        """
        return hidden_size and number of heads
        """
        raise NotImplementedError

    def mlp(self):
        """
        Returns mlp intermediate and output
        weight: (intermediate, hidden) and (hidden, intermediate)
        bias: (intermediate) and (hidden)
        """
        raise NotImplementedError

    def layerNorm(self):
        """
        Returns LayerNorms used in transformer layer
        Post-Attention and pre/post layer norm
        gamma and beta with shape: (hidden)
        """
        raise NotImplementedError

    def get_param_names(self):
        """
        Returns all the transformer parameter names to
        be loaded from checkpoint files. The order of
        the names is as follows:
            1. Attention weights and biases;
            2. MLP weights and biases;
            3. LayerNorm weights and biases;
        In addition to the parameter names, we require two
        more parameters to help read the the data correctly
        from the checkpoint and split the qkv heads in the
        right order:
            1. `use_load_prefix` (Default: False): this specifies
                whether we need to use the name of first abstraction
                layer of the model for searching the parameter's name
                in a checkpoint file. For more information of how this
                is used please see
                https://github.com/microsoft/DeepSpeed/blob/fix-ckpt-loading/deepspeed/module_inject/load_checkpoint.py#L341
            2. `split_qkv` (Default: True): we use this flag when splitting
                the qkv parameter into heads. If it is False, it means the heads
                of q, k, and v are stored together and needs to split in the
                DeepSpeed-Inference API.
        """
        raise NotImplementedError


class HFBertLayerPolicy(TransformerPolicy):
    def __init__(self, client_module, inference=False):
        super().__init__(inference, pre_attn_norm=False)
        self.client_module = client_module
        self.cuda_graph_supported = True

        if HFBertLayerPolicy._orig_layer_class is None:
            try:
                import transformers
                HFBertLayerPolicy._orig_layer_class = [
                    transformers.models.bert.modeling_bert.BertLayer,
                    transformers.models.roberta.modeling_roberta.RobertaLayer
                ]
            except:
                HFBertLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attention.self.query.weight.shape[1], \
                self.client_module.attention.self.num_attention_heads

    def attention(self):
        qw = self.client_module.attention.self.query.weight
        qb = self.client_module.attention.self.query.bias
        kw = self.client_module.attention.self.key.weight
        kb = self.client_module.attention.self.key.bias
        vw = self.client_module.attention.self.value.weight
        vb = self.client_module.attention.self.value.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=False)

        return self.linear_layer, \
               qkvw, \
               qkvb, \
               self.client_module.attention.output.dense.weight, \
               self.client_module.attention.output.dense.bias, \
               self.scale_attention, \
               self.is_megatron_v2

    def mlp(self):
        if self.pre_attn_norm:
            intermediate_ff = self.client_module.intermediate.dense_act
        else:
            intermediate_ff = self.client_module.intermediate.dense

        return self.linear_layer, intermediate_ff.weight, intermediate_ff.bias, \
            self.client_module.output.dense.weight, \
            self.client_module.output.dense.bias

    def layerNorm(self):
        if self.pre_attn_norm:
            attention_layernorm = self.client_module.PostAttentionLayerNorm
            transformer_layernorm = self.client_module.PreAttentionLayerNorm
        else:
            attention_layernorm = self.client_module.attention.output.LayerNorm
            transformer_layernorm = self.client_module.output.LayerNorm
        return attention_layernorm.weight, \
               attention_layernorm.bias, \
               transformer_layernorm.weight, \
               transformer_layernorm.bias


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
                self.client_module.self_attn.num_heads

    def attention(self):
        qw = self.client_module.self_attn.q_proj.weight
        qb = self.client_module.self_attn.q_proj.bias
        kw = self.client_module.self_attn.k_proj.weight
        kb = self.client_module.self_attn.k_proj.bias
        vw = self.client_module.self_attn.v_proj.weight
        vb = self.client_module.self_attn.v_proj.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=False)

        return self.linear_layer, \
               qkvw, \
               qkvb, \
               self.client_module.self_attn.out_proj.weight, \
               self.client_module.self_attn.out_proj.bias, \
               self.scale_attention, \
               self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
            self.client_module.mlp.fc1.weight, \
            self.client_module.mlp.fc1.bias, \
            self.client_module.mlp.fc2.weight, \
            self.client_module.mlp.fc2.bias

    def layerNorm(self):
        return self.client_module.layer_norm2.weight, \
               self.client_module.layer_norm2.bias, \
               self.client_module.layer_norm1.weight, \
               self.client_module.layer_norm1.bias


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

        return self.linear_layer, \
                qkvw, \
                None, \
                self.client_module.attn.attention.out_proj.weight, \
                self.client_module.attn.attention.out_proj.bias, \
                self.scale_attention, \
               self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
                self.client_module.mlp.c_fc.weight, \
                self.client_module.mlp.c_fc.bias, \
                self.client_module.mlp.c_proj.weight, \
                self.client_module.mlp.c_proj.bias

    def layerNorm(self):
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


class HFGPTJLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        super().__init__(inference, scale_attention=True)
        self.client_module = client_module
        try:
            import transformers
            HFGPTJLayerPolicy._orig_layer_class = transformers.models.gptj.modeling_gptj.GPTJBlock
        except:
            HFGPTJLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.q_proj.weight.shape[1], \
                self.client_module.attn.num_attention_heads

    def attention(self):
        qw = self.client_module.attn.q_proj.weight
        kw = self.client_module.attn.k_proj.weight
        vw = self.client_module.attn.v_proj.weight

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

        return self.linear_layer, \
            qkvw, \
            None, \
            self.client_module.attn.out_proj.weight, \
            None, \
            self.scale_attention, \
            self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
                self.client_module.mlp.fc_in.weight, \
                self.client_module.mlp.fc_in.bias, \
                self.client_module.mlp.fc_out.weight, \
                self.client_module.mlp.fc_out.bias

    def layerNorm(self):
        return None, \
               None, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias

    def get_param_names(self):
        return 'attn.q_proj.weight', \
               'attn.k_proj.weight', \
               'attn.v_proj.weight', \
               'attn.out_proj.weight', \
               'mlp.fc_in.weight', \
               'mlp.fc_in.bias', \
               'mlp.fc_out.weight', \
               'mlp.fc_out.bias', \
               'ln_1.weight', \
               'ln_1.bias', \
               self.use_load_prefix, \
               self.split_qkv


class MegatronLayerPolicy(TransformerPolicy):
    _orig_layer_class = None
    version = 0
    moe_type = 'standard'
    megatron_v2 = True

    def __init__(self, client_module, inference=True):
        super().__init__(inference, megatron_v2=MegatronLayerPolicy.megatron_v2)
        self.client_module = client_module
        # we use megatron version to differentiate between the old and new
        # megatron-lm source code
        if MegatronLayerPolicy._orig_layer_class is None:
            if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
                MegatronLayerPolicy._orig_layer_class = None
            else:
                try:
                    from megatron.model.transformer import ParallelTransformerLayer
                    MegatronLayerPolicy._orig_layer_class = ParallelTransformerLayer
                except ImportError:
                    MegatronLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attention.query_key_value.weight.shape[1], \
                self.client_module.attention.num_attention_heads

    def attention(self):
        if self.inference:
            if MegatronLayerPolicy.version == 0:
                attention = self.client_module.attention
            else:
                attention = self.client_module.self_attention

        return self.linear_layer, \
                attention.query_key_value.weight, \
                attention.query_key_value.bias, \
                attention.dense.weight, \
                attention.dense.bias, \
                self.scale_attention, \
                self.is_megatron_v2

    def mlp(self, moe_type='standard'):
        from deepspeed.moe.utils import has_moe_layers
        moe, _ = has_moe_layers(self.client_module)

        if moe:
            moe_experts = self.client_module.mlp.deepspeed_moe.experts.deepspeed_experts if moe_type == 'standard' else \
                            self.client_module.mlp.moe.deepspeed_moe.experts.deepspeed_experts
            num_experts = len(moe_experts)
            if moe_type == 'standard':
                return self.linear_layer, \
                    [moe_experts[i].dense_h_to_4h.weight for i in range(num_experts)], \
                    [moe_experts[i].dense_h_to_4h.bias for i in range(num_experts)], \
                    [moe_experts[i].dense_4h_to_h.weight for i in range(num_experts)], \
                    [moe_experts[i].dense_4h_to_h.bias for i in range(num_experts)]
            else:

                return self.linear_layer, \
                    [moe_experts[i].dense_h_to_4h.weight for i in range(num_experts)], \
                    [moe_experts[i].dense_h_to_4h.bias for i in range(num_experts)], \
                    [moe_experts[i].dense_4h_to_h.weight for i in range(num_experts)], \
                    [moe_experts[i].dense_4h_to_h.bias for i in range(num_experts)], \
                    self.client_module.mlp.mlp.dense_h_to_4h.weight, \
                    self.client_module.mlp.mlp.dense_h_to_4h.bias, \
                    self.client_module.mlp.mlp.dense_4h_to_h.weight, \
                    self.client_module.mlp.mlp.dense_4h_to_h.bias, \
                    self.client_module.mlp.coefficient.weight

        else:
            return self.linear_layer, \
                self.client_module.mlp.dense_h_to_4h.weight, \
                self.client_module.mlp.dense_h_to_4h.bias, \
                self.client_module.mlp.dense_4h_to_h.weight, \
                self.client_module.mlp.dense_4h_to_h.bias

    def layerNorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias


class HFGPT2LayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        # HuggingFace GPT2 uses convolutional layer instead of linear layer
        super().__init__(inference, linear_layer=False)
        self.client_module = client_module
        try:
            import transformers
            HFGPT2LayerPolicy._orig_layer_class = transformers.models.gpt2.modeling_gpt2.GPT2Block
        except:
            HFGPT2LayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.embed_dim, \
                self.client_module.attn.num_heads

    def attention(self):
        return self.linear_layer, \
                self.client_module.attn.c_attn.weight, \
                self.client_module.attn.c_attn.bias, \
                self.client_module.attn.c_proj.weight, \
                self.client_module.attn.c_proj.bias, \
                self.scale_attention, \
                self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
            self.client_module.mlp.c_fc.weight, \
            self.client_module.mlp.c_fc.bias, \
            self.client_module.mlp.c_proj.weight, \
            self.client_module.mlp.c_proj.bias

    def layerNorm(self):
        return self.client_module.ln_2.weight, \
               self.client_module.ln_2.bias, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias


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
        except:
            BLOOMLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.self_attention.hidden_size, \
                self.client_module.self_attention.num_heads

    def attention(self):
        return self.linear_layer, \
                self.client_module.self_attention.query_key_value.weight, \
                self.client_module.self_attention.query_key_value.bias, \
                self.client_module.self_attention.dense.weight, \
                self.client_module.self_attention.dense.bias, \
                self.scale_attention, \
                self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
            self.client_module.mlp.dense_h_to_4h.weight, \
            self.client_module.mlp.dense_h_to_4h.bias, \
            self.client_module.mlp.dense_4h_to_h.weight, \
            self.client_module.mlp.dense_4h_to_h.bias

    def layerNorm(self):
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


class GPTNEOXLayerPolicy(TransformerPolicy):
    _orig_layer_class = None
    version = 0

    def __init__(self, client_module, inference=True, megatron_v2=True, split_qkv=False):
        super().__init__(inference, megatron_v2=megatron_v2)
        self.client_module = client_module
        if GPTNEOXLayerPolicy._orig_layer_class is None:
            if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
                GPTNEOXLayerPolicy._orig_layer_class = None
            else:
                try:
                    from transformers import GPTNeoXLayer
                    GPTNEOXLayerPolicy._orig_layer_class = GPTNeoXLayer
                except ImportError:
                    GPTNEOXLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        if GPTNEOXLayerPolicy.version == 0:
            attention = self.client_module.attention
        else:
            attention = self.client_module.self_attention

        return self.client_module.attention.query_key_value.weight.shape[1], \
                self.client_module.attention.num_attention_heads

    def attention(self):
        if GPTNEOXLayerPolicy.version == 0:
            attention = self.client_module.attention
        else:
            attention = self.client_module.self_attention

        return self.linear_layer, \
                attention.query_key_value.weight, \
                attention.query_key_value.bias, \
                attention.dense.weight, \
                attention.dense.bias, \
                self.scale_attention, \
                self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
            self.client_module.mlp.dense_h_to_4h.weight, \
            self.client_module.mlp.dense_h_to_4h.bias, \
            self.client_module.mlp.dense_4h_to_h.weight, \
            self.client_module.mlp.dense_4h_to_h.bias

    def layerNorm(self):
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias

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


class HFOPTLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True, use_load_prefix=True):
        super().__init__(inference,
                         linear_layer=True,
                         mlp_act_func_type=ActivationFuncType.ReLU,
                         pre_attn_norm=True)
        self.client_module = client_module
        try:
            import transformers
            HFOPTLayerPolicy._orig_layer_class = transformers.models.opt.modeling_opt.OPTDecoderLayer
            if isinstance(TransformerPolicy.hf_model_config,
                          transformers.models.opt.configuration_opt.OPTConfig):
                self.pre_attn_norm = TransformerPolicy.hf_model_config.do_layer_norm_before
        except:
            HFOPTLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.self_attn.embed_dim, \
                self.client_module.self_attn.num_heads

    def attention(self):
        qw = self.client_module.self_attn.q_proj.weight
        qb = self.client_module.self_attn.q_proj.bias

        kw = self.client_module.self_attn.k_proj.weight
        kb = self.client_module.self_attn.k_proj.bias

        vw = self.client_module.self_attn.v_proj.weight
        vb = self.client_module.self_attn.v_proj.bias

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)
        qkvb = Parameter(torch.cat((qb, kb, vb), dim=0), requires_grad=False)

        return self.linear_layer, \
            qkvw, \
            qkvb, \
            self.client_module.self_attn.out_proj.weight, \
            self.client_module.self_attn.out_proj.bias, \
            self.scale_attention, \
            self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
            self.client_module.fc1.weight, \
            self.client_module.fc1.bias, \
            self.client_module.fc2.weight, \
            self.client_module.fc2.bias

    def layerNorm(self):
        return self.client_module.final_layer_norm.weight, \
            self.client_module.final_layer_norm.bias, \
            self.client_module.self_attn_layer_norm.weight, \
            self.client_module.self_attn_layer_norm.bias

    def get_param_names(self):
        return 'self_attn.q_proj.weight', \
               'self_attn.q_proj.bias', \
               'self_attn.k_proj.weight', \
               'self_attn.k_proj.bias', \
               'self_attn.v_proj.weight', \
               'self_attn.v_proj.bias', \
               'self_attn.out_proj.weight', \
               'self_attn.out_proj.bias', \
               'fc1.weight', \
               'fc1.bias', \
               'fc2.weight', \
               'fc2.bias', \
               'self_attn_layer_norm.weight', \
               'self_attn_layer_norm.bias', \
               'final_layer_norm.weight', \
               'final_layer_norm.bias', \
               self.use_load_prefix, \
               self.split_qkv


# transformer-based policies
replace_policies = [
    HFBertLayerPolicy,
    HFGPTNEOLayerPolicy,
    GPTNEOXLayerPolicy,
    HFGPTJLayerPolicy,
    MegatronLayerPolicy,
    HFGPT2LayerPolicy,
    BLOOMLayerPolicy,
    HFOPTLayerPolicy,
    HFCLIPLayerPolicy,
]

# non-transformer-based policies
generic_policies = [UNetPolicy, VAEPolicy]
