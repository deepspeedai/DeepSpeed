# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from deepspeed.utils.types import ActivationFuncType, NormType
import torch
from deepspeed.accelerator import get_accelerator

transformer_param_names = (
        'attn_qkvw', \
        'attn_qkvb', \
        'attn_ow' , \
        'attn_ob', \
        'inter_w', \
        'inter_b', \
        'output_w', \
        'output_b', \
        'attn_nw', \
        'attn_nb', \
        'norm_w', \
        'norm_b')


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
            split_qkv=True,
            # Type of normalization to perform
            norm_type=NormType.LayerNorm):
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
        self.norm_type = norm_type

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


# TODO (lekurile): This function exists in base container as well, consolidate as some point
def transpose(data):
    with torch.no_grad():
        data = data.contiguous()
        data1 = data.transpose(-1, -2).reshape(-1)
        data.reshape(-1).copy_(data1)
        data1 = None
    return data.reshape(data.shape[-1], data.shape[-2])


# TODO (lekurile): This function exists in megatron feature container as well, consolidate as some point
def _transpose(x, heads=1, mp_replace=None):
    heads = heads // mp_replace.mp_size  # type: ignore
    outer_dim = -1
    attention_head_size = x.shape[outer_dim] // heads
    new_x_shape = x.size()[:outer_dim] + (heads, attention_head_size)
    x_1 = x.view(*new_x_shape)
    (q, k, v) = torch.split(x_1, (x_1.shape[-1] // 3), dim=-1)
    if len(q.shape) > 2:
        new_shape = (q.shape[0], ) + (-1, )
        return torch.cat((q.reshape(new_shape), k.reshape(new_shape), v.reshape(new_shape)),
                         dim=outer_dim).reshape(x.shape)
    else:
        return torch.cat((q.reshape(-1), k.reshape(-1), v.reshape(-1)), dim=-1).reshape(x.shape)


# This checks if the parameter exits in the checkpoint file and maybe copies it into the corresponding destination tensor.
# Note that not all parameters are saved in one checkpoint, that's why we always need to check if they exist!
def maybe_copy(module,
               sd,
               weight_quantizer,
               mp_replace,
               dst_name,
               src_name,
               qkv=False,
               megatron_v2=False,
               split_qkv=False,
               heads=1):
    if src_name in sd:
        dst = getattr(module, dst_name)
        tmp = sd[src_name]
        if len(dst.shape) == 1:
            if split_qkv:
                dst = mp_replace.strided_copy(dst, tmp, num_splits=3)
            else:
                dst = mp_replace.copy(dst, tmp)
            if qkv and megatron_v2:
                dst = torch.nn.parameter.Parameter(_transpose(dst, heads=heads, mp_replace=mp_replace).contiguous())
        else:
            if split_qkv:
                dst = mp_replace.strided_copy(dst, weight_quantizer.quantize(tmp if weight_quantizer.q_int8 else \
                                                (transpose(tmp).contiguous())), num_splits=3, int8=weight_quantizer.q_int8)
            else:
                if qkv and megatron_v2:
                    tmp = _transpose(transpose(tmp), heads=heads, mp_replace=mp_replace).contiguous()
                    if weight_quantizer.q_int8:
                        tmp = transpose(tmp)
                dst = mp_replace.copy(dst, weight_quantizer.quantize(tmp if weight_quantizer.q_int8 else \
                                                transpose(tmp)), int8=weight_quantizer.q_int8)
        setattr(module, dst_name, dst)


# Extending the maybe_copy function for when the q, k, and v are in separate parameters!
def maybe_copy_qkv(module, sd, weight_quantizer, mp_replace, dst_name, src_names, split_qkv=False):
    if src_names[0] in sd:
        q = sd[src_names[0]]
        k = sd[src_names[1]]
        v = sd[src_names[2]]
        qkv_data = torch.cat((q, k, v), dim=0)
        dst = getattr(module, dst_name)
        if len(dst.shape) == 1:
            if split_qkv:
                dst = mp_replace.strided_copy(dst, qkv_data.contiguous(), num_splits=3)
            else:
                dst = mp_replace.copy(dst, qkv_data)
        else:
            if split_qkv:
                dst = mp_replace.strided_copy(dst, weight_quantizer.quantize(qkv_data.to(get_accelerator().device_name()) if weight_quantizer.q_int8 else \
                                                ((transpose(qkv_data)).contiguous())), num_splits=3, int8=weight_quantizer.q_int8)
            else:
                dst = mp_replace.copy(dst, weight_quantizer.quantize(qkv_data.to(get_accelerator().device_name()) if weight_quantizer.q_int8 else \
                                                transpose(qkv_data)), int8=weight_quantizer.q_int8)
        setattr(module, dst_name, dst)


# Extending the `maybe_copy` function for when mlp1 is in separate parameters for GeGLU
def maybe_copy_geglu(module, sd, weight_quantizer, mp_replace, dst_name, src_names):
    if src_names[0] in sd:
        reg_proj = sd[src_names[0]]
        gate_proj = sd[src_names[1]]

        mlp1_data = torch.cat((reg_proj, gate_proj), dim=0)
        dst = getattr(module, dst_name)

        dst = mp_replace.strided_copy(dst, weight_quantizer.quantize(mlp1_data.to(get_accelerator().device_name()) if weight_quantizer.q_int8 else \
                                            transpose(mlp1_data)), num_splits=2, int8=weight_quantizer.q_int8)
        setattr(module, dst_name, dst)


def pack_lora_weights(p):
    return [
        p.lora_right_weight, \
        p.lora_left_weight, \
        p.lora_scaling
    ]


def maybe_get_lora(p):
    if hasattr(p, 'lora_right_weight'):
        lora_param = pack_lora_weights(p)
    else:
        lora_param = []
    return lora_param
