# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.
from abc import ABC

import torch

import deepspeed
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from deepspeed.accelerator import get_accelerator

# If the intermediate size attribute is set DEFAULT_INTERMEDIATE_SIZE
# it is assumed the intermediate size is 4x the embedding dimension
DEFAULT_INTERMEDIATE_SIZE = -1


class BaseConvolutionContainer(ABC):
    # not implemented
    def __init__(self):
        pass


class BaseTransformerContainer(ABC):

    def __init__(self, policy, config, model_config, layer_id, child):
        self.policy = policy
        self.config = config
        self.model_config = model_config
        self.layer_id = layer_id
        self.child = child

        self.megatron_v2 = self.policy.is_megatron_v2
        self.scale_attention = self.policy.scale_attention
        self.ckpt_load_enabled = False

        # configuration for models. todo: can this be moved to a pydantic model config?
        self.hidden_size = None
        self.intermediate_size = None
        self.num_attention_heads = None
        self.mp_size = self.config.tensor_parallel.tp_size
        self.pre_layer_norm = self.model_config.do_layer_norm_before if \
            hasattr(self.model_config, 'do_layer_norm_before') else self.policy.pre_attn_norm
        self.dtype = self.config.dtype
        self.attn_linear_layer = self.policy.linear_layer
        self.mlp_linear_layer = self.policy.linear_layer
        self.return_tuple = self.config.return_tuple
        self.triangular_masking = True
        self.local_attention = ((self.model_config.attention_layers[self.layer_id] == "local") if hasattr(
            self.model_config, 'attention_layers') else False)
        self.window_size = getattr(self.model_config, "window_size", 1)
        self.mlp_act_func_type = self.policy.mlp_act_func_type
        self.norm_type = self.policy.norm_type
        self.training_mp_size = self.config.training_mp_size
        self.bigscience_bloom = False
        self.max_out_tokens = self.config.max_out_tokens
        self.min_out_tokens = self.config.min_out_tokens
        self.scale_attn_by_inverse_layer_idx = getattr(self.config, "scale_attn_by_inverse_layer_idx", False)
        self.use_mup = self.policy.use_mup
        self.return_single_tuple = False
        self.rotary_dim = self.get_rotary_dim()
        self.mlp_after_attn = (self.rotary_dim is None or self.rotary_dim < 0)

        # Attention tensors
        self.qkvw = None
        self.qkvb = None
        self.dense_w = None
        self.dense_b = None
        # MLP tensors
        self._h4h_w = None
        self._h4h_b = None
        self._4hh_w = None
        self._4hh_b = None
        # LayerNorm tensors
        self.attn_nw = None
        self.attn_nb = None
        self.input_nw = None
        self.input_nb = None

        self.mp_group = None
        self.use_triton = False

        # Triton
        self.use_triton = config.use_triton and deepspeed.HAS_TRITON

    def create_ds_model_config(self):
        self.set_hidden_heads(*self.policy.get_hidden_heads())
        assert self.num_attention_heads % self.mp_size == 0,\
                "To run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!" +\
                "This is because the attention computation is partitioned evenly among the parallel GPUs."

        self.ds_model_config = DeepSpeedInferenceConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            heads=self.num_attention_heads,
            layer_norm_eps=self.layernorm_epsilon,
            dtype=self.dtype,
            pre_layer_norm=self.pre_layer_norm,
            norm_type=self.norm_type,
            mp_size=self.mp_size,
            return_tuple=self.return_tuple,
            triangular_masking=self.triangular_masking,
            local_attention=self.local_attention,
            window_size=self.window_size,
            rotary_dim=self.rotary_dim,
            mlp_after_attn=self.mlp_after_attn,
            mlp_act_func_type=self.mlp_act_func_type,
            training_mp_size=self.training_mp_size,
            bigscience_bloom=self.bigscience_bloom,
            max_out_tokens=self.max_out_tokens,
            min_out_tokens=self.min_out_tokens,
            scale_attn_by_inverse_layer_idx=self.scale_attn_by_inverse_layer_idx,
            use_mup=self.use_mup,
            return_single_tuple=self.return_single_tuple,
            set_empty_params=self.config.set_empty_params,
            transposed_mode=self.config.transposed_mode,
            use_triton=self.use_triton,
            triton_autotune=self.config.triton_autotune)

        if self.use_triton and deepspeed.HAS_TRITON:
            from .bert import DS_BERTContainer
            if not isinstance(self, DS_BERTContainer):
                raise NotImplementedError("Triton kernels are only for BERT-like models yet")

            if not self.config.triton_autotune:
                from deepspeed.ops.transformer.inference.triton.matmul_ext import fp16_matmul
                fp16_matmul.skip_autotune()

        return self.ds_model_config

    def check_meta_tensor_support(self):
        if hasattr(self.qkvw, 'is_meta'):
            if self.qkvw.is_meta:
                assert self.ckpt_load_enabled, "Meta tensors are not supported for this model currently."
        else:
            raise NotImplementedError("Meta tensor support is not available, please upgrade to torch 1.10+")

    def initialize_tensors(self, enable_training=False):
        # Set the tensors from policy (user module) to container (DS module)
        self.set_attention(*self.policy.attention(enable_training=enable_training))
        self.set_mlp(*self.policy.mlp(enable_training=enable_training))
        self.set_layernorm(*self.policy.layernorm())
        #self.check_meta_tensor_support()

    def convert_to_required_dtype(self):
        # Note: converting tensors to fp16 requires that we do it in-place using self.__dict__ and not make a list/dict copy
        if self.dtype in [torch.half, torch.bfloat16]:
            for k, v in self.__dict__.items():
                # The list comprehension is used for MoE tensor lists
                if isinstance(v, list) and all((isinstance(tensor, torch.Tensor) \
                   or isinstance(tensor, torch.nn.Parameter)) for tensor in v):
                    self.__dict__[k] = [moe_tensor.to(self.dtype) for moe_tensor in v]

                if isinstance(v, torch.Tensor) or isinstance(v, torch.nn.Parameter):
                    self.__dict__[k] = v.to(self.dtype)

    def get_rotary_dim(self):
        if hasattr(self.model_config, 'rotary_dim'):
            return self.model_config.rotary_dim
        if hasattr(self.child, 'attention') and hasattr(self.child.attention, 'rotary_ndims'):
            return self.child.attention.rotary_ndims
        return -1

    def set_moe(self, moe=False):
        self.moe = moe

    def set_tensor_parallel_config(self, mp_size, mp_group):
        self.mp_size = mp_size
        self.mp_group = mp_group

    def set_quantization_config(self, quantizer):
        self.quantizer = quantizer

    def set_hidden_heads(self, hidden_size, num_attention_heads, epsilon, intermediate_size):
        """
        Args:
            hidden_size: embedding dimension of the model
            num_attention_heads: number of attention heads in the model
            epsilon: epsilon value for layer norm (same value used for all norms)
            intermediate_size: Size of MLP projection. If `DEFAULT_INTERMEDIATE_SIZE` is passed
                it is assumed to be `4 * hidden_size`
        """
        self.hidden_size = hidden_size
        if intermediate_size == DEFAULT_INTERMEDIATE_SIZE:
            self.intermediate_size = 4 * hidden_size
        else:
            self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.layernorm_epsilon = epsilon

    def set_attention(self, qkvw, qkvb, dense_w, dense_b):
        self.qkvw = qkvw
        self.qkvb = qkvb
        self.dense_w = dense_w
        self.dense_b = dense_b

    def set_mlp(self, _h4h_w, _h4h_b, _4hh_w, _4hh_b):
        self._h4h_w = _h4h_w
        self._h4h_b = _h4h_b
        self._4hh_w = _4hh_w
        self._4hh_b = _4hh_b

    def set_layernorm(self, attn_nw, attn_nb, input_nw, input_nb):
        self.attn_nw = attn_nw
        self.attn_nb = attn_nb
        self.input_nw = input_nw
        self.input_nb = input_nb

    def apply_weight_quantization(self):
        # quantize attention weights
        self.attention_quantization()

        # quantize mlp weights
        self.mlp_quantization()

    def attention_quantization(self):
        self.module.attention.attn_qkvw = self.quantizer.quantize(self.module.attention.attn_qkvw)
        self.module.attention.attn_ow = self.quantizer.quantize(self.module.attention.attn_ow)

    def mlp_quantization(self):
        self.module.mlp.inter_w = self.quantizer.quantize(self.module.mlp.inter_w)
        self.module.mlp.output_w = self.quantizer.quantize(self.module.mlp.output_w)

    def apply_tensor_parallelism(self, mp_replace):
        # setup the new Attention module
        self.attention_qkv_mp(mp_replace)
        self.attention_o_mp(mp_replace)

        # setup the new MLP module
        self.mlp_inter_mp(mp_replace)
        self.mlp_output_mp(mp_replace)

        # Apply weight quantization
        # TODO(cmikeh2): Re-enable this once verified
        #self.apply_weight_quantization()

    def attention_qkv_mp(self, mp_replace, reversed_dim=False):
        self.module.attention.attn_qkvw = mp_replace.strided_copy(self.module.attention.attn_qkvw,
                                                                  self.qkvw,
                                                                  num_splits=3,
                                                                  int8=reversed_dim)
        self.module.attention.attn_qkvb = mp_replace.strided_copy(self.module.attention.attn_qkvb,
                                                                  self.qkvb,
                                                                  num_splits=3,
                                                                  int8=reversed_dim)

    def attention_o_mp(self, mp_replace, reversed_dim=False):
        self.module.attention.attn_ow = mp_replace.copy(self.module.attention.attn_ow, self.dense_w, int8=reversed_dim)
        self.module.attention.attn_ob = mp_replace.copy(self.module.attention.attn_ob,
                                                        self.dense_b,
                                                        int8=reversed_dim,
                                                        allocate_tensor=reversed_dim)

    def mlp_inter_mp(self, mp_replace, reversed_dim=False):
        self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w, self._h4h_w, int8=reversed_dim)
        self.module.mlp.inter_b = mp_replace.copy(self.module.mlp.inter_b, self._h4h_b, int8=reversed_dim)

    def mlp_output_mp(self, mp_replace, reversed_dim=False):
        self.module.mlp.output_w = mp_replace.copy(self.module.mlp.output_w, self._4hh_w, int8=reversed_dim)
        self.module.mlp.output_b = mp_replace.copy(self.module.mlp.output_b,
                                                   self._4hh_b,
                                                   int8=reversed_dim,
                                                   allocate_tensor=reversed_dim)

    def copy_data_to_new_module(self):
        params = {'attn_nw': self.attn_nw, 'attn_nb': self.attn_nb}
        for key in params:
            if params[key] is None:
                setattr(self.module.mlp, key, None)
            else:
                setattr(self.module.mlp, key,
                        torch.nn.parameter.Parameter(params[key].to(get_accelerator().current_device_name())))

        params = {'norm_w': self.input_nw, 'norm_b': self.input_nb}
        for key in params:
            if params[key] is None:
                setattr(self.module, key, None)
            else:
                setattr(self.module, key,
                        torch.nn.parameter.Parameter(params[key].to(get_accelerator().current_device_name())))

    def transpose(self):
        self.transpose_attention()
        self.transpose_mlp()

    def transpose_attention(self):
        if self.attn_linear_layer:
            self.qkvw = self.transpose_impl(self.qkvw.data)
            self.dense_w = self.transpose_impl(self.dense_w.data)

    def transpose_mlp(self):
        if self.mlp_linear_layer:
            self._h4h_w = self.transpose_impl(self._h4h_w.data)
            self._4hh_w = self.transpose_impl(self._4hh_w.data)

    def transpose_impl(self, data):
        data = data.contiguous()
        data.reshape(-1).copy_(data.transpose(-1, -2).contiguous().reshape(-1))
        data = data.reshape(data.shape[-1], data.shape[-2])
        data.to(get_accelerator().current_device_name())
        return data

    def get_all_params(self):
        params = [
            self.attn_nw,
            self.attn_nb,
            self.input_nw,
            self.input_nb,
        ]

        params.extend(self.get_attn_params())
        params.extend(self.get_mlp_params())

        return params

    def get_attn_params(self):
        return [self.qkvw, self.qkvb, self.dense_w, self.dense_b]

    def get_mlp_params(self):
        return [self._h4h_w, self._h4h_b, self._4hh_w, self._4hh_b]
