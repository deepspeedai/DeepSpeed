# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.
from abc import ABC
import torch

from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from deepspeed.accelerator import get_accelerator


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
        self.num_attention_heads = None
        self.mp_size = self.config.tensor_parallel.tp_size
        self.pre_layer_norm = self.model_config.do_layer_norm_before if \
            hasattr(self.model_config, 'do_layer_norm_before') else self.policy.pre_attn_norm
        self.fp16 = False
        self.attn_linear_layer = self.policy.linear_layer
        self.mlp_linear_layer = self.policy.linear_layer
        self.return_tuple = self.config.return_tuple
        self.triangular_masking = True
        self.local_attention = ((self.model_config.attention_layers[self.layer_id] == "local") if hasattr(
            self.model_config, 'attention_layers') else False)
        self.window_size = getattr(self.model_config, "window_size", 1)
        self.mlp_act_func_type = self.policy.mlp_act_func_type
        self.training_mp_size = self.config.training_mp_size
        self.bigscience_bloom = False
        self.max_out_tokens = self.config.max_out_tokens
        self.min_out_tokens = self.config.min_out_tokens
        self.scale_attn_by_inverse_layer_idx = getattr(self.config, "scale_attn_by_inverse_layer_idx", False)
        self.use_mup = self.policy.use_mup
        self.return_single_tuple = False
        self.rotary_dim = self.model_config.rotary_dim if hasattr(self.model_config, 'rotary_dim') \
                          else self.child.attention.rotary_ndims if \
                          hasattr(self.child, 'attention') and hasattr(self.child.attention,'rotary_ndims') else -1
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

    def create_ds_model_config(self):
        self.set_hidden_heads(*self.policy.get_hidden_heads())
        assert self.num_attention_heads % self.mp_size == 0,\
                "To run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!" +\
                "This is because the attention computation is partitioned evenly among the parallel GPUs."

        self.ds_model_config = DeepSpeedInferenceConfig(
            hidden_size=self.hidden_size,
            heads=self.num_attention_heads,
            layer_norm_eps=self.layernorm_epsilon,
            fp16=self.fp16,
            pre_layer_norm=self.pre_layer_norm,
            mp_size=self.mp_size,
            q_int8=self.quantize if hasattr(self, 'quantize') else False,
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
            transposed_mode=self.config.transposed_mode)

        return self.ds_model_config

    def initialize_tensors(self, enable_training=False):
        # Set the tensors from policy (user module) to container (DS module)
        self.set_attention(*self.policy.attention(enable_training=enable_training))
        self.set_mlp(*self.policy.mlp())
        self.set_layernorm(*self.policy.layernorm())
        self.set_lora_params(self.policy.get_lora_params())
        self.q_k_v = self.policy.get_q_k_v()
        if self.q_k_v is not None:
            self.set_q_k_v(*self.q_k_v)

    def convert_to_required_dtype(self, dtype):
        # Note: converting tensors to fp16 requires that we do it in-place using self.__dict__ and not make a list/dict copy
        if dtype == torch.half:
            for k, v in self.__dict__.items():
                # The list comprehension is used for MoE tensor lists
                if isinstance(v, list) and all((isinstance(tensor, torch.Tensor) \
                   or isinstance(tensor, torch.nn.Parameter)) for tensor in v):
                    self.__dict__[k] = [moe_tensor.half() for moe_tensor in v]

                if isinstance(v, torch.Tensor) or isinstance(v, torch.nn.Parameter):
                    self.__dict__[k] = v.half()

    def set_dtype(self, fp16=False):
        self.fp16 = fp16

    def set_moe(self, moe=False):
        self.moe = moe

    def set_tensor_parallel_config(self, mp_size, mp_group):
        self.mp_size = mp_size
        self.mp_group = mp_group

    def set_quantization_config(self, quantize, quantizer):
        self.quantize = quantize
        self.quantizer = quantizer

    def set_hidden_heads(self, hidden_size, num_attention_heads, epsilon):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.layernorm_epsilon = epsilon

    def set_attention(self, qkvw, qkvb, dense_w, dense_b):
        self.qkvw = qkvw
        self.qkvb = qkvb
        self.dense_w = dense_w
        self.dense_b = dense_b

    def set_lora_params(self, lora_params):
        self.lora_params = lora_params

    def set_q_k_v(self, qw, qb, kw, kb, vw, vb):
        self.qw = qw
        self.qb = qb
        self.kw = kw
        self.kb = kb
        self.vw = vw
        self.vb = vb

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

    def apply_tensor_parallelism(self, mp_replace=None, mp_group=None, tp_size=None):
        reversed_dim = False
        if mp_replace is None:
            from deepspeed.module_inject import ReplaceWithTensorSlicing
            mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group, mp_size=tp_size, out_dim=0, in_dim=1)
            reversed_dim = True
        # setup the new Attention module
        if self.module.attention.attn_qkvw is None:
            self.attention_q_k_v_mp(mp_replace, reversed_dim=reversed_dim)
        else:
            self.attention_qkv_mp(mp_replace, reversed_dim=reversed_dim)
        self.attention_o_mp(mp_replace, reversed_dim=reversed_dim)

        # setup the new MLP module
        self.mlp_inter_mp(mp_replace, reversed_dim=reversed_dim)
        self.mlp_output_mp(mp_replace, reversed_dim=reversed_dim)

        # Apply weight quantization
        #self.apply_weight_quantization()

    def attention_qkv_mp(self, mp_replace, reversed_dim=False):
        if reversed_dim:
            self.module.attention.attn_qkvw = mp_replace.qkv_copy(
                self.module.attention.attn_qkvw[:self.qkvw.shape[0] // mp_replace.mp_size],
                self.qkvw,
                int8=reversed_dim)
            self.module.attention.attn_qkvb = mp_replace.qkv_copy(
                self.module.attention.attn_qkvb[:self.qkvw.shape[0] // mp_replace.mp_size],
                self.qkvb,
                int8=reversed_dim)
        else:
            self.module.attention.attn_qkvw = mp_replace.qkv_copy(self.module.attention.attn_qkvw,
                                                                  self.qkvw,
                                                                  int8=reversed_dim)
            self.module.attention.attn_qkvb = mp_replace.qkv_copy(self.module.attention.attn_qkvb,
                                                                  self.qkvb,
                                                                  int8=reversed_dim)

    def attention_q_k_v_mp(self, mp_replace, reversed_dim=False):
        self.module.attention.attn_qw = mp_replace.copy(self.module.attention.attn_qw[:self.qw.shape[0] //
                                                                                      mp_replace.mp_size],
                                                        self.qw,
                                                        int8=reversed_dim,
                                                        allocat_tensor=reversed_dim)
        self.module.attention.attn_kw = mp_replace.copy(self.module.attention.attn_kw[:self.qw.shape[0] //
                                                                                      mp_replace.mp_size],
                                                        self.kw,
                                                        int8=reversed_dim,
                                                        allocat_tensor=reversed_dim)
        self.module.attention.attn_vw = mp_replace.copy(self.module.attention.attn_vw[:self.qw.shape[0] //
                                                                                      mp_replace.mp_size],
                                                        self.vw,
                                                        int8=reversed_dim,
                                                        allocat_tensor=reversed_dim)
        self.module.attention.attn_qb = mp_replace.copy(
            self.module.attention.attn_qb[:self.qw.shape[0] // mp_replace.mp_size],
            self.qb,
            int8=reversed_dim,
            allocat_tensor=reversed_dim) if self.module.attention.attn_qb is not None else None
        self.module.attention.attn_kb = mp_replace.copy(
            self.module.attention.attn_kb[:self.qw.shape[0] // mp_replace.mp_size],
            self.kb,
            int8=reversed_dim,
            allocat_tensor=reversed_dim) if self.module.attention.attn_kb is not None else None
        self.module.attention.attn_vb = mp_replace.copy(
            self.module.attention.attn_vb[:self.qw.shape[0] // mp_replace.mp_size],
            self.vb,
            int8=reversed_dim,
            allocat_tensor=reversed_dim) if self.module.attention.attn_vb is not None else None

    def attention_o_mp(self, mp_replace, reversed_dim=False):
        if reversed_dim:
            self.module.attention.attn_ow = mp_replace.copy(self.module.attention.attn_ow[:, :self.dense_w.shape[1] //
                                                                                          mp_replace.mp_size],
                                                            self.dense_w,
                                                            int8=reversed_dim,
                                                            allocat_tensor=reversed_dim)
        else:
            self.module.attention.attn_ow = mp_replace.copy(self.module.attention.attn_ow,
                                                            self.dense_w,
                                                            int8=reversed_dim)
        self.module.attention.attn_ob = mp_replace.copy(self.module.attention.attn_ob,
                                                        self.dense_b,
                                                        int8=reversed_dim,
                                                        allocat_tensor=reversed_dim)

    def mlp_inter_mp(self, mp_replace, reversed_dim=False):
        if reversed_dim:
            self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w[:self._h4h_w.shape[0] //
                                                                              mp_replace.mp_size],
                                                      self._h4h_w,
                                                      int8=reversed_dim,
                                                      allocat_tensor=reversed_dim)
            self.module.mlp.inter_b = mp_replace.copy(
                self.module.mlp.inter_b[:self._h4h_w.shape[0] // mp_replace.mp_size],
                self._h4h_b,
                int8=reversed_dim,
                allocat_tensor=reversed_dim) if self.module.mlp.inter_b is not None else None
        else:
            self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w, self._h4h_w, int8=reversed_dim)
            self.module.mlp.inter_b = mp_replace.copy(self.module.mlp.inter_b, self._h4h_b, int8=reversed_dim)

    def mlp_output_mp(self, mp_replace, reversed_dim=False):
        if reversed_dim:
            self.module.mlp.output_w = mp_replace.copy(self.module.mlp.output_w[:, :self._4hh_w.shape[1] //
                                                                                mp_replace.mp_size],
                                                       self._4hh_w,
                                                       int8=reversed_dim,
                                                       allocat_tensor=reversed_dim)
        else:
            self.module.mlp.output_w = mp_replace.copy(self.module.mlp.output_w, self._4hh_w, int8=reversed_dim)
        self.module.mlp.output_b = mp_replace.copy(self.module.mlp.output_b,
                                                   self._4hh_b,
                                                   int8=reversed_dim,
                                                   allocat_tensor=reversed_dim)

    def release_qkv(self):
        del self.module.attention.attn_qkvw
        del self.module.attention.attn_qkvb
        self.module.attention.attn_qkvw = self.qkvw
        self.module.attention.attn_qkvb = self.qkvb
        if self.module.attention.attn_qw is not None:
            qkv_data = [self.module.attention.attn_qw.data, \
                        self.module.attention.attn_qb.data if self.module.attention.attn_qb is not None else None, \
                        self.module.attention.attn_kw.data, \
                        self.module.attention.attn_kb.data if self.module.attention.attn_kb is not None else None, \
                        self.module.attention.attn_vw.data, \
                        self.module.attention.attn_vb.data if self.module.attention.attn_vb is not None else None]
            for data in qkv_data:
                del data

            self.module.attention.attn_qw = self.qw
            self.module.attention.attn_qb = self.qb
            self.module.attention.attn_kw = self.kw
            self.module.attention.attn_kb = self.kb
            self.module.attention.attn_vw = self.vw
            self.module.attention.attn_vb = self.vb

    def release_memory(self):
        self.release_qkv()
        del self.module.attention.attn_ow
        del self.module.attention.attn_ob
        self.module.attention.attn_ow = self.dense_w
        self.module.attention.attn_ob = self.dense_b
        del self.module.mlp.inter_w
        del self.module.mlp.inter_b
        del self.module.mlp.output_w
        del self.module.mlp.output_b
        self.module.mlp.inter_w = self._h4h_w
        self.module.mlp.inter_b = self._h4h_b
        self.module.mlp.output_w = self._4hh_w
        self.module.mlp.output_b = self._4hh_b

    def copy_data_to_new_module(self):
        if self.attn_nw is None:
            self.module.mlp.attn_nw = self.attn_nw
            self.module.mlp.attn_nb = self.attn_nb
        else:
            self.module.mlp.attn_nw.data.copy_(self.attn_nw.to(get_accelerator().current_device_name()))
            self.module.mlp.attn_nb.data.copy_(self.attn_nb.to(get_accelerator().current_device_name()))

        self.module.norm_w.data.copy_(self.input_nw.to(get_accelerator().current_device_name()))
        self.module.norm_b.data.copy_(self.input_nb.to(get_accelerator().current_device_name()))

    def align_merged_qkv(self):
        if hasattr(self, '_align_merged_qkv'):
            self._align_merged_qkv()

    def partition_merged_qkv(self):
        if hasattr(self, '_partition_merged_qkv'):
            self._partition_merged_qkv()

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

    def reset_qkv_experimental(self):
        if self.module.attention.attn_qkvw is None:
            self.module.attention.attn_qkvw = torch.empty(self.qw.shape[0] * 3,
                                                          self.qw.shape[0],
                                                          dtype=self.qw.dtype,
                                                          device=self.qw.device)
            self.module.attention.attn_qkvb = torch.empty(self.qw.shape[0] * 3,
                                                          dtype=self.qw.dtype,
                                                          device=self.qw.device)
        self.module.attention.attn_qkvw.data[:self.qw.shape[0]] = self.qw.data
        self.module.attention.attn_qkvb.data[:self.qw.shape[0]] = self.qb.data
        self.module.attention.attn_qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kw.data
        self.module.attention.attn_qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kb.data
        self.module.attention.attn_qkvw.data[2 * self.qw.shape[0]:] = self.vw.data
        self.module.attention.attn_qkvb.data[2 * self.qw.shape[0]:] = self.vb.data

        qkv_data = [self.qw.data, \
                    self.qb.data, \
                    self.kw.data, \
                    self.kb.data, \
                    self.vw.data, \
                    self.vb.data]

        self.qw.data = self.module.attention.attn_qkvw.data[:self.qw.shape[0]]
        self.qb.data = self.module.attention.attn_qkvb.data[:self.qw.shape[0]]
        self.kw.data = self.module.attention.attn_qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]]
        self.kb.data = self.module.attention.attn_qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]]
        self.vw.data = self.module.attention.attn_qkvw.data[2 * self.qw.shape[0]:]
        self.vb.data = self.module.attention.attn_qkvb.data[2 * self.qw.shape[0]:]

        for data in qkv_data:
            del data

    def reset_qkv(self):
        self.qkvw.data[:self.qw.shape[0]] = self.qw.data
        self.qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kw.data
        self.qkvw.data[2 * self.qw.shape[0]:] = self.vw.data
        if self.qkvb is not None:
            self.qkvb.data[:self.qw.shape[0]] = self.qb.data
            self.qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kb.data
            self.qkvb.data[2 * self.qw.shape[0]:] = self.vb.data

        qkv_data = [self.qw.data, \
                    self.qb.data if self.qb is not None else None, \
                    self.kw.data, \
                    self.kb.data if self.kb is not None else None, \
                    self.vw.data, \
                    self.vb.data if self.vb is not None else None]

        self.qw.data = self.qkvw.data[:self.qw.shape[0]]
        self.kw.data = self.qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]]
        self.vw.data = self.qkvw.data[2 * self.qw.shape[0]:]

        if self.qkvb is not None:
            self.qb.data = self.qkvb.data[:self.qw.shape[0]]
            self.kb.data = self.qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]]
            self.vb.data = self.qkvb.data[2 * self.qw.shape[0]:]

        for data in qkv_data:
            del data

    def set_params_wo_copy(self, Z3_enabled=False):
        self.module.mlp.attn_nw = self.attn_nw
        self.module.mlp.attn_nb = self.attn_nb
        self.module.norm_w = self.input_nw
        self.module.norm_b = self.input_nb
        self.module.mlp.inter_w = self._h4h_w
        self.module.mlp.inter_b = self._h4h_b
        self.module.mlp.output_w = self._4hh_w
        self.module.mlp.output_b = self._4hh_b
        self.module.attention.attn_ow = self.dense_w
        self.module.attention.attn_ob = self.dense_b
        if not Z3_enabled or self.q_k_v is None:
            self.module.attention.attn_qkvw = self.qkvw
            self.module.attention.attn_qkvb = self.qkvb
        if self.q_k_v is not None:
            if Z3_enabled:
                self.module.attention.attn_qw = self.qw
                self.module.attention.attn_qb = self.qb
                self.module.attention.attn_kw = self.kw
                self.module.attention.attn_kb = self.kb
                self.module.attention.attn_vw = self.vw
                self.module.attention.attn_vb = self.vb
            else:
                self.qw.data = self.qkvw[:self.qw.shape[0], :]
                self.kw.data = self.qkvw[self.qw.shape[0]:2 * self.qw.shape[0], :]
                self.vw.data = self.qkvw[self.qw.shape[0] * 2:, :]
                if self.qkvb is not None:
                    self.qb.data = self.qkvb[:self.qw.shape[0]]
                    self.kb.data = self.qkvb[self.qw.shape[0]:2 * self.qw.shape[0]]
                    self.vb.data = self.qkvb[self.qw.shape[0] * 2:]

    def get_lora_params(self):
        return self.lora_params

    def get_all_params(self):
        if self.q_k_v is not None:
            return [
                self.attn_nw, self.attn_nb, self.input_nw, self.input_nb, self._h4h_w, self._h4h_b, self._4hh_w,
                self._4hh_b, self.qw, self.qb, self.kw, self.kb, self.vw, self.vb, self.dense_w, self.dense_b
            ]
        else:
            return [
                self.attn_nw, self.attn_nb, self.input_nw, self.input_nb, self._h4h_w, self._h4h_b, self._4hh_w,
                self._4hh_b, self.qkvw, self.qkvb, self.dense_w, self.dense_b
            ]
