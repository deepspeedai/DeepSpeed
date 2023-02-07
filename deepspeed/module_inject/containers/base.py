# Create a container object to save model-specific tensors using the policy file above.
from abc import ABC
import torch

from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig


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
        self.pre_layer_norm = self.policy.pre_attn_norm
        self.fp16 = False
        self.attn_linear_layer = self.policy.linear_layer
        self.mlp_linear_layer = self.policy.linear_layer
        self.layer_norm_eps = self.model_config.layer_norm_eps if \
            hasattr(self.model_config, 'layer_norm_eps') else (self.model_config.layer_norm_epsilon if \
            hasattr(self.model_config, 'layer_norm_epsilon') else self.model_config.layernorm_epsilon if \
            hasattr(self.model_config, 'layernorm_epsilon') else 1.0e-12)
        self.return_tuple = self.config.return_tuple
        self.triangular_masking = True
        self.local_attention = ((self.model_config.attention_layers[self.layer_id]
                                 == "local") if hasattr(self.model_config,
                                                        'attention_layers') else False)
        self.window_size = getattr(self.model_config, "window_size", 1)
        self.mlp_act_func_type = self.policy.mlp_act_func_type
        self.training_mp_size = self.config.training_mp_size
        self.bigscience_bloom = False
        self.max_out_tokens = self.config.max_out_tokens
        self.scale_attn_by_inverse_layer_idx = getattr(
            self.config,
            "scale_attn_by_inverse_layer_idx",
            False)
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

    def create_ds_model_config(self):
        self.set_hidden_heads(*self.policy.get_hidden_heads())
        assert self.num_attention_heads % self.mp_size == 0,\
                "To run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!" +\
                "This is because the attention computation is partitioned evenly among the parallel GPUs."

        self.ds_model_config = DeepSpeedInferenceConfig(
            hidden_size=self.hidden_size,
            heads=self.num_attention_heads,
            layer_norm_eps=self.layer_norm_eps,
            fp16=self.fp16,
            pre_layer_norm=self.pre_layer_norm,
            mp_size=self.mp_size,
            q_int8=self.quantize,
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
            scale_attn_by_inverse_layer_idx=self.scale_attn_by_inverse_layer_idx,
            use_mup=self.use_mup,
            return_single_tuple=self.return_single_tuple,
        )

        return self.ds_model_config

    def initialize_tensors(self):
        # Set the tensors from policy (user module) to container (DS module)
        self.set_attention(*self.policy.attention())
        self.set_mlp(*self.policy.mlp())
        self.set_layernorm(*self.policy.layernorm())

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

    def set_hidden_heads(self, hidden_size, num_attention_heads):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

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
        self.module.attention.attn_qkvw = self.quantizer.quantize(
            self.module.attention.attn_qkvw)
        self.module.attention.attn_ow = self.quantizer.quantize(
            self.module.attention.attn_ow)

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
        self.apply_weight_quantization()

    def attention_qkv_mp(self, mp_replace):
        self.module.attention.attn_qkvw = mp_replace.qkv_copy(
            self.module.attention.attn_qkvw,
            self.qkvw)
        self.module.attention.attn_qkvb = mp_replace.qkv_copy(
            self.module.attention.attn_qkvb,
            self.qkvb)

    def attention_o_mp(self, mp_replace):
        self.module.attention.attn_ow = mp_replace.copy(self.module.attention.attn_ow,
                                                        self.dense_w)
        self.module.attention.attn_ob = mp_replace.copy(self.module.attention.attn_ob,
                                                        self.dense_b)

    def mlp_inter_mp(self, mp_replace):
        self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w, self._h4h_w)
        self.module.mlp.inter_b = mp_replace.copy(self.module.mlp.inter_b, self._h4h_b)

    def mlp_output_mp(self, mp_replace):
        self.module.mlp.output_w = mp_replace.copy(self.module.mlp.output_w, self._4hh_w)
        self.module.mlp.output_b = mp_replace.copy(self.module.mlp.output_b, self._4hh_b)

    def copy_data_to_new_module(self):
        if self.attn_nw is None:
            self.module.mlp.attn_nw = self.attn_nw
            self.module.mlp.attn_nb = self.attn_nb
        else:
            self.module.mlp.attn_nw.data.copy_(
                self.attn_nw.to(torch.cuda.current_device()))
            self.module.mlp.attn_nb.data.copy_(
                self.attn_nb.to(torch.cuda.current_device()))

        self.module.norm_w.data.copy_(self.input_nw.to(torch.cuda.current_device()))
        self.module.norm_b.data.copy_(self.input_nb.to(torch.cuda.current_device()))

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
        data.to(torch.cuda.current_device())
        return data
