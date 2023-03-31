# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.
from .base import *
from deepspeed import comm as dist
import deepspeed.ops.transformer as transformer_inference
from deepspeed.accelerator import get_accelerator


class BaseTransformerMoEContainer(BaseTransformerContainer):

    def __init__(self, **kwargs):
        # Call the init function of the parent class to initialize the tensors and configs from parent class
        super().__init__(**kwargs)

        self.num_experts = self.policy.get_num_experts()
        self.ep_world_size = dist.get_world_size()
        self.local_ep_size = 1 if self.num_experts < self.ep_world_size else self.num_experts // self.ep_world_size

        self.layer_norm_eps = self.config.layer_norm_eps if hasattr(self.config, 'layer_norm_eps') else 1e-12,

        # MoE models will have a list of mlp related tensors
        self._h4h_w = []
        self._h4h_b = []
        self._4hh_w = []
        self._4hh_b = []

        # Residual MoE needs extra parameters
        self._res_h4h_w = None
        self._res_h4h_b = None
        self._res_4hh_w = None
        self._res_4hh_b = None
        self._res_coef = None

    def create_ds_model_config(self):
        self.set_hidden_heads(*self.policy.get_hidden_heads())
        assert self.num_attention_heads % self.mp_size == 0,\
                "To run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!" +\
                "This is because the attention computation is partitioned evenly among the parallel GPUs."

        self.ds_model_config = transformer_inference.DeepSpeedMoEInferenceConfig(
            hidden_size=self.hidden_size,
            heads=self.num_attention_heads,
            layer_norm_eps=self.layer_norm_eps,
            fp16=self.fp16,
            pre_layer_norm=self.pre_layer_norm,
            mp_size=self.mp_size,
            q_int8=self.quantize,
            moe_experts=self.local_ep_size,
            global_experts=self.num_experts,
            mlp_type=self.config.moe.type,
            scale_attn_by_inverse_layer_idx=self.scale_attn_by_inverse_layer_idx,
        )

        return self.ds_model_config

    def initialize_tensors(self):
        # Set the tensors from policy (user module) to container (DS module)
        self.set_attention(*self.policy.attention())
        self.set_mlp(self.config.moe.type)
        self.set_layernorm(*self.policy.layernorm())

    def set_mlp(self, config_moe_type):
        if config_moe_type == 'standard':
            self._h4h_w, self._h4h_b, \
            self._4hh_w, self._4hh_b = self.policy.mlp()
        else:
            self._h4h_w, self._h4h_b, self._4hh_w, \
            self._4hh_b, self._res_h4h_w, self._res_h4h_b, \
            self._res_4hh_w, self._res_4hh_b, \
            self._res_coef = self.policy.mlp(config_moe_type)

    def transpose(self):
        self.transpose_attention()
        self.transpose_mlp()

        if self.config.moe.type == 'residual':
            self.transpose_residual()

    def transpose_mlp(self):
        self._h4h_w = [self.transpose_impl(moe_w1.data) for moe_w1 in self._h4h_w]
        self._4hh_w = [self.transpose_impl(moe_w1.data) for moe_w1 in self._4hh_w]

    def transpose_residual(self):
        self._res_h4h_w.data = self.transpose_impl(self._res_h4h_w.data)
        self._res_4hh_w.data = self.transpose_impl(self._res_4hh_w.data)
        self._res_coef.data = self.transpose_impl(self._res_coef.data)

    def apply_tensor_parallelism(self, mp_replace):
        # setup the new Attention module
        self.attention_qkv_mp(mp_replace)
        self.attention_o_mp(mp_replace)

        # quantize attention weights
        self.attention_quantization()

        # setup the new MLP module
        self.mlp_mp()

    def mlp_mp(self):
        gpu_index = dist.get_rank()
        for ep_index in range(self.local_ep_size):
            # mlp inter
            self.module.mlp[ep_index].inter_w.data = self._h4h_w[gpu_index * self.local_ep_size + ep_index].to(
                get_accelerator().current_device_name())
            self.module.mlp[ep_index].inter_b.data = self._h4h_b[gpu_index * self.local_ep_size + ep_index].to(
                get_accelerator().current_device_name())

            # mlp output
            self.module.mlp[ep_index].output_w.data = self._4hh_w[gpu_index * self.local_ep_size + ep_index].to(
                get_accelerator().current_device_name())
            self.module.mlp[ep_index].output_b.data = self._4hh_b[gpu_index * self.local_ep_size + ep_index].to(
                get_accelerator().current_device_name())

    def copy_data_to_new_module(self):
        self.module.attn_nw.data = self.attn_nw.to(get_accelerator().current_device_name())
        self.module.attn_nb.data = self.attn_nb.to(get_accelerator().current_device_name())

        self.module.norm_w.data.copy_(self.input_nw.to(get_accelerator().current_device_name()))
        self.module.norm_b.data.copy_(self.input_nb.to(get_accelerator().current_device_name()))

        if self.config.moe.type == 'residual':
            self.module.res_mlp.inter_w.data = self._res_h4h_w.to(get_accelerator().current_device_name())
            self.module.res_mlp.inter_b.data = self._res_h4h_b.to(get_accelerator().current_device_name())
            self.module.res_mlp.output_w.data = self._res_4hh_w.to(get_accelerator().current_device_name())
            self.module.res_mlp.output_b.data = self._res_4hh_b.to(get_accelerator().current_device_name())
            self.module.res_coef.data = self._res_coef.to(get_accelerator().current_device_name())
