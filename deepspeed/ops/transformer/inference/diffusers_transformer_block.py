# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn as nn

from deepspeed import module_inject
from .diffusers_attention import DeepSpeedDiffusersAttention
from .bias_add import nhwc_bias_add
from .diffusers_2d_transformer import Diffusers2DTransformerConfig
from deepspeed.ops.op_builder import InferenceBuilder, SpatialInferenceBuilder
from deepspeed.utils.types import ActivationFuncType

# Ops will be loaded on demand
transformer_cuda_module = None
spatial_cuda_module = None


def load_transformer_module():
    global transformer_cuda_module
    if transformer_cuda_module is None:
        transformer_cuda_module = InferenceBuilder().load()
    return transformer_cuda_module


def load_spatial_module():
    global spatial_cuda_module
    if spatial_cuda_module is None:
        spatial_cuda_module = SpatialInferenceBuilder().load()
    return spatial_cuda_module


class DeepSpeedDiffusersTransformerBlock(nn.Module):

    def __init__(self, equivalent_module: nn.Module, config: Diffusers2DTransformerConfig):
        super(DeepSpeedDiffusersTransformerBlock, self).__init__()
        self.quantizer = module_inject.GroupQuantizer(q_int8=config.int8_quantization)
        # Ensure ops are built by the time we start running
        self.config = config

        self.ff1_w = self.quantizer.quantize(
            nn.Parameter(equivalent_module.ff.net[0].proj.weight.data, requires_grad=False))
        self.ff1_b = nn.Parameter(equivalent_module.ff.net[0].proj.bias.data, requires_grad=False)
        self.ff2_w = self.quantizer.quantize(nn.Parameter(equivalent_module.ff.net[2].weight.data,
                                                          requires_grad=False))
        self.ff2_b = nn.Parameter(equivalent_module.ff.net[2].bias.data, requires_grad=False)

        self.norm1_g = nn.Parameter(equivalent_module.norm1.weight.data, requires_grad=False)
        self.norm1_b = nn.Parameter(equivalent_module.norm1.bias.data, requires_grad=False)
        self.norm1_eps = equivalent_module.norm1.eps

        self.norm2_g = nn.Parameter(equivalent_module.norm2.weight.data, requires_grad=False)
        self.norm2_b = nn.Parameter(equivalent_module.norm2.bias.data, requires_grad=False)
        self.norm2_eps = equivalent_module.norm2.eps

        self.norm3_g = nn.Parameter(equivalent_module.norm3.weight.data, requires_grad=False)
        self.norm3_b = nn.Parameter(equivalent_module.norm3.bias.data, requires_grad=False)
        self.norm3_eps = equivalent_module.norm3.eps

        self.attn_1 = equivalent_module.attn1
        self.attn_2 = equivalent_module.attn2

        # Pull the bias in if we can
        if isinstance(self.attn_1, DeepSpeedDiffusersAttention):
            self.attn_1.do_out_bias = False
            self.attn_1_bias = self.attn_1.attn_ob
        else:
            self.attn_1_bias = nn.Parameter(torch.zeros_like(self.norm2_g), requires_grad=False)

        # Pull the bias in if we can
        if isinstance(self.attn_2, DeepSpeedDiffusersAttention):
            self.attn_2.do_out_bias = False
            self.attn_2_bias = self.attn_2.attn_ob
        else:
            self.attn_2_bias = nn.Paramaeter(torch.zeros_like(self.norm3_g), requires_grad=False)

        self.transformer_cuda_module = load_transformer_module()
        load_spatial_module()

    def forward(self, hidden_states, context=None, timestep=None, **kwargs):
        # In v0.12.0 of diffuser, several new kwargs were added. Capturing
        # those with kwargs to maintain backward compatibility

        # In v0.11.0 of diffusers, the kwarg was changed from 'context' to 'encoder_hidden_states'
        # This is so we can support older and newer versions of diffusers
        if "encoder_hidden_states" in kwargs and kwargs["encoder_hidden_states"] is not None:
            context = kwargs["encoder_hidden_states"]

        out_norm_1 = self.transformer_cuda_module.layer_norm(hidden_states, self.norm1_g, self.norm1_b, self.norm1_eps)
        out_attn_1 = self.attn_1(out_norm_1)

        out_norm_2, out_attn_1 = self.transformer_cuda_module.layer_norm_residual_store_pre_ln_res(
            out_attn_1, self.attn_1_bias, hidden_states, self.norm2_g, self.norm2_b, self.norm2_eps)
        out_attn_2 = self.attn_2(out_norm_2, context=context)
        out_norm_3, out_attn_2 = self.transformer_cuda_module.layer_norm_residual_store_pre_ln_res(
            out_attn_2, self.attn_2_bias, out_attn_1, self.norm3_g, self.norm3_b, self.norm3_eps)

        out_ff1 = nn.functional.linear(out_norm_3, self.ff1_w)
        out_geglu = self.transformer_cuda_module.gated_activation(out_ff1, self.ff1_b, ActivationFuncType.GATED_GELU)

        out_ff2 = nn.functional.linear(out_geglu, self.ff2_w)
        return nhwc_bias_add(out_ff2, self.ff2_b, other=out_attn_2)
