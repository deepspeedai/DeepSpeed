# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .base_moe import *
from .features.megatron import MegatronContainer
from deepspeed.model_implementations.transformers.ds_megatron_gpt import DeepSpeedMegatronGPTInference
import torch
from .megatron_gpt import MegatronLayerPolicy
from packaging import version as pkg_version


class DS_MegatronGPTMoEContainer(MegatronContainer, BaseTransformerMoEContainer):

    def __init__(self, policy, config, model_config, layer_id):
        super().__init__(policy, config, model_config, layer_id)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedMegatronGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention

        if self.megatron_v2:
            self.module.config.rotate_half = True
            self.module.config.rotate_every_two = False

        return self.module


# TODO: Megatron GPT MoE inherits from Megatron policy and replaces mlp
# TODO: Generalize MoE overall goal, expand beyond Megatron
class MegatronMoELayerPolicy(MegatronLayerPolicy):
    _orig_layer_class = None
    version = 0
    moe_type = 'standard'
    num_experts = 1

    def __init__(self, client_module, inference=True):
        super().__init__(inference)
        self.client_module = client_module
        # we use megatron version to differentiate between the old and new
        # megatron-lm source code
        if MegatronMoELayerPolicy._orig_layer_class is None:
            if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
                MegatronMoELayerPolicy._orig_layer_class = None
            else:
                try:
                    from megatron.model.transformer import ParallelTransformerLayer
                    MegatronMoELayerPolicy._orig_layer_class = ParallelTransformerLayer
                except ImportError:
                    MegatronMoELayerPolicy._orig_layer_class = None

    def get_num_experts(self):
        return self.num_experts

    def mlp(self, moe_type='standard', enable_training=False):
        # for now, all of this is tightly coupled to megatron-deepspeed moe implementation
        # todo: think and refactor this to be more general

        #from deepspeed.moe.utils import has_moe_layers
        #moe, _ = has_moe_layers(self.client_module)

        moe_experts = self.client_module.mlp.deepspeed_moe.experts.deepspeed_experts if moe_type == 'standard' else \
                        self.client_module.mlp.moe.deepspeed_moe.experts.deepspeed_experts
        num_experts = len(moe_experts)
        self.num_experts = num_experts

        if moe_type == 'standard':
            return [moe_experts[i].dense_h_to_4h.weight for i in range(num_experts)], \
                    [moe_experts[i].dense_h_to_4h.bias for i in range(num_experts)], \
                    [moe_experts[i].dense_4h_to_h.weight for i in range(num_experts)], \
                    [moe_experts[i].dense_4h_to_h.bias for i in range(num_experts)]
        else:
            return [moe_experts[i].dense_h_to_4h.weight for i in range(num_experts)], \
                    [moe_experts[i].dense_h_to_4h.bias for i in range(num_experts)], \
                    [moe_experts[i].dense_4h_to_h.weight for i in range(num_experts)], \
                    [moe_experts[i].dense_4h_to_h.bias for i in range(num_experts)], \
                    self.client_module.mlp.mlp.dense_h_to_4h.weight, \
                    self.client_module.mlp.mlp.dense_h_to_4h.bias, \
                    self.client_module.mlp.mlp.dense_4h_to_h.weight, \
                    self.client_module.mlp.mlp.dense_4h_to_h.bias, \
                    self.client_module.mlp.coefficient.weight
