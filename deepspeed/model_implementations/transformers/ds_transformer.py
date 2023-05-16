# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn as nn
from deepspeed import comm as dist
from deepspeed.utils.logging import log_dist

from deepspeed.ops.transformer.inference.ds_mlp import DeepSpeedMLP
from deepspeed.ops.transformer.inference.ds_attention import DeepSpeedSelfAttention, BloomSelfAttention
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder

inference_module = None


class DeepSpeedTransformerInference(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
            mp_group: Model parallelism group initialized on the modeling side.
            quantize_scales: This argument groups all the layers' scales used for quantization
            quantize_groups: Number of groups used for quantizing the model
            merge_count: Shows the number of model-parallel checkpoints merged before running inference.
                We use this argument to control the quantization scale for the model parameters if a bigger
                quantize-grouping than 1 is used.
            mlp_extra_grouping: This flag is used to show a 2x higher number of groups used for the MLP part
                of a Transformer layer. We use this feature for quantization to reduce the convergence impact
                for specific downstream tasks.
    """
    layer_id = 0

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        super(DeepSpeedTransformerInference, self).__init__()

        self.config = config
        self.config.layer_id = DeepSpeedTransformerInference.layer_id
        DeepSpeedTransformerInference.layer_id += 1

        data_type = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        global inference_module
        if inference_module is None:
            builder = InferenceBuilder()
            inference_module = builder.load()

        if DeepSpeedTransformerInference.layer_id == 1:
            log_dist(f"DeepSpeed-Inference config: {self.config.__dict__}", [0])

        if self.config.bigscience_bloom:
            self.attention = BloomSelfAttention(self.config, mp_group, quantize_scales, quantize_groups, merge_count)
        else:
            self.attention = DeepSpeedSelfAttention(self.config, mp_group, quantize_scales, quantize_groups,
                                                    merge_count)
        self.mlp = DeepSpeedMLP(self.config, mp_group, quantize_scales, quantize_groups, merge_count,
                                mlp_extra_grouping)

        device = get_accelerator().current_device_name()  # if config.bigscience_bloom else 'cpu'
        if self.config.set_empty_params:
            self.norm_w = None
            self.norm_b = None
        else:
            self.norm_w = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type, device=device),
                                       requires_grad=False)
            self.norm_b = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type, device=device),
                                       requires_grad=False)
        self.layer_past = None
        try:
            if config.dtype == torch.float32:
                self.allocate_workspace = inference_module.allocate_workspace_fp32
            elif config.dtype == torch.bfloat16:
                self.allocate_workspace = inference_module.allocate_workspace_bf16
            else:
                self.allocate_workspace = inference_module.allocate_workspace_fp32
            self._alloc_workspace = True
        except AttributeError:
            self.allocate_workspace = None
            self._alloc_workspace = False

    @classmethod
    def reset_cache(cls):
        if inference_module is not None:
            inference_module.reset_cache()

    def forward(
            self,
            input=None,
            input_mask=None,
            attention_mask=None,
            attn_mask=None,
            head_mask=None,
            layer_past=None,
            get_key_value=False,
            get_present=False,
            encoder_output=None,
            enc_dec_attn_mask=None,
            x=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            alibi=None,
            output_attentions=False,
            # TODO(arashb): 'layer_head_mask' and 'past_key_value' are only added to satisfy the OPT models API.
            # This needs to be redesigned later!
            layer_head_mask=None,
            past_key_value=None,
            **kwargs):

        if x is not None:
            input = x
        if "hidden_states" in kwargs:
            input = kwargs["hidden_states"]

        input_mask = (input_mask if attn_mask is None else attn_mask) if attention_mask is None else attention_mask

        # Allocate memory only on first layer forward
        if self.config.layer_id == 0 and self._alloc_workspace:
            self.allocate_workspace(self.config.hidden_size, self.config.heads,
                                    input.size()[1],
                                    input.size()[0], DeepSpeedTransformerInference.layer_id, self.config.mp_size,
                                    self.config.bigscience_bloom,
                                    dist.get_rank() if dist.is_initialized() else 0, self.config.max_out_tokens,
                                    self.config.min_out_tokens)
            self._alloc_workspace = False

        get_present = (get_present or get_key_value or use_cache)
        input_mask = input_mask if attention_mask is None else attention_mask

        # We set the prev key/value to None when there is a prompt
        if input.shape[1] > 1:
            self.layer_past = None
        layer_past = layer_past if layer_past is not None else self.layer_past
        head_mask = layer_head_mask if layer_head_mask is not None else head_mask

        attn_mask = None
        if isinstance(input, tuple):
            attn_mask = input[1]
            input = input[0]
        input_type = input.dtype

        if (self.config.dtype in [torch.float16, torch.bfloat16, torch.int8]) \
            and input.dtype == torch.float:
            target_dtype = torch.half if self.dtype == torch.int8 else self.dtype
            input = input.to(target_dtype)

        with torch.no_grad():
            attention_output, key, value, context_outputtn_ctx, inp_norm = \
                                     self.attention(input,
                                              input_mask,
                                              head_mask,
                                              layer_past,
                                              get_present,
                                              encoder_hidden_states,
                                              encoder_attention_mask,
                                              output_attentions,
                                              self.norm_w,
                                              self.norm_b,
                                              alibi)

            presents = (key, value)
            self.layer_past = presents if layer_past is None else None
            output = self.mlp(attention_output, input, inp_norm, self.attention.attn_ob)

            if not self.config.pre_layer_norm:
                output = inference_module.layer_norm(output, self.norm_w, self.norm_b, self.config.epsilon)

            output = output.to(input_type)
        if get_present:
            output = (output, presents)

        if self.config.return_single_tuple:
            return (output, )
        elif self.config.return_tuple:
            return output if type(output) is tuple else (output, attn_mask)
        else:
            return output
