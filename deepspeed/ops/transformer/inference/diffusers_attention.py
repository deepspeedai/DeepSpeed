# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
from torch.autograd import Function
import torch.nn as nn
from packaging import version as pkg_version
from deepspeed.utils.logging import log_dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder

# Cuda modules will be imported if needed
inference_module = None
minus_inf = -10000.0
triton_flash_attn = None


def load_triton_flash_attn():
    global triton_flash_attn
    try:
        import triton
    except ImportError:
        raise ImportError("Please install triton 2.0+ or `pip install deepspeed[sd]`")

    if pkg_version.parse(triton.__version__) < pkg_version.parse("2.0"):
        raise ImportError("Please install triton 2.0+ or `pip install deepspeed[sd]`")

    from .triton_ops import triton_flash_attn


class DeepSpeedDiffusersAttentionFunction(Function):

    @staticmethod
    def forward(ctx, input, context, input_mask, config, attn_qkvw, attn_qw, attn_kw, attn_vw, attn_qkvb,
                num_attention_heads_per_partition, norm_factor, hidden_size_per_partition, attn_ow, attn_ob,
                do_out_bias, score_context_func, linear_func, triton_flash_attn_kernel):

        def _transpose_for_context(x):
            x = x.permute(0, 2, 1, 3)
            new_x_layer_shape = x.size()[:-2] + \
                                      (hidden_size_per_partition,)
            return x.reshape(*new_x_layer_shape)

        def _transpose_for_scores(x):
            attention_head_size = x.shape[-1] // num_attention_heads_per_partition
            new_x_shape = x.size()[:-1] + (num_attention_heads_per_partition, attention_head_size)
            x = x.reshape(*new_x_shape)
            x = x.permute(0, 2, 1, 3)
            return x.contiguous()

        def selfAttention_fp(input, context, input_mask):
            if config.fp16 and input.dtype == torch.float32:
                input = input.half()
            head_size = input.shape[-1] // config.heads
            do_flash_attn = (head_size <= 128)
            scale = (1 / norm_factor) * (1 / norm_factor)
            if do_flash_attn and context == None:
                qkv_out = linear_func(input, attn_qkvw, attn_qkvb if attn_qkvb is not None else attn_qkvw, attn_qkvb
                                      is not None, do_flash_attn, config.heads, False)

                context_layer = triton_flash_attn_kernel(qkv_out[0], qkv_out[1], qkv_out[2], scale,
                                                         input.shape[-2] % 128 == 0)
                context_layer = _transpose_for_context(context_layer[:, :, :, :head_size])

            else:
                do_flash_attn = False
                if context is not None:
                    query = torch.matmul(input, attn_qw)
                    key = torch.matmul(context, attn_kw)
                    value = torch.matmul(context, attn_vw)
                else:
                    qkv = torch.matmul(input, attn_qkvw)
                    query, key, value = qkv.chunk(3, dim=-1)
                    query = query.contiguous()
                    key = key.contiguous()
                    value = value.contiguous()
                query, key, value = inference_module.pad_transform_fp16(query, key, value, config.heads, do_flash_attn)
                attention_scores = (torch.matmul(query, key.transpose(-1, -2)) * scale).softmax(dim=-1)
                context_layer = _transpose_for_context(torch.matmul(attention_scores, value))

            output = linear_func(context_layer, attn_ow, attn_ob, do_out_bias, False, config.heads, False)
            return output

        output = selfAttention_fp(input, context, input_mask)

        return output

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2, grad_output3):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedDiffusersAttention(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
    """
    layer_id = 0

    def __init__(
        self,
        config,
    ):
        super(DeepSpeedDiffusersAttention, self).__init__()

        self.config = config
        self.config.layer_id = DeepSpeedDiffusersAttention.layer_id
        DeepSpeedDiffusersAttention.layer_id += 1
        device = get_accelerator().current_device_name() if config.bigscience_bloom else 'cpu'
        qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3

        data_type = self.config.dtype
        data_type_fp = torch.half if self.config.dtype == torch.int8 else self.config.dtype
        global inference_module
        if inference_module is None:
            builder = InferenceBuilder()
            inference_module = builder.load()

        if DeepSpeedDiffusersAttention.layer_id == 1:
            log_dist(f"DeepSpeed-Attention config: {self.config.__dict__}", [0])

        self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                  qkv_size_per_partition,
                                                  dtype=data_type,
                                                  device=device),
                                      requires_grad=False)
        self.attn_kw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_vw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_qw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device),
                                      requires_grad=False)
        out_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)

        self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                    requires_grad=False)
        self.do_out_bias = True

        if triton_flash_attn is None:
            load_triton_flash_attn()
        self.triton_flash_attn_kernel = triton_flash_attn()
        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.norm_factor = math.sqrt(math.sqrt(self.config.hidden_size // self.config.heads))

        if self.config.scale_attn_by_inverse_layer_idx is True:
            self.norm_factor *= math.sqrt(self.config.layer_id + 1)
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt2/modeling_gpt2.py#L191

        if self.config.dtype in [torch.float16, torch.int8]:
            self.score_context_func = inference_module.softmax_context_fp16
            self.linear_func = inference_module.linear_layer_fp16
            self.allocate_workspace = inference_module.allocate_workspace_fp16
        else:
            self.score_context_func = inference_module.softmax_context_fp32
            self.linear_func = inference_module.linear_layer_fp32
            self.allocate_workspace = inference_module.allocate_workspace_fp32

    def forward(self, input, context=None, input_mask=None):
        if self.config.layer_id == 0:
            self.allocate_workspace(self.config.hidden_size, self.config.heads,
                                    input.size()[1],
                                    input.size()[0], DeepSpeedDiffusersAttention.layer_id, self.config.mp_size, False,
                                    0, self.config.max_out_tokens, self.config.min_out_tokens)
        output = DeepSpeedDiffusersAttentionFunction.apply(input, context, input_mask, self.config, self.attn_qkvw,
                                                           self.attn_qw, self.attn_kw, self.attn_vw, self.attn_qkvb,
                                                           self.num_attention_heads_per_partition, self.norm_factor,
                                                           self.hidden_size_per_partition, self.attn_ow, self.attn_ob,
                                                           self.do_out_bias, self.score_context_func, self.linear_func,
                                                           self.triton_flash_attn_kernel)

        return output
