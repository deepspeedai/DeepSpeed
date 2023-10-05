# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from torch import nn
from deepspeed.model_implementations.transformers.ds_bloom import DeepSpeedBloomInference
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
from deepspeed.model_implementations.transformers.ds_bert import DeepSpeedBERTInference
from deepspeed.model_implementations.transformers.ds_megatron_gpt import DeepSpeedMegatronGPTInference
from deepspeed.model_implementations.transformers.ds_opt import DeepSpeedOPTInference
from deepspeed.model_implementations.transformers.ds_llama2 import DeepSpeedLlama2Inference

import deepspeed.ops.transformer as transformer_inference
from .layers import LinearLayer, Normalize, EmbeddingLayer, OPTEmbedding, RMSNormalize
import torch
import gc
from deepspeed.accelerator import get_accelerator
import re


def load_model_with_checkpoint(r_module,
                               sd,
                               mp_replace,
                               ckpt_type,
                               ckpt_mp_size,
                               weight_quantizer=None,
                               rank=0,
                               container=None):
    error_msgs = []

    def prefix_check():
        # if keys start with 'model.' or 'transformer.', don't skip level 0 prefix
        for key in sd[0].keys():
            # OPT models
            if re.match("^model[.]", key):
                return False
            # BLOOM models
            if re.match("^transformer[.]", key):
                return False
        return True

    skip_level_0_prefix = prefix_check() and container.policy.use_load_prefix

    def transpose(data):
        with torch.no_grad():
            data = data.contiguous()
            data1 = data.transpose(-1, -2).reshape(-1)
            data.reshape(-1).copy_(data1)
            data1 = None
        return data.reshape(data.shape[-1], data.shape[-2])

    def load(module, prefix):
        args = (sd[0], prefix, {}, True, [], [], error_msgs)

        if hasattr(module, 'weight'):
            module.weight = mp_replace.copy(module.weight.data, sd[0][prefix + 'weight'])
        if prefix + 'bias' in sd[0].keys():
            if module.bias.data.is_meta:
                # meta tensor cannot be casted or copied to, so we need to replace it with a normal tensor here
                module.bias = torch.nn.parameter.Parameter(data=torch.empty_like(module.bias.data, device="cpu"),
                                                           requires_grad=module.bias.data.requires_grad)
            module.bias = mp_replace.copy(module.bias.data, sd[0][prefix + 'bias'])
        args = None
        gc.collect()

    def load_transformer_layer(module, prefix):
        if ckpt_type == "tp":

            def load_parameters(module, prefix):
                for n, p in module.named_parameters():
                    if prefix + n in sd[0] and len(n.split('.')) == 1:
                        if type(sd[0][prefix + n]) is list:
                            tmp_data, scale = sd[0][prefix + n]
                            tmp_data = tmp_data
                            scale = scale.to(get_accelerator().current_device_name())
                            # set the quantizer number of groups using the checkpoint scale shape
                            weight_quantizer.num_groups = scale.shape[0]
                        else:
                            tmp_data = sd[0][prefix + n].to(get_accelerator().current_device_name())
                            scale = None
                        src_shape = tmp_data.shape
                        dst_shape = p.shape
                        inner_dim = 1 if tmp_data.dtype == torch.int8 else 0
                        outer_dim = 0 if tmp_data.dtype == torch.int8 else 1
                        if (len(src_shape) == 2 and len(dst_shape) == 2):
                            if (src_shape[inner_dim] == dst_shape[0] and src_shape[outer_dim] == dst_shape[1]):
                                if tmp_data.dtype != torch.int8:
                                    p = weight_quantizer.quantize(
                                        transpose(tmp_data) if weight_quantizer.q_int8 else tmp_data)
                                else:
                                    p = torch.nn.parameter.Parameter(tmp_data, requires_grad=False)
                                    p.scale = scale
                                setattr(module, n, p)
                            else:
                                dim = inner_dim if src_shape[inner_dim] != dst_shape[0] else outer_dim
                                dim1 = 0 if src_shape[inner_dim] != dst_shape[0] else 1
                                if src_shape[dim] > dst_shape[dim1]:
                                    weight_partition = torch.split(tmp_data, dst_shape[dim1], dim=dim)[rank].to(
                                        get_accelerator().current_device_name())
                                    assert tmp_data.dtype != torch.int8 or scale.numel() > weight_quantizer.num_groups * (rank+1), \
                                        '''ERROR: We require the quantization scales for larger TP-size when loading INT8 checkpoint!\
                                           Please use the FP16 checkpoint to generate INT8 checkpoint with the sharding parameters!'''
                                    scale = scale.view(-1)[weight_quantizer.num_groups * (rank + 1):].reshape(
                                        weight_quantizer.num_groups, -1).contiguous()
                                else:
                                    assert tmp_data.dtype != torch.int8, \
                                        '''Merging of the checkpoints are not supported when using INT8 checkpoint! \
                                          Please use a as many GPUs as TP-size for the checkpoint'''
                                    all_data = [
                                        sd[j][prefix + n] if type(sd[j][prefix + n]) is list else sd[j][prefix + n].to(
                                            get_accelerator().current_device_name()) for j in range(len(sd))
                                    ]
                                    # Check if the weight tensor is for the QKV parameter
                                    if src_shape[1] == (3 * src_shape[0]) // ckpt_mp_size:
                                        qkv_size = src_shape[outer_dim] // 3
                                        src_split = [
                                            torch.split(src[0].data, qkv_size, dim=outer_dim) for src in all_data
                                        ]

                                        weight_partition = torch.cat([
                                            torch.cat([qkv_s[i] for qkv_s in src_split], axis=outer_dim)
                                            for i in range(len(src_split[0]))
                                        ],
                                                                     dim=dim)
                                    else:
                                        weight_partition = torch.cat([
                                            ad[0].to(get_accelerator().current_device_name())
                                            if type(ad) is list else ad for ad in all_data
                                        ],
                                                                     dim=dim)
                                    if tmp_data.dtype == torch.int8:
                                        scale = torch.cat(
                                            [ad[1].to(get_accelerator().current_device_name()) for ad in all_data],
                                            dim=dim)

                                if tmp_data.dtype != torch.int8:
                                    weight_partition = weight_quantizer.quantize(
                                        transpose(weight_partition), \
                                        parallel_dim=(0 if dim == 1 else 1)) if weight_quantizer.q_int8 else \
                                        weight_quantizer.quantize(weight_partition)
                                else:
                                    weight_partition = torch.nn.parameter.Parameter(weight_partition,
                                                                                    requires_grad=False)
                                    weight_partition.scale = scale
                                setattr(module, n, weight_partition)
                        else:
                            if src_shape[0] == dst_shape[0]:
                                p.data.copy_(tmp_data)
                            else:
                                if src_shape[0] > dst_shape[0]:
                                    bias_split = torch.split(tmp_data, dst_shape[-1])[rank].to(
                                        get_accelerator().current_device_name()).contiguous()
                                    p.data.copy_(bias_split)
                                else:
                                    # Check if the weight tensor is for the QKV parameter
                                    if src_shape[0] == (3 * r_module.config.hidden_size) // ckpt_mp_size:
                                        qkv_size = src_shape[0] // 3
                                        src_split = [
                                            torch.split(sd[j][prefix + n], qkv_size, dim=0) for j in range(len(sd))
                                        ]

                                        p.data.copy_(
                                            torch.cat([
                                                torch.cat([qkv_s[i] for qkv_s in src_split], axis=0)
                                                for i in range(len(src_split[0]))
                                            ],
                                                      dim=0).to(get_accelerator().current_device_name()).contiguous())
                                    else:
                                        p.data.copy_(
                                            torch.cat([sd[j][prefix + n] for j in range(len(sd))],
                                                      dim=0).to(get_accelerator().current_device_name()).contiguous())

            load_parameters(module, prefix)
            for n, child in module.named_children():
                load_parameters(child, prefix + n + '.')
        else:
            container.load_params(module, sd[0], weight_quantizer, mp_replace, prefix)

    try:
        import transformers
        OPTLearnedPositionalEmbedding = transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding
        if hasattr(transformers.models, "llama"):
            LlamaRMSNorm = transformers.models.llama.modeling_llama.LlamaRMSNorm
        else:
            LlamaRMSNorm = None
    except:
        OPTLearnedPositionalEmbedding = None
    try:
        from fairscale.nn.model_parallel.layers import (
            ColumnParallelLinear,
            ParallelEmbedding,
            RowParallelLinear,
        )
    except:
        ColumnParallelLinear = None
        ParallelEmbedding = None
        RowParallelLinear = None
    try:
        from llama.model import RMSNorm
    except:
        RMSNorm = None
    layer_policies = {
        nn.Linear: load,
        nn.Embedding: load,
        nn.LayerNorm: load,
        EmbeddingLayer: load,
        LinearLayer: load,
        Normalize: load,
        transformer_inference.DeepSpeedTransformerInference: load_transformer_layer,
        DeepSpeedBloomInference: load_transformer_layer,
        DeepSpeedGPTInference: load_transformer_layer,
        DeepSpeedBERTInference: load_transformer_layer,
        DeepSpeedMegatronGPTInference: load_transformer_layer,
        DeepSpeedOPTInference: load_transformer_layer,
        DeepSpeedLlama2Inference: load_transformer_layer,
        OPTLearnedPositionalEmbedding: load,
        OPTEmbedding: load,
        LlamaRMSNorm: load,
        RMSNormalize: load,
        ColumnParallelLinear: load,
        ParallelEmbedding: load,
        RowParallelLinear: load,
        RMSNorm: load
    }

    all_ds_ids = {}

    def load_module_recursive(module, prefix='', level=0):
        for name, child in module.named_children():
            if child.__class__ in layer_policies:
                checking_key = prefix + name + '.'
                if not any(checking_key in item for item in sd[0].keys()):
                    if hasattr(child, 'weight') and \
                        (hasattr(child.weight, 'ds_id') and \
                        child.weight.ds_id in all_ds_ids):
                        prefix1 = all_ds_ids[child.weight.ds_id]
                        if child.__class__ is nn.Linear:
                            child = LinearLayer(weight=all_ds_ids[child.weight.ds_id])
                            setattr(module, name, child)
                    continue
                child_params = list(child.parameters())
                if len(child_params) > 0 and (child_params[0].numel() == 0 or child_params[0].is_meta):
                    if child.weight.is_meta:
                        ds_shape = child.weight.shape
                    else:
                        ds_shape = child.weight.ds_shape
                    if child.__class__ is nn.LayerNorm:
                        child = Normalize(dim=ds_shape[-1], dtype=child.weight.dtype, eps=child.eps)
                        setattr(module, name, child)
                    elif child.__class__ in [nn.Linear, ColumnParallelLinear, RowParallelLinear]:
                        child = LinearLayer(weight_shape=child.weight.shape, bias=child.bias)
                        setattr(module, name, child)
                    elif child.__class__ is OPTLearnedPositionalEmbedding:
                        child = OPTEmbedding(weight_shape=ds_shape)
                        setattr(module, name, child)
                    elif child.__class__ in [LlamaRMSNorm, RMSNorm]:
                        child = RMSNormalize(dim=ds_shape[-1],
                                             dtype=child.weight.dtype,
                                             eps=child.eps if hasattr(child, 'eps') else child.variance_epsilon)
                        setattr(module, name, child)
                    else:
                        ds_id = None
                        if hasattr(child.weight, 'ds_id'):
                            ds_id = child.weight.ds_id
                        child = EmbeddingLayer(weight_shape=ds_shape, dtype=child.weight.dtype)
                        if ds_id is not None:
                            all_ds_ids[ds_id] = child.weight
                        setattr(module, name, child)
                layer_policies[child.__class__](child, prefix + name + '.')
            else:
                load_module_recursive(
                    child,
                    prefix if (level == 0 and ckpt_type == 'pp') and skip_level_0_prefix else \
                    prefix + name + '.',
                    level + 1)

    load_module_recursive(r_module)

    for sd_ in sd:
        del sd_
    sd = None
    gc.collect()
