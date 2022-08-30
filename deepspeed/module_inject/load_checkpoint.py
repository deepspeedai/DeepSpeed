from torch import nn
import deepspeed.ops.transformer as transformer_inference
from ..runtime.zero import GatheredParameters
from .layers import LinearLayer, Normalize, EmbeddingLayer
import torch
import gc


def load_model_with_checkpoint(r_module,
                               sd,
                               mp_replace,
                               ckpt_type,
                               weight_quantizer=None,
                               rank=0):
    error_msgs = []

    def transpose(data):
        data1 = data.transpose(-1, -2).reshape(-1)
        data.reshape(-1).copy_(data1)
        data1 = None
        return data.reshape(data.shape[-1], data.shape[-2])

    def load(module, prefix):
        args = (sd[0], prefix, {}, True, [], [], error_msgs)

        if len(list(module.parameters())) > 0 and list(
                module.parameters())[0].numel() == 0:
            with GatheredParameters(list(module.parameters(recurse=False)),
                                    modifier_rank=0):
                module._load_from_sd(*args)
        else:
            if hasattr(module, 'weight'):
                module.weight = mp_replace.copy(module.weight.data,
                                                sd[0][prefix + 'weight'])
            if prefix + 'bias' in sd[0].keys():
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
                            scale = scale.to(torch.cuda.current_device())
                        else:
                            tmp_data = sd[0][prefix + n].to(torch.cuda.current_device())
                            scale = None
                        src_shape = tmp_data.shape
                        dst_shape = p.shape
                        inner_dim = 1 if tmp_data.dtype == torch.int8 else 0
                        outer_dim = 0 if tmp_data.dtype == torch.int8 else 1
                        if (len(src_shape) == 2 and len(dst_shape) == 2):
                            if (src_shape[inner_dim] == dst_shape[0]
                                    and src_shape[outer_dim] == dst_shape[1]):
                                if tmp_data.dtype != torch.int8:
                                    p = weight_quantizer.quantize(
                                        transpose(tmp_data) if weight_quantizer.
                                        q_int8 else tmp_data)
                                else:
                                    p = torch.nn.parameter.Parameter(tmp_data,
                                                                     requires_grad=False)
                                    p.scale = scale
                                setattr(module, n, p)
                            else:
                                dim = inner_dim if src_shape[inner_dim] != dst_shape[
                                    0] else outer_dim
                                dim1 = 0 if src_shape[inner_dim] != dst_shape[0] else 1
                                if src_shape[dim] > dst_shape[dim1]:
                                    weight_partition = torch.split(
                                        tmp_data,
                                        dst_shape[dim1],
                                        dim=dim)[rank].to(torch.cuda.current_device())
                                    assert tmp_data.dtype != torch.int8 or scale.numel() > weight_quantizer.num_groups * (rank+1), \
                                        '''ERROR: We require the quantization scales for larger TP-size when loading INT8 checkpoint!\
                                           Please use the FP16 checkpoint to generate INT8 checkpoint with the sharding parameters!'''
                                    scale = scale.view(
                                        -1)[weight_quantizer.num_groups *
                                            (rank + 1):].reshape(
                                                weight_quantizer.num_groups,
                                                -1).contiguous()
                                else:
                                    assert tmp_data.dtype != torch.int8, \
                                        '''Merging of the checkpoints are not supported when using INT8 checkpoint! \
                                           Please use a as many GPUs as TP-size for the checkpoint'''
                                    all_data = [
                                        sd[j][prefix +
                                              n] if type(sd[j][prefix + n]) is list else
                                        sd[j][prefix + n].to(torch.cuda.current_device())
                                        for j in range(len(sd))
                                    ]
                                    weight_partition = torch.cat([
                                        ad[0].to(torch.cuda.current_device())
                                        if type(ad) is list else ad for ad in all_data
                                    ],
                                                                 dim=dim)
                                    if tmp_data.dtype == torch.int8:
                                        scale = torch.cat([
                                            ad[1].to(torch.cuda.current_device())
                                            for ad in all_data
                                        ],
                                                          dim=dim)

                                if tmp_data.dtype != torch.int8:
                                    weight_partition = weight_quantizer.quantize(
                                        transpose(weight_partition), \
                                        parallel_dim=(0 if dim == 1 else 1)) if weight_quantizer.q_int8 else \
                                        weight_quantizer.quantize(weight_partition)
                                else:
                                    weight_partition = torch.nn.parameter.Parameter(
                                        weight_partition,
                                        requires_grad=False)
                                    weight_partition.scale = scale
                                setattr(module, n, weight_partition)
                        else:
                            if src_shape[0] == dst_shape[0]:
                                p.data.copy_(tmp_data)
                            else:
                                if src_shape[0] > dst_shape[0]:
                                    bias_split = torch.split(
                                        tmp_data,
                                        dst_shape[-1])[rank].to(
                                            torch.cuda.current_device()).contiguous()
                                    p.data.copy_(bias_split)
                                else:
                                    p.data.copy_(
                                        torch.cat(
                                            [sd[j][prefix + n] for j in range(len(sd))],
                                            dim=0).to(torch.cuda.current_device()).
                                        contiguous())

            load_parameters(module, prefix)
            for n, child in module.named_children():
                load_parameters(child, prefix + n + '.')
        else:
            module.norm_w.data.copy_(sd[0][prefix + 'input_layernorm.' + 'weight'])
            module.norm_b.data.copy_(sd[0][prefix + 'input_layernorm.' + 'bias'])
            module.attention.attn_qkvw = mp_replace.copy(module.attention.attn_qkvw,
                weight_quantizer.quantize(sd[0][prefix + 'self_attention.query_key_value.' + 'weight']) if weight_quantizer.q_int8 else \
                weight_quantizer.quantize(transpose(sd[0][prefix + 'self_attention.query_key_value.' + 'weight'])))
            module.attention.attn_qkvb = mp_replace.copy(
                module.attention.attn_qkvb.data,
                sd[0][prefix + 'self_attention.query_key_value.' + 'bias'])
            module.attention.attn_ow = mp_replace.copy(module.attention.attn_ow,
                weight_quantizer.quantize(sd[0][prefix + 'self_attention.dense.' + 'weight']) if weight_quantizer.q_int8 else \
                weight_quantizer.quantize(transpose(sd[0][prefix + 'self_attention.dense.' + 'weight'])))
            module.attention.attn_ob = mp_replace.copy(
                module.attention.attn_ob.data,
                sd[0][prefix + 'self_attention.dense.' + 'bias'])
            module.mlp.attn_nw.data.copy_(sd[0][prefix + 'post_attention_layernorm.' +
                                                'weight'])
            module.mlp.attn_nb.data.copy_(sd[0][prefix + 'post_attention_layernorm.' +
                                                'bias'])
            module.mlp.inter_w = mp_replace.copy(module.mlp.inter_w,
                weight_quantizer.quantize(sd[0][prefix + 'mlp.dense_h_to_4h.' + 'weight']) if weight_quantizer.q_int8 else \
                weight_quantizer.quantize(transpose(sd[0][prefix + 'mlp.dense_h_to_4h.' + 'weight'])))
            module.mlp.inter_b = mp_replace.copy(
                module.mlp.inter_b.data,
                sd[0][prefix + 'mlp.dense_h_to_4h.' + 'bias'])
            module.mlp.output_w = mp_replace.copy(module.mlp.output_w,
                weight_quantizer.quantize(sd[0][prefix + 'mlp.dense_4h_to_h.' + 'weight']) if weight_quantizer.q_int8 else \
                weight_quantizer.quantize(transpose(sd[0][prefix + 'mlp.dense_4h_to_h.' + 'weight'])))
            module.mlp.output_b = mp_replace.copy(
                module.mlp.output_b.data,
                sd[0][prefix + 'mlp.dense_4h_to_h.' + 'bias'])

    layer_policies = {
        nn.Linear: load,
        nn.Embedding: load,
        nn.LayerNorm: load,
        EmbeddingLayer: load,
        LinearLayer: load,
        Normalize: load,
        transformer_inference.DeepSpeedTransformerInference: load_transformer_layer
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
                if len(child_params) > 0 and (child_params[0].numel() == 0
                                              or child_params[0].is_meta):
                    if child.weight.is_meta:
                        ds_shape = child.weight.shape
                    else:
                        ds_shape = child.weight.ds_shape

                    if child.__class__ is nn.LayerNorm:
                        child = Normalize(dim=ds_shape[-1],
                                          dtype=child.weight.dtype,
                                          eps=child.eps)
                        setattr(module, name, child)
                    elif child.__class__ is nn.Linear:
                        child = LinearLayer(weight=child.weight, bias=child.bias)
                        setattr(module, name, child)
                    else:
                        ds_id = None
                        if hasattr(child.weight, 'ds_id'):
                            ds_id = child.weight.ds_id
                        child = EmbeddingLayer(weight_shape=ds_shape,
                                               dtype=child.weight.dtype)
                        if ds_id is not None:
                            all_ds_ids[ds_id] = child.weight
                        setattr(module, name, child)

                layer_policies[child.__class__](child, prefix + name + '.')
            else:
                load_module_recursive(
                    child,
                    prefix if level == 0 and ckpt_type == 'pp' else prefix + name + '.',
                    level + 1)

    load_module_recursive(r_module)

    #XXX: hack to tie embedding w. lm_head for BLOOM, need to revist soon
    embedding_weight = None
    for n, p in r_module.named_parameters():
        if "word_embeddings." in n:
            embedding_weight = p
    assert hasattr(r_module, 'lm_head'), "attempting to set lm_head but it doesn't exist"
    r_module.lm_head.weight = embedding_weight
    for sd_ in sd:
        del sd_
    sd = None
    gc.collect()
