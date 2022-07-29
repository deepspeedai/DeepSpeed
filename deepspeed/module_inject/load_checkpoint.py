from torch import nn
import deepspeed.ops.transformer as transformer_inference
from ..runtime.zero import GatheredParameters
from .layers import LinearLayer, Normalize, EmbeddingLayer
import torch


def load_model_with_checkpoint(r_module, sd, mp_replace, ckpt_type, rank=0):
    error_msgs = []

    def transpose(data):
        data1 = data.transpose(-1, -2).reshape(-1)
        data.reshape(-1).copy_(data1)
        data1 = None
        return data.reshape(data.shape[-1], data.shape[-2])

    def load(module, prefix):
        args = (sd, prefix, {}, True, [], [], error_msgs)

        if len(list(module.parameters())) > 0 and list(
                module.parameters())[0].numel() == 0:
            with GatheredParameters(list(module.parameters(recurse=False)),
                                    modifier_rank=0):
                module._load_from_sd(*args)
        else:
            if hasattr(module, 'weight'):
                module.weight = mp_replace.copy(module.weight.data,
                                                sd[prefix + 'weight'])
            if prefix + 'bias' in sd.keys():
                module.bias = mp_replace.copy(module.bias.data, sd[prefix + 'bias'])

    def load_transformer_layer(module, prefix):
        if ckpt_type == "tp":

            def load_parameters(module, prefix):
                for n, p in module.named_parameters():
                    if len(n.split('.')) == 1:
                        src_shape = sd[prefix + n].shape
                        dst_shape = p.shape

                        if (len(src_shape) == 2 and len(dst_shape) == 2):
                            if src_shape[0] == dst_shape[0] and src_shape[
                                    1] == dst_shape[1]:
                                p.data.copy_(sd[prefix + n])
                            else:
                                if src_shape[0] != dst_shape[0]:
                                    weight_split = torch.split(
                                        sd[prefix + n],
                                        dst_shape[0],
                                        dim=0)[rank].to(
                                            torch.cuda.current_device()).contiguous()
                                else:
                                    weight_split = torch.split(
                                        sd[prefix + n],
                                        dst_shape[1],
                                        dim=1)[rank].to(
                                            torch.cuda.current_device()).contiguous()
                                p.data.copy_(weight_split.contiguous())
                        else:
                            if src_shape[0] == dst_shape[0]:
                                p.data.copy_(sd[prefix + n])
                            else:
                                bias_split = torch.split(
                                    sd[prefix + n],
                                    dst_shape[-1])[rank].to(
                                        torch.cuda.current_device()).contiguous()
                                p.data.copy_(bias_split)

            load_parameters(module, prefix)
            for n, child in module.named_children():
                load_parameters(child, prefix + n + '.')
        else:
            module.norm_w.data.copy_(sd[prefix + 'input_layernorm.' + 'weight'])
            module.norm_b.data.copy_(sd[prefix + 'input_layernorm.' + 'bias'])
            module.attention.attn_qkvw = mp_replace.copy(
                module.attention.attn_qkvw.data,
                transpose(sd[prefix + 'self_attention.query_key_value.' + 'weight']))
            module.attention.attn_qkvb = mp_replace.copy(
                module.attention.attn_qkvb.data,
                sd[prefix + 'self_attention.query_key_value.' + 'bias'])
            module.attention.attn_ow = mp_replace.copy(
                module.attention.attn_ow.data,
                transpose(sd[prefix + 'self_attention.dense.' + 'weight']))
            module.attention.attn_ob = mp_replace.copy(
                module.attention.attn_ob.data,
                sd[prefix + 'self_attention.dense.' + 'bias'])
            module.mlp.attn_nw.data.copy_(sd[prefix + 'post_attention_layernorm.' +
                                             'weight'])
            module.mlp.attn_nb.data.copy_(sd[prefix + 'post_attention_layernorm.' +
                                             'bias'])
            module.mlp.inter_w = mp_replace.copy(
                module.mlp.inter_w.data,
                transpose(sd[prefix + 'mlp.dense_h_to_4h.' + 'weight']))
            module.mlp.inter_b = mp_replace.copy(
                module.mlp.inter_b.data,
                sd[prefix + 'mlp.dense_h_to_4h.' + 'bias'])
            module.mlp.output_w = mp_replace.copy(
                module.mlp.output_w.data,
                transpose(sd[prefix + 'mlp.dense_4h_to_h.' + 'weight']))
            module.mlp.output_b = mp_replace.copy(
                module.mlp.output_b.data,
                sd[prefix + 'mlp.dense_4h_to_h.' + 'bias'])

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
                if not any(checking_key in item for item in sd.keys()):
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

    del sd
    sd = None
