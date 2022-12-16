from torch import nn
import deepspeed.ops.transformer as transformer_inference
from .layers import LinearLayer, Normalize, EmbeddingLayer, OPTEmbedding
import torch
import gc


def load_model_with_checkpoint(r_module,
                               sd,
                               mp_replace,
                               ckpt_type,
                               weight_quantizer=None,
                               rank=0,
                               param_names=None,
                               transformer_config=None,
                               megatron_v2=False):
    error_msgs = []

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

            def _transpose(x):
                heads = transformer_config.heads // mp_replace.mp_size
                attention_head_size = x.shape[-1] // heads
                new_x_shape = x.size()[:-1] + (heads, attention_head_size)
                x_1 = x.view(*new_x_shape)
                (q, k, v) = torch.split(x_1, (x_1.shape[-1] // 3), dim=(x_1.dim() - 1))
                if len(q.shape) > 2:
                    return torch.cat((q.reshape(q.shape[0],
                                                -1),
                                      k.reshape(q.shape[0],
                                                -1),
                                      v.reshape(q.shape[0],
                                                -1)),
                                     dim=-1).reshape(x.shape)
                else:
                    return torch.cat((q.reshape(-1),
                                      k.reshape(-1),
                                      v.reshape(-1)),
                                     dim=-1).reshape(x.shape)

            # This checks if the parameter exits in the checkpoint file and maybe copies it into the corresponding destination tensor.
            # Note that not all parameters are saved in one checkpoint, that's why we always need to check if they exist!
            def maybe_copy(module,
                           dst_name,
                           src_name,
                           qkv=False,
                           megatron_v2=False,
                           split_qkv=False):
                if src_name in sd[0]:
                    dst = getattr(module, dst_name)
                    tmp = sd[0][src_name].cuda()
                    if len(dst.shape) == 1:
                        if split_qkv:
                            dst = mp_replace.qkv_copy(dst, tmp)
                        else:
                            dst = mp_replace.copy(dst, tmp)
                        if qkv and megatron_v2:
                            dst = torch.nn.parameter.Parameter(
                                _transpose(dst).contiguous())
                    else:
                        if split_qkv:
                            dst = weight_quantizer.quantize(mp_replace.qkv_copy(dst, tmp if weight_quantizer.q_int8 else \
                                                            (transpose(tmp).contiguous())))
                        else:
                            dst = weight_quantizer.quantize(mp_replace.copy(dst, tmp if weight_quantizer.q_int8 else \
                                                            transpose(tmp)))
                        if qkv and megatron_v2:
                            scale1 = dst.scale
                            dst = torch.nn.parameter.Parameter(
                                _transpose(dst).contiguous())
                            dst.scale = scale1
                    setattr(module, dst_name, dst)

            # Extending the maybe_copy function for when the q, k, and v are in separate parameters!
            def maybe_copy_qkv(module, dst_name, src_names, split_qkv=False):
                if src_names[0] in sd[0]:
                    q = sd[0][src_names[0]]
                    k = sd[0][src_names[1]]
                    v = sd[0][src_names[2]]
                    qkv_data = torch.cat((q, k, v), dim=0)
                    dst = getattr(module, dst_name)
                    if len(dst.shape) == 1:
                        if split_qkv:
                            dst = mp_replace.qkv_copy(dst,
                                                      (qkv_data.cuda()).contiguous())
                        else:
                            dst = mp_replace.copy(dst, qkv_data.cuda())
                    else:
                        if split_qkv:
                            dst = weight_quantizer.quantize(mp_replace.qkv_copy(dst, qkv_data.cuda() if weight_quantizer.q_int8 else \
                                                            ((transpose(qkv_data.cuda())).contiguous())))
                        else:
                            dst = weight_quantizer.quantize(mp_replace.copy(dst, qkv_data.cuda() if weight_quantizer.q_int8 else \
                                                            transpose(qkv_data.cuda())))
                    setattr(module, dst_name, dst)

            if len(param_names) == 14:
                qkv_w, qkv_b, attn_ow, attn_ob, \
                mlp_intw, mlp_intb, mlp_ow, mlp_ob, \
                inp_normw, inp_normb, attn_nw, attn_nb, _, split_qkv = param_names
            elif len(param_names) < 14:
                q_w, k_w, v_w, attn_ow, \
                mlp_intw, mlp_intb, mlp_ow, mlp_ob, \
                inp_normw, inp_normb, _, split_qkv = param_names
            else:
                q_w, q_b, k_w, k_b, v_w, v_b, attn_ow, attn_ob, \
                mlp_intw, mlp_intb, mlp_ow, mlp_ob, \
                inp_normw, inp_normb, attn_nw, attn_nb, _, split_qkv = param_names

            maybe_copy(module, 'norm_w', prefix + inp_normw)
            maybe_copy(module, 'norm_b', prefix + inp_normb)
            if len(param_names) == 14:
                maybe_copy(module.attention,
                           'attn_qkvw',
                           prefix + qkv_w,
                           qkv=True,
                           megatron_v2=megatron_v2,
                           split_qkv=split_qkv)
                maybe_copy(module.attention,
                           'attn_qkvb',
                           prefix + qkv_b,
                           qkv=True,
                           megatron_v2=megatron_v2,
                           split_qkv=split_qkv)
            elif len(param_names) < 14:
                maybe_copy_qkv(module.attention,
                               'attn_qkvw',
                               [prefix + q_w,
                                prefix + k_w,
                                prefix + v_w],
                               split_qkv=split_qkv)
            else:
                maybe_copy_qkv(module.attention,
                               'attn_qkvw',
                               [prefix + q_w,
                                prefix + k_w,
                                prefix + v_w],
                               split_qkv=split_qkv)
                maybe_copy_qkv(module.attention,
                               'attn_qkvb',
                               [prefix + q_b,
                                prefix + k_b,
                                prefix + v_b],
                               split_qkv=split_qkv)
            maybe_copy(module.attention, 'attn_ow', prefix + attn_ow)
            if len(param_names) >= 14:
                maybe_copy(module.attention, 'attn_ob', prefix + attn_ob)
                maybe_copy(module.mlp, 'attn_nw', prefix + attn_nw)
                maybe_copy(module.mlp, 'attn_nb', prefix + attn_nb)
            maybe_copy(module.mlp, 'inter_w', prefix + mlp_intw)
            maybe_copy(module.mlp, 'inter_b', prefix + mlp_intb)
            maybe_copy(module.mlp, 'output_w', prefix + mlp_ow)
            maybe_copy(module.mlp, 'output_b', prefix + mlp_ob)

    try:
        import transformers
        OPTLearnedPositionalEmbedding = transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding
    except:
        OPTLearnedPositionalEmbedding = None
    layer_policies = {
        nn.Linear: load,
        nn.Embedding: load,
        nn.LayerNorm: load,
        EmbeddingLayer: load,
        LinearLayer: load,
        Normalize: load,
        transformer_inference.DeepSpeedTransformerInference: load_transformer_layer,
        OPTLearnedPositionalEmbedding: load,
        OPTEmbedding: load
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
                        child = LinearLayer(weight_shape=child.weight.shape,
                                            bias=child.bias)
                        setattr(module, name, child)
                    elif child.__class__ is OPTLearnedPositionalEmbedding:
                        child = OPTEmbedding(weight_shape=ds_shape)
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
                    prefix if (level == 0 and ckpt_type == 'pp') and param_names[-2] else \
                    prefix + name + '.',
                    level + 1)

    load_module_recursive(r_module)

    #XXX: hack to tie embedding w. lm_head for BLOOM, need to revist soon
    embedding_weight = None
    for n, p in r_module.named_parameters():
        if "word_embeddings." in n or "embed_tokens." in n:
            embedding_weight = p

    if embedding_weight is not None:
        r_module.lm_head.weight = embedding_weight
    for sd_ in sd:
        del sd_
    sd = None
    gc.collect()
