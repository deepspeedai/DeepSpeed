# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from deepspeed.utils.logging import warning_once
from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list, get_num_kv_heads, get_n_embd


def split_by_qkvlist_and_refuse(qkv_list, split_size, split_dim=0, cat_dim=0):
    qkv_split_list = [torch.split(mat, split_size, dim=split_dim) for mat in qkv_list]
    tp_fusedqkv_list = [
        torch.cat([qkv_s[i] for qkv_s in qkv_split_list], dim=cat_dim) for i in range(len(qkv_split_list[0]))
    ]
    return tp_fusedqkv_list


def require_tp_fused_qkvw(name, mp_size):
    fused_qkvw_name_list = ['qkv_proj', 'query_key_value', 'attn.Wqkv', 'self_attn.W_pack', 'c_attn']

    if mp_size == 1:
        return False
    for fused_name in fused_qkvw_name_list:
        if fused_name in name:
            return True
    return False


def prepare_tp_fused_qkvw(module, src, mp_size, gpu_index):

    module_str = str(module).strip()
    if src is None:
        return
    fused_type_dict = {
        'CodeGenBlock': 'codegentype',
        'BloomBlock': 'bloomtype',
        'GLMBlock': 'glmtype',
        "MPTBlock": 'glmtype',
        "MptBlock": 'glmtype',
        "BaichuanLayer": 'glmtype',
        "QWenBlock": 'qwentype',
        "FalconDecoderLayer": 'bloomtype',
        "GPTBigCodeBlock": 'bigcodetype',
        "DecoderLayer": 'glmtype',
    }

    def _codegen_type_transpose(input, mp_size, codegen_mp_num=4):
        # codegen_mp_num defined in https://github.com/huggingface/transformers/blob/main/src/transformers/models/codegen/modeling_codegen.py
        assert get_num_kv_heads() % (
            mp_size * codegen_mp_num) == 0, "codgen autoTP requires num_kv_heads % (mp_size*codegen_mp_num) == 0"
        #input : [3*hidden_dim, hidden_dim](weight) or [3*hidden_dim](bias)

        shape = input.shape
        dst_shape = get_shard_size(shape[0], mp_size)
        num_mp_blocks = input.reshape(codegen_mp_num, shape[0] // codegen_mp_num, shape[1])

        #num_mp_blocks : [codegen_mp_num, 3*hidden_dim/codegen_mp_num, :]
        src_split = list(torch.split(num_mp_blocks, num_mp_blocks.shape[1] // 3, dim=1))
        src_split = [x.reshape(codegen_mp_num * mp_size, -1, shape[1]) for x in src_split]

        split_fusedqkv = split_by_qkvlist_and_refuse(src_split, get_shard_size(shape[0] // 3, mp_size), 0, 1)
        tp_fuseqkv_weight = torch.cat(split_fusedqkv, dim=0).reshape(shape[0], -1)

        return tp_fuseqkv_weight[gpu_index * dst_shape:(gpu_index + 1) * dst_shape]

    def _glm_type_transpose(input, mp_size):
        #input : [3*hidden_dim, hidden_dim](weight) or [3*hidden_dim](bias)

        shape = input.shape
        src_split = torch.split(input, shape[0] // 3, dim=0)

        split_fusedqkv = split_by_qkvlist_and_refuse(src_split, get_shard_size_list(shape[0] // 3, mp_size))
        return split_fusedqkv[gpu_index]

    def _bloom_type_transpose(input, mp_size):
        shape = input.shape

        split_fusedqkv = input.split(get_shard_size_list(shape[0], mp_size), dim=0)
        return split_fusedqkv[gpu_index]

    def _qwen_type_transpose(input, mp_size, module):
        if not hasattr(module, "_ds_fusedqkv_entered"):
            # Adjust splitting absolute value variables
            setattr(module, "_ds_fusedqkv_entered", True)
            module.attn.split_size = get_shard_size(module.attn.split_size, mp_size)
        return _glm_type_transpose(input, mp_size)

    def _bigcode_type_transpose(input, mp_size):
        n_embd = get_n_embd()
        q = input[:n_embd]
        kv = input[n_embd:]
        shape = q.shape
        split_q = q.split(get_shard_size_list(shape[0], mp_size), dim=0)
        return torch.cat((split_q[gpu_index], kv), dim=0)

    def _transpose_fused_qkvw(src, mp_size, fused_qkv_type=None, module=None):

        # suppose num_heads=n, q(n)_w means the n-th q head linear weight, the weight format are as following
        # bloomtype: [q(1)_w,k(1)_w,v(1)_w,q(2)_w,k(2)_w,v(2)_w,...,q(n)_w,k(n)_w,v(n)_w]
        # glmtype:  [q(1)_w, q(2)_w,...,q(n)_w,k(1)_w,k(2)_w,...,k(n)_w,v(1)_w,v(2)_w,...,v(n)_w]
        # codegentype: [q(1)_w,q(2)_w,...,q(n/t)_w,k(1)_w,k(2)_w,...,k(n/t)_w,v(1)_2,v(2)_w,...v(n/t)_w,q(n/t+1)_w,...], where t is a const defined in model file.

        if fused_qkv_type == 'bloomtype':
            return _bloom_type_transpose(src, mp_size)
        elif fused_qkv_type == 'codegentype':
            return _codegen_type_transpose(src, mp_size)
        elif fused_qkv_type == 'glmtype':
            return _glm_type_transpose(src, mp_size)
        elif fused_qkv_type == 'qwentype':
            return _qwen_type_transpose(src, mp_size, module)
        elif fused_qkv_type == 'bigcodetype':
            return _bigcode_type_transpose(src, mp_size)

        raise ValueError("unknown fused_qkv_type")

    module_name_matches = [k for k in fused_type_dict.keys() if k in module_str]
    if module_name_matches:
        # There can be overlap with matches (e.g., "DecoderLayer" and "FalconDecoderLayer").
        # We take the longest matching module_name
        module_name = max(module_name_matches, key=len)
        fused_type = fused_type_dict[module_name]
        return _transpose_fused_qkvw(src, mp_size, fused_type, module)
    warning_once(f"Unrecognized fusedkqv weight type, default to using bloom type,"
                 f"please check in prepare_tp_fused_qkvw() to avoid potential calculation errors")
    return _bloom_type_transpose(src, mp_size)
