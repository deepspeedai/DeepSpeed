# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from deepspeed.utils.logging import warning_once
import re


def split_by_qkvlist_and_refuse(qkv_list, split_size, split_dim=0, cat_dim=0):
    qkv_split_list = [torch.split(mat, split_size, dim=split_dim) for mat in qkv_list]
    tp_fusedqkv_list = [
        torch.cat([qkv_s[i] for qkv_s in qkv_split_list], dim=cat_dim) for i in range(len(qkv_split_list[0]))
    ]
    return tp_fusedqkv_list


def require_tp_fused_qkvw(name, mp_size):
    fused_qkvw_name_list = ['qkv_proj', 'query_key_value', 'attn.Wqkv']

    if mp_size == 1:
        return False
    for fused_name in fused_qkvw_name_list:
        if fused_name in name:
            return True
    return False


def prepare_tp_fused_qkvw(module_str, src, mp_size, gpu_index):
    if src == None:
        return
    fused_type_dict = {
        'CodeGenBlock': 'codegentype',
        'BloomBlock': 'bloomtype',
        'GLMBlock': 'glmtype',
        "MPTBlock": 'glmtype',
        "MptBlock": 'glmtype',
    }

    def _codegen_type_transpose(input, mp_size, codegen_mp_num=4):
        # codegen_mp_num defined in https://github.com/huggingface/transformers/blob/main/src/transformers/models/codegen/modeling_codegen.py
        #TODO: assert num_heads % (mp_size*codegen_mp_num) == 0
        #input : [3*hidden_dim, hidden_dim](weight) or [3*hidden_dim](bias)

        shape = input.shape
        dst_shape = shape[0] // mp_size
        num_mp_blocks = input.reshape(codegen_mp_num, shape[0] // codegen_mp_num, shape[1])

        #num_mp_blocks : [codegen_mp_num, 3*hidden_dim/codegen_mp_num, :]
        src_split = list(torch.split(num_mp_blocks, num_mp_blocks.shape[1] // 3, dim=1))
        src_split = [x.reshape(codegen_mp_num * mp_size, -1, shape[1]) for x in src_split]

        split_fusedqkv = split_by_qkvlist_and_refuse(src_split, shape[0] // 3 // mp_size, 0, 1)
        tp_fuseqkv_weight = torch.cat(split_fusedqkv, dim=0).reshape(shape[0], -1)

        return tp_fuseqkv_weight[gpu_index * dst_shape:(gpu_index + 1) * dst_shape]

    def _glm_type_transpose(input, mp_size):
        #input : [3*hidden_dim, hidden_dim](weight) or [3*hidden_dim](bias)

        shape = input.shape
        dst_shape = shape[0] // mp_size
        src_split = torch.split(input, shape[0] // 3, dim=0)

        split_fusedqkv = split_by_qkvlist_and_refuse(src_split, shape[0] // 3 // mp_size)
        tp_fuseqkv_weight = torch.cat(split_fusedqkv, dim=0)

        return tp_fuseqkv_weight[gpu_index * dst_shape:(gpu_index + 1) * dst_shape]

    def _bloom_type_transpose(input, mp_size):
        shape = input.shape
        dst_shape = shape[0] // mp_size
        return input[gpu_index * dst_shape:(gpu_index + 1) * dst_shape]

    def _transpose_fused_qkvw(src, mp_size, fused_qkv_type=None):

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

        raise ValueError("unknown fused_qkv_type")

    for module_name, fused_type in fused_type_dict.items():
        if re.search(module_name, module_str):
            return _transpose_fused_qkvw(src, mp_size, fused_type)
    warning_once(f"Unrecognized fusedkqv weight type, default to using bloom type,"
                 f"please check in prepare_tp_fused_qkvw() to avoid potential calculation errors")
    return _bloom_type_transpose(src, mp_size)
