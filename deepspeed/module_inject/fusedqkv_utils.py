# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from deepspeed.utils.logging import warning_once
from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list, get_num_kv_heads, get_n_embd, get_num_attention_heads


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
        "Phi3DecoderLayer": "phi3type"
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

        # For chatglm2 & chatglm3(kv_heads=2), need to special handle.
        if get_num_kv_heads() == 2:
            shape = input.shape
            hidden_dim = get_n_embd()
            kv_dim = (shape[0] - hidden_dim) // get_num_kv_heads()
            q = input[:hidden_dim]
            k = input[hidden_dim:hidden_dim + kv_dim]
            v = input[hidden_dim + kv_dim:]
            q_split = q.split(get_shard_size_list(q.shape[0], mp_size), dim=0)
            k_split = k.split(get_shard_size_list(k.shape[0], mp_size), dim=0)
            v_split = v.split(get_shard_size_list(v.shape[0], mp_size), dim=0)
            return torch.cat((q_split[gpu_index], k_split[gpu_index], v_split[gpu_index]), dim=0)
        else:
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

    def _phi3_type_transpose(input, mp_size):
        num_kv_heads = get_num_kv_heads()
        num_heads = get_num_attention_heads()
        hidden_size = input.shape[1]
        head_dim = hidden_size // num_heads
        q_pos = input.shape[0] - 2 * num_kv_heads * head_dim
        q = input[:q_pos]
        k = input[q_pos:q_pos + num_kv_heads * head_dim]
        v = input[q_pos + num_kv_heads * head_dim:]
        split_q = q.split(get_shard_size_list(q.shape[0], mp_size), dim=0)
        split_k = k.split(get_shard_size_list(k.shape[0], mp_size), dim=0)
        split_v = v.split(get_shard_size_list(v.shape[0], mp_size), dim=0)
        return torch.cat((split_q[gpu_index], split_k[gpu_index], split_v[gpu_index]), dim=0)

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
        elif fused_qkv_type == 'phi3type':
            return _phi3_type_transpose(src, mp_size)

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


# For share qk type:
# q = [q1,...,q_{n/4}, q_{n/2+1},...,q_{3n/4}, k1,...,k_{n/4}, k_{n/2+1},...,k_{3n/4}]
# k = [q_{n/4+1},...,q_{n/2}, q_{3n/4+1},...,qn, k_{n/4+1},...,k_{n/2}, k{3n/4+1},...,kn]
# Avoid modifying the modeling code. We adjust the value and oproj weight to fit this qk type.
def shard_value_with_share_qk(
        weight,
        bias,
        rank,
        world_size,
        shard_value=True  # True -> shard_value; False -> shard_oproj
):
    if shard_value:
        total_size = weight.shape[0]
        weight_cat_dim = 0
    else:
        total_size = weight.shape[1]
        weight_cat_dim = 1
    num_heads = get_num_kv_heads()
    head_dim = total_size // num_heads
    assert (num_heads % world_size == 0)
    if world_size > num_heads // 2:
        RuntimeError(f"world_size {world_size} is larger than half of num_heads {num_heads}")
    head_per_rank = num_heads // world_size
    q_head_start = rank * head_per_rank
    # mapping q_head to v_head
    v_head_ids = []
    i = 0
    # mapping neighbor q_head to v_head
    while i < head_per_rank:
        v_head_ids.append(q_head_start // 2)
        q_head_start += 2
        i = i + 2

    # mapping neighbor k_head to v_head
    v_head_ids.extend([i + num_heads // 2 for i in v_head_ids])
    sharded_weight = []
    sharded_bias = []
    for head_id in v_head_ids:
        if shard_value:
            sharded_weight.append(weight[head_id * head_dim:(head_id + 1) * head_dim])
            if bias is not None:
                sharded_bias.append(bias.data[head_id * head_dim:(head_id + 1) * head_dim])
        else:
            sharded_weight.append(weight[:, head_id * head_dim:(head_id + 1) * head_dim])
    sharded_weight = torch.cat(sharded_weight, dim=weight_cat_dim)
    if bias is not None:
        if shard_value:
            sharded_bias = torch.cat(sharded_bias, dim=0)
        else:
            bias = bias / float(world_size)
        return torch.nn.Parameter(sharded_weight), torch.nn.Parameter(sharded_bias)
    else:
        return torch.nn.Parameter(sharded_weight), None


# For phi3 with chunk mlp, adjust the weight order.
def shard_chunk_mlp(
    weight,
    bias,
    rank,
    world_size,
):
    weight_gate, weight_states = weight.chunk(2, dim=0)
    total_size = weight_gate.shape[0]
    split_weight_gate = weight_gate.split(get_shard_size_list(total_size, world_size, "mlp"), dim=0)
    split_weight_states = weight_states.split(get_shard_size_list(total_size, world_size, "mlp"), dim=0)
    shard_weight = torch.cat((split_weight_gate[rank], split_weight_states[rank]), dim=0)
    if bias is not None:
        bias_gate, bias_states = bias.chunk(2, dim=0)
        split_bias_gate = bias_gate.split(get_shard_size_list(total_size, world_size, "mlp"), dim=0)
        split_bias_states = bias_states.split(get_shard_size_list(total_size, world_size, "mlp"), dim=0)
        return shard_weight, torch.cat((split_bias_gate[rank], split_bias_states[rank]), dim=0)

    return shard_weight, None
