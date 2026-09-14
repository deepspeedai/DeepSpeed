# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from deepspeed.utils.logging import warning_once
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size, get_shard_size_list


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


FUSED_QKV_TYPE_DICT = {
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


def get_fused_qkv_type(module):
    module_str = str(module).strip()
    module_name_matches = [k for k in FUSED_QKV_TYPE_DICT.keys() if k in module_str]
    if not module_name_matches:
        return None
    # There can be overlap with matches (e.g., "DecoderLayer" and "FalconDecoderLayer").
    # We take the longest matching module_name
    return FUSED_QKV_TYPE_DICT[max(module_name_matches, key=len)]


def fused_qkv_subparam_sizes(module, weight_shape, meta: AutoTPMeta):
    """Sizes of the sub-parameters a fused qkv weight is cut into, or None.

    ``prepare_tp_fused_qkvw`` splits most layouts into q/k/v (or a single block) and
    concatenates this rank's piece of each, which is exactly the sub-parameter layout the
    gather and the checkpoint conversion need in order to undo the split. The remaining
    layouts interleave or replicate blocks instead, so they have no such description and
    return ``None``.
    """
    fused_type = get_fused_qkv_type(module)
    if fused_type is None:
        # prepare_tp_fused_qkvw falls back to the bloom layout for unrecognized modules.
        fused_type = 'bloomtype'
    total_size = weight_shape[0]
    if fused_type == 'bloomtype':
        return (total_size, )
    if fused_type in ('glmtype', 'qwentype'):
        if meta.num_kv_heads == 2:
            hidden_dim = meta.n_embd
            kv_dim = (total_size - hidden_dim) // meta.num_kv_heads
            return (hidden_dim, kv_dim, kv_dim)
        third = total_size // 3
        return (third, third, third)
    if fused_type == 'phi3type':
        head_dim = weight_shape[1] // meta.num_attention_heads
        kv_dim = meta.num_kv_heads * head_dim
        return (total_size - 2 * kv_dim, kv_dim, kv_dim)
    # codegentype interleaves blocks across ranks and bigcodetype replicates the kv block,
    # so neither is a per-sub-parameter split.
    return None


def set_fused_qkv_shard_state(module, shard_widths, tp_index):
    """Tell the model how wide this rank's query block is, for layouts that record it.

    QWen reads ``attn.split_size`` to unpack query, key and value out of the fused projection
    output, so it has to follow the frozen shard widths rather than the full model width.
    """
    if get_fused_qkv_type(module) != 'qwentype':
        return
    query_width = shard_widths[0][tp_index]
    if query_width == 0:
        # QWen unpacks three tensors from mixed_x_layer.split(self.attn.split_size), and a split
        # size of zero yields a single tensor instead. That model code lives outside DeepSpeed,
        # so an empty attention shard cannot be made to work here.
        raise RuntimeError(f"AutoTP cannot shard this QWen attention across {len(shard_widths[0])} ranks: "
                           f"rank {tp_index} would receive an empty query/key/value shard. Reduce the "
                           f"tensor parallel size so that every rank holds at least one query column.")
    module.attn.split_size = query_width


def prepare_tp_fused_qkvw(module, src, mp_size, gpu_index, meta: AutoTPMeta):

    if src is None:
        return

    def _codegen_type_transpose(input, mp_size, codegen_mp_num=4, meta=meta):
        # codegen_mp_num defined in https://github.com/huggingface/transformers/blob/main/src/transformers/models/codegen/modeling_codegen.py
        assert meta.num_kv_heads % (
            mp_size * codegen_mp_num) == 0, "codgen autoTP requires num_kv_heads % (mp_size*codegen_mp_num) == 0"
        #input : [3*hidden_dim, hidden_dim](weight) or [3*hidden_dim](bias)

        shape = input.shape
        dst_shape = get_shard_size(shape[0], mp_size, meta, rank=gpu_index)
        num_mp_blocks = input.reshape(codegen_mp_num, shape[0] // codegen_mp_num, shape[1])

        #num_mp_blocks : [codegen_mp_num, 3*hidden_dim/codegen_mp_num, :]
        src_split = list(torch.split(num_mp_blocks, num_mp_blocks.shape[1] // 3, dim=1))
        src_split = [x.reshape(codegen_mp_num * mp_size, -1, shape[1]) for x in src_split]

        split_fusedqkv = split_by_qkvlist_and_refuse(src_split,
                                                     get_shard_size(shape[0] // 3, mp_size, meta, rank=gpu_index), 0,
                                                     1)
        tp_fuseqkv_weight = torch.cat(split_fusedqkv, dim=0).reshape(shape[0], -1)

        return tp_fuseqkv_weight[gpu_index * dst_shape:(gpu_index + 1) * dst_shape]

    def _glm_type_transpose(input, mp_size, meta=meta):
        #input : [3*hidden_dim, hidden_dim](weight) or [3*hidden_dim](bias)

        # For chatglm2 & chatglm3(kv_heads=2), need to special handle.
        if meta.num_kv_heads == 2:
            shape = input.shape
            hidden_dim = meta.n_embd
            kv_dim = (shape[0] - hidden_dim) // meta.num_kv_heads
            q = input[:hidden_dim]
            k = input[hidden_dim:hidden_dim + kv_dim]
            v = input[hidden_dim + kv_dim:]
            q_split = q.split(get_shard_size_list(q.shape[0], mp_size, meta), dim=0)
            k_split = k.split(get_shard_size_list(k.shape[0], mp_size, meta), dim=0)
            v_split = v.split(get_shard_size_list(v.shape[0], mp_size, meta), dim=0)
            return torch.cat((q_split[gpu_index], k_split[gpu_index], v_split[gpu_index]), dim=0)
        else:
            shape = input.shape
            src_split = torch.split(input, shape[0] // 3, dim=0)

            split_fusedqkv = split_by_qkvlist_and_refuse(src_split, get_shard_size_list(shape[0] // 3, mp_size, meta))
            return split_fusedqkv[gpu_index]

    def _bloom_type_transpose(input, mp_size, meta=meta):
        shape = input.shape

        split_fusedqkv = input.split(get_shard_size_list(shape[0], mp_size, meta), dim=0)
        return split_fusedqkv[gpu_index]

    def _bigcode_type_transpose(input, mp_size, meta=meta):
        n_embd = meta.n_embd
        # A missing hidden size would slice as input[:None], handing the whole fused weight to q
        # and leaving kv empty rather than failing.
        assert n_embd is not None
        q = input[:n_embd]
        kv = input[n_embd:]
        shape = q.shape
        split_q = q.split(get_shard_size_list(shape[0], mp_size, meta), dim=0)
        return torch.cat((split_q[gpu_index], kv), dim=0)

    def _phi3_type_transpose(input, mp_size, meta=meta):
        num_kv_heads = meta.num_kv_heads
        num_heads = meta.num_attention_heads
        hidden_size = input.shape[1]
        head_dim = hidden_size // num_heads
        q_pos = input.shape[0] - 2 * num_kv_heads * head_dim
        q = input[:q_pos]
        k = input[q_pos:q_pos + num_kv_heads * head_dim]
        v = input[q_pos + num_kv_heads * head_dim:]
        split_q = q.split(get_shard_size_list(q.shape[0], mp_size, meta), dim=0)
        split_k = k.split(get_shard_size_list(k.shape[0], mp_size, meta), dim=0)
        split_v = v.split(get_shard_size_list(v.shape[0], mp_size, meta), dim=0)
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
            return _glm_type_transpose(src, mp_size)
        elif fused_qkv_type == 'bigcodetype':
            return _bigcode_type_transpose(src, mp_size)
        elif fused_qkv_type == 'phi3type':
            return _phi3_type_transpose(src, mp_size)

        raise ValueError("unknown fused_qkv_type")

    fused_type = get_fused_qkv_type(module)
    if fused_type is not None:
        return _transpose_fused_qkvw(src, mp_size, fused_type, module)
    warning_once("Unrecognized fusedkqv weight type, default to using bloom type,"
                 "please check in prepare_tp_fused_qkvw() to avoid potential calculation errors")
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
        shard_value,  # True -> shard_value; False -> shard_oproj
        meta: AutoTPMeta):
    if shard_value:
        total_size = weight.shape[0]
        weight_cat_dim = 0
    else:
        total_size = weight.shape[1]
        weight_cat_dim = 1
    num_heads = meta.num_kv_heads
    head_dim = total_size // num_heads
    assert (num_heads % world_size == 0)
    if world_size > num_heads // 2:
        raise RuntimeError(f"world_size {world_size} is larger than half of num_heads {num_heads}")
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
