# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import comm as dist
global num_kv_heads


def set_num_kv_heads(num):
    global num_kv_heads
    num_kv_heads = num


def set_num_attention_heads(num):
    global num_attention_heads
    num_attention_heads = num


def set_n_embd(num):
    global n_embd
    n_embd = num


def set_tp_grain_size(num):
    global tp_grain_size
    tp_grain_size = num


def get_num_kv_heads():
    global num_kv_heads
    if 'num_kv_heads' in globals():
        return num_kv_heads
    return None


def get_num_attention_heads():
    global num_attention_heads
    return num_attention_heads


def get_shard_size(total_size, mp_size, name=None, rank=None):
    global num_kv_heads
    last_linear = ["lm_head", "embed_out"]
    # MoE MLP layer use near even division will get better perf.
    moe_mlp_layer = ["gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
    not_moe_mlp_layer = True
    if name != None and any(s in str(name) for s in moe_mlp_layer):
        not_moe_mlp_layer = False
    # When we have num_kv_heads defined, uneven division is possible, otherwise enforce near even division
    if rank == None:
        rank = dist.get_rank()
    if num_kv_heads != None and total_size % num_kv_heads == 0 and "mlp" not in str(name) and str(
            name) not in last_linear and not_moe_mlp_layer:
        my_slices = (num_kv_heads // mp_size) + (1 if rank < (num_kv_heads % mp_size) else 0)
        return total_size * my_slices // num_kv_heads
    else:
        if total_size >= tp_grain_size:
            grain_size = total_size // tp_grain_size
            return (grain_size // mp_size + (1 if rank < (grain_size % mp_size) else 0)) * tp_grain_size
        else:
            return total_size // mp_size + (1 if rank < (total_size % mp_size) else 0)


def get_n_embd():
    global n_embd
    return n_embd


def get_shard_size_list(total_size, mp_size, name=None):
    shard_sizes = []
    for i in range(mp_size):
        shard_sizes.append(get_shard_size(total_size, mp_size, name, i))
    return shard_sizes
