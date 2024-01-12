# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import comm as dist
global num_kv_heads


def set_num_kv_heads(num):
    global num_kv_heads
    num_kv_heads = num


def set_n_embd(num):
    global n_embd
    n_embd = num


def get_num_kv_heads():
    global num_kv_heads
    return num_kv_heads


def get_shard_size(total_size, mp_size, name=None, rank=None):
    global num_kv_heads
    last_linear = ["lm_head", "embed_out"]
    # When we have num_kv_heads defined, uneven division is possible, otherwise enforce near even division
    if rank == None:
        rank = dist.get_rank()
    if num_kv_heads != None and total_size % num_kv_heads == 0 and "mlp" not in str(name) and str(
            name) not in last_linear:
        my_slices = (num_kv_heads // mp_size) + (1 if rank < (num_kv_heads % mp_size) else 0)
        return total_size * my_slices // num_kv_heads
    else:
        if total_size >= 64:
            grain_size = total_size // 64
            return (grain_size // mp_size + (1 if rank < (grain_size % mp_size) else 0)) * 64
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
