# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import comm as dist
global num_kv_heads


def set_num_kv_heads(num):
    global num_kv_heads
    num_kv_heads = num


def get_num_kv_heads():
    global num_kv_heads
    return num_kv_heads


def get_shard_size(total_size, mp_size, rank=None):
    global num_kv_heads
    # When we have num_kv_heads defined, uneven division is possible, otherwise enforce even division
    if num_kv_heads != None:
        if (rank == None):
            rank = dist.get_rank()
        my_slices = (num_kv_heads // mp_size) + (1 if rank < (num_kv_heads % mp_size) else 0)
        return total_size * my_slices // num_kv_heads
    else:
        if total_size % mp_size == 0:
            return total_size // mp_size
        else:
            assert False, f"Number of attention heads ({total_size}) must be divisible by mp_size ({mp_size})"


def get_shard_size_list(total_size, mp_size):
    shard_sizes = []
    for i in range(mp_size):
        shard_sizes.append(get_shard_size(total_size, mp_size, i))
    return shard_sizes
