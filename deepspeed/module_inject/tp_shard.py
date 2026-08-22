# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import comm as dist

# Defaults for optional TP globals. These can be overridden by setters.
num_kv_heads = None
num_attention_heads = None
n_embd = None
tp_grain_size = 1


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


def get_shard_size(total_size, mp_size, name=None, rank=None, mp_group=None, num_kv_heads=None):
    """Size of one shard of ``total_size`` split across a tensor-parallel group of ``mp_size``.

    ``rank`` is the rank *within the tensor-parallel group*, i.e. in ``[0, mp_size)``, matching
    ``dist.get_rank(group=mp_group)`` and the index used by ``get_shard_size_list``. It is not a
    global rank.

    ``num_kv_heads`` overrides the module-level global of the same name when set. Passing it
    explicitly lets callers split fused sub-parameters (Q/K/V) against their respective head
    counts without reimplementing the KV-head-aligned partition logic at the call site. When
    ``None`` (default), the function falls back to the global ``num_kv_heads`` set via
    ``set_num_kv_heads`` for backward compatibility.
    """
    if num_kv_heads is None:
        num_kv_heads = globals()["num_kv_heads"]
    last_linear = ["lm_head", "embed_out"]
    # MoE MLP layer use near even division will get better perf.
    moe_mlp_layer = ["gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
    not_moe_mlp_layer = True
    if name != None and any(s in str(name) for s in moe_mlp_layer):
        not_moe_mlp_layer = False
    # When we have num_kv_heads defined, uneven division is possible, otherwise enforce near even division
    if rank is None:
        if mp_group is not None:
            rank = dist.get_rank(group=mp_group)
        else:
            world_size = dist.get_world_size()
            if world_size != mp_size:
                raise ValueError("get_shard_size requires a group-local rank or process group when mp_size "
                                 f"({mp_size}) differs from the distributed world size ({world_size}).")
            rank = dist.get_rank()
    if num_kv_heads is not None and total_size % num_kv_heads == 0 and "mlp" not in str(name) and \
        str(name) not in last_linear and not_moe_mlp_layer:
        my_slices = (num_kv_heads // mp_size) + (1 if rank < (num_kv_heads % mp_size) else 0)
        return total_size * my_slices // num_kv_heads
    else:
        if total_size >= tp_grain_size:
            grain_size, remainder = divmod(total_size, tp_grain_size)
            shard_size = (grain_size // mp_size + (1 if rank < (grain_size % mp_size) else 0)) * tp_grain_size
            if rank == mp_size - 1:
                # Quantizing to tp_grain_size would otherwise drop total_size % tp_grain_size
                # and silently truncate the dimension. Giving that tail to the last rank keeps
                # every other rank aligned for the compute kernels.
                shard_size += remainder
            return shard_size
        else:
            return total_size // mp_size + (1 if rank < (total_size % mp_size) else 0)


def get_n_embd():
    global n_embd
    return n_embd


def get_shard_size_list(total_size, mp_size, name=None, num_kv_heads=None):
    shard_sizes = []
    if num_kv_heads is None:
        num_kv_heads = globals()["num_kv_heads"]
    for i in range(mp_size):
        shard_sizes.append(get_shard_size(total_size, mp_size, name, i, num_kv_heads=num_kv_heads))
    # Shards must tile the dimension exactly, otherwise the partitioned weights no longer
    # reconstruct the original tensor.
    assert sum(shard_sizes) == total_size, (
        f"AutoTP shard sizes {shard_sizes} for layer '{name}' do not sum to the dimension size "
        f"{total_size} with tp_size={mp_size}, tp_grain_size={tp_grain_size} and "
        f"num_kv_heads={num_kv_heads}.")
    return shard_sizes
