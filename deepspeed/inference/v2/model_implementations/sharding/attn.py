# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional, Tuple


def get_local_heads(shard_rank: int,
                    num_shards: int,
                    n_heads_q: int,
                    n_heads_kv: Optional[int] = None) -> Tuple[int, int]:
    """
    Helper to determine the number of local heads of a given shard.

    Args:
        shard_rank (int): The rank of the shard.
        num_shards (int): The total number of shards that attention is distributed over.
        n_heads_q (int): The number of query heads.
        n_heads_kv (int): The number of key/value heads. If not passed, it is assumed that
            the number of query and key/value heads are the same.
    """
    if n_heads_q < num_shards:
        raise ValueError("There must be at least as many attention heads as there are shards.")

    if n_heads_kv is None or n_heads_kv == n_heads_q:
        # MHA attention
        base_heads = n_heads_q // num_shards
        extra_heads = n_heads_q % num_shards

        if shard_rank < extra_heads:
            return (base_heads + 1), (base_heads + 1)
        else:
            return base_heads, base_heads
    else:
        # GQA attention
        if n_heads_q % n_heads_kv != 0:
            raise ValueError("Must be an even ratio between query and key/value heads.")

        if n_heads_kv < num_shards and num_shards % n_heads_kv != 0:
            raise ValueError(
                "If splitting a group across multiple shards, we must be able to distribute the groups evenly.")

        if n_heads_kv >= num_shards and n_heads_kv % num_shards != 0:
            raise ValueError("If parallelizing groups, must be able to evenly distribute them.")

        q_ratio = n_heads_q // n_heads_kv

        if n_heads_kv >= num_shards:
            local_kv_heads = n_heads_kv // num_shards
            local_q_heads = local_kv_heads * q_ratio
            return local_q_heads, local_kv_heads
        else:
            group_sharding_size = num_shards // n_heads_kv
            group_rank_idx = shard_rank % group_sharding_size

            base_heads = q_ratio // group_sharding_size
            extra_heads = q_ratio % group_sharding_size

            if group_rank_idx < extra_heads:
                return (base_heads + 1), 1
            else:
                return base_heads, 1
