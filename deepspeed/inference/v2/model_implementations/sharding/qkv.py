# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch

from .types import ShardingType
from .utils import shard_param, get_shard_endpoints


def shard_qkv_param(param: torch.Tensor,
                    shard_rank: int,
                    num_shards: int,
                    head_size: int,
                    n_heads_q: Optional[int] = None,
                    n_heads_kv: Optional[int] = None) -> Optional[torch.Tensor]:
    """
    Utility method for sharding a QKV parameter. Both biases and weights are supported. It is assumed
    that the layout of the parameter is such that all Q heads, all K heads, and all V heads
    are contiguous with respect to each other.

    Args:
        param (torch.Tensor): The parameter to shard.
        shard_rank (int): Which shard of the partitioned tensor to return.
        num_shards (int): The total number of shards the parameter is distributed across.
        head_size (int): The size of each head.
        n_heads_q (int): The number of query heads. This only needs to be passed if the number
             of query and key/value heads are different. If passed without n_heads_kv, default
             MHA partitioning will be used.
        n_heads_kv (int): The number of key/value heads. This only needs to be passed if the number
                of query and key/value heads are different. This argument should not be passed without
                n_heads_q (we want to explicitly opt into GQA sharding).
    """
    if n_heads_kv is not None and n_heads_q is None:
        raise ValueError("n_heads_kv should not be passed without n_heads_q")

    if n_heads_q is None:
        # Guaranteed to be in MHA
        if param.shape[0] // 3 % head_size != 0:
            raise ValueError("MHA param shape is not correct")
        n_heads_q = param.shape[0] // head_size // 3
        mha_sharding = True
    else:
        mha_sharding = n_heads_q == n_heads_kv

    if n_heads_q < num_shards:
        raise ValueError("There must be at least as many query heads as there are shards.")

    if mha_sharding:
        return shard_param(param,
                           ShardingType.OUTER_DIMENSION,
                           shard_rank,
                           num_shards,
                           num_concatenated_matrices=3,
                           granularity=head_size)
    else:
        if n_heads_q % n_heads_kv != 0:
            raise ValueError("Must be an even ratio between query and key/value heads.")

        if param.shape[0] != head_size * (n_heads_q + 2 * n_heads_kv):
            raise ValueError("GQA param shape is not correct")

        # 32 KV heads, 16 shards for example
        if n_heads_kv >= num_shards and n_heads_kv % num_shards != 0:
            raise ValueError("Currently do not support uneven partitioning of KV heads for GQA.")

        # 8 KV heads, 16 shards for example
        if n_heads_kv < num_shards and num_shards % n_heads_kv != 0:
            raise ValueError("Currently do not support distributing KV heads across different numbers of shards.")
        else:
            even_kv_sharding = n_heads_kv >= num_shards

        if param is None:
            return None

        q_param = param[:head_size * n_heads_q]
        kv_param = param[head_size * n_heads_q:]

        if even_kv_sharding:
            # This is equivalent to the original sharding algorithm since n_heads_q = C * n_heads_kv.
            # If n_heads_kv % num_shards == 0, then n_heads_q % num_shards == 0.
            q_param = shard_param(q_param, ShardingType.OUTER_DIMENSION, shard_rank, num_shards, granularity=head_size)
            kv_param = shard_param(kv_param,
                                   ShardingType.OUTER_DIMENSION,
                                   shard_rank,
                                   num_shards,
                                   num_concatenated_matrices=2,
                                   granularity=head_size)
            return torch.cat([q_param, kv_param], dim=0)
        else:
            # We will first do a sharding on the KV and Q to map to the one KV shard per group of Q.
            q_sharding_degree = num_shards // n_heads_kv

            kv_head = shard_rank // q_sharding_degree
            k_param = kv_param[kv_head * head_size:(kv_head + 1) * head_size]
            v_param = kv_param[(n_heads_kv + kv_head) * head_size:(n_heads_kv + kv_head + 1) * head_size]

            q_sharding_rank = shard_rank % q_sharding_degree
            q_factor = n_heads_q // n_heads_kv

            q_chunk = q_param[q_factor * kv_head * head_size:q_factor * (kv_head + 1) * head_size]

            q_param = shard_param(q_chunk,
                                  ShardingType.OUTER_DIMENSION,
                                  q_sharding_rank,
                                  q_sharding_degree,
                                  granularity=head_size)

            return torch.cat([q_param, k_param, v_param], dim=0)


def qkv_out_features(in_features: int,
                     shard_rank: int,
                     num_shards: int,
                     head_size: int,
                     n_heads_q: Optional[int] = None,
                     n_heads_kv: Optional[int] = None) -> int:
    """
    Helper to calculate the expected output projection dimension of a QKV projection matrix.

    Args:
        in_features (int): The model dimension.
        shard_rank (int): Which rank to return the corresponding size for.
        num_shards (int): The total number of shards the parameter is distributed across.
        head_size (int): The size of each head.
        n_heads_q (int): The number of query heads. This only needs to be passed if the number
             of query and key/value heads are different. If passed without n_heads_kv, default
             MHA partitioning will be used.
        n_heads_kv (int): The number of key/value heads. This only needs to be passed if the number
            of query and key/value heads are different. This argument cannot be passed without also
            passing n_heads_q (we want to explicitly opt into GQA sharding).
    """
    if n_heads_kv is not None and n_heads_q is None:
        raise ValueError("n_heads_kv should not be passed without n_heads_q")

    mha_sharding = n_heads_kv is None or n_heads_q == n_heads_kv

    if n_heads_q is not None and in_features != head_size * n_heads_q:
        raise ValueError("in_features is not consistent with n_heads_q and head_size")

    if mha_sharding:
        endpoints = get_shard_endpoints(in_features, shard_rank, num_shards, granularity=head_size)
        return (endpoints[1] - endpoints[0]) * 3
    else:
        if n_heads_kv >= num_shards:
            if n_heads_kv % num_shards != 0:
                raise ValueError("The KV heads must be evenly distributed across the shards.")

            n_local_groups = n_heads_kv // num_shards
            group_size = n_heads_q // n_heads_kv

            return n_local_groups * head_size * (2 + group_size)
        else:
            if num_shards % n_heads_kv != 0:
                raise ValueError("A shared KV head must always partition across the same number of shards.")

            q_split_degree = num_shards // n_heads_kv
            q_split_rank = shard_rank % q_split_degree
            split_granularity = (n_heads_q // n_heads_kv) * head_size

            q_endpoints = get_shard_endpoints(split_granularity, q_split_rank, q_split_degree, granularity=head_size)

            return (q_endpoints[1] - q_endpoints[0]) + 2 * head_size
