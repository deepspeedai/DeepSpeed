# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch

from .types import ShardingType
from .utils import shard_param, get_shard_endpoints


def shard_attn_out_param(param: torch.Tensor,
                         shard_rank: int,
                         num_shards: int,
                         head_size: int,
                         n_heads_q: Optional[int] = None,
                         n_heads_kv: Optional[int] = None) -> Optional[torch.Tensor]:
    """
    Utility method for sharding an attention output parameter.
    """
    if len(param.shape) == 1:
        # We will do the bias addition on the 0th rank only rather than scale the parameter and
        # implicitly reconstruct this in the distributed reduce.
        return param if shard_rank == 0 else None

    assert n_heads_kv is None or (n_heads_q is not None
                                  and n_heads_kv is not None), "n_heads_kv should not be passed without n_heads_q"

    mha_sharding = n_heads_kv is None or n_heads_q == n_heads_kv

    if mha_sharding:
        return shard_param(param, ShardingType.INNER_DIMENSION, shard_rank, num_shards, granularity=head_size)
    else:
        assert param.shape[0] == head_size * n_heads_q, "GQA param shape is not correct"

        # 32 KV heads, 16 shards for example
        even_kv_sharding = n_heads_kv % num_shards == 0

        # 8 KV heads, 16 shards for example
        even_kv_distribution = num_shards % n_heads_kv == 0

        assert even_kv_sharding or even_kv_distribution, "No partitioning algorithm for this yet."

        if even_kv_sharding:
            # Same as original sharding scenario
            return shard_param(param, ShardingType.INNER_DIMENSION, shard_rank, num_shards, granularity=head_size)
        else:
            # We will first do a sharding on the KV and Q to map to the one KV shard per group of Q.
            q_sharding_degree = num_shards // n_heads_kv

            kv_head = shard_rank // q_sharding_degree

            q_sharding_rank = shard_rank % q_sharding_degree
            q_factor = n_heads_q // n_heads_kv

            q_chunk = param[..., q_factor * kv_head * head_size:q_factor * (kv_head + 1) * head_size]

            return shard_param(q_chunk,
                               ShardingType.INNER_DIMENSION,
                               q_sharding_rank,
                               q_sharding_degree,
                               granularity=head_size)


def attn_out_in_features(out_features: int,
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
        head_size (int): The size of each attention head.
        n_heads_q (int): The number of query heads on the model. This only needs to be passed if the number
            of query and key/value heads are different. If passed without n_heads_kv, default
            MHA partitioning will be used.
        n_heads_kv (int): The number of key and value heads on the model. This only needs to be passed
            if the number of query and key/value heads are different. This argument cannot be passed without
            also passing n_heads_q (we want to explicitly opt into GQA sharding).
    """
    assert n_heads_kv is None or (n_heads_q is not None
                                  and n_heads_kv is not None), "n_heads_kv should not be passed without n_heads_q"

    mha_sharding = n_heads_kv is None or n_heads_q == n_heads_kv

    if mha_sharding:
        endpoints = get_shard_endpoints(out_features, shard_rank, num_shards, granularity=head_size)
        return endpoints[1] - endpoints[0]
    else:
        if n_heads_kv >= num_shards:
            assert n_heads_kv % num_shards == 0, "No partitioning algorithm for this yet."
            n_local_groups = n_heads_kv // num_shards
            group_size = n_heads_q // n_heads_kv

            return n_local_groups * head_size * group_size
        else:
            assert num_shards % n_heads_kv == 0, "No partitioning algorithm for this yet."
            q_split_degree = num_shards // n_heads_kv
            q_split_rank = shard_rank % q_split_degree
            split_granularity = (n_heads_q // n_heads_kv) * head_size

            q_endpoints = get_shard_endpoints(split_granularity, q_split_rank, q_split_degree, granularity=head_size)

            return q_endpoints[1] - q_endpoints[0]
