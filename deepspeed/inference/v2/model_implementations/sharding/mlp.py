# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import torch

from .types import ShardingType, DEFAULT_SHARD_GRANULARITY
from .utils import shard_param, get_shard_endpoints


def shard_mlp_1_param(param: torch.Tensor,
                      shard_rank: int,
                      num_shards: int,
                      gated: bool = False,
                      is_moe: bool = False) -> torch.Tensor:
    """
    Utility method for sharding an MLP 1 parameter. Both biases and weights are supported, as well
    as for fused weights for MoE.

    Args:
        param (torch.Tensor): The parameter to shard.
        shard_rank (int): Which shard of the partitioned tensor to return.
        num_shards (int): The total number of shards the parameter is distributed across.
        gated (bool): Whether or not the parameter is from a gated MLP.
    """
    bias_dims = 2 if is_moe else 1

    if gated:
        return shard_param(param,
                           ShardingType.OUTER_DIMENSION,
                           shard_rank,
                           num_shards,
                           granularity=DEFAULT_SHARD_GRANULARITY * 2,
                           bias_dims=bias_dims)
    else:
        return shard_param(param, ShardingType.OUTER_DIMENSION, shard_rank, num_shards, bias_dims=bias_dims)


def shard_mlp_2_param(param: torch.Tensor,
                      shard_rank: int,
                      num_shards: int,
                      is_moe: bool = False) -> Optional[torch.Tensor]:
    """
    Utility method for sharding an MLP 2 parameter.

    Args:
        param (torch.Tensor): The parameter to shard.
        shard_rank (int): Which shard of the partitioned tensor to return.
        num_shards (int): The total number of shards the parameter is distributed across.
        is_moe (bool): Whether or not the parameter is from a MoE model.
    """
    bias_dim_size = 2 if is_moe else 1

    if len(param.shape) == bias_dim_size:
        # We will do the bias addition on the 0th rank only rather than scale the parameter and
        # implicitly reconstruct this in the distributed reduce.
        return param if shard_rank == 0 else None

    return shard_param(param, ShardingType.INNER_DIMENSION, shard_rank, num_shards)


def sharded_intermediate_dim(intermediate_size: int, num_shards: int, shard_rank: int) -> int:
    """
    Utility method for getting the size of the intermediate dimension of a sharded MLP.

    Args:
        intermediate_size (int): The size of the intermediate dimension.
        num_shards (int): The total number of shards the parameter is distributed across.
        shard_rank (int): Which shard of the partitioned tensor to return.
    """
    endpoints = get_shard_endpoints(intermediate_size, shard_rank, num_shards)
    return endpoints[1] - endpoints[0]
