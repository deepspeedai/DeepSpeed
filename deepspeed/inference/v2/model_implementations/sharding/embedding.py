# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .types import ShardingType
from .utils import shard_param, get_shard_endpoints


def shard_embedding_param(param: torch.Tensor, shard_rank: int, num_shards: int) -> torch.Tensor:
    """
    Utility method for sharding an embedding parameter.

    Args:
        param (torch.Tensor): The parameter to shard. Should be of shape [vocab_size, model_dim]
        shard_rank (int): Which shard of the partitioned tensor to return.
        num_shards (int): The total number of shards the parameter is distributed across.
    """
    return shard_param(param, ShardingType.INNER_DIMENSION, shard_rank, num_shards)


def sharded_embedding_dim(embedding_size: int, shard_rank: int, num_shards: int) -> int:
    """
    Utility method for getting the size of the embedding dimension of a sharded embedding.

    Args:
        embedding_size (int): The size of the embedding.
        shard_rank (int): Which shard of the partitioned tensor to return.
        num_shards (int): The total number of shards the parameter is distributed across.
    """
    start_idx, end_idx = get_shard_endpoints(embedding_size, shard_rank, num_shards)
    return end_idx - start_idx
