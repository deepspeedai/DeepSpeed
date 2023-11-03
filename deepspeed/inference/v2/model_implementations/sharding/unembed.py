# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from .types import ShardingType
from .utils import shard_param, get_shard_endpoints


def shard_unembed_param(param: torch.Tensor, shard_rank: int, num_shards: int) -> torch.Tensor:
    """
    Utility method for sharding an unembed parameter. We shard unembeddings on the vocab dimension
    with the expectation of an all-gather to produce the full results.

    TODO(cmikeh2): Really ideal would be if MII could have access to the comm and we would do
    an A2A and sharded sampling.

    Args:
        param (torch.Tensor): The parameter to shard. Should be of shape [vocab_size, model_dim]
        shard_rank (int): Which shard of the partitioned tensor to return.
        num_shards (int): The total number of shards the parameter is distributed across.

    Returns:
        torch.Tensor: The sharded parameter of shape [sharded_vocab_size, model_dim]
    """
    return shard_param(param, ShardingType.OUTER_DIMENSION, shard_rank, num_shards, granularity=1)


def sharded_unembed_dim(vocab_size: int, shard_rank: int, num_shards: int) -> int:
    """
    Utility method for determining the sharded vocab size of a sharded unembed parameter.

    Args:
        vocab_size (int): The size of the vocabulary.
        shard_rank (int): Which shard of the partitioned tensor to return.
        num_shards (int): The total number of shards the parameter is distributed across.
    """
    start_idx, end_idx = get_shard_endpoints(vocab_size, shard_rank, num_shards, granularity=1)
    return end_idx - start_idx
