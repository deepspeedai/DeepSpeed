# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional, Tuple

import torch

from .types import ShardingType, DEFAULT_SHARD_GRANULARITY


def get_shard_endpoints(dim_size: int,
                        shard_rank: int,
                        num_shards: int,
                        granularity: int = DEFAULT_SHARD_GRANULARITY) -> Tuple[int, int]:
    """
    Given a dimension to shard with size dim_size, return the start and end indices of the slice
    that belong to the given rank.

    The typical use of this is as an internal helper function, so see if there is a higher level
    API that better suits the application.

    Args:
        dim_size (int): The size of the dimension to shard.
        shard_rank (int): The rank of the shard to return.
        num_shards (int): Total number of shards the dimension will be distributed across.
        granularity (int): The minimum alignment of the shard endpoints. This is used to support
            non-even head counts as well as align dimensions to cleaner GEMM boundaries.
    """
    assert dim_size % granularity == 0, "Dimension size must be divisible by granularity"

    total_chunks = dim_size // granularity
    base_chunks_per_rank = total_chunks // num_shards
    remainder_chunks = total_chunks % num_shards

    start_chunk_id = shard_rank * base_chunks_per_rank + min(shard_rank, remainder_chunks)
    end_chunk_id = start_chunk_id + base_chunks_per_rank + (1 if shard_rank < remainder_chunks else 0)

    return start_chunk_id * granularity, end_chunk_id * granularity


def shard_param(param: Optional[torch.Tensor],
                shard_mode: ShardingType,
                shard_rank: int,
                num_shards: int,
                num_concatenated_matrices: int = 1,
                granularity: int = 32,
                bias_dims: int = 1) -> torch.Tensor:
    """
    Utility for sharding a parameter. This will return the slice of the parameter that should
    exist on the given shard_rank given the sharding configuration. The workflow here is
    to find the minimum bounded Tensor to shard, get the slicing endpoints, and then concatenate
    as needed.

    The typical use of this is as an internal helper function, so see if there is a higher level
    API that better suits the application.

    Args:
        param (torch.Tensor): The parameter to shard.
        shard_mode (ShardingType): The type of sharding to apply. See ShardingType for more context.
        shard_rank (int): The rank of the shard to return.
        num_shards (int): Total number of shards the parameter will be distrbuted across.
        num_concatenated_matrices (int): The number of matrices that have been concatenated together in the original
            parameter. An example of this is a fused QKV projection matrix, where the `num_concatenated_matrices`
            argument would be 3.
        granularity (int): The minimum alignment of the shard endpoints. For attention projection matrices, this
            should be set to the head size to support non-even sharding.
        bias_dims (int): The number of dimensions that are considered bias dimensions. This is used to support
            sharding of MoE and non-MoE biases on the same codepath.
    """
    assert shard_rank < num_shards, "Shard rank must be less than num_shards"

    # Easier to hide this inside of the sharding logic than to add checks in every model
    # implementation.
    if param is None:
        return None

    if num_shards == 1:
        # Trivial case of no sharding.
        return param

    if shard_mode == ShardingType.OUTER_DIMENSION:

        def get_matrices(dim_idx: int) -> torch.Tensor:
            dim_size = param.size(dim_idx) // num_concatenated_matrices
            start_channel_id, end_channel_id = get_shard_endpoints(dim_size, shard_rank, num_shards, granularity)
            return torch.chunk(param, num_concatenated_matrices, dim=dim_idx), start_channel_id, end_channel_id

        if param.ndim == bias_dims:
            # Special case for bias parameters.
            matrices, start_channel_id, end_channel_id = get_matrices(dim_idx=-1)
            return torch.cat([mat[..., start_channel_id:end_channel_id] for mat in matrices], dim=-1)
        else:
            # General case for weight parameters. This assumes MoE parameters are stored in the format of
            # [num_experts, out_features, in_features]
            matrices, start_channel_id, end_channel_id = get_matrices(dim_idx=-2)
            return torch.cat([mat[..., start_channel_id:end_channel_id, :] for mat in matrices], dim=-2)

    elif shard_mode == ShardingType.INNER_DIMENSION:
        dim_size = param.size(-1) // num_concatenated_matrices
        start_channel_id, end_channel_id = get_shard_endpoints(dim_size, shard_rank, num_shards, granularity)
        matrices = torch.chunk(param, num_concatenated_matrices, dim=-1)
        return torch.cat([mat[..., start_channel_id:end_channel_id] for mat in matrices], dim=-1)
