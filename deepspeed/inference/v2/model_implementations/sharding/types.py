# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum

DEFAULT_SHARD_GRANULARITY = 32


class ShardingType(Enum):
    # Inner dimension sharding corresponds to splitting the Tensor along the K-dimension
    # of a matrix multiplication. This would be used for attention_output or MLP2.
    INNER_DIMENSION = 1

    # Outer dimension sharding corresponds to splitting the Tensor along the N-dimension
    # of a matrix multiplication. This would be used for the QKV and MLP1 projections.
    OUTER_DIMENSION = 0
