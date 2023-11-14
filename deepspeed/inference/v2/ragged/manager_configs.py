# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from enum import Enum
from typing import Tuple

from deepspeed.pydantic_v1 import PositiveInt, validator

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from ..inference_utils import DtypeEnum


class KVCacheType(Enum):

    DENSE = "dense"
    """
    Dense KV-cache. This is the default type.
    """

    LOCAL = "local"
    """
    KV-cache that attends to only a local (trailing) window of tokens.
    """


class KVCacheConfig(DeepSpeedConfigModel):

    type: KVCacheType = KVCacheType.DENSE
    """
    Type of KV-cache to use. This may inform the allocator of the expected access/retention pattern
    to enable more efficient memory management.
    """

    block_size: int = 128
    """
    Number of tokens that may be contained in each cache block.
    """

    num_allocation_groups: PositiveInt = 1
    """
    Allocation groups are assumed to be able to use the same allocation block size because
    the allocation granularity is the same but the number of blocks required in each group
    may differ.

    As a concrete example, consider a model with alternating layers of local and global
    attention (such as GPTNeo). The local attention layers do not require the same number
    of cache blocks as the global layer. However, a static partitioning scheme is sub-optimal since the ratio of local to global KV-cache blocks is not constant across
    the range of sequence lengths that may be encountered.

    NOTE: In theory, this functionality could be used to do per-head and per-layer
    KV-cache allocation, but it is likely the allocator will struggle with managing that
    many blocks.

    NOTE: This will need to be primarily understood and handled by the model implementation
    itself, rather than the KV cache manager. However, I'd like to make this explicit.
    """

    cache_shape: Tuple[PositiveInt, PositiveInt, PositiveInt]
    """
    The shape of the cache per token. The first dimension is the number of individual
    caches, the second is the number of heads, and the third is the head size. The number
    of caches argument here is per allocation group.
    """

    cache_dtype: DtypeEnum = DtypeEnum.fp16
    """
    Data type of the KV-cache.
    """

    max_blocks_per_allocation_group: PositiveInt = 64
    """
    Maximum number of blocks that can be associated with an allocation group.
    """


"""
The config above is a little confusing so let's use a couple of concrete examples of
usage:

Model 1: Llama-13B with a block size of 256

Llama is uniform attention so we have a single allocation group. The cache shape is
(40 layers, 40 heads, 128 head size)

```python
llama_kv_config = KVCacheConfig(block_size=256,
                                num_allocation_groups=1,
                                cache_shape=(40, 40, 128))
```

Model 2: GPTNeo-2.7B with a block size of 128

GPTNeo has alternating local and global attention layers. We have two allocation groups.
There are 16 layers of each type with 20 heads apiece at 128 head size.

```python
gptneo_kv_config = KVCacheConfig(num_allocation_groups=2, cache_shape=(16, 20, 128))
```
"""


class AllocationMode(Enum):
    """
    Helper class to describe memory allocation strategies for the KV-cache.
    """

    RESERVE = "reserve"
    """
    Reserve a small amount of memory for non-KV cache allocations.
    """

    ALLOCATE = "allocate"
    """
    Allocate an explicit number of KV blocks.
    """


class MemoryConfig(DeepSpeedConfigModel):

    mode: AllocationMode = AllocationMode.RESERVE

    size: PositiveInt = 1_000_000_000
    """
    Parameter for each of the modes.

    If mode is RESERVE, this is the amount of memory in bytes to reserve after allocating the
    KV-cache. If in a tensor-parallel regime, this amount is guaranteed to be reserved on
    all devices.

    If mode is ALLOCATE, this is the number of blocks to allocate for the KV-cache. This may
    require tuning for model/GPU setups.
    """


class DSStateManagerConfig(DeepSpeedConfigModel):

    max_tracked_sequences: PositiveInt = 2048
    """
    How many sequences this engine will track simultaneously. This limit should be greater
    than the ``max_ragged_sequence_count``.
    """

    max_ragged_batch_size: PositiveInt = 768
    """
    The maximum number of tokens that can be contained in a single ragged batch. Passing
    a larger value than this will raise an exception that must be handled by the runtime.
    """

    max_ragged_sequence_count: PositiveInt = 512
    """
    The maximum number of sequences that can compose a batch. This limitation is only
    relevant under CUDA graphing scenarios currently, where the maximum number of blocks
    is largely bound by the total number of sequences in the ragged batch. This number cannot
    be larger than ``max_tracked_sequences`` or ``max_ragged_batch_size``.
    """

    max_context: PositiveInt = 8192
    """
    The maximum number of tokens (inclusive of generation) that can be contained in a single
    sequence. Currently used to bound the size of the KV cache metadata.
    """

    memory_config: MemoryConfig = MemoryConfig()
    """
    Directive for how to manage the creation of the KV-cache. See MemoryConfig for more
    details.
    """

    offload: bool = False
    """
    Enable tracking for offloading KV-cache to host memory. Currently unsupported.
    """

    @validator("max_ragged_sequence_count")
    def max_ragged_sequence_count_validator(cls, v: int, values: dict):
        # If the attributes below failed their validation they won't appear in the values dict.
        if "max_tracked_sequences" in values and v > values["max_tracked_sequences"]:
            raise ValueError("max_ragged_sequence_count must be less than max_tracked_sequences")
        if "max_ragged_batch_size" in values and v > values["max_ragged_batch_size"]:
            raise ValueError("max_ragged_sequence_count must be less than max_ragged_batch_size")
        return v
