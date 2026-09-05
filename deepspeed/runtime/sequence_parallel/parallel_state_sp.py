# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
This is a slimmed-down version of parallel_state.py (mpu) from Megatron-Deepspeed
"""

from deepspeed import comm as dist

# Sequence parallel groups to handle both data and sequence parallelisms.
# These groups are used to reduce gradients and shard parameters and optimizer stages for ZeRO.
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_DATA_PARALLEL_GROUP = None
_CREATED_SEQUENCE_PARALLEL_GROUPS = []


def initialize_sequence_parallel(sequence_parallel_size: int) -> None:
    """Initialize sequence parallel groups."""

    assert dist.is_initialized()
    world_size: int = dist.get_world_size()

    if world_size < sequence_parallel_size:
        raise RuntimeError(f"world_size ({world_size}) is less than sequence_parallel_size {sequence_parallel_size}")

    if sequence_parallel_size <= 1:
        raise ValueError(f"sequence_parallel_size must be greater than 1, got {sequence_parallel_size}")

    if world_size % sequence_parallel_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by sequence_parallel_size {sequence_parallel_size})")

    data_parallel_size: int = world_size // sequence_parallel_size
    sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    num_sequence_data_parallel_groups: int = world_size // sequence_parallel_size // data_parallel_size

    rank = dist.get_rank()

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_DATA_PARALLEL_GROUP
    global _CREATED_SEQUENCE_PARALLEL_GROUPS
    assert _SEQUENCE_PARALLEL_GROUP is None, "sequence parallel group is already initialized"
    assert _SEQUENCE_DATA_PARALLEL_GROUP is None, "sequence data parallel group is already initialized"
    assert not _CREATED_SEQUENCE_PARALLEL_GROUPS, "sequence parallel process groups are already initialized"

    created_groups = []
    try:
        for i in range(num_sequence_parallel_groups):
            ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
            group = dist.new_group(ranks)
            if rank in ranks:
                _SEQUENCE_PARALLEL_GROUP = group
                created_groups.append(group)

        # Build the sequence data parallel groups.
        for i in range(num_sequence_data_parallel_groups):
            ranks = range(i * sequence_data_parallel_size, (i + 1) * sequence_data_parallel_size)
            group = dist.new_group(ranks)
            if rank in ranks:
                _SEQUENCE_DATA_PARALLEL_GROUP = group
                created_groups.append(group)
    except Exception:
        if dist.is_initialized():
            for group in reversed(created_groups):
                dist.destroy_process_group(group)
        _SEQUENCE_PARALLEL_GROUP = None
        _SEQUENCE_DATA_PARALLEL_GROUP = None
        raise

    _CREATED_SEQUENCE_PARALLEL_GROUPS = created_groups


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, "sequence parallel group is not initialized"
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_data_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_DATA_PARALLEL_GROUP is not None, "sequence data parallel group is not initialized"
    return _SEQUENCE_DATA_PARALLEL_GROUP


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return dist.get_world_size(group=get_sequence_parallel_group())


def get_sequence_data_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return dist.get_world_size(group=get_sequence_data_parallel_group())


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return dist.get_rank(group=get_sequence_parallel_group())


def get_sequence_data_parallel_rank():
    """Return my rank for the sequence data parallel group."""
    return dist.get_rank(group=get_sequence_data_parallel_group())


def destroy_sequence_parallel() -> None:
    """Destroy process groups created for sequence parallelism and clear local state."""
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_DATA_PARALLEL_GROUP
    global _CREATED_SEQUENCE_PARALLEL_GROUPS

    created_groups = _CREATED_SEQUENCE_PARALLEL_GROUPS
    _CREATED_SEQUENCE_PARALLEL_GROUPS = []
    _SEQUENCE_PARALLEL_GROUP = None
    _SEQUENCE_DATA_PARALLEL_GROUP = None
    if dist.is_initialized():
        for group in reversed(created_groups):
            dist.destroy_process_group(group)


# since we only have 1 additional dimension over DP, we can just alias MP with SP
get_model_parallel_rank = get_sequence_parallel_rank
get_model_parallel_world_size = get_sequence_parallel_world_size
get_model_parallel_group = get_sequence_parallel_group
