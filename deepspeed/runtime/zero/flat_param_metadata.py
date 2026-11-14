# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Metadata describing the original parameters behind a flat optimizer tensor.

ZeRO-1/2 (including the BF16 optimizer wrapper) present each optimizer
parameter group as a single flat local FP32 partition. The metadata in this
module lets custom optimizers recover the original parameter shapes, tensor
boundaries, and global offsets without changing that flat layout.
"""

from dataclasses import dataclass
from typing import Tuple

import torch

__all__ = [
    "ZeROFlatParamMetadata",
    "ZeROFlatPartitionMetadata",
    "get_zero_flat_partition_metadata",
]


@dataclass(frozen=True)
class ZeROFlatParamMetadata:
    """Describes one original parameter inside a ZeRO-1/2 flat partition.

    Attributes:
        original_index: Index of the parameter in the optimizer group's
            original (pre-round-robin) parameter order.
        flat_index: Index of the parameter in the flat, round-robin reordered
            group.
        shape: Original parameter shape.
        numel: Number of elements in the original parameter.
        flat_offset: Offset of the parameter relative to the start of the
            padded flat parameter group.
        padding: Alignment padding inserted after this parameter in the flat
            group.
        local_offset: Offset of the parameter relative to the start of the
            local optimizer partition. ``-1`` when the parameter does not
            overlap this rank's partition.
        local_numel: Number of parameter elements that overlap the local
            partition.
        requires_grad: Whether the original parameter required gradients.
    """

    original_index: int
    flat_index: int
    shape: torch.Size
    numel: int
    flat_offset: int
    padding: int
    local_offset: int
    local_numel: int
    requires_grad: bool


@dataclass(frozen=True)
class ZeROFlatPartitionMetadata:
    """Original parameter layout behind a ZeRO-1/2 optimizer flat partition.

    ZeRO-1/2 replace each optimizer parameter group with one flat FP32 master
    partition. This metadata lets custom optimizers recover the original tensor
    shapes, boundaries, and global offsets without changing the flat layout.

    Attributes:
        group_index: Index of the optimizer parameter group.
        global_numel: Number of elements in the padded flat parameter group,
            including alignment padding.
        partition_start: Global offset of this rank's local partition inside
            the flat parameter group.
        partition_numel: Number of elements in this rank's local partition.
        dp_group: Data parallel process group that owns the group.
        dp_world_size: Size of ``dp_group``.
        params: Per-parameter layout descriptions ordered by ``flat_index``.
    """

    group_index: int
    global_numel: int
    partition_start: int
    partition_numel: int
    dp_group: object
    dp_world_size: int
    params: Tuple[ZeROFlatParamMetadata, ...]


def get_zero_flat_partition_metadata(param):
    """Return the original-layout metadata for a ZeRO-1/2 flat optimizer parameter.

    Args:
        param (``torch.Tensor``): A parameter from a ZeRO-1/2 optimizer param
            group. Both the standard ZeRO optimizer and the BF16 optimizer
            wrapper expose this metadata.

    Returns:
        ``ZeROFlatPartitionMetadata`` or ``None`` if ``param`` is not a
        ZeRO-1/2 flat partition.

    Example:
        >>> from deepspeed.runtime.zero.flat_param_metadata import (
        ...     get_zero_flat_partition_metadata)
        >>> for group in optimizer.param_groups:
        ...     for param in group["params"]:
        ...         meta = get_zero_flat_partition_metadata(param)
        ...         if meta is not None:
        ...             for p in meta.params:
        ...                 print(p.original_index, tuple(p.shape), p.local_offset)
    """
    return getattr(param, "ds_zero_flat_partition_metadata", None)
