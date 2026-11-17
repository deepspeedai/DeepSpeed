# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Region copies for prepared affine transfers.

A prepared transfer (``prepare_transfers``) runs a list of ``(destination, source)`` view copies per
round: the packs into the send arenas before a round, the unpacks out of the receive arenas after it,
and the copies that stay on a process. ``prepare_transfers(..., copier=...)`` turns each list into one
callable once, at prepare time. ``sequential_copies`` is the default and runs one ``copy_`` per region.
``fused_copies`` describes the list as a table of tiles and copies every tile of the list with one
Triton kernel launch, which removes the per-region launch and Python cost a round otherwise pays.

The kernel copies bits, so it serves every dtype, and it takes regions whose last dimension is
contiguous with at most one outer stride. Other views, host tensors, and devices without Triton keep
``copy_``. Triton is imported only when ``fused_copies`` first builds a table on such a device.
"""

from typing import Callable, Sequence, Tuple

import torch

__all__ = ['sequential_copies', 'fused_copies']

CopyList = Sequence[Tuple[torch.Tensor, torch.Tensor]]

# Elements one program copies at most. Long regions are cut into tiles of this size so that one program
# never copies a large region alone while the rest of the device idles.
TILE_ELEMENTS = 1 << 15
_ALIGN_BYTES = 16


def _nothing() -> None:
    pass


def sequential_copies(pairs: CopyList) -> Callable[[], None]:
    """One ``copy_`` per region, in order."""
    pairs = tuple(pairs)
    if not pairs:
        return _nothing

    def run() -> None:
        for destination, source in pairs:
            destination.copy_(source)

    return run


def fused_copies(pairs: CopyList) -> Callable[[], None]:
    """Every region of ``pairs`` in one kernel launch (per alignment class), where the device allows it.

    The table holds the regions' addresses, so it is valid as long as the views' storage stays where it
    is, which a prepared transfer already requires of its buffers and checks before each run.
    """
    pairs = tuple(pairs)
    if not pairs:
        return _nothing
    if not _triton_usable(pairs[0][0]):
        return sequential_copies(pairs)
    return _TileTable(pairs).launch


def _triton_usable(tensor: torch.Tensor) -> bool:
    import deepspeed
    return tensor.device.type in ('cuda', 'xpu') and deepspeed.HAS_TRITON


def _rows(view: torch.Tensor):
    """``(rows, columns, row stride)`` for a view made of equally spaced contiguous rows, or None."""
    shape = [size for size in view.shape if size != 1]
    strides = [stride for size, stride in zip(view.shape, view.stride()) if size != 1]
    if not shape:
        return 1, 1, 1
    if strides[-1] != 1:
        return None
    columns = shape[-1]
    index = len(shape) - 2
    while index >= 0 and strides[index] == columns:
        columns *= shape[index]
        index -= 1
    if index < 0:
        return 1, columns, columns
    if index > 0:
        return None
    return shape[0], columns, strides[0]


def _signed(address: int) -> int:
    """An address as the int64 with the same bits; device addresses on XPU can have the top bit set."""
    return address - (1 << 64) if address >= 1 << 63 else address


class _TileTable:
    """A copy list as device-resident tile tables, one per alignment class, plus the regions it can't tile."""

    def __init__(self, pairs: CopyList):
        import triton
        from ._affine_copy_kernel import copy_tiles
        self.kernel = copy_tiles
        device = pairs[0][0].device
        columns_by_class = {True: [[] for _ in range(6)], False: [[] for _ in range(6)]}
        self.fallback = []
        element = None
        for destination, source in pairs:
            if element is None:
                element = source.element_size()
            target_rows, source_rows = _rows(destination), _rows(source)
            # A side that is one contiguous run follows the other side's rows.
            if (target_rows is not None and source_rows is not None and target_rows[0] == 1
                    and source_rows[0] * source_rows[1] == target_rows[1]):
                target_rows = (source_rows[0], source_rows[1], source_rows[1])
            if (target_rows is not None and source_rows is not None and source_rows[0] == 1
                    and target_rows[0] * target_rows[1] == source_rows[1]):
                source_rows = (target_rows[0], target_rows[1], target_rows[1])
            if (target_rows is None or source_rows is None or target_rows[:2] != source_rows[:2]
                    or destination.device != device or source.device != device or source.element_size() != element
                    or destination.element_size() != element):
                self.fallback.append((destination, source))
                continue
            rows, columns = target_rows[0], target_rows[1]
            vector = _ALIGN_BYTES // element
            aligned = (source.data_ptr() % _ALIGN_BYTES == 0 and destination.data_ptr() % _ALIGN_BYTES == 0
                       and (rows == 1 or (source_rows[2] % vector == 0 and target_rows[2] % vector == 0)))
            width = min(columns, TILE_ELEMENTS)
            rows_per_tile = max(1, TILE_ELEMENTS // width)
            for first in range(0, rows, rows_per_tile):
                count = min(rows_per_tile, rows - first)
                for column in range(0, columns, width):
                    span = min(width, columns - column)
                    table = columns_by_class[aligned and span % vector == 0]
                    table[0].append(_signed(source.data_ptr() + (first * source_rows[2] + column) * element))
                    table[1].append(_signed(destination.data_ptr() + (first * target_rows[2] + column) * element))
                    table[2].append(source_rows[2])
                    table[3].append(target_rows[2])
                    table[4].append(count)
                    table[5].append(span)
        self.launches = []
        for aligned, table in columns_by_class.items():
            if not table[4]:
                continue
            block_columns = min(1024, triton.next_power_of_2(max(table[5])))
            block_rows = max(1, 4096 // block_columns)
            arguments = [torch.tensor(values, dtype=torch.int64, device=device) for values in table]
            self.launches.append((len(table[4]), arguments, block_rows, block_columns, aligned))
        self.element = element
        self.tiles = sum(launch[0] for launch in self.launches)
        # Compile now, inside prepare's agreement, so a kernel that cannot build refuses the batch on
        # every process instead of failing one process in the middle of a run.
        for tiles, arguments, block_rows, block_columns, aligned in self.launches:
            try:
                copy_tiles.warmup(*arguments,
                                  ELEMENT=element,
                                  VECTOR=_ALIGN_BYTES // element,
                                  ROWS=block_rows,
                                  COLUMNS=block_columns,
                                  ALIGNED=aligned,
                                  num_warps=8,
                                  grid=(tiles, ))
            except Exception as error:
                raise RuntimeError(f'the fused copy kernel does not build here: {error}') from error

    def launch(self) -> None:
        for tiles, arguments, block_rows, block_columns, aligned in self.launches:
            self.kernel[(tiles, )](*arguments,
                                   ELEMENT=self.element,
                                   VECTOR=_ALIGN_BYTES // self.element,
                                   ROWS=block_rows,
                                   COLUMNS=block_columns,
                                   ALIGNED=aligned,
                                   num_warps=8)
        for destination, source in self.fallback:
            destination.copy_(source)
