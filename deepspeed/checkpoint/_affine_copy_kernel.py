# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The Triton kernel behind ``affine_transfer_copy.fused_copies``; imported only when it is used."""

import triton
import triton.language as tl


@triton.jit
def copy_tiles(source_addresses, destination_addresses, source_strides, destination_strides, row_counts, column_counts,
               ELEMENT: tl.constexpr, VECTOR: tl.constexpr, ROWS: tl.constexpr, COLUMNS: tl.constexpr,
               ALIGNED: tl.constexpr):
    """Each program copies one tile: ``rows`` rows of ``columns`` contiguous elements."""
    tile = tl.program_id(0)
    if ELEMENT == 1:
        kind = tl.int8
    elif ELEMENT == 2:
        kind = tl.int16
    elif ELEMENT == 4:
        kind = tl.int32
    else:
        kind = tl.int64
    source = tl.load(source_addresses + tile).to(tl.pointer_type(kind))
    destination = tl.load(destination_addresses + tile).to(tl.pointer_type(kind))
    source_stride = tl.load(source_strides + tile)
    destination_stride = tl.load(destination_strides + tile)
    rows = tl.load(row_counts + tile)
    columns = tl.load(column_counts + tile)
    if ALIGNED:
        # Every tile of an aligned table starts on 16 bytes, with rows and strides whole multiples of
        # 16 bytes, which lets the compiler use vector loads and stores.
        source = tl.multiple_of(source, 16)
        destination = tl.multiple_of(destination, 16)
        source_stride = tl.multiple_of(source_stride, VECTOR)
        destination_stride = tl.multiple_of(destination_stride, VECTOR)
        columns = tl.multiple_of(columns, VECTOR)
    row_offsets = tl.arange(0, ROWS)[:, None]
    column_offsets = tl.arange(0, COLUMNS)[None, :]
    for row_start in range(0, rows, ROWS):
        row = row_start + row_offsets
        for column_start in range(0, columns, COLUMNS):
            column = column_start + column_offsets
            mask = (row < rows) & (column < columns)
            value = tl.load(source + row * source_stride + column, mask=mask)
            tl.store(destination + row * destination_stride + column, value, mask=mask)
