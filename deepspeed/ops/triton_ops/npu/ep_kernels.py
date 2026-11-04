# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ascend (NPU) variant of the MoE row-gather kernel.

The portable kernel computes the source address from the raw padding index
before applying the load mask. On Ascend the masked lanes can still be checked
against the computed address, so a padding index below zero or past the last
source row faults the device. This variant keeps every computed address inside
the tensor while leaving the mask to zero the padding rows, and returns before
launching when the source has no rows.

The dispatch annotation on the portable ``gather_rows`` selects this function
on Ascend; the portable implementation stays unchanged.
"""

import torch

from deepspeed.ops.triton_ops._triton import _TRITON_AVAILABLE, current_device, triton, tl

_GATHER_ROWS = 4


def _gather_rows_eager(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Padding-aware reference used when Triton is unavailable."""
    out = src.new_zeros((index.numel(), src.shape[1]))
    valid = (index >= 0) & (index < src.shape[0])
    if valid.any():
        out[valid] = src.index_select(0, index[valid].to(torch.long))
    return out


if _TRITON_AVAILABLE:

    @triton.jit
    def _gather_rows_kernel(
        src_ptr,
        index_ptr,
        out_ptr,
        num_out_rows,
        num_src_rows,
        hidden,
        ROWS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        row_mask = rows < num_out_rows
        # Keep every computed address in range even where the access is masked:
        # Ascend may still check the address of a masked lane.
        safe_rows = tl.where(row_mask, rows, 0)
        index = tl.load(index_ptr + safe_rows, mask=row_mask, other=-1).to(tl.int64)
        valid = (index >= 0) & (index < num_src_rows)
        safe_index = tl.where(valid, index, 0)
        for start in range(0, hidden, BLOCK_H):
            columns = start + tl.arange(0, BLOCK_H)
            column_mask = columns < hidden
            safe_columns = tl.where(column_mask, columns, 0)
            mask = row_mask[:, None] & column_mask[None, :]
            values = tl.load(src_ptr + safe_index[:, None] * hidden + safe_columns[None, :],
                             mask=mask & valid[:, None],
                             other=0.0)
            tl.store(out_ptr + safe_rows[:, None] * hidden + safe_columns[None, :], values, mask=mask)


def gather_rows(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``out[r] = src[index[r]]``, with zero rows at padding entries."""
    src = src.contiguous()
    index = index.contiguous()
    if src.shape[0] == 0:
        return src.new_zeros((index.numel(), src.shape[1]))
    out = torch.empty((index.numel(), src.shape[1]), dtype=src.dtype, device=src.device)
    if out.numel() == 0:
        return out
    if not _TRITON_AVAILABLE:
        return _gather_rows_eager(src, index)
    hidden = src.shape[1]
    with current_device(src):
        _gather_rows_kernel[(triton.cdiv(index.numel(), _GATHER_ROWS), )](
            src,
            index,
            out,
            index.numel(),
            src.shape[0],
            hidden,
            ROWS=_GATHER_ROWS,
            BLOCK_H=min(1024, triton.next_power_of_2(hidden)),
        )
    return out
