# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple

import torch
from deepspeed.accelerator import get_accelerator

TOLERANCES = None


def get_tolerances():
    global TOLERANCES
    if TOLERANCES is None:
        TOLERANCES = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}
        if get_accelerator().is_bf16_supported():
            # Note: BF16 tolerance is higher than FP16 because of the lower precision (7 (+1) bits vs
            # 10 (+1) bits)
            TOLERANCES[torch.bfloat16] = (4.8e-1, 3.2e-2)
    return TOLERANCES


DTYPES = None


def get_dtypes(include_float=True):
    global DTYPES
    if DTYPES is None:
        DTYPES = [torch.float16, torch.float32] if include_float else [torch.float16]
        try:
            if get_accelerator().is_bf16_supported():
                DTYPES.append(torch.bfloat16)
        except (AssertionError, AttributeError):
            pass
    return DTYPES


def allclose(x, y, tolerances: Tuple[int, int] = None):
    assert x.dtype == y.dtype
    if tolerances is None:
        rtol, atol = get_tolerances()[x.dtype]
    else:
        rtol, atol = tolerances
    return torch.allclose(x, y, rtol=rtol, atol=atol)


CANARY_VALUE = 1000.0

# The generic bf16 tolerance (rtol 0.48) is loose enough to accept a row normalized with its neighbour's
# statistics. A row norm's bf16 output is accurate to well under this, so the norm tests use it instead.
NORM_BF16_TOLERANCES = (3e-2, 1e-2)


def norm_tolerances(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return NORM_BF16_TOLERANCES
    return None


def allocate_output_with_canary(rows: int, channels: int, dtype: torch.dtype, canary_rows: int = 256):
    """Allocate a ``[rows, channels]`` output that is directly followed in memory by ``canary_rows`` rows
    filled with ``CANARY_VALUE``. A kernel that writes past the last row overwrites the canary, which plain
    pytest can then detect (the value is finite, so NaN written past the end is detected as well). The default
    covers the largest sub-block schedule (256 rows per block), so even a full block of overrun stays inside
    the canary instead of spilling into neighbouring allocations.
    Returns (output, canary)."""
    buffer = torch.full((rows + canary_rows, channels),
                        CANARY_VALUE,
                        dtype=dtype,
                        device=get_accelerator().current_device_name())
    return buffer[:rows], buffer[rows:]


def assert_rows_independent(run_rows, batched_outputs: Tuple[torch.Tensor, ...]):
    """Every row of a batched launch must match the same row run on its own, within the norm tolerance for
    its dtype. Rows that share a thread block must not see each other's statistics; a per-row reference alone
    cannot tell that apart from ordinary rounding at bf16 precision. ``run_rows(start, end)`` runs the kernel on
    input rows ``[start, end)`` and returns its outputs in the same order as ``batched_outputs``."""
    rows = batched_outputs[0].size(0)
    for row in range(rows):
        for single, batched in zip(run_rows(row, row + 1), batched_outputs):
            assert allclose(single, batched[row:row + 1],
                            norm_tolerances(single.dtype)), f"row {row} differs when run on its own"
