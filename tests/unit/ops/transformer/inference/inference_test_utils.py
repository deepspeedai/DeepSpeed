# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple

import pytest
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


def run_with_output_canaries(op, out_numel, dtype, num_outputs=1, canary_numel=0):
    """Run ``op``, which allocates ``num_outputs`` outputs of ``out_numel`` elements each, so that every
    output is directly followed in memory by a canary tensor owned by the caller and filled with
    ``CANARY_VALUE`` (finite, so that NaN written past the end is detected as well).

    A kernel that writes past the end of an output overwrites its canary, which plain pytest can then
    detect. The layout is made deterministic with a private caching-allocator pool: one placeholder
    block per output, each followed by its canary; the placeholders are freed so that the outputs
    allocated inside ``op`` reuse them (best fit, lowest address first). Returns (outputs, canaries).
    """
    if not hasattr(torch.cuda, "use_mem_pool"):  #ignore-cuda
        pytest.skip("requires torch.cuda.MemPool")  #ignore-cuda
    device = get_accelerator().current_device_name()
    nbytes = out_numel * torch.empty((), dtype=dtype).element_size()
    pool = torch.cuda.MemPool()  #ignore-cuda
    with torch.cuda.use_mem_pool(pool):  #ignore-cuda
        placeholders, canaries = [], []
        for _ in range(num_outputs):
            placeholders.append(torch.empty(out_numel, dtype=dtype, device=device))
            canaries.append(torch.full((canary_numel, ), CANARY_VALUE, dtype=dtype, device=device))
        expected_ptrs = [t.data_ptr() for t in placeholders]
        del placeholders
        outputs = op()
    get_accelerator().synchronize()

    output_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    # The check is only meaningful if each output sits right in front of its canary.
    assert [t.data_ptr() for t in output_list] == expected_ptrs, "outputs did not reuse the placeholder blocks"
    for t, canary in zip(output_list, canaries):
        assert 0 <= canary.data_ptr() - (t.data_ptr() + nbytes) < 512
    return outputs, canaries
