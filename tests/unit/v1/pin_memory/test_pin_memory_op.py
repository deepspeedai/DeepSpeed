# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import PinMemoryBuilder

if not deepspeed.ops.__compatible_ops__.get(PinMemoryBuilder.NAME, True):
    pytest.skip('Skip tests since pin_memory is not compatible', allow_module_level=True)

NUM_ELEMS = 4096


def test_pin_handle_alloc_free_and_is_pinned():
    # pin_memory must work without libaio / async_io.
    handle = PinMemoryBuilder().load().pin_handle()
    buffer = handle.new_cpu_locked_tensor(NUM_ELEMS, torch.empty(0, dtype=torch.float))
    try:
        assert handle.is_pinned(buffer)
        assert handle.is_pinned(buffer.narrow(0, 16, 16))
        assert not handle.is_pinned(torch.empty(NUM_ELEMS, dtype=torch.float))
    finally:
        assert handle.free_cpu_locked_tensor(buffer) is True
