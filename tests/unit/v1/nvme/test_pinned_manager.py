# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import AsyncIOBuilder

if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
    pytest.skip('Skip tests since async-io is not compatible', allow_module_level=True)

BLOCK_SIZE = 1024
QUEUE_DEPTH = 2
IO_PARALLEL = 1
NUM_ELEMS = 4 * BLOCK_SIZE


def _new_handle():
    return AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, False, False, IO_PARALLEL)


def test_narrow_of_locked_buffer_is_pinned():
    handle = _new_handle()
    buffer = handle.new_cpu_locked_tensor(NUM_ELEMS, torch.empty(0, dtype=torch.float))
    try:
        assert handle.is_pinned(buffer)
        # A slice/view falls inside the locked range, so range-based recognition
        # must report it as pinned too (exact-base matching would miss it).
        assert handle.is_pinned(buffer.narrow(0, BLOCK_SIZE, BLOCK_SIZE))
    finally:
        handle.free_cpu_locked_tensor(buffer)


def test_buffer_is_shared_across_handles():
    handle_a = _new_handle()
    handle_b = _new_handle()
    buffer = handle_a.new_cpu_locked_tensor(NUM_ELEMS, torch.empty(0, dtype=torch.float))
    try:
        # The pinned-tensor manager is process-wide, so a buffer locked through one
        # handle must be recognized by any other handle.
        assert handle_b.is_pinned(buffer)
    finally:
        handle_a.free_cpu_locked_tensor(buffer)


def test_unmanaged_buffer_is_not_pinned():
    handle = _new_handle()
    assert not handle.is_pinned(torch.empty(NUM_ELEMS, dtype=torch.float))


def test_pin_handle_buffer_recognized_by_aio_handle():
    # Buffers allocated via the standalone pin_memory op must share the same
    # process-wide manager as async_io, so aio_handle.is_pinned sees them.
    from deepspeed.ops.op_builder import PinMemoryBuilder
    if not deepspeed.ops.__compatible_ops__.get(PinMemoryBuilder.NAME, True):
        pytest.skip('Skip tests since pin_memory is not compatible')

    pin_handle = PinMemoryBuilder().load().pin_handle()
    aio_handle = _new_handle()
    buffer = pin_handle.new_cpu_locked_tensor(NUM_ELEMS, torch.empty(0, dtype=torch.float))
    try:
        assert pin_handle.is_pinned(buffer)
        assert aio_handle.is_pinned(buffer)
        assert aio_handle.is_pinned(buffer.narrow(0, BLOCK_SIZE, BLOCK_SIZE))
    finally:
        pin_handle.free_cpu_locked_tensor(buffer)


def test_aio_handle_buffer_recognized_by_pin_handle():
    from deepspeed.ops.op_builder import PinMemoryBuilder
    if not deepspeed.ops.__compatible_ops__.get(PinMemoryBuilder.NAME, True):
        pytest.skip('Skip tests since pin_memory is not compatible')

    pin_handle = PinMemoryBuilder().load().pin_handle()
    aio_handle = _new_handle()
    buffer = aio_handle.new_cpu_locked_tensor(NUM_ELEMS, torch.empty(0, dtype=torch.float))
    try:
        assert aio_handle.is_pinned(buffer)
        assert pin_handle.is_pinned(buffer)
    finally:
        aio_handle.free_cpu_locked_tensor(buffer)
