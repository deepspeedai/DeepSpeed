# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import filecmp
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import GDSBuilder
from unit.common import DistributedTest

KILO_BYTE = 1024 * 256
BLOCK_SIZE = KILO_BYTE
QUEUE_DEPTH = 2
IO_SIZE = 4 * BLOCK_SIZE
IO_PARALLEL = 2

if not deepspeed.ops.__compatible_ops__[GDSBuilder.NAME]:
    pytest.skip('Skip tests since gds is not compatible', allow_module_level=True)


def _get_local_rank():
    if get_accelerator().is_available():
        return dist.get_rank()
    return 0


def _do_ref_write(tmpdir, index=0):
    file_suffix = f'{_get_local_rank()}_{index}'
    ref_file = os.path.join(tmpdir, f'_py_random_{file_suffix}.pt')
    ref_buffer = os.urandom(IO_SIZE)
    with open(ref_file, 'wb') as f:
        f.write(ref_buffer)

    return ref_file, ref_buffer


def _get_test_write_file(tmpdir, index):
    file_suffix = f'{_get_local_rank()}_{index}'
    return os.path.join(tmpdir, f'_gds_write_random_{file_suffix}.pt')


def _get_test_write_file_and_device_buffer(tmpdir, ref_buffer, gds_handle, index=0):
    test_file = _get_test_write_file(tmpdir, index)
    test_buffer = get_accelerator().ByteTensor(list(ref_buffer))
    gds_handle.pin_device_tensor(test_buffer)
    return test_file, test_buffer


def _validate_handle_state(handle, single_submit, overlap_events):
    assert handle.get_single_submit() == single_submit
    assert handle.get_overlap_events() == overlap_events
    assert handle.get_thread_count() == IO_PARALLEL
    assert handle.get_block_size() == BLOCK_SIZE
    assert handle.get_queue_depth() == QUEUE_DEPTH


@pytest.mark.parametrize("single_submit", [True, False])
@pytest.mark.parametrize("overlap_events", [True, False])
class TestRead(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    def test_parallel_read(self, tmpdir, single_submit, overlap_events):

        h = GDSBuilder().load().gds_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        gds_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device=get_accelerator().device_name())
        h.pin_device_tensor(gds_buffer)

        _validate_handle_state(h, single_submit, overlap_events)

        ref_file, _ = _do_ref_write(tmpdir)
        read_status = h.sync_pread(gds_buffer, ref_file)
        assert read_status == 1

        with open(ref_file, 'rb') as f:
            ref_buffer = list(f.read())
        assert ref_buffer == gds_buffer.tolist()

        h.unpin_device_tensor(gds_buffer)

    def test_async_read(self, tmpdir, single_submit, overlap_events):

        h = GDSBuilder().load().gds_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        gds_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device=get_accelerator().device_name())
        h.pin_device_tensor(gds_buffer)

        _validate_handle_state(h, single_submit, overlap_events)

        ref_file, _ = _do_ref_write(tmpdir)
        read_status = h.async_pread(gds_buffer, ref_file)
        assert read_status == 0

        wait_status = h.wait()
        assert wait_status == 1

        with open(ref_file, 'rb') as f:
            ref_buffer = list(f.read())
        assert ref_buffer == gds_buffer.tolist()

        h.unpin_device_tensor(gds_buffer)


@pytest.mark.parametrize("single_submit", [True, False])
@pytest.mark.parametrize("overlap_events", [True, False])
class TestWrite(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    def test_parallel_write(self, tmpdir, single_submit, overlap_events):

        ref_file, ref_buffer = _do_ref_write(tmpdir)
        h = GDSBuilder().load().gds_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        gds_file, gds_buffer = _get_test_write_file_and_device_buffer(tmpdir, ref_buffer, h)

        _validate_handle_state(h, single_submit, overlap_events)

        write_status = h.sync_pwrite(gds_buffer, gds_file)
        assert write_status == 1

        h.unpin_device_tensor(gds_buffer)

        assert os.path.isfile(gds_file)

        filecmp.clear_cache()
        assert filecmp.cmp(ref_file, gds_file, shallow=False)

    def test_async_write(self, tmpdir, single_submit, overlap_events):
        ref_file, ref_buffer = _do_ref_write(tmpdir)

        h = GDSBuilder().load().gds_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
        gds_file, gds_buffer = _get_test_write_file_and_device_buffer(tmpdir, ref_buffer, h)

        _validate_handle_state(h, single_submit, overlap_events)

        write_status = h.async_pwrite(gds_buffer, gds_file)
        assert write_status == 0

        wait_status = h.wait()
        assert wait_status == 1

        h.unpin_device_tensor(gds_buffer)

        assert os.path.isfile(gds_file)

        filecmp.clear_cache()
        assert filecmp.cmp(ref_file, gds_file, shallow=False)


@pytest.mark.sequential
class TestAsyncQueue(DistributedTest):
    world_size = 1
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize("async_queue", [2, 3])
    def test_read(self, tmpdir, async_queue):

        ref_files = []
        for i in range(async_queue):
            f, _ = _do_ref_write(tmpdir, i)
            ref_files.append(f)

        single_submit = True
        overlap_events = True
        h = GDSBuilder().load().gds_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        gds_buffers = [
            torch.empty(IO_SIZE, dtype=torch.uint8, device=get_accelerator().device_name()) for _ in range(async_queue)
        ]
        for buf in gds_buffers:
            h.pin_device_tensor(buf)

        _validate_handle_state(h, single_submit, overlap_events)

        for i in range(async_queue):
            read_status = h.async_pread(gds_buffers[i], ref_files[i])
            assert read_status == 0

        wait_status = h.wait()
        assert wait_status == async_queue

        for i in range(async_queue):
            with open(ref_files[i], 'rb') as f:
                ref_buffer = list(f.read())
            assert ref_buffer == gds_buffers[i].tolist()

        for t in gds_buffers:
            h.unpin_device_tensor(t)

    @pytest.mark.parametrize("async_queue", [2, 3])
    def test_write(self, tmpdir, async_queue):
        ref_files = []
        ref_buffers = []
        for i in range(async_queue):
            f, buf = _do_ref_write(tmpdir, i)
            ref_files.append(f)
            ref_buffers.append(buf)

        single_submit = True
        overlap_events = True
        h = GDSBuilder().load().gds_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        gds_files = []
        gds_buffers = []
        for i in range(async_queue):
            f, buf = _get_test_write_file_and_device_buffer(tmpdir, ref_buffers[i], h, i)
            gds_files.append(f)
            gds_buffers.append(buf)

        _validate_handle_state(h, single_submit, overlap_events)

        for i in range(async_queue):
            read_status = h.async_pwrite(gds_buffers[i], gds_files[i])
            assert read_status == 0

        wait_status = h.wait()
        assert wait_status == async_queue

        for t in gds_buffers:
            h.unpin_device_tensor(t)

        for i in range(async_queue):
            assert os.path.isfile(gds_files[i])

            filecmp.clear_cache()
            assert filecmp.cmp(ref_files[i], gds_files[i], shallow=False)


@pytest.mark.parametrize("use_new_api", [True, False])
class TestLockDeviceTensor(DistributedTest):
    world_size = 2
    reuse_dist_env = True
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    def test_pin_device_tensor(self, use_new_api):

        h = GDSBuilder().load().gds_handle(BLOCK_SIZE, QUEUE_DEPTH, True, True, IO_PARALLEL)

        unpinned_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device=get_accelerator().device_name())
        if use_new_api:
            pinned_buffer = h.new_pinned_device_tensor(unpinned_buffer.numel(), unpinned_buffer)
        else:
            pinned_buffer = torch.empty_like(unpinned_buffer)
            h.pin_device_tensor(pinned_buffer)

        assert unpinned_buffer.device == pinned_buffer.device
        assert unpinned_buffer.dtype == pinned_buffer.dtype
        assert unpinned_buffer.numel() == pinned_buffer.numel()

        if use_new_api:
            h.free_pinned_device_tensor(pinned_buffer)
        else:
            h.unpin_device_tensor(pinned_buffer)
