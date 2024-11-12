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
from deepspeed.ops.op_builder import AsyncIOBuilder
from unit.common import DistributedTest

KILO_BYTE = 1024
BLOCK_SIZE = KILO_BYTE
QUEUE_DEPTH = 2
IO_SIZE = 4 * BLOCK_SIZE
IO_PARALLEL = 2

if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
    pytest.skip('Skip tests since async-io is not compatible', allow_module_level=True)


def _skip_for_invalid_environment(use_cuda_pinned_tensor=True):
    if get_accelerator().device_name() != 'cuda':
        if use_cuda_pinned_tensor:
            pytest.skip("torch.pin_memory is only supported in CUDA environments.")


def _get_local_rank():
    if get_accelerator().is_available():
        return dist.get_rank()
    return 0


def _do_ref_write(tmpdir, index=0, file_size=IO_SIZE):
    file_suffix = f'{_get_local_rank()}_{index}'
    ref_file = os.path.join(tmpdir, f'_py_random_{file_suffix}.pt')
    ref_buffer = os.urandom(file_size)
    with open(ref_file, 'wb') as f:
        f.write(ref_buffer)

    return ref_file, ref_buffer


def _get_file_path(tmpdir, file_prefix, index=0):
    file_suffix = f'{_get_local_rank()}_{index}'
    return os.path.join(tmpdir, f'{file_prefix}_{file_suffix}.pt')


def _get_test_write_file(tmpdir, index):
    file_suffix = f'{_get_local_rank()}_{index}'
    return os.path.join(tmpdir, f'_aio_write_random_{file_suffix}.pt')


def _get_test_write_file_and_unpinned_tensor(tmpdir, ref_buffer, index=0):
    test_file = _get_test_write_file(tmpdir, index)
    test_buffer = get_accelerator().ByteTensor(list(ref_buffer))
    return test_file, test_buffer


def _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer, aio_handle=None, index=0):
    test_file = _get_test_write_file(tmpdir, index)
    if aio_handle is None:
        test_buffer = get_accelerator().pin_memory(torch.ByteTensor(list(ref_buffer)))
    else:
        tmp_buffer = torch.ByteTensor(list(ref_buffer))
        test_buffer = aio_handle.new_cpu_locked_tensor(len(ref_buffer), tmp_buffer)
        test_buffer.data.copy_(tmp_buffer)

    return test_file, test_buffer


def _validate_handle_state(handle, single_submit, overlap_events):
    assert handle.get_single_submit() == single_submit
    assert handle.get_overlap_events() == overlap_events
    assert handle.get_intra_op_parallelism() == IO_PARALLEL
    assert handle.get_block_size() == BLOCK_SIZE
    assert handle.get_queue_depth() == QUEUE_DEPTH


@pytest.mark.parametrize("use_cuda_pinned_tensor", [True, False])
@pytest.mark.parametrize("single_submit", [True, False])
@pytest.mark.parametrize("overlap_events", [True, False])
class TestRead(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize("use_unpinned_tensor", [True, False])
    def test_parallel_read(self, tmpdir, use_cuda_pinned_tensor, single_submit, overlap_events, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)

        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        if use_unpinned_tensor:
            aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device=get_accelerator().device_name())
        elif use_cuda_pinned_tensor:
            aio_buffer = get_accelerator().pin_memory(torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu'))
        else:
            aio_buffer = h.new_cpu_locked_tensor(IO_SIZE, torch.empty(0, dtype=torch.uint8))

        _validate_handle_state(h, single_submit, overlap_events)

        ref_file, _ = _do_ref_write(tmpdir)
        read_status = h.sync_pread(aio_buffer, ref_file, 0)
        assert read_status == 1

        with open(ref_file, 'rb') as f:
            ref_buffer = list(f.read())
        assert ref_buffer == aio_buffer.tolist()

        if not use_cuda_pinned_tensor:
            h.free_cpu_locked_tensor(aio_buffer)

    @pytest.mark.parametrize("use_unpinned_tensor", [True, False])
    def test_async_read(self, tmpdir, use_cuda_pinned_tensor, single_submit, overlap_events, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)

        use_cpu_locked_tensor = False
        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        if use_unpinned_tensor:
            aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device=get_accelerator().device_name())
        elif use_cuda_pinned_tensor:
            aio_buffer = get_accelerator().pin_memory(torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu'))
        else:
            aio_buffer = h.new_cpu_locked_tensor(IO_SIZE, torch.empty(0, dtype=torch.uint8))
            use_cpu_locked_tensor = True

        _validate_handle_state(h, single_submit, overlap_events)

        ref_file, _ = _do_ref_write(tmpdir)
        read_status = h.async_pread(aio_buffer, ref_file, 0)
        assert read_status == 0

        wait_status = h.wait()
        assert wait_status == 1

        with open(ref_file, 'rb') as f:
            ref_buffer = list(f.read())
        assert ref_buffer == aio_buffer.tolist()

        if use_cpu_locked_tensor:
            h.free_cpu_locked_tensor(aio_buffer)


@pytest.mark.parametrize("use_cuda_pinned_tensor", [True, False])
@pytest.mark.parametrize("single_submit", [True, False])
@pytest.mark.parametrize("overlap_events", [True, False])
class TestWrite(DistributedTest):
    world_size = 1
    reuse_dist_env = True
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize("use_unpinned_tensor", [True, False])
    def test_parallel_write(self, tmpdir, use_cuda_pinned_tensor, single_submit, overlap_events, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)

        ref_file, ref_buffer = _do_ref_write(tmpdir)
        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        if use_unpinned_tensor:
            aio_file, aio_buffer = _get_test_write_file_and_unpinned_tensor(tmpdir, ref_buffer)
        if use_cuda_pinned_tensor:
            aio_file, aio_buffer = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer)
        else:
            aio_file, aio_buffer = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer, h)

        _validate_handle_state(h, single_submit, overlap_events)

        write_status = h.sync_pwrite(aio_buffer, aio_file, 0)
        assert write_status == 1

        if not use_cuda_pinned_tensor:
            h.free_cpu_locked_tensor(aio_buffer)

        assert os.path.isfile(aio_file)

        filecmp.clear_cache()
        assert filecmp.cmp(ref_file, aio_file, shallow=False)

    @pytest.mark.parametrize("use_unpinned_tensor", [True, False])
    def test_async_write(self, tmpdir, use_cuda_pinned_tensor, single_submit, overlap_events, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)

        ref_file, ref_buffer = _do_ref_write(tmpdir)

        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)
        use_cpu_locked_tensor = False
        if use_unpinned_tensor:
            aio_file, aio_buffer = _get_test_write_file_and_unpinned_tensor(tmpdir, ref_buffer)
        elif use_cuda_pinned_tensor:
            aio_file, aio_buffer = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer)
        else:
            aio_file, aio_buffer = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffer, h)
            use_cpu_locked_tensor = True

        _validate_handle_state(h, single_submit, overlap_events)

        write_status = h.async_pwrite(aio_buffer, aio_file, 0)
        assert write_status == 0

        wait_status = h.wait()
        assert wait_status == 1

        if use_cpu_locked_tensor:
            h.free_cpu_locked_tensor(aio_buffer)

        assert os.path.isfile(aio_file)

        filecmp.clear_cache()
        assert filecmp.cmp(ref_file, aio_file, shallow=False)


@pytest.mark.sequential
@pytest.mark.parametrize("use_cuda_pinned_tensor", [True, False])
@pytest.mark.parametrize("use_unpinned_tensor", [True, False])
class TestAsyncQueue(DistributedTest):
    world_size = 1
    requires_cuda_env = False
    if not get_accelerator().is_available():
        init_distributed = False
        set_dist_env = False

    @pytest.mark.parametrize("async_queue", [2, 3])
    def test_read(self, tmpdir, async_queue, use_cuda_pinned_tensor, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)

        ref_files = []
        for i in range(async_queue):
            f, _ = _do_ref_write(tmpdir, i)
            ref_files.append(f)

        single_submit = True
        overlap_events = True
        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        use_cpu_locked_tensor = False
        if use_unpinned_tensor:
            aio_buffers = [
                torch.empty(IO_SIZE, dtype=torch.uint8, device=get_accelerator().device_name())
                for _ in range(async_queue)
            ]
        elif use_cuda_pinned_tensor:
            aio_buffers = [
                get_accelerator().pin_memory(torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu'))
                for _ in range(async_queue)
            ]
        else:
            tmp_tensor = torch.empty(0, dtype=torch.uint8)
            aio_buffers = [h.new_cpu_locked_tensor(IO_SIZE, tmp_tensor) for _ in range(async_queue)]
            use_cpu_locked_tensor = True

        _validate_handle_state(h, single_submit, overlap_events)

        for i in range(async_queue):
            read_status = h.async_pread(aio_buffers[i], ref_files[i], 0)
            assert read_status == 0

        wait_status = h.wait()
        assert wait_status == async_queue

        for i in range(async_queue):
            with open(ref_files[i], 'rb') as f:
                ref_buffer = list(f.read())
            assert ref_buffer == aio_buffers[i].tolist()

        if use_cpu_locked_tensor:
            for t in aio_buffers:
                h.free_cpu_locked_tensor(t)

    @pytest.mark.parametrize("async_queue", [2, 3])
    def test_write(self, tmpdir, use_cuda_pinned_tensor, async_queue, use_unpinned_tensor):
        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)

        ref_files = []
        ref_buffers = []
        for i in range(async_queue):
            f, buf = _do_ref_write(tmpdir, i)
            ref_files.append(f)
            ref_buffers.append(buf)

        single_submit = True
        overlap_events = True
        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, single_submit, overlap_events, IO_PARALLEL)

        aio_files = []
        aio_buffers = []
        for i in range(async_queue):
            if use_unpinned_tensor:
                f, buf = _get_test_write_file_and_unpinned_tensor(tmpdir, ref_buffers[i], i)
            elif use_cuda_pinned_tensor:
                f, buf = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffers[i], None, i)
            else:
                f, buf = _get_test_write_file_and_pinned_tensor(tmpdir, ref_buffers[i], h, i)
            aio_files.append(f)
            aio_buffers.append(buf)

        use_cpu_locked_tensor = not (use_unpinned_tensor or use_cuda_pinned_tensor)

        _validate_handle_state(h, single_submit, overlap_events)

        for i in range(async_queue):
            read_status = h.async_pwrite(aio_buffers[i], aio_files[i], 0)
            assert read_status == 0

        wait_status = h.wait()
        assert wait_status == async_queue

        if use_cpu_locked_tensor:
            for t in aio_buffers:
                h.free_cpu_locked_tensor(t)

        for i in range(async_queue):
            assert os.path.isfile(aio_files[i])

            filecmp.clear_cache()
            assert filecmp.cmp(ref_files[i], aio_files[i], shallow=False)


@pytest.mark.parametrize("use_cuda_pinned_tensor", [True, False])
@pytest.mark.parametrize('file_partitions', [[1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1]])
class TestAsyncFileOffset(DistributedTest):
    world_size = 1

    def test_offset_write(self, tmpdir, file_partitions, use_cuda_pinned_tensor):

        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        ref_file = _get_file_path(tmpdir, '_py_random')
        aio_file = _get_file_path(tmpdir, '_aio_random')
        partition_unit_size = BLOCK_SIZE
        file_size = sum(file_partitions) * partition_unit_size

        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, True, True, IO_PARALLEL)

        if use_cuda_pinned_tensor:
            data_buffer = torch.ByteTensor(list(os.urandom(file_size))).pin_memory()
        else:
            data_buffer = h.new_cpu_locked_tensor(file_size, torch.empty(0, dtype=torch.uint8))

        file_offsets = []
        next_offset = 0
        for i in range(len(file_partitions)):
            file_offsets.append(next_offset)
            next_offset += file_partitions[i] * partition_unit_size

        ref_fd = open(ref_file, 'wb')
        for i in range(len(file_partitions)):
            src_buffer = torch.narrow(data_buffer, 0, file_offsets[i], file_partitions[i] * partition_unit_size)

            ref_fd.write(src_buffer.numpy().tobytes())
            ref_fd.flush()

            assert 1 == h.sync_pwrite(buffer=src_buffer, filename=aio_file, file_offset=file_offsets[i])

            filecmp.clear_cache()
            assert filecmp.cmp(ref_file, aio_file, shallow=False)

        ref_fd.close()

        if not use_cuda_pinned_tensor:
            h.free_cpu_locked_tensor(data_buffer)

    def test_offset_read(self, tmpdir, file_partitions, use_cuda_pinned_tensor):

        _skip_for_invalid_environment(use_cuda_pinned_tensor=use_cuda_pinned_tensor)
        partition_unit_size = BLOCK_SIZE
        file_size = sum(file_partitions) * partition_unit_size
        ref_file, _ = _do_ref_write(tmpdir, 0, file_size)
        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, True, True, IO_PARALLEL)

        if use_cuda_pinned_tensor:
            data_buffer = torch.zeros(file_size, dtype=torch.uint8, device='cpu').pin_memory()
        else:
            data_buffer = h.new_cpu_locked_tensor(file_size, torch.empty(0, dtype=torch.uint8))

        file_offsets = []
        next_offset = 0
        for i in range(len(file_partitions)):
            file_offsets.append(next_offset)
            next_offset += file_partitions[i] * partition_unit_size

        with open(ref_file, 'rb') as ref_fd:
            for i in range(len(file_partitions)):
                ref_fd.seek(file_offsets[i])
                bytes_to_read = file_partitions[i] * partition_unit_size
                ref_buf = list(ref_fd.read(bytes_to_read))

                dst_tensor = torch.narrow(data_buffer, 0, 0, bytes_to_read)
                assert 1 == h.sync_pread(dst_tensor, ref_file, file_offsets[i])
                assert dst_tensor.tolist() == ref_buf

        if not use_cuda_pinned_tensor:
            h.free_cpu_locked_tensor(data_buffer)
