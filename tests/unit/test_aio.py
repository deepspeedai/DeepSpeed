import pytest
import os
import filecmp
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.ops.aio import AsyncIOBuilder
from .common import distributed_test

MEGA_BYTE = 1024**2
BLOCK_SIZE = MEGA_BYTE
QUEUE_DEPTH = 2
IO_SIZE = 16 * MEGA_BYTE
IO_PARALLEL = 2


def _skip_if_no_aio():
    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        pytest.skip('Skip tests since async-io is not compatible')


def _do_ref_write(tmpdir, index=0):
    file_suffix = f'{dist.get_rank()}_{index}'
    ref_file = os.path.join(tmpdir, f'_py_random_{file_suffix}.pt')
    ref_buffer = os.urandom(IO_SIZE)
    with open(ref_file, 'wb') as f:
        f.write(ref_buffer)

    return ref_file, ref_buffer


def _get_test_file_and_buffer(tmpdir, ref_buffer, cuda_device, index=0):
    file_suffix = f'{dist.get_rank()}_{index}'
    test_file = os.path.join(tmpdir, f'_aio_write_random_{file_suffix}.pt')
    if cuda_device:
        test_buffer = torch.cuda.ByteTensor(list(ref_buffer))
    else:
        test_buffer = torch.ByteTensor(list(ref_buffer)).pin_memory()

    return test_file, test_buffer


def _validate_handle_state(handle, single_submit, overlap_events):
    assert handle.get_single_submit() == single_submit
    assert handle.get_overlap_events() == overlap_events
    assert handle.get_thread_count() == IO_PARALLEL
    assert handle.get_block_size() == BLOCK_SIZE
    assert handle.get_queue_depth() == QUEUE_DEPTH


@pytest.mark.parametrize('single_submit, overlap_events',
                         [(False,
                           False),
                          (False,
                           True),
                          (True,
                           False),
                          (True,
                           True)])
def test_parallel_read(tmpdir, single_submit, overlap_events):
    _skip_if_no_aio()

    @distributed_test(world_size=[2])
    def _test_parallel_read(single_submit, overlap_events):
        ref_file, _ = _do_ref_write(tmpdir)

        aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu').pin_memory()
        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE,
                                               QUEUE_DEPTH,
                                               single_submit,
                                               overlap_events,
                                               IO_PARALLEL)

        _validate_handle_state(h, single_submit, overlap_events)

        read_status = h.sync_pread(aio_buffer, ref_file)
        assert read_status == 1

        with open(ref_file, 'rb') as f:
            ref_buffer = list(f.read())
        assert ref_buffer == aio_buffer.tolist()

    _test_parallel_read(single_submit, overlap_events)


@pytest.mark.parametrize('single_submit, overlap_events, cuda_device',
                         [(False,
                           False,
                           False),
                          (False,
                           True,
                           False),
                          (True,
                           False,
                           False),
                          (True,
                           True,
                           False),
                          (False,
                           False,
                           True),
                          (True,
                           True,
                           True)])
def test_async_read(tmpdir, single_submit, overlap_events, cuda_device):

    _skip_if_no_aio()

    @distributed_test(world_size=[2])
    def _test_async_read(single_submit, overlap_events, cuda_device):
        ref_file, _ = _do_ref_write(tmpdir)

        if cuda_device:
            aio_buffer = torch.empty(IO_SIZE, dtype=torch.uint8, device='cuda')
        else:
            aio_buffer = torch.empty(IO_SIZE,
                                     dtype=torch.uint8,
                                     device='cpu').pin_memory()

        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE,
                                               QUEUE_DEPTH,
                                               single_submit,
                                               overlap_events,
                                               IO_PARALLEL)

        _validate_handle_state(h, single_submit, overlap_events)

        read_status = h.async_pread(aio_buffer, ref_file)
        assert read_status == 0

        wait_status = h.wait()
        assert wait_status == 1

        with open(ref_file, 'rb') as f:
            ref_buffer = list(f.read())
        assert ref_buffer == aio_buffer.tolist()

    _test_async_read(single_submit, overlap_events, cuda_device)


@pytest.mark.parametrize('single_submit, overlap_events',
                         [(False,
                           False),
                          (False,
                           True),
                          (True,
                           False),
                          (True,
                           True)])
def test_parallel_write(tmpdir, single_submit, overlap_events):

    _skip_if_no_aio()

    @distributed_test(world_size=[2])
    def _test_parallel_write(single_submit, overlap_events):
        ref_file, ref_buffer = _do_ref_write(tmpdir)

        aio_file, aio_buffer = _get_test_file_and_buffer(tmpdir, ref_buffer, False)

        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE,
                                               QUEUE_DEPTH,
                                               single_submit,
                                               overlap_events,
                                               IO_PARALLEL)

        _validate_handle_state(h, single_submit, overlap_events)

        write_status = h.sync_pwrite(aio_buffer, aio_file)
        assert write_status == 1

        assert os.path.isfile(aio_file)

        filecmp.clear_cache()
        assert filecmp.cmp(ref_file, aio_file, shallow=False)

    _test_parallel_write(single_submit, overlap_events)


@pytest.mark.parametrize('single_submit, overlap_events, cuda_device',
                         [(False,
                           False,
                           False),
                          (False,
                           True,
                           False),
                          (True,
                           False,
                           False),
                          (True,
                           True,
                           False),
                          (False,
                           False,
                           True),
                          (True,
                           True,
                           True)])
def test_async_write(tmpdir, single_submit, overlap_events, cuda_device):

    _skip_if_no_aio()

    @distributed_test(world_size=[2])
    def _test_async_write(single_submit, overlap_events, cuda_device):
        ref_file, ref_buffer = _do_ref_write(tmpdir)

        aio_file, aio_buffer = _get_test_file_and_buffer(tmpdir, ref_buffer, cuda_device)

        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE,
                                               QUEUE_DEPTH,
                                               single_submit,
                                               overlap_events,
                                               IO_PARALLEL)

        _validate_handle_state(h, single_submit, overlap_events)

        write_status = h.async_pwrite(aio_buffer, aio_file)
        assert write_status == 0

        wait_status = h.wait()
        assert wait_status == 1

        assert os.path.isfile(aio_file)

        filecmp.clear_cache()
        assert filecmp.cmp(ref_file, aio_file, shallow=False)

    _test_async_write(single_submit, overlap_events, cuda_device)


@pytest.mark.parametrize('async_queue, cuda_device',
                         [(2,
                           False),
                          (4,
                           False),
                          (2,
                           True),
                          (4,
                           True)])
def test_async_queue_read(tmpdir, async_queue, cuda_device):

    _skip_if_no_aio()

    @distributed_test(world_size=[2])
    def _test_async_queue_read(async_queue, cuda_device):
        ref_files = []
        for i in range(async_queue):
            f, _ = _do_ref_write(tmpdir, i)
            ref_files.append(f)

        aio_buffers = []
        for i in range(async_queue):
            if cuda_device:
                buf = torch.empty(IO_SIZE, dtype=torch.uint8, device='cuda')
            else:
                buf = torch.empty(IO_SIZE, dtype=torch.uint8, device='cpu').pin_memory()
            aio_buffers.append(buf)

        single_submit = True
        overlap_events = True
        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE,
                                               QUEUE_DEPTH,
                                               single_submit,
                                               overlap_events,
                                               IO_PARALLEL)

        _validate_handle_state(h, single_submit, overlap_events)

        for i in range(async_queue):
            read_status = h.async_pread(aio_buffers[i], ref_files[i])
            assert read_status == 0

        wait_status = h.wait()
        assert wait_status == async_queue

        for i in range(async_queue):
            with open(ref_files[i], 'rb') as f:
                ref_buffer = list(f.read())
            assert ref_buffer == aio_buffers[i].tolist()

    _test_async_queue_read(async_queue, cuda_device)


@pytest.mark.parametrize('async_queue, cuda_device',
                         [(2,
                           False),
                          (7,
                           False),
                          (2,
                           True),
                          (7,
                           True)])
def test_async_queue_write(tmpdir, async_queue, cuda_device):

    _skip_if_no_aio()

    @distributed_test(world_size=[2])
    def _test_async_queue_write(async_queue, cuda_device):
        ref_files = []
        ref_buffers = []
        for i in range(async_queue):
            f, buf = _do_ref_write(tmpdir, i)
            ref_files.append(f)
            ref_buffers.append(buf)

        aio_files = []
        aio_buffers = []
        for i in range(async_queue):
            f, buf = _get_test_file_and_buffer(tmpdir, ref_buffers[i], cuda_device, i)
            aio_files.append(f)
            aio_buffers.append(buf)

        single_submit = True
        overlap_events = True
        h = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE,
                                               QUEUE_DEPTH,
                                               single_submit,
                                               overlap_events,
                                               IO_PARALLEL)

        _validate_handle_state(h, single_submit, overlap_events)

        for i in range(async_queue):
            read_status = h.async_pwrite(aio_buffers[i], aio_files[i])
            assert read_status == 0

        wait_status = h.wait()
        assert wait_status == async_queue

        for i in range(async_queue):
            assert os.path.isfile(aio_files[i])

            filecmp.clear_cache()
            assert filecmp.cmp(ref_files[i], aio_files[i], shallow=False)

    _test_async_queue_write(async_queue, cuda_device)
