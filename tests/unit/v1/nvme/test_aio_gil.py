# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import sys
import threading

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder

BLOCK_SIZE = 4096
QUEUE_DEPTH = 2
INTRA_OP_PARALLELISM = 2
MAX_BLOCKING_CALLS = 32
CONCURRENT_READ_TIMEOUT = 30
CONCURRENT_MANAGER_OPS = 32


def _require_aio_cuda():
    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        pytest.skip("async_io is not compatible with this environment")
    if get_accelerator().device_name() != "cuda" or not get_accelerator().is_available():
        pytest.skip("CUDA-pinned memory is required")


def _invoke_blocking_aio(handle, operation_name, buffer, filename):
    if operation_name == "pread":
        return handle.pread(buffer, filename, False, False, 0)
    if operation_name == "pwrite":
        return handle.pwrite(buffer, filename, False, False, 0)
    if operation_name == "sync_pread":
        return handle.sync_pread(buffer, filename, 0)
    if operation_name == "sync_pwrite":
        return handle.sync_pwrite(buffer, filename, 0)
    raise AssertionError(f"unexpected operation: {operation_name}")


@pytest.mark.parametrize("operation_name", ["pread", "pwrite", "sync_pread", "sync_pwrite"])
def test_blocking_aio_releases_gil(tmp_path, operation_name):
    _require_aio_cuda()

    payload = os.urandom(BLOCK_SIZE)
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(payload)
    target_path = tmp_path / "target.bin"

    handle = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, True, True, INTRA_OP_PARALLELISM)
    if "pread" in operation_name:
        buffer = torch.zeros(BLOCK_SIZE, dtype=torch.uint8, device="cpu").pin_memory()
        filename = str(source_path)
    else:
        buffer = torch.tensor(list(payload), dtype=torch.uint8, device="cpu").pin_memory()
        filename = str(target_path)
        assert _invoke_blocking_aio(handle, operation_name, buffer, filename) == 1

    peer_ready = threading.Event()
    peer_ran = threading.Event()
    gate = threading.Lock()
    gate.acquire()

    def run_peer():
        peer_ready.set()
        with gate:
            peer_ran.set()

    peer = threading.Thread(target=run_peer)
    peer.start()
    assert peer_ready.wait(timeout=5)

    original_switch_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(60)
        gate.release()
        for _ in range(MAX_BLOCKING_CALLS):
            assert _invoke_blocking_aio(handle, operation_name, buffer, filename) == 1
            if peer_ran.is_set():
                break
        assert peer_ran.is_set(), f"{operation_name} held the GIL across all blocking calls"
    finally:
        sys.setswitchinterval(original_switch_interval)
        if gate.locked():
            gate.release()
        peer.join(timeout=5)

    assert not peer.is_alive()
    if "pread" in operation_name:
        assert bytes(buffer.tolist()) == payload
    else:
        assert target_path.read_bytes() == payload


@pytest.mark.parametrize("use_pinned_memory", [True, False])
def test_concurrent_blocking_reads_on_shared_handle(tmp_path, use_pinned_memory):
    _require_aio_cuda()

    payloads = [os.urandom(BLOCK_SIZE), os.urandom(BLOCK_SIZE)]
    source_paths = []
    buffers = []
    for index, payload in enumerate(payloads):
        source_path = tmp_path / f"source-{index}.bin"
        source_path.write_bytes(payload)
        source_paths.append(source_path)
        if use_pinned_memory:
            buffers.append(torch.zeros(BLOCK_SIZE, dtype=torch.uint8, device="cpu").pin_memory())
        else:
            buffers.append(torch.zeros(BLOCK_SIZE, dtype=torch.uint8, device=get_accelerator().device_name()))

    handle = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, True, True, INTRA_OP_PARALLELISM)
    start = threading.Barrier(3, timeout=CONCURRENT_READ_TIMEOUT)
    statuses = [None, None]
    errors = []

    def run_read(index):
        try:
            start.wait()
            statuses[index] = handle.sync_pread(buffers[index], str(source_paths[index]), 0)
        except Exception as error:
            errors.append(error)

    readers = [threading.Thread(target=run_read, args=(index, ), daemon=True) for index in range(2)]
    for reader in readers:
        reader.start()
    start.wait()
    for reader in readers:
        reader.join(timeout=CONCURRENT_READ_TIMEOUT)

    assert not errors
    assert all(not reader.is_alive() for reader in readers)
    assert statuses == [1, 1]
    assert [bytes(buffer.tolist()) for buffer in buffers] == payloads


def test_concurrent_unpinned_read_and_locked_tensor_management(tmp_path):
    _require_aio_cuda()

    payload = os.urandom(BLOCK_SIZE)
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(payload)
    buffer = torch.zeros(BLOCK_SIZE, dtype=torch.uint8, device=get_accelerator().device_name())
    example_tensor = torch.empty(1, dtype=torch.uint8, device="cpu")
    handle = AsyncIOBuilder().load().aio_handle(BLOCK_SIZE, QUEUE_DEPTH, True, True, INTRA_OP_PARALLELISM)
    start = threading.Barrier(3, timeout=CONCURRENT_READ_TIMEOUT)
    errors = []

    def run_reads():
        try:
            start.wait()
            for _ in range(CONCURRENT_MANAGER_OPS):
                assert handle.sync_pread(buffer, str(source_path), 0) == 1
        except Exception as error:
            errors.append(error)

    def manage_locked_tensors():
        try:
            start.wait()
            for _ in range(CONCURRENT_MANAGER_OPS):
                locked_tensor = handle.new_cpu_locked_tensor(BLOCK_SIZE, example_tensor)
                assert handle.free_cpu_locked_tensor(locked_tensor)
        except Exception as error:
            errors.append(error)

    workers = [
        threading.Thread(target=run_reads, daemon=True),
        threading.Thread(target=manage_locked_tensors, daemon=True),
    ]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=CONCURRENT_READ_TIMEOUT)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert bytes(buffer.tolist()) == payload
