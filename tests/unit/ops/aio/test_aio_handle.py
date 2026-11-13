# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import threading

import pytest

import deepspeed
from deepspeed.ops.op_builder import AsyncIOBuilder

BLOCK_SIZE = 4096
QUEUE_DEPTH = 2
INTRA_OP_PARALLELISM = 2
MAX_HANDLE_DESTRUCTORS = 32


def test_aio_handle_destructor_releases_gil():
    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        pytest.skip("async_io is not compatible with this environment")

    aio_handle = AsyncIOBuilder().load().aio_handle
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

    original_switch_interval = sys.getswitchinterval()
    try:
        assert peer_ready.wait(timeout=5)
        sys.setswitchinterval(60)
        gate.release()
        for _ in range(MAX_HANDLE_DESTRUCTORS):
            handle = aio_handle(BLOCK_SIZE, QUEUE_DEPTH, True, True, INTRA_OP_PARALLELISM)
            del handle
            if peer_ran.is_set():
                break
        assert peer_ran.is_set(), "aio_handle destructor held the GIL while joining its worker threads"
    finally:
        sys.setswitchinterval(original_switch_interval)
        if gate.locked():
            gate.release()
        peer.join(timeout=5)

    assert not peer.is_alive()
