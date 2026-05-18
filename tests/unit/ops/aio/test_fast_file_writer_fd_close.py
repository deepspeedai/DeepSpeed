# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression test for FastFileWriter file-descriptor cleanup.

Without proper os.close() in _fini(), every save through FastFileWriter
leaks one fd pointing at the just-written file. When the user later
unlinks the file (e.g. checkpoint rotation), the leaked fd holds the
inode in the filesystem's orphan list, so blocks are never freed and
the filesystem eventually reports ENOSPC even though `ls` shows only
N files on disk.

These tests assert that after FastFileWriter.close() returns, no fd
in /proc/self/fd points at the (possibly already-unlinked) file.
"""
import os
import sys
import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed.io import FastFileWriter, FastFileWriterConfig

if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
    pytest.skip("async_io op is not compatible on this system", allow_module_level=True)

if not sys.platform.startswith("linux"):
    pytest.skip("test uses /proc/self/fd which is Linux-only", allow_module_level=True)

if get_accelerator().device_name() != 'cuda':
    pytest.skip("FastFileWriter requires CUDA-pinned tensors", allow_module_level=True)

BLOCK_SIZE = 1 * 1024 * 1024
QUEUE_DEPTH = 8
PINNED_BYTES = 8 * 1024 * 1024
PAYLOAD_BYTES = 1 * 1024 * 1024


def _count_deleted_fds(target_dir):
    """How many fds in /proc/self/fd point at a now-deleted file located
    under target_dir? Restricting to target_dir avoids false positives
    from unrelated deleted fds in the test process."""
    pid = os.getpid()
    n = 0
    for entry in os.listdir(f"/proc/{pid}/fd"):
        try:
            target = os.readlink(f"/proc/{pid}/fd/{entry}")
        except OSError:
            continue
        if target.startswith(str(target_dir)) and target.endswith("(deleted)"):
            n += 1
    return n


def _build_writer(file_path):
    aio = AsyncIOBuilder().load(verbose=False).aio_handle(block_size=BLOCK_SIZE,
                                                          queue_depth=QUEUE_DEPTH,
                                                          single_submit=False,
                                                          overlap_events=False,
                                                          intra_op_parallelism=1)
    pinned = torch.zeros(PINNED_BYTES, dtype=torch.uint8).pin_memory()
    cfg = FastFileWriterConfig(dnvme_handle=aio,
                               pinned_tensor=pinned,
                               double_buffer=True,
                               num_parallel_writers=1,
                               writer_rank=0)
    return FastFileWriter(file_path=str(file_path), config=cfg)


def test_close_releases_fd_after_unlink(tmp_path):
    """Single save + unlink must not leave a deleted-fd reference."""
    target = tmp_path / "ckpt_single.pt"
    buf = torch.zeros(PAYLOAD_BYTES, dtype=torch.uint8)

    before = _count_deleted_fds(tmp_path)
    w = _build_writer(target)
    torch.save(obj=buf, f=w)
    w.close()
    os.unlink(target)
    after = _count_deleted_fds(tmp_path)

    assert after == before, (f"FastFileWriter leaked an fd: deleted-fd count went "
                             f"from {before} to {after} after a single save+close+unlink. "
                             f"This indicates _fini() did not os.close(self._aio_fd).")


@pytest.mark.parametrize("n_iters", [5, 20])
def test_rotation_loop_does_not_leak(tmp_path, n_iters):
    """N iterations of save+close+unlink should leave zero deleted-fds.
    Mirrors the real checkpoint-rotation workload that originally
    surfaced this bug."""
    buf = torch.zeros(PAYLOAD_BYTES, dtype=torch.uint8)
    before = _count_deleted_fds(tmp_path)

    for i in range(n_iters):
        path = tmp_path / f"ckpt_{i}.pt"
        w = _build_writer(path)
        torch.save(obj=buf, f=w)
        w.close()
        os.unlink(path)

    after = _count_deleted_fds(tmp_path)
    assert after == before, (f"FastFileWriter leaked {after - before} fd(s) over {n_iters} "
                             f"save+close+unlink iterations (expected 0).")
