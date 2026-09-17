# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import logging

import torch

from deepspeed.utils.pin_memory_tracker import (
    _fmt_bytes,
    _tracker,
    pinned_memory_summary,
    track_pinned_memory,
)


def test_track_accumulates_and_resets():
    _tracker.reset()
    track_pinned_memory(100)
    track_pinned_memory(2**30)
    assert _tracker._bytes == 100 + 2**30
    assert _tracker._calls == 2
    _tracker.reset()
    assert _tracker._bytes == 0 and _tracker._calls == 0


def test_summary_does_not_raise():
    _tracker.reset()
    track_pinned_memory(2**30)
    pinned_memory_summary("unit-test")
    _tracker.reset()


def test_fmt_bytes():
    assert _fmt_bytes(512) == "512 B"
    assert _fmt_bytes(2048) == "2.0 KB"
    assert _fmt_bytes(2**20) == "1.00 MB"
    assert _fmt_bytes(2**30).endswith("GB")


def test_torch_tensor_nbytes_is_consistent():
    t = torch.zeros(1024, dtype=torch.float32)
    track_pinned_memory(t.nbytes)
    assert _tracker._bytes == 4096
    _tracker.reset()


def test_checkpoint_thresholds_double_from_32gb():
    _tracker.reset()
    gb = 1024**3
    assert _tracker._next_checkpoint == 32 * gb
    track_pinned_memory(30 * gb)  # below 32 GB -> no crossing
    assert _tracker._next_checkpoint == 32 * gb
    track_pinned_memory(10 * gb)  # 40 GB -> crosses 32
    assert _tracker._next_checkpoint == 64 * gb
    track_pinned_memory(100 * gb)  # 140 GB -> crosses 64 and 128 in one call
    assert _tracker._next_checkpoint == 256 * gb
    _tracker.reset()


def test_checkpoint_emits_info(caplog):
    # The DeepSpeed logger does not propagate, so flip propagation so caplog
    # (root-based) can observe the checkpoint INFO records.
    _tracker.reset()
    ds_logger = logging.getLogger("DeepSpeed")
    old_prop = ds_logger.propagate
    ds_logger.propagate = True
    try:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="DeepSpeed"):
            track_pinned_memory(33 * (1024**3))  # crosses the 32 GB checkpoint
            track_pinned_memory(5 * (1024**3))  # 38 GB, no new checkpoint
        checkpoints = [r.message for r in caplog.records if "checkpoint" in r.message]
        assert len(checkpoints) == 1
        assert "32" in checkpoints[0]
    finally:
        ds_logger.propagate = old_prop
        _tracker.reset()
