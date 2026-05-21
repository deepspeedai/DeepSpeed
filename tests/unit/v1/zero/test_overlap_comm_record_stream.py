# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from contextlib import nullcontext

import torch

import deepspeed.runtime.zero.stage_1_and_2 as zero_stage12
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer


class _FakeTensor:

    def __init__(self):
        self.recorded_streams = []
        self.copied_from = None

    def copy_(self, other):
        self.copied_from = other
        return self

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


class _FakeAccelerator:

    def __init__(self, resolves_data_dependency, current_device_name="cpu"):
        self._resolves_data_dependency = resolves_data_dependency
        self._current_device_name = current_device_name

    def resolves_data_dependency(self):
        return self._resolves_data_dependency

    def stream(self, stream):
        return nullcontext()

    def current_stream(self):
        return object()

    def current_device_name(self):
        return self._current_device_name

    def synchronize(self):
        return None


def _build_overlap_optimizer(monkeypatch, *, resolves_data_dependency):
    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.overlap_comm = True
    optimizer.reduction_stream = object()
    optimizer.dp_process_group = object()
    optimizer.previous_reduced_grads = {}

    allreduced = _FakeTensor()
    synced = [_FakeTensor(), _FakeTensor()]

    optimizer.allreduce_bucket = lambda *args, **kwargs: allreduced
    optimizer.unflatten = lambda allreduced_tensor, small_bucket: synced

    monkeypatch.setattr(
        zero_stage12,
        "get_accelerator",
        lambda: _FakeAccelerator(resolves_data_dependency),
    )
    monkeypatch.setattr(zero_stage12.dist, "get_rank", lambda group=None: 0)
    return optimizer, allreduced, synced


def test_allreduce_and_copy_records_stream_for_overlap_comm(monkeypatch):
    optimizer, allreduced, synced = _build_overlap_optimizer(monkeypatch, resolves_data_dependency=False)
    bucket = [_FakeTensor(), _FakeTensor()]

    optimizer.allreduce_and_copy(bucket, torch.float16)

    assert allreduced.recorded_streams == [optimizer.reduction_stream]
    for buf, expected_synced in zip(bucket, synced):
        assert buf.copied_from is expected_synced
        assert buf.recorded_streams == [optimizer.reduction_stream]


def test_allreduce_and_copy_with_multiple_ranks_records_only_local_buffers(monkeypatch):
    optimizer, allreduced, synced = _build_overlap_optimizer(monkeypatch, resolves_data_dependency=False)
    bucket = [_FakeTensor(), _FakeTensor()]

    optimizer.allreduce_and_copy_with_multiple_ranks(
        bucket,
        torch.float16,
        bucket_ranks=[0, 1],
    )

    assert allreduced.recorded_streams == [optimizer.reduction_stream]
    assert bucket[0].copied_from is synced[0]
    assert bucket[0].recorded_streams == [optimizer.reduction_stream]
    assert bucket[1].copied_from is None
    assert bucket[1].recorded_streams == []
