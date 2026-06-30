# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed.runtime.zero import partition_parameters as pp
from deepspeed.runtime.zero.partition_parameters import _partition_chunk_overlap

from unit.common import DistributedTest


def _aligned_numel(numel, num_partitions):
    remainder = numel % num_partitions
    return numel + ((num_partitions - remainder) if remainder else 0)


def _stream_one_partition(full_flat, rank, num_partitions, chunk_numel):
    """Rebuild a single rank's partition by streaming the flattened parameter in
    fixed-size chunks, mirroring ``_partition_param_streaming``. Padding elements
    (beyond the real numel) are left as the ``-1`` sentinel so the test can assert
    they are never written."""
    ds_numel = full_flat.numel()
    partition_numel = _aligned_numel(ds_numel, num_partitions) // num_partitions
    partition_start = partition_numel * rank
    out = torch.full((partition_numel, ), -1.0)
    offset = 0
    while offset < ds_numel:
        cur = min(chunk_numel, ds_numel - offset)
        overlap = _partition_chunk_overlap(offset, cur, partition_start, partition_numel)
        if overlap is not None:
            dst_offset, src_offset, numel = overlap
            chunk = full_flat.narrow(0, offset, cur)
            out.narrow(0, dst_offset, numel).copy_(chunk.narrow(0, src_offset, numel))
        offset += cur
    return out


@pytest.mark.parametrize("numel,num_partitions,chunk_numel", [
    (64, 4, 8),
    (64, 4, 7),
    (60, 8, 5),
    (10, 4, 3),
    (1, 2, 4),
    (100, 3, 1),
    (128, 1, 16),
])
def test_streamed_partitions_match_direct_slicing(numel, num_partitions, chunk_numel):
    full = torch.arange(numel, dtype=torch.float32)
    partition_numel = _aligned_numel(numel, num_partitions) // num_partitions
    aligned = partition_numel * num_partitions

    rebuilt = torch.full((aligned, ), -1.0)
    for rank in range(num_partitions):
        partition = _stream_one_partition(full, rank, num_partitions, chunk_numel)
        rebuilt.narrow(0, rank * partition_numel, partition_numel).copy_(partition)

    # The real (non-padded) region must match the original parameter exactly.
    assert torch.equal(rebuilt.narrow(0, 0, numel), full)
    # Padding elements must never be written by the streaming copy.
    if aligned > numel:
        padding = rebuilt.narrow(0, numel, aligned - numel)
        assert torch.all(padding == -1.0)


class TestStreamingPartitionMatchesStandard(DistributedTest):
    world_size = 2

    def test_streaming_matches_standard(self):

        def build(chunk_size):
            config = {
                "train_batch_size": self.world_size,
                "zero_optimization": {
                    "stage": 3,
                    "stage3_partition_stream_chunk_size": chunk_size,
                },
            }
            torch.manual_seed(1234)
            with deepspeed.zero.Init(config_dict_or_path=config):
                linear = torch.nn.Linear(64, 64, bias=False)
            return linear

        # Reference: the standard broadcast-then-partition path (streaming disabled).
        reference = build(0)
        reference_partition = reference.weight.ds_tensor.detach().clone()

        # Streaming the same parameter (4096 elements) in 512-element chunks must
        # produce a byte-identical partition while actually exercising the new path.
        streaming_calls = {"count": 0}
        original = pp.Init._partition_param_streaming

        def counting_stream(self, param, *args, **kwargs):
            streaming_calls["count"] += 1
            return original(self, param, *args, **kwargs)

        pp.Init._partition_param_streaming = counting_stream
        try:
            streamed = build(512)
        finally:
            pp.Init._partition_param_streaming = original

        assert streaming_calls["count"] >= 1, "streaming partition path was not exercised"
        assert torch.equal(streamed.weight.ds_tensor, reference_partition)

    def test_streaming_via_module_path(self):
        # zero.Init(module=...) decides whether to stream on plain torch parameters,
        # before they are converted to ZeRO params. The decision must therefore not
        # require ZeRO metadata (e.g. a per-parameter process group) that is only
        # attached during conversion.

        def build(chunk_size):
            config = {
                "train_batch_size": self.world_size,
                "zero_optimization": {
                    "stage": 3,
                    "stage3_partition_stream_chunk_size": chunk_size,
                },
            }
            torch.manual_seed(99)
            linear = torch.nn.Linear(64, 64, bias=True)  # built on the host, then converted
            deepspeed.zero.Init(module=linear, config_dict_or_path=config)
            return linear

        reference = build(0)
        # weight (4096 elements) streams; bias (64) stays on the standard path but is
        # still passed through the pre-conversion stream check.
        streamed = build(512)
        assert torch.equal(streamed.weight.ds_tensor, reference.weight.ds_tensor)
        assert torch.equal(streamed.bias.ds_tensor, reference.bias.ds_tensor)
