# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Transfer analysis is checked against hand-derived counts and against the executor's own stats."""

import math
import json
import os
from dataclasses import asdict

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map
from deepspeed.checkpoint.affine_transfer import plan_transfer
from deepspeed.checkpoint.affine_transfer_analysis import analyze_transfers, expected_stats, report_to_dict
from deepspeed.checkpoint.affine_transfer_calibration import INTRA_NODE, LOCAL, node_path_of
from deepspeed.checkpoint.affine_transfer_executor import (StagingBudget, TransferEndpoints, TransferRequest,
                                                           execute_transfers)
from unit.common import DistributedTest

FP32 = 4


def _uneven_handoff():
    """8 elements, sources [4, 4] on processes 0-1, targets [2, 6] on processes 2-3.

    By hand: target 0 takes source 0[0:2]; target 1 takes source 0[2:4] and source 1[0:4].
    With 2 elements per round, 1->3 needs two rounds and every other pair one.
    """
    source = contiguous_split_map((8, ), [4, 4], 0)
    target = contiguous_split_map((8, ), [2, 6], 0)
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 2, 1: 3})
    return plan_transfer(target, source), endpoints


def test_volume_rounds_and_messages_match_hand_count():
    plan, endpoints = _uneven_handoff()
    report = analyze_transfers([plan], endpoints, FP32, StagingBudget(2 * FP32))

    assert (report.segments, report.chunks, report.rounds) == (3, 4, 2)
    assert (report.moved_bytes, report.remote_bytes, report.local_copy_bytes) == (32, 32, 0)
    by_process = {entry.process: entry for entry in report.processes}
    assert (by_process[0].sent_bytes, by_process[0].send_messages, by_process[0].rounds) == (16, 2, 1)
    assert (by_process[1].sent_bytes, by_process[1].send_messages, by_process[1].rounds) == (16, 2, 2)
    assert (by_process[2].received_bytes, by_process[2].rounds) == (8, 1)
    assert (by_process[3].received_bytes, by_process[3].receive_messages, by_process[3].receive_peers) == (24, 3, 2)
    # One-dimensional slices are packed on both sides, so nothing is staged.
    assert sum(entry.pack_bytes + entry.unpack_bytes + entry.staging_bytes for entry in report.processes) == 0


def test_strided_source_is_packed_before_sending():
    """Rows to columns on two processes: each sends a 2x2 block that is strided in its source."""
    source = contiguous_split_map((4, 4), [2, 2], 0)
    target = contiguous_split_map((4, 4), [2, 2], 1)
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
    report = analyze_transfers([plan_transfer(target, source)], endpoints, FP32)

    assert (report.local_copy_bytes, report.remote_bytes) == (32, 32)
    for entry in report.processes:
        assert (entry.pack_bytes, entry.unpack_bytes, entry.staging_bytes) == (16, 0, 16)


def test_shared_plan_counts_once_per_request():
    plan, endpoints = _uneven_handoff()
    single = analyze_transfers([plan], endpoints, FP32, StagingBudget(2 * FP32))
    double = analyze_transfers([plan, plan], endpoints, FP32, StagingBudget(2 * FP32))
    assert (double.segments, double.chunks, double.remote_bytes) == (6, 8, 64)
    # A larger batch packs into the same per-peer arenas, so it cannot need fewer rounds.
    assert double.rounds >= single.rounds


def test_report_is_json_ready_and_labels_paths():
    """Rows to columns on two processes of one node: half is copied locally, half crosses the node."""
    source = contiguous_split_map((4, 4), [2, 2], 0)
    target = contiguous_split_map((4, 4), [2, 2], 1)
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
    report = analyze_transfers([plan_transfer(target, source)], endpoints, FP32, path_of=node_path_of(2))
    summary = report_to_dict(report, include_messages=True)
    assert summary['paths'] == {INTRA_NODE: {'messages': 2, 'bytes': 32}, LOCAL: {'messages': 0, 'bytes': 32}}
    assert summary['pairs'] == {'0->1': {'messages': 1, 'bytes': 16}, '1->0': {'messages': 1, 'bytes': 16}}
    assert len(summary['messages']) == len(report.messages) == 2


def test_empty_batch_summary_keeps_its_keys():
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
    summary = report_to_dict(analyze_transfers([], endpoints, FP32))
    for name in ('segments', 'messages'):
        stats = summary['granularity'][name]
        assert set(stats) == {'count', 'sum', 'min', 'p50', 'p90', 'max', 'mean'}
        assert stats['count'] == 0 and stats['max'] is None
    json.dumps(summary)


def test_local_copies_use_the_custom_local_label():
    """A path_of that names each process's local copies, as calibrate_links records them."""
    source = contiguous_split_map((4, 4), [2, 2], 0)
    target = contiguous_split_map((4, 4), [2, 2], 1)
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})

    def path_of(from_process, to_process):
        return f'hbm{from_process}' if from_process == to_process else 'link'

    report = analyze_transfers([plan_transfer(target, source)], endpoints, FP32, path_of=path_of)
    summary = report_to_dict(report)
    assert summary['paths'] == {
        'link': {
            'messages': 2,
            'bytes': 32
        },
        'hbm0': {
            'messages': 0,
            'bytes': 16
        },
        'hbm1': {
            'messages': 0,
            'bytes': 16
        },
    }


def _join_transfer_group():
    if not dist.is_initialized():
        dist.init_distributed(dist_backend=get_accelerator().communication_backend_name(), init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', dist.get_rank()))
    get_accelerator().set_device(local_rank)
    return torch.device(get_accelerator().current_device_name())


@pytest.mark.skipif(get_accelerator().device_count() < 4, reason='requires four visible accelerator devices')
class TestPredictionMatchesExecutor(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize('peer_elements', [3, 64])
    def test_expected_stats_equal_executor_stats(self, peer_elements):
        """A batch of a strided and a replicated parameter, so staging, chunking and local copies all occur."""
        device = _join_transfer_group()
        rank = dist.get_rank()
        dtype = torch.float32
        endpoints = TransferEndpoints(source={0: 0, 1: 1, 2: 2, 3: 3}, target={0: 3, 1: 2, 2: 1, 3: 0})
        cases = [
            (contiguous_split_map((8, 8), [2, 2, 2, 2], 0), contiguous_split_map((8, 8), [2, 2, 2, 2], 1)),
            (replicated_map((8, 4), 4), contiguous_split_map((8, 4), [1, 3, 2, 2], 0)),
        ]
        requests, plans = [], []
        for index, (source, target) in enumerate(cases):
            numel = math.prod(source.logical_shape)
            full = torch.arange(numel, dtype=dtype, device=device).reshape(source.logical_shape)
            sources = {
                location: source.extract(full, location).contiguous()
                for location, process in endpoints.source.items() if process == rank
            }
            targets = {
                location: torch.full(shape, float('nan'), dtype=dtype, device=device)
                for location, shape in target.shard_shapes.items() if endpoints.target[location] == rank
            }
            plan = plan_transfer(target, source)
            plans.append(plan)
            requests.append(TransferRequest(f'param-{index}', plan, sources, targets))

        budget = StagingBudget(peer_elements * 4)
        stats = execute_transfers(requests, endpoints, dtype, budget)
        report = analyze_transfers(plans, endpoints, 4, budget)
        assert asdict(stats) == expected_stats(report, rank)


@pytest.mark.skipif(get_accelerator().device_count() < 2, reason='requires two visible accelerator devices')
class TestPredictionWithInPlaceCopies(DistributedTest):
    world_size = 2

    def test_local_copy_bytes_over_count_by_the_in_place_bytes(self):
        """Each process keeps its own shard of two parameters; one is moved in place, the other copied.

        The analysis counts both local copies, the executor only the one it runs. Every other field agrees.
        """
        device = _join_transfer_group()
        rank = dist.get_rank()
        dtype = torch.float32
        split = contiguous_split_map((8, ), [4, 4], 0)
        endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
        plan = plan_transfer(split, split)
        shard = torch.arange(4, dtype=dtype, device=device) + 4 * rank
        other = torch.full((4, ), float('nan'), dtype=dtype, device=device)
        requests = [
            TransferRequest('in-place', plan, {rank: shard}, {rank: shard}),
            TransferRequest('copied', plan, {rank: shard.clone()}, {rank: other}),
        ]

        stats = execute_transfers(requests, endpoints, dtype)
        expected = expected_stats(analyze_transfers([plan, plan], endpoints, FP32), rank)

        in_place_bytes = shard.numel() * FP32
        assert expected['local_copy_bytes'] == 2 * in_place_bytes
        assert expected['local_copy_bytes'] - stats.local_copy_bytes == in_place_bytes
        assert {
            key: value
            for key, value in asdict(stats).items() if key != 'local_copy_bytes'
        } == {
            key: value
            for key, value in expected.items() if key != 'local_copy_bytes'
        }
        assert torch.equal(other, shard)
