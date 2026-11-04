# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Check transfer payloads against extraction from the logical tensor."""

import math
import os
from dataclasses import replace

import pytest
import torch
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.affine_transfer_executor import (AffineTransferExecutionError, StagingBudget,
                                                           TransferEndpoints, TransferRequest, execute_transfer,
                                                           execute_transfers)
from unit.common import DistributedTest

UNWRITTEN = float('nan')
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _join_transfer_group():
    if not dist.is_initialized():
        dist.init_distributed(dist_backend=get_accelerator().communication_backend_name(), init_method='env://')


def _device():
    local_rank = int(os.environ.get('LOCAL_RANK', dist.get_rank()))
    get_accelerator().set_device(local_rank)
    return torch.device(get_accelerator().current_device_name())


def _full(rows, cols, dtype):
    """Identical markers that stay distinct after fp16 or bf16 conversion."""
    values = torch.arange(rows * cols, dtype=torch.float32, device=_device()).reshape(rows, cols).to(dtype)
    flat = values.reshape(-1)
    assert torch.unique(flat).numel() == flat.numel(), f'{dtype} markers collided; this case cannot detect a swap'
    return values


def _targets(map_, endpoints, dtype):
    return {
        location: torch.full(shape, UNWRITTEN, dtype=dtype, device=_device())
        for location, shape in map_.shard_shapes.items() if endpoints.target[location] == dist.get_rank()
    }


def _assert_matches(plan_targets, target_map, full):
    for location, buffer in plan_targets.items():
        want = target_map.extract(full, location)
        assert torch.equal(buffer, want), (f'target location {location}: {int((~torch.eq(buffer, want)).sum())} '
                                           f'of {want.numel()} elements differ from the extracted shard')


def _row_to_column(world_size, tp_size=2):
    rows, cols = 8 * tp_size, 8 * tp_size
    source = contiguous_split_map((rows, cols), [rows // tp_size] * tp_size, 0)
    target = contiguous_split_map((rows, cols), [cols // world_size] * world_size, 1)
    return source, target


class TestB2Transfer(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('dtype', DTYPES, ids=['fp32', 'fp16', 'bf16'])
    def test_pure_copy_lands_every_element(self, dtype):
        _join_transfer_group()
        source, target = _row_to_column(self.world_size)
        full = _full(16, 16, dtype)
        endpoints = TransferEndpoints(source={loc: loc
                                              for loc in source.shard_shapes},
                                      target={loc: loc
                                              for loc in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)

        plan = plan_transfer(target, source, PlanBudget())
        stats = execute_transfer(plan, source_buffers, target_buffers, endpoints, dtype)

        _assert_matches(target_buffers, target, full)
        assert stats.chunks > 0
        assert sum(segment.numel for segment in plan.segments) == math.prod(target.logical_shape), \
            'the plan must cover the parameter exactly once'

    @pytest.mark.parametrize('batched', [False, True])
    def test_scalar_plan_crosses_processes(self, batched):
        _join_transfer_group()
        rank = dist.get_rank()
        map_ = replicated_map((), 1)
        plan, endpoints = plan_transfer(map_, map_), TransferEndpoints({0: 0}, {0: 1})
        sources = {0: torch.tensor(3.0, device=_device())} if rank == 0 else {}
        storage = torch.full((3, ), -9.0, device=_device())
        targets = {0: storage[1].view(())} if rank == 1 else {}
        if batched:
            execute_transfers([TransferRequest('scalar', plan, sources, targets)], endpoints, torch.float32,
                              StagingBudget(4))
        else:
            execute_transfer(plan, sources, targets, endpoints, torch.float32, StagingBudget(4))
        expected = torch.tensor([-9.0, 3.0 if rank == 1 else -9.0, -9.0], device=_device())
        assert torch.equal(storage, expected)

    def test_ranks_disagreeing_about_the_schedule_refuse_everywhere(self):
        _join_transfer_group()
        """A lone refusal would strand its peers, so disagreement has to become a collective failure.

        One rank sizes its staging differently, which changes the chunk schedule and therefore the
        digest. Nobody may post communication after that, so this asserts that all ranks raise rather
        than that some rank does.
        """
        source, target = _row_to_column(self.world_size)
        dtype = torch.float32
        full = _full(16, 16, dtype)
        endpoints = TransferEndpoints(source={loc: loc
                                              for loc in source.shard_shapes},
                                      target={loc: loc
                                              for loc in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)
        budget = StagingBudget(bytes_per_peer=4 << 20 if dist.get_rank() == 0 else 1 << 20)

        with pytest.raises(AffineTransferExecutionError, match='transfer schedule'):
            execute_transfer(plan_transfer(target, source, PlanBudget()), source_buffers, target_buffers, endpoints,
                             dtype, budget)

    def test_shared_plan_batch_keeps_distinct_payloads_and_revalidates(self):
        _join_transfer_group()
        rank, world = dist.get_rank(), dist.get_world_size()
        map_ = contiguous_split_map((4 * world, 8), [4] * world, 0)
        plan = plan_transfer(map_, map_)
        endpoints = TransferEndpoints({location: location
                                       for location in range(world)},
                                      {location: (location + world // 2) % world
                                       for location in range(world)})
        target_location = (rank + world // 2) % world
        requests, checks = [], []
        for index in range(6):
            full = _full(4 * world, 8, torch.float32) + index * 32 * world
            source = map_.extract(full, rank).contiguous()
            target = torch.full((4, 8), UNWRITTEN, device=_device())
            requests.append(TransferRequest(str(index), plan, {rank: source}, {target_location: target}))
            checks.append((target, map_.extract(full, target_location)))
        previous_rounds = 0
        # The smallest budget spans 48 rounds, crossing the bounded CUDA work retention interval.
        for budget in (1024, 32, 16):
            stats = execute_transfers(requests, endpoints, torch.float32, StagingBudget(budget))
            assert stats.rounds > previous_rounds
            assert stats.sent_bytes == stats.received_bytes == 6 * 4 * 8 * 4
            previous_rounds = stats.rounds
            for request, (target, expected) in zip(requests, checks):
                assert torch.equal(target, expected)
                request.source_buffers[rank].add_(1)
                expected.add_(1)
                target.fill_(UNWRITTEN)
        if rank == 0:
            requests[1].target_buffers[target_location] = checks[1][0][:1]
        with pytest.raises(AffineTransferExecutionError, match='rejected the transfer batch'):
            execute_transfers(requests, endpoints, torch.float32, StagingBudget(1024))
        assert all(torch.isnan(target).all().item() for target, _ in checks)

    def test_mixed_packed_and_strided_regions_preserve_storage_windows(self):
        _join_transfer_group()
        world, rank = dist.get_world_size(), dist.get_rank()
        width = 4 * world
        endpoints = TransferEndpoints({location: location
                                       for location in range(world)},
                                      {location: (location + world // 2) % world
                                       for location in range(world)})
        requests, checks, guards = [], [], []
        # Flattening a noncontiguous region or ignoring a shard's storage offset corrupts this batch.
        for index, (source_axis, target_axis) in enumerate(((0, 0), (0, 1), (1, 0))):
            full = _full(width, width, torch.float32) + index * width * width
            source_map = contiguous_split_map((width, width), [4] * world, source_axis)
            target_map = contiguous_split_map((width, width), [4] * world, target_axis)
            sources, targets = {}, {}
            for map_, bindings, buffers, is_source in ((source_map, endpoints.source, sources, True),
                                                       (target_map, endpoints.target, targets, False)):
                for location, shape in map_.shard_shapes.items():
                    if bindings[location] != rank:
                        continue
                    size = math.prod(shape)
                    storage = torch.full((size + 12, ), -17.0, device=_device())
                    buffer = storage[5:5 + size].view(shape)
                    if is_source:
                        buffer.copy_(map_.extract(full, location))
                    buffers[location] = buffer
                    guards.append((storage, size))
            requests.append(TransferRequest(f'mixed-{index}', plan_transfer(target_map, source_map), sources, targets))
            checks.append((targets, target_map, full))
        if not get_accelerator().is_synchronized_device():
            stream = get_accelerator().Stream()
            stream.wait_stream(get_accelerator().current_stream())
            with get_accelerator().stream(stream):
                execute_transfers(requests, endpoints, torch.float32, StagingBudget(256))
        else:
            execute_transfers(requests, endpoints, torch.float32, StagingBudget(256))
        # The return contract makes targets ready even after leaving the caller's stream context.
        for targets, target_map, full in checks:
            _assert_matches(targets, target_map, full)
        for storage, size in guards:
            assert torch.all(storage[:5] == -17).item()
            assert torch.all(storage[5 + size:] == -17).item()

    def test_batch_parameter_order_must_match(self):
        _join_transfer_group()
        source, target = _row_to_column(self.world_size)
        full = _full(16, 16, torch.float32)
        endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
        sources = {
            location: source.extract(full, location).contiguous()
            for location in source.shard_shapes if endpoints.source[location] == dist.get_rank()
        }
        plan = plan_transfer(target, source)
        first = TransferRequest('first', plan, sources, _targets(target, endpoints, torch.float32))
        second = TransferRequest('second', plan, sources, _targets(target, endpoints, torch.float32))
        requests = [first, second] if dist.get_rank() == 0 else [second, first]
        with pytest.raises(AffineTransferExecutionError, match='batch schedule'):
            execute_transfers(requests, endpoints, torch.float32)


class TestB2DisjointEndpoints(DistributedTest):
    """Every process is either a reader or a writer, never both.

    A transfer that only appears to work by falling back to local copies fails here. Two ranks fit the
    two-device pull-request runners; four fit the CPU multi-rank nightly and give each writer two readers.
    """

    @pytest.mark.parametrize('world_size', [2, 4])
    def test_source_and_target_ranks_do_not_overlap(self, world_size):
        _join_transfer_group()
        source, target = _row_to_column(2)
        dtype = torch.float32
        full = _full(16, 16, dtype)
        half = world_size // 2
        # Readers are the lower half of the ranks and writers the upper half.
        endpoints = TransferEndpoints(source={location: location % half
                                              for location in source.shard_shapes},
                                      target={location: half + location % half
                                              for location in target.shard_shapes})
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)

        stats = execute_transfer(plan_transfer(target, source, PlanBudget()), source_buffers, target_buffers,
                                 endpoints, dtype)

        # With no process holding both ends, each element crosses the transport exactly once.
        assert stats.local_copy_bytes == 0
        assert stats.sent_bytes == sum(buffer.numel() for buffer in source_buffers.values()) * 4
        assert stats.received_bytes == sum(buffer.numel() for buffer in target_buffers.values()) * 4
        _assert_matches(target_buffers, target, full)


class TestB2TransferCPU:

    def test_shared_plan_uses_each_buffer_and_rechecks_each_call(self):
        map_ = replicated_map((3, 4), 1)
        plan = plan_transfer(map_, map_)
        endpoints = TransferEndpoints({0: 0}, {0: 0})
        sources = [torch.arange(12, dtype=torch.float32).view(3, 4) + i * 100 for i in range(2)]
        targets = [torch.full((3, 4), -1.0) for _ in sources]
        requests = [
            TransferRequest(str(i), plan, {0: source}, {0: target})
            for i, (source, target) in enumerate(zip(sources, targets))
        ]
        for budget, chunks in ((48, 2), (8, 12)):
            for source, target in zip(sources, targets):
                source.add_(1)
                target.fill_(-1)
            stats = execute_transfers(requests, endpoints, torch.float32, StagingBudget(budget))
            assert stats.chunks == chunks
            assert stats.local_copy_bytes == 96
            for source, target in zip(sources, targets):
                assert torch.equal(target, source)
        for target in targets:
            target.fill_(-1)
        # Reusing geometry must not reuse a different request's buffer validation.
        requests[1].target_buffers[0] = targets[1][:1]
        with pytest.raises(AffineTransferExecutionError):
            execute_transfers(requests, endpoints, torch.float32, StagingBudget(48))
        for target in targets:
            assert torch.equal(target, torch.full((3, 4), -1.0))

    @pytest.mark.parametrize('shape', [(3, 7, 11), (13, 17, 1), (2, 5, 7, 3)])
    def test_nd_tiles_fit_a_small_packed_budget_and_preserve_guards(self, shape):
        # Last-axis-only splitting rejects these valid shapes even when one element fits.
        numel = math.prod(shape)
        full = torch.arange(numel, dtype=torch.float32).reshape(shape)
        source_storage = torch.full((numel + 10, ), -7.0)
        target_storage = torch.full((numel + 12, ), -9.0)
        source = source_storage[5:5 + numel].view(shape)
        target = target_storage[6:6 + numel].view(shape)
        source.copy_(full)
        map_ = replicated_map(shape, 1)
        request = TransferRequest('expert', plan_transfer(map_, map_), {0: source}, {0: target})
        stats = execute_transfers([request], TransferEndpoints({0: 0}, {0: 0}), torch.float32,
                                  StagingBudget(bytes_per_peer=28))
        assert torch.equal(target, full)
        assert torch.equal(target_storage[:6], torch.full((6, ), -9.0))
        assert torch.equal(target_storage[-6:], torch.full((6, ), -9.0))
        assert stats.local_copy_bytes == numel * 4
        assert stats.chunks > 1

    @pytest.mark.parametrize('shape', [(4, 7, 11), (4, 5, 7, 3)])
    def test_nd_strided_pieces_land_in_the_target_map(self, shape):
        source_map = contiguous_split_map(shape, [2, 2], 0)
        last = shape[-1]
        target_map = contiguous_split_map(shape, [last // 2, last - last // 2], len(shape) - 1)
        full = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        sources = {rank: source_map.extract(full, rank).contiguous() for rank in source_map.shard_shapes}
        targets = {rank: torch.full(local_shape, -1.0) for rank, local_shape in target_map.shard_shapes.items()}
        endpoints = TransferEndpoints({0: 0, 1: 0}, {0: 0, 1: 0})
        execute_transfer(plan_transfer(target_map, source_map), sources, targets, endpoints, torch.float32,
                         StagingBudget(bytes_per_peer=20))
        for rank, target in targets.items():
            assert torch.equal(target, target_map.extract(full, rank))

    @pytest.mark.parametrize('invalid', [
        'source_noncontiguous',
        'target_noncontiguous',
        'missing_source',
        'missing_target',
        'negative_offset',
        'short_stride',
        'zero_stride',
        'wrong_dtype',
        'zero_budget',
        'sub_element_budget',
        'unowned_source',
        'unowned_target',
        'invalid_endpoint',
    ])
    @pytest.mark.parametrize('batched', [False, True])
    def test_invalid_buffers_and_metadata_are_rejected_before_any_write(self, invalid, batched):
        map_ = replicated_map((2, 3), 1)
        plan = plan_transfer(map_, map_)
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        target = torch.full((2, 3), -11.0)
        targets = {0: target}
        sources = {0: source}
        endpoints = TransferEndpoints({0: 0}, {0: 0})
        budget = StagingBudget(16)
        if invalid == 'source_noncontiguous':
            sources[0] = source.t()
        elif invalid == 'target_noncontiguous':
            targets[0] = target.t()
        elif invalid == 'missing_source':
            sources = {}
        elif invalid == 'missing_target':
            targets = {}
        elif invalid in ('negative_offset', 'short_stride', 'zero_stride'):
            changes = {
                'negative_offset': {
                    'source_offset': -1
                },
                'short_stride': {
                    'source_strides': (1, )
                },
                'zero_stride': {
                    'target_strides': (0, 1)
                }
            }
            segment = replace(plan.segments[0], **changes[invalid])
            plan = replace(plan, segments=(segment, ))
        elif invalid == 'wrong_dtype':
            sources[0] = source.to(torch.bfloat16)
        elif invalid == 'zero_budget':
            budget = StagingBudget(0)
        elif invalid == 'sub_element_budget':
            budget = StagingBudget(3)
        elif invalid == 'unowned_source':
            sources = {1: source}
        elif invalid == 'unowned_target':
            targets = {1: target}
        elif invalid == 'invalid_endpoint':
            endpoints = TransferEndpoints({0: 1}, {0: 0})
        first_target = torch.full((2, 3), -13.0)
        with pytest.raises(AffineTransferExecutionError):
            if batched:
                first = TransferRequest('first', plan_transfer(map_, map_), {0: source}, {0: first_target})
                second = TransferRequest('second', plan, sources, targets)
                execute_transfers([first, second], endpoints, torch.float32, budget)
            else:
                execute_transfer(plan, sources, targets, endpoints, torch.float32, budget)
        assert torch.equal(target, torch.full((2, 3), -11.0))
        assert torch.equal(first_target, torch.full((2, 3), -13.0))

    @pytest.mark.parametrize('alias', ['read_write', 'write_write'])
    def test_aliases_between_requests_are_rejected_before_any_write(self, alias):
        map_ = replicated_map((6, ), 1)
        plan = plan_transfer(map_, map_)
        shared = torch.full((6, ), -11.0)
        other = torch.full((6, ), -13.0)
        first = TransferRequest('first', plan, {0: torch.arange(6, dtype=torch.float32)}, {0: shared})
        second = TransferRequest('second', plan,
                                 {0: shared if alias == 'read_write' else torch.arange(6, dtype=torch.float32)},
                                 {0: other if alias == 'read_write' else shared})
        with pytest.raises(AffineTransferExecutionError):
            execute_transfers([first, second], TransferEndpoints({0: 0}, {0: 0}), torch.float32)
        assert torch.equal(shared, torch.full((6, ), -11.0))
        assert torch.equal(other, torch.full((6, ), -13.0))

    def test_disjoint_windows_of_one_storage_are_valid(self):
        map_ = replicated_map((6, ), 1)
        storage = torch.arange(12, dtype=torch.float32)
        source, target = storage[:6], storage[6:]
        expected = source.clone()
        execute_transfer(plan_transfer(map_, map_), {0: source}, {0: target}, TransferEndpoints({0: 0}, {0: 0}),
                         torch.float32, StagingBudget(8))
        assert torch.equal(target, expected)
        assert torch.equal(source, expected)

    def test_frozen_parameters_keep_their_identity_and_storage(self):
        map_ = replicated_map((2, 3), 1)
        source = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        target = torch.nn.Parameter(torch.full((2, 3), -1.0))
        identity, pointer = id(target), target.data_ptr()
        execute_transfer(plan_transfer(map_, map_), {0: source}, {0: target}, TransferEndpoints({0: 0}, {0: 0}),
                         torch.float32, StagingBudget(8))
        assert id(target) == identity and target.data_ptr() == pointer
        assert target.requires_grad and target.grad is None
        assert torch.equal(target, source)

    @pytest.mark.parametrize('batched', [False, True])
    def test_scalar_plan_preserves_storage_offsets_and_guards(self, batched):
        source_storage = torch.tensor([-7.0, 3.0, -7.0])
        target_storage = torch.full((3, ), -9.0)
        source, target = source_storage[1].view(()), target_storage[1].view(())
        map_ = replicated_map((), 1)
        plan, endpoints = plan_transfer(map_, map_), TransferEndpoints({0: 0}, {0: 0})
        if batched:
            execute_transfers([TransferRequest('scalar', plan, {0: source}, {0: target})], endpoints, torch.float32,
                              StagingBudget(4))
        else:
            execute_transfer(plan, {0: source}, {0: target}, endpoints, torch.float32, StagingBudget(4))
        assert torch.equal(target_storage, torch.tensor([-9.0, 3.0, -9.0]))
        assert torch.equal(source_storage, torch.tensor([-7.0, 3.0, -7.0]))


class TestB2ResidencyAndOffload:
    """Test CPU <-> Accelerator residency combinations, uneven partitions, padding, and offloaded optimizer states."""

    @pytest.mark.parametrize("src_device_type, dst_device_type", [
        ("cpu", "cuda"),
        ("cuda", "cpu"),
        ("cpu", "cpu"),
    ])
    def test_residency_combinations_with_uneven_partitions_and_guards(self, src_device_type, dst_device_type):
        acc = get_accelerator()
        if (src_device_type == "cuda" or dst_device_type == "cuda") and not acc.is_available():
            pytest.skip("Accelerator not available")

        src_device = torch.device(acc.current_device_name() if src_device_type == "cuda" else "cpu")
        dst_device = torch.device(acc.current_device_name() if dst_device_type == "cuda" else "cpu")

        # Uneven partitions: 23 rows x 7 cols = 161 elements
        # Source map: 3 uneven row partitions: [8, 8, 7]
        # Target map: 3 uneven row partitions: [10, 8, 5]
        logical_shape = (23, 7)
        total_numel = 23 * 7
        source_map = contiguous_split_map(logical_shape, [8, 8, 7], partition_dim=0)
        target_map = contiguous_split_map(logical_shape, [10, 8, 5], partition_dim=0)
        plan = plan_transfer(target_map, source_map)

        full = torch.arange(total_numel, dtype=torch.float32).reshape(logical_shape)

        # Allocate storage with guard offsets to verify bounds and storage offsets
        source_buffers = {}
        source_guards = []
        for rank, shape in source_map.shard_shapes.items():
            numel = math.prod(shape)
            storage = torch.full((numel + 18, ), -42.0, device=src_device)
            # Offset 7, length numel, followed by 11 guard elements
            shard = storage[7:7 + numel].view(shape)
            shard.copy_(source_map.extract(full, rank).to(src_device))
            source_buffers[rank] = shard
            source_guards.append((storage, numel, 7))

        target_buffers = {}
        target_guards = []
        for rank, shape in target_map.shard_shapes.items():
            numel = math.prod(shape)
            storage = torch.full((numel + 14, ), -99.0, device=dst_device)
            # Offset 5, length numel, followed by 9 guard elements
            shard = storage[5:5 + numel].view(shape)
            target_buffers[rank] = shard
            target_guards.append((storage, numel, 5))

        endpoints = TransferEndpoints({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0})

        # Use small budget of 32 bytes (8 fp32 elements) to force bounded chunking across pinned staging
        budget = StagingBudget(bytes_per_peer=32)
        stats = execute_transfers([TransferRequest("param_uneven", plan, source_buffers, target_buffers)], endpoints,
                                  torch.float32, budget)

        assert stats.chunks >= 20, f"Expected chunking under 32-byte budget, got {stats.chunks}"

        # Verify data landed bit-exactly
        for rank, target in target_buffers.items():
            expected = target_map.extract(full, rank).to(dst_device)
            assert torch.equal(target, expected), f"Rank {rank} target elements mismatch"

        # Verify guard boundaries are completely untouched
        for storage, numel, offset in target_guards:
            assert torch.all(storage[:offset] == -99.0).item(), "Target prefix guard corrupted"
            assert torch.all(storage[offset + numel:] == -99.0).item(), "Target suffix guard corrupted"

        for storage, numel, offset in source_guards:
            assert torch.all(storage[:offset] == -42.0).item(), "Source prefix guard corrupted"
            assert torch.all(storage[offset + numel:] == -42.0).item(), "Source suffix guard corrupted"

    def test_mixed_residency_batch(self):
        acc = get_accelerator()
        if not acc.is_available():
            pytest.skip("Accelerator not available")
        gpu_device = torch.device(acc.current_device_name())
        cpu_device = torch.device("cpu")

        # Three requests in one batch:
        # req 0: CPU -> GPU (e.g. CPU-offloaded optimizer state -> GPU model param)
        # req 1: GPU -> CPU (e.g. GPU model param -> CPU-offloaded optimizer state)
        # req 2: CPU -> CPU (e.g. CPU-offloaded moment -> CPU-offloaded moment resharding)
        shape = (12, 4)
        src_map = contiguous_split_map(shape, [6, 6], partition_dim=0)
        dst_map = contiguous_split_map(shape, [4, 8], partition_dim=0)
        plan = plan_transfer(dst_map, src_map)

        endpoints = TransferEndpoints({0: 0, 1: 0}, {0: 0, 1: 0})
        budget = StagingBudget(bytes_per_peer=32)  # 8 elements per chunk

        configs = [
            ("cpu_to_gpu", cpu_device, gpu_device),
            ("gpu_to_cpu", gpu_device, cpu_device),
            ("cpu_to_cpu", cpu_device, cpu_device),
        ]

        requests = []
        oracles = []
        for key, s_dev, d_dev in configs:
            full = torch.randn(shape, dtype=torch.float32)
            s_bufs = {r: src_map.extract(full, r).clone().to(s_dev) for r in src_map.shard_shapes}
            d_bufs = {
                r: torch.full(dst_map.shard_shapes[r], -1.0, dtype=torch.float32, device=d_dev)
                for r in dst_map.shard_shapes
            }
            requests.append(TransferRequest(key, plan, s_bufs, d_bufs))
            oracles.append((d_bufs, full, d_dev))

        execute_transfers(requests, endpoints, torch.float32, budget)

        for d_bufs, full, d_dev in oracles:
            for r, buf in d_bufs.items():
                expected = dst_map.extract(full, r).to(d_dev)
                assert torch.equal(buf, expected)

    @pytest.mark.parametrize("src_device_type, dst_device_type", [
        ("cpu", "cuda"),
        ("cuda", "cpu"),
        ("cpu", "cpu"),
    ])
    def test_zero3_padding_drop_with_uneven_shards(self, src_device_type, dst_device_type):
        from deepspeed.checkpoint.zero_affine import zero_storage_affine_map
        acc = get_accelerator()
        if (src_device_type == "cuda" or dst_device_type == "cuda") and not acc.is_available():
            pytest.skip("Accelerator not available")

        src_device = torch.device(acc.current_device_name() if src_device_type == "cuda" else "cpu")
        dst_device = torch.device(acc.current_device_name() if dst_device_type == "cuda" else "cpu")

        # 19 elements under ZeRO-3 with partition size 6 across 4 ranks:
        # Rank 0: [0:6] (6 elements)
        # Rank 1: [6:12] (6 elements)
        # Rank 2: [12:18] (6 elements)
        # Rank 3: [18:19] (1 element, trailing 5 elements are padding in storage!)
        source_map = zero_storage_affine_map(19, [0, 1, 2, 3], partition_numel=6)
        target_map = contiguous_split_map((19, ), [7, 7, 5], partition_dim=0)
        plan = plan_transfer(target_map, source_map)

        full = torch.arange(19.0, dtype=torch.float32)

        source_buffers = {}
        for r, shape in source_map.shard_shapes.items():
            storage = torch.full((6, ), -777.0, device=src_device)
            valid = storage[:shape[0]]
            valid.copy_(source_map.extract(full, r).to(src_device))
            source_buffers[r] = valid

        target_buffers = {}
        target_storage = {}
        for r, shape in target_map.shard_shapes.items():
            storage = torch.full((shape[0] + 4, ), -888.0, device=dst_device)
            valid = storage[2:2 + shape[0]]
            target_buffers[r] = valid
            target_storage[r] = storage

        endpoints = TransferEndpoints({0: 0, 1: 0, 2: 0, 3: 0}, {0: 0, 1: 0, 2: 0})
        execute_transfers([TransferRequest("z3_padding", plan, source_buffers, target_buffers)], endpoints,
                          torch.float32, StagingBudget(bytes_per_peer=16))

        for r, target in target_buffers.items():
            expected = target_map.extract(full, r).to(dst_device)
            assert torch.equal(target, expected)

        # Padding in target storage remained intact
        for r, storage in target_storage.items():
            assert torch.all(storage[:2] == -888.0).item()
            assert torch.all(storage[2 + target_map.shard_shapes[r][0]:] == -888.0).item()

    def test_live_offloaded_optimizer_states_in_place_update(self):
        acc = get_accelerator()
        gpu_device = torch.device(acc.current_device_name() if acc.is_available() else "cpu")
        cpu_device = torch.device("cpu")

        # Simulate ZeRO-2 CPU-offloaded optimizer states:
        # Master weight (fp32) and Adam moments (exp_avg, exp_avg_sq) on CPU
        numel = 24
        full_exp_avg = torch.randn(numel, dtype=torch.float32)

        source_map = contiguous_split_map((numel, ), [12, 12], partition_dim=0)
        target_map = contiguous_split_map((numel, ), [8, 8, 8], partition_dim=0)

        plan = plan_transfer(target_map, source_map)
        endpoints = TransferEndpoints({0: 0, 1: 0}, {0: 0, 1: 0, 2: 0})

        # Transfer CPU offloaded moments to GPU target shards
        cpu_source_exp_avg = {
            0: source_map.extract(full_exp_avg, 0).clone().to(cpu_device),
            1: source_map.extract(full_exp_avg, 1).clone().to(cpu_device)
        }
        gpu_target_exp_avg = {
            r: torch.zeros(target_map.shard_shapes[r], device=gpu_device)
            for r in target_map.shard_shapes
        }

        execute_transfer(plan, cpu_source_exp_avg, gpu_target_exp_avg, endpoints, torch.float32,
                         StagingBudget(bytes_per_peer=16))

        for r, target in gpu_target_exp_avg.items():
            assert torch.equal(target, target_map.extract(full_exp_avg, r).to(gpu_device))

        # Reverse transfer: GPU target back into CPU-offloaded moments, modifying source in-place
        reverse_plan = plan_transfer(source_map, target_map)
        reverse_endpoints = TransferEndpoints({0: 0, 1: 0, 2: 0}, {0: 0, 1: 0})

        # Scale target on GPU
        for target in gpu_target_exp_avg.values():
            target.mul_(2.0)

        execute_transfer(reverse_plan, gpu_target_exp_avg, cpu_source_exp_avg, reverse_endpoints, torch.float32,
                         StagingBudget(bytes_per_peer=16))

        # Assert in-place update landed in CPU source tensors
        for r, src in cpu_source_exp_avg.items():
            expected = (source_map.extract(full_exp_avg, r) * 2.0).to(cpu_device)
            assert torch.equal(src, expected)
