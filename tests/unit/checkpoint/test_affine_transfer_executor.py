# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Check transfer payloads against extraction from the logical tensor."""

import gc
import math
import os
from dataclasses import replace

import pytest
import torch
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import (ParamAffineMap, contiguous_split_map, nest_affine_maps, replicated_map,
                                         union_affine_maps)
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.affine_transfer_executor import (AffineTransferExecutionError, StagingBudget,
                                                           TransferEndpoints, TransferRequest, TransferWindow,
                                                           _is_packed, execute_transfer, execute_transfers,
                                                           prepare_transfers, schedule_transfers)
from deepspeed.checkpoint.affine_transfer_copy import fused_copies, sequential_copies
from deepspeed.checkpoint.zero_affine import zero_storage_affine_map
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


def _mixed_batch(world, rank):
    """Packed and strided regions over guarded storage windows; endpoints shift by half the world."""
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
    return endpoints, requests, checks, guards


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
        endpoints, requests, checks, guards = _mixed_batch(dist.get_world_size(), dist.get_rank())
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


class TestB2Schedule(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('world_size', [2, 4])
    def test_schedule_is_what_the_executor_runs(self, world_size):
        _join_transfer_group()
        rank = dist.get_rank()
        endpoints, requests, checks, _ = _mixed_batch(world_size, rank)
        # 16 elements per message: single-region rounds, some packed on one side only.
        budget = StagingBudget(64)
        element_size = torch.empty((), dtype=torch.float32).element_size()
        plans = [request.plan for request in requests]
        schedules = schedule_transfers(plans, endpoints, budget, element_size)
        assert sorted(schedules) == list(range(world_size))
        assert schedule_transfers(plans, endpoints, budget, element_size, process=rank) == {rank: schedules[rank]}
        mine = schedules[rank]
        stats = execute_transfers(requests, endpoints, torch.float32, budget)
        for targets, target_map, full in checks:
            _assert_matches(targets, target_map, full)
        assert mine.rounds == stats.rounds
        assert sum(r.elements for rounds in mine.sends.values() for r in rounds) * element_size == stats.sent_bytes
        assert sum(r.elements for rounds in mine.receives.values()
                   for r in rounds) * element_size == stats.received_bytes
        assert mine.local_copy_elements * element_size == stats.local_copy_bytes
        for process, schedule in schedules.items():
            # Each message is one round on both sides; only the in-place decision is per side.
            for peer, rounds in schedule.sends.items():
                assert [(r.elements, r.chunks) for r in rounds] == [(r.elements, r.chunks)
                                                                    for r in schedules[peer].receives[process]]
            for peer, rounds in schedule.receives.items():
                staged = [r.elements for r in rounds if not r.direct]
                assert schedule.receive_arena_elements.get(peer) == (max(staged) if staged else None)
        every = [r for s in schedules.values() for side in (s.sends, s.receives) for rs in side.values() for r in rs]
        # Otherwise this batch would not exercise both the in-place and the staged path.
        assert any(r.direct for r in every) and any(not r.direct for r in every)


def _refill(requests, offset):
    """Shift every source by ``offset`` in place, keeping its storage."""
    for request in requests:
        for buffer in request.source_buffers.values():
            buffer.add_(offset)


class TestB2Prepared(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize('world_size', [2, 4])
    def test_repeated_runs_move_the_current_contents(self, world_size):
        _join_transfer_group()
        rank = dist.get_rank()
        endpoints, requests, checks, guards = _mixed_batch(world_size, rank)
        # 16 elements per message: packed and in-place rounds, as in the schedule test.
        budget = StagingBudget(64)
        prepared = prepare_transfers(requests, endpoints, torch.float32, budget)
        once = execute_transfers(requests, endpoints, torch.float32, budget)
        for run in range(3):
            if run:
                _refill(requests, 1000.0)
            stats = prepared.run()
            assert stats == once
            for targets, target_map, full in checks:
                _assert_matches(targets, target_map, full + 1000.0 * run)
        for storage, size in guards:
            assert torch.equal(storage[:5], torch.full((5, ), -17.0, device=storage.device))
            assert torch.equal(storage[5 + size:], torch.full((12 - 5, ), -17.0, device=storage.device))

    @pytest.mark.parametrize('world_size', [2, 4])
    @pytest.mark.parametrize('fused', [False, True])
    def test_a_copier_runs_every_copy_list(self, world_size, fused):
        _join_transfer_group()
        rank = dist.get_rank()
        endpoints, requests, checks, guards = _mixed_batch(world_size, rank)
        budget = StagingBudget(64)
        lists = []

        def copier(pairs):
            pairs = list(pairs)
            lists.append(pairs)
            return (fused_copies if fused else sequential_copies)(pairs)

        prepared = prepare_transfers(requests, endpoints, torch.float32, budget, copier=copier)
        once = execute_transfers(requests, endpoints, torch.float32, budget)
        # A pack list and an unpack list per round, then the local copies.
        assert len(lists) == 2 * once.rounds + 1
        for run in range(3):
            if run:
                _refill(requests, 1000.0)
            assert prepared.run() == once
            for targets, target_map, full in checks:
                _assert_matches(targets, target_map, full + 1000.0 * run)
        for storage, size in guards:
            assert torch.equal(storage[:5], torch.full((5, ), -17.0, device=storage.device))
            assert torch.equal(storage[5 + size:], torch.full((12 - 5, ), -17.0, device=storage.device))

    @pytest.mark.parametrize('world_size', [2, 4])
    def test_a_released_handle_stages_again_on_its_next_run(self, world_size):
        _join_transfer_group()
        endpoints, requests, checks, guards = _mixed_batch(world_size, dist.get_rank())
        budget = StagingBudget(64)
        prepared = prepare_transfers(requests, endpoints, torch.float32, budget, copier=fused_copies)
        once = execute_transfers(requests, endpoints, torch.float32, budget)
        for run in range(4):
            if run:
                _refill(requests, 1000.0)
            prepared.release_staging()
            assert prepared.run(check=run % 2 == 0) == once
            for targets, target_map, full in checks:
                _assert_matches(targets, target_map, full + 1000.0 * run)
        for storage, size in guards:
            assert torch.equal(storage[:5], torch.full((5, ), -17.0, device=storage.device))
            assert torch.equal(storage[5 + size:], torch.full((12 - 5, ), -17.0, device=storage.device))

    def test_one_process_may_release_on_its_own(self):
        _join_transfer_group()
        endpoints, requests, checks, _ = _mixed_batch(self.world_size, dist.get_rank())
        budget = StagingBudget(64)
        prepared = prepare_transfers(requests, endpoints, torch.float32, budget)
        once = execute_transfers(requests, endpoints, torch.float32, budget)
        for run in range(3):
            if run:
                _refill(requests, 1000.0)
            if dist.get_rank() == run % self.world_size:
                prepared.release_staging()
            assert prepared.run(check=run != 1) == once
            for targets, target_map, full in checks:
                _assert_matches(targets, target_map, full + 1000.0 * run)

    def test_a_process_that_cannot_stage_again_stops_every_process(self):
        _join_transfer_group()
        rank = dist.get_rank()
        endpoints, requests, checks, _ = _mixed_batch(self.world_size, rank)
        prepared = prepare_transfers(requests, endpoints, torch.float32, StagingBudget(64))
        prepared.run()
        _refill(requests, 1000.0)
        prepared.release_staging()
        if rank == self.world_size - 1:

            def out_of_memory():
                raise RuntimeError('out of memory for the staging')

            prepared._restage = out_of_memory
        with pytest.raises(AffineTransferExecutionError, match='could not allocate the released staging'):
            prepared.run()
        for targets, target_map, full in checks:
            _assert_matches(targets, target_map, full)
        # Still released, so the next run stages again and moves the current contents.
        if rank == self.world_size - 1:
            del prepared._restage
        prepared.run()
        for targets, target_map, full in checks:
            _assert_matches(targets, target_map, full + 1000.0)

    @pytest.mark.parametrize('world_size', [2, 4])
    def test_a_buffer_moved_on_one_process_stops_every_process(self, world_size):
        _join_transfer_group()
        rank = dist.get_rank()
        source, target = _row_to_column(world_size)
        full = _full(16, 16, torch.float32)
        endpoints = TransferEndpoints({location: location
                                       for location in source.shard_shapes},
                                      {location: location
                                       for location in target.shard_shapes})
        sources = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == rank
        }
        targets = _targets(target, endpoints, torch.float32)
        request = TransferRequest('weight', plan_transfer(target, source), sources, targets)
        prepared = prepare_transfers([request], endpoints, torch.float32, StagingBudget(64))
        prepared.run()
        _assert_matches(targets, target, full)

        # The last process re-points one target in place, as a reload into new storage would.
        moved = None
        if rank == world_size - 1:
            location = next(iter(targets))
            old = targets[location].data
            targets[location].data = torch.full_like(old, -3.0)
            moved = (location, old)
        before = {location: buffer.clone() for location, buffer in targets.items()}
        for buffer in sources.values():
            buffer.add_(1.0)
        with pytest.raises(AffineTransferExecutionError, match='moved or changed'):
            prepared.run()
        for location, buffer in targets.items():
            assert torch.equal(buffer, before[location])
        if moved is not None:
            assert torch.equal(moved[1], target.extract(full, moved[0]))

    @pytest.mark.parametrize('world_size', [2, 4])
    def test_requests_rebuilt_from_the_live_state_are_checked(self, world_size):
        _join_transfer_group()
        rank = dist.get_rank()
        endpoints, requests, checks, _ = _mixed_batch(world_size, rank)
        prepared = prepare_transfers(requests, endpoints, torch.float32, StagingBudget(64))

        def rebuilt(view):
            return [
                replace(request,
                        source_buffers={
                            location: view(buffer)
                            for location, buffer in request.source_buffers.items()
                        },
                        target_buffers=dict(request.target_buffers)) for request in requests
            ]

        # Fresh views of the same storage, as accessors return them, pass.
        prepared.run(rebuilt(lambda buffer: buffer.view(buffer.shape)))
        for targets, target_map, full in checks:
            _assert_matches(targets, target_map, full)
        # A source copied to new storage on one process stops every process.
        moved = rebuilt(lambda buffer: buffer.clone() if rank == 0 else buffer)
        with pytest.raises(AffineTransferExecutionError, match='moved or changed'):
            prepared.run(moved)
        # So does a batch that is not the prepared one.
        with pytest.raises(AffineTransferExecutionError, match='moved or changed'):
            prepared.run(requests[:-1] if rank == 0 else requests)
        prepared.run(requests)

    def test_processes_running_different_handles_are_refused(self):
        _join_transfer_group()
        rank = dist.get_rank()
        endpoints, requests, _, _ = _mixed_batch(self.world_size, rank)
        whole = prepare_transfers(requests, endpoints, torch.float32, StagingBudget(64))
        first = prepare_transfers(requests[:1], endpoints, torch.float32, StagingBudget(64))
        with pytest.raises(AffineTransferExecutionError, match='not running the same prepared transfer'):
            (whole if rank == 0 else first).run()
        # The refusal came before any data moved, so both handles still run.
        assert whole.run() == execute_transfers(requests, endpoints, torch.float32, StagingBudget(64))
        first.run()

    def test_handles_prepared_alike_are_told_apart(self):
        # Same keys, plans and endpoints, so the same schedule and digest; only the buffers differ.
        _join_transfer_group()
        rank = dist.get_rank()
        endpoints, one, checks, _ = _mixed_batch(self.world_size, rank)
        _, other, other_checks, _ = _mixed_batch(self.world_size, rank)
        _refill(other, 1000.0)
        first = prepare_transfers(one, endpoints, torch.float32, StagingBudget(64))
        second = prepare_transfers(other, endpoints, torch.float32, StagingBudget(64))
        with pytest.raises(AffineTransferExecutionError, match='not running the same prepared transfer'):
            (first if rank == 0 else second).run()
        for targets, _, _ in checks + other_checks:
            for buffer in targets.values():
                assert torch.equal(buffer, torch.full_like(buffer, -17.0))
        first.run()
        second.run()
        for targets, target_map, full in checks:
            _assert_matches(targets, target_map, full)
        for targets, target_map, full in other_checks:
            _assert_matches(targets, target_map, full + 1000.0)


def _window_batch(window, placement, dtype):
    """A row-to-column batch whose sources, targets, both or neither are carved from ``window``."""
    world_size, rank = dist.get_world_size(), dist.get_rank()
    source, target = _row_to_column(world_size)
    endpoints = TransferEndpoints(source={loc: loc
                                          for loc in source.shard_shapes},
                                  target={loc: loc
                                          for loc in target.shard_shapes})
    in_source = placement in ('source', 'both_pull', 'both_push')
    in_target = placement in ('target', 'both_pull', 'both_push')
    sources, targets = {}, {}
    for map_, bindings, buffers, windowed in ((source, endpoints.source, sources, in_source),
                                              (target, endpoints.target, targets, in_target)):
        for location, shape in map_.shard_shapes.items():
            if bindings[location] == rank:
                buffer = window.tensor(shape, dtype) if windowed else torch.empty(shape, dtype=dtype, device=_device())
                buffers[location] = buffer.fill_(UNWRITTEN)
    request = TransferRequest('weight', plan_transfer(target, source, PlanBudget()), sources, targets)
    windowed = (set(source.shard_shapes) if in_source else set(), set(target.shard_shapes) if in_target else set())
    return endpoints, request, source, target, windowed


class TestWindowTransfer(DistributedTest):
    # With four processes the copiers have several peers each, which exercises the staggered copy order.
    world_size = [2, 4]

    @pytest.mark.parametrize('copier', [sequential_copies, fused_copies])
    @pytest.mark.parametrize('placement', ['source', 'target', 'both_pull', 'both_push', 'neither'])
    def test_window_shards_are_copied_in_place_on_every_run(self, placement, copier):
        if get_accelerator().device_name() not in ('cuda', 'xpu'):
            pytest.skip('symmetric memory needs a CUDA or XPU device')
        _join_transfer_group()
        dtype = torch.bfloat16
        world_size, rank = dist.get_world_size(), dist.get_rank()
        # Shard sizes differ between processes, so each one asks for its own window size.
        window = TransferWindow((1 << 20) + rank * 4096, direction='push' if placement == 'both_push' else 'pull')
        endpoints, request, source, target, windowed = _window_batch(window, placement, dtype)
        budget = StagingBudget()
        prepared = prepare_transfers([request], endpoints, dtype, budget, copier=copier, window=window)
        schedule = schedule_transfers([request.plan], endpoints, budget, 2, rank, [windowed])[rank]

        full = _full(16, 16, dtype)
        # Each run must see the sources as rewritten in place since the previous one, unchecked runs included.
        for run, values in enumerate((full, full.flip(0), full.flip(1))):
            for location, buffer in request.source_buffers.items():
                buffer.copy_(source.extract(values, location))
            stats = prepared.run(check=run == 0)
            _assert_matches(request.target_buffers, target, values)
        for buffer in request.target_buffers.values():
            buffer.fill_(UNWRITTEN)
        assert execute_transfers([request], endpoints, dtype, budget, window=window) == stats
        _assert_matches(request.target_buffers, target, values)

        assert schedule.rounds == stats.rounds
        assert sum(r.elements for rounds in schedule.sends.values() for r in rounds) * 2 == stats.sent_bytes
        # Processes 0 and 1 hold the source rows; a target block is local only where its process holds them,
        # which is 1/world_size of the matrix in total.
        remote = full.numel() * (world_size - 1) // world_size * full.element_size()
        moved = torch.tensor([stats.mapped_bytes, stats.sent_bytes, stats.received_bytes], device=_device())
        dist.all_reduce(moved)
        if placement == 'neither':
            assert moved.tolist() == [0, remote, remote]
        else:
            assert moved.tolist() == [remote, 0, 0], 'window shards must bypass staged messages'

    def test_ranks_disagreeing_about_the_window_refuse_everywhere(self):
        if get_accelerator().device_name() not in ('cuda', 'xpu'):
            pytest.skip('symmetric memory needs a CUDA or XPU device')
        _join_transfer_group()
        dtype = torch.bfloat16
        # Each rank would make itself the copier, so both would write every target.
        window = TransferWindow(1 << 20, direction='pull' if dist.get_rank() == 0 else 'push')
        endpoints, request, _, _, _ = _window_batch(window, 'both_pull', dtype)
        with pytest.raises(AffineTransferExecutionError, match='disagree'):
            prepare_transfers([request], endpoints, dtype, window=window)
        for buffer in request.target_buffers.values():
            assert buffer.isnan().all(), 'a refused batch must not write'

    @pytest.mark.parametrize('direction', ['pull', 'push'])
    @pytest.mark.parametrize('prepared', [False, True])
    def test_a_window_region_written_here_while_a_peer_reads_it_is_refused(self, direction, prepared):
        if get_accelerator().device_name() not in ('cuda', 'xpu'):
            pytest.skip('symmetric memory needs a CUDA or XPU device')
        _join_transfer_group()
        dtype, rank = torch.float32, dist.get_rank()
        world_size = dist.get_world_size()
        rows, cols = 4 * world_size, 8
        band = rows // world_size
        split = contiguous_split_map((rows, cols), [band] * world_size, 0)
        # Each process sends its band to the next and receives the previous one's, into the same memory:
        # every operation crosses processes, so every copy goes through the window and none is in a round.
        endpoints = TransferEndpoints(source={location: location
                                              for location in range(world_size)},
                                      target={location: (location + 1) % world_size
                                              for location in range(world_size)})
        window = TransferWindow(1 << 20, direction=direction)
        shard = window.tensor((band, cols), dtype)
        shard.copy_(_full(rows, cols, dtype)[rank * band:(rank + 1) * band])
        before = shard.clone()
        held = (rank - 1) % world_size
        request = TransferRequest('weight', plan_transfer(split, split), {rank: shard}, {held: shard})
        call = prepare_transfers if prepared else execute_transfers
        with pytest.raises(AffineTransferExecutionError, match='rejected the transfer batch') as refused:
            call([request], endpoints, dtype, window=window)
        assert 'a source and a target overlap' in str(refused.value.__cause__)
        assert torch.equal(shard, before), 'a refused batch must not write'


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


class TestB2RelayBandInPlace(DistributedTest):
    """Stage two of a relay: each rank holds one band of the weight inside its own full copy.

    The band is a view of the rank's destination, so the band onto itself is the same memory and is
    skipped, while the bands from the peers land around it.
    """

    @pytest.mark.parametrize('world_size', [2, 4])
    @pytest.mark.parametrize('swapped', [False, True])
    @pytest.mark.parametrize('prepared', [False, True])
    def test_every_rank_completes_its_copy_around_its_own_band(self, world_size, swapped, prepared):
        _join_transfer_group()
        rank, dtype = dist.get_rank(), torch.float32
        rows, cols = 4 * world_size, 6
        full = _full(rows, cols, dtype)
        band = rows // world_size
        source = contiguous_split_map((rows, cols), [band] * world_size, 0)
        target = replicated_map((rows, cols), world_size)
        endpoints = TransferEndpoints(source={location: location
                                              for location in range(world_size)},
                                      target={location: location
                                              for location in range(world_size)})
        mine = torch.full((rows, cols), UNWRITTEN, dtype=dtype, device=_device())
        mine[rank * band:(rank + 1) * band] = full[rank * band:(rank + 1) * band]
        # Swapped: the view is of a peer's rows, which that peer's band then writes.
        held = (rank + 1) % world_size if swapped else rank
        view = mine[held * band:(held + 1) * band]
        before = mine.clone()

        def call():
            plan, budget = plan_transfer(target, source), StagingBudget(band * cols * 4)
            if not prepared:
                return execute_transfer(plan, {rank: view}, {rank: mine}, endpoints, dtype, budget)
            # The band is kept in place by the decision made at prepare, for every run.
            handle = prepare_transfers([TransferRequest('weight', plan, {rank: view}, {rank: mine})], endpoints, dtype,
                                       budget)
            handle.run()
            assert torch.equal(mine, full)
            mine.copy_(before)
            return handle.run()

        if swapped:
            with pytest.raises(AffineTransferExecutionError, match='rejected the transfer batch') as refused:
                call()
            assert 'a source and a target overlap' in str(refused.value.__cause__)
            assert torch.equal(mine.isnan(), before.isnan()) and torch.equal(mine.nan_to_num(), before.nan_to_num())
            return
        stats = call()
        assert torch.equal(mine, full)
        assert stats.local_copy_bytes == 0
        assert stats.sent_bytes == stats.received_bytes == (world_size - 1) * band * cols * 4

    def test_a_prepared_band_rebuilt_at_run_is_checked(self):
        # Whether the band onto itself is copied was decided at prepare. A band that is no longer that view
        # of the destination, or newly is one, is refused on every process before any data moves.
        _join_transfer_group()
        rank, world_size, dtype = dist.get_rank(), dist.get_world_size(), torch.float32
        rows, cols = 4 * world_size, 6
        full = _full(rows, cols, dtype)
        band = rows // world_size
        own = slice(rank * band, (rank + 1) * band)
        plan = plan_transfer(replicated_map((rows, cols), world_size),
                             contiguous_split_map((rows, cols), [band] * world_size, 0))
        bindings = {location: location for location in range(world_size)}
        endpoints, budget = TransferEndpoints(bindings, dict(bindings)), StagingBudget(band * cols * 4)
        mine = torch.full((rows, cols), UNWRITTEN, dtype=dtype, device=_device())
        mine[own] = full[own]
        start = mine.clone()
        kept = [TransferRequest('weight', plan, {rank: mine[own]}, {rank: mine})]
        apart = [TransferRequest('weight', plan, {rank: full[own].clone()}, {rank: mine})]
        in_place = prepare_transfers(kept, endpoints, dtype, budget)
        copied = prepare_transfers(apart, endpoints, dtype, budget)
        assert in_place.run([replace(kept[0], source_buffers={rank: mine[own]})]).local_copy_bytes == 0
        assert torch.equal(mine, full)
        assert copied.run(apart).local_copy_bytes == band * cols * 4
        assert torch.equal(mine, full)

        mine.copy_(start)
        last = rank == world_size - 1
        rebuilt = [
            # The band copied to new storage on one process: the copy it skips would now be needed.
            (in_place, [replace(kept[0], source_buffers={rank: mine[own].clone()})] if last else kept),
            # The band as a negative view on one process.
            (in_place, [replace(kept[0], source_buffers={rank: torch._neg_view(mine[own])})] if last else kept),
            # A band prepared on its own storage, now a view of the destination on one process.
            (copied, kept if last else apart),
        ]
        for handle, requests in rebuilt:
            with pytest.raises(AffineTransferExecutionError, match='moved or changed'):
                handle.run(requests)
        assert torch.equal(mine.isnan(), start.isnan()) and torch.equal(mine.nan_to_num(), start.nan_to_num())


class TestB2LazyViewsRefused(DistributedTest):
    """A conjugate or negative view is refused before data moves, wherever its copy would run.

    Such a view of a destination is not the destination itself, so it cannot stand for a relay band,
    and point-to-point transport does not carry the lazy bit.
    """
    world_size = 2

    @staticmethod
    def _values(kind):
        if kind == 'conj':
            return torch.complex(torch.arange(6.), torch.arange(1., 7.)).to(_device())
        return torch.arange(1., 7., device=_device())

    @staticmethod
    def _lazy(buffer, kind):
        return buffer.conj() if kind == 'conj' else torch._neg_view(buffer)

    @staticmethod
    def _transfer(prepared, plan, sources, targets, endpoints, dtype):
        if prepared:
            # Refused when the handle is prepared, before it can run.
            return prepare_transfers([TransferRequest('weight', plan, sources, targets)], endpoints, dtype)
        return execute_transfer(plan, sources, targets, endpoints, dtype)

    @pytest.mark.parametrize('kind', ['conj', 'neg'])
    @pytest.mark.parametrize('prepared', [False, True])
    def test_a_lazy_view_of_the_destination_is_refused(self, kind, prepared):
        _join_transfer_group()
        rank, world = dist.get_rank(), dist.get_world_size()
        map_ = replicated_map((6, ), world)
        own = {location: location for location in range(world)}
        endpoints = TransferEndpoints(own, dict(own))
        base = self._values(kind)
        before = base.clone()
        with pytest.raises(AffineTransferExecutionError, match='rejected the transfer batch') as refused:
            self._transfer(prepared, plan_transfer(map_, map_), {rank: self._lazy(base, kind)}, {rank: base},
                           endpoints, base.dtype)
        assert 'conjugate or negative view' in str(refused.value.__cause__)
        assert torch.equal(base, before)

    @pytest.mark.parametrize('kind', ['conj', 'neg'])
    @pytest.mark.parametrize('side', ['source', 'target'])
    @pytest.mark.parametrize('prepared', [False, True])
    def test_a_lazy_view_crossing_processes_is_refused_on_both(self, kind, side, prepared):
        _join_transfer_group()
        rank = dist.get_rank()
        map_ = replicated_map((6, ), 1)
        endpoints = TransferEndpoints({0: 0}, {0: 1})
        values = self._values(kind)
        store = torch.zeros_like(values)
        if rank == 0:
            sources, targets = {0: self._lazy(values, kind) if side == 'source' else values}, {}
        else:
            sources, targets = {}, {0: self._lazy(store, kind) if side == 'target' else store}
        with pytest.raises(AffineTransferExecutionError, match='rejected the transfer batch') as refused:
            self._transfer(prepared, plan_transfer(map_, map_), sources, targets, endpoints, values.dtype)
        if rank == (0 if side == 'source' else 1):
            assert 'conjugate or negative view' in str(refused.value.__cause__)
        assert not store.any()


class TestB2TwoLayoutsOnOneProcess(DistributedTest):
    """A process that holds a parameter in two layouts serves each of its buffers under its own location."""
    world_size = 2

    def test_a_union_source_reads_both_buffers_of_one_process(self):
        _join_transfer_group()
        rows, cols = 8, 6
        numel = rows * cols
        zero = nest_affine_maps(replicated_map((rows, cols), 1),
                                {0: zero_storage_affine_map(numel, [100, 101], numel // 2)})
        # Process 0 also keeps row shard 0 of a tensor-parallel copy that is still current.
        layout = contiguous_split_map((rows, cols), [4, 4], 0)
        shard = ParamAffineMap((rows, cols), {0: layout.shard_shapes[0]}, {0: layout.pieces_by_rank[0]})
        source = union_affine_maps([zero, shard])
        target = contiguous_split_map((rows, cols), [3, 3], 1)
        endpoints = TransferEndpoints(source={0: 0, 100: 0, 101: 1}, target={0: 1, 1: 0})
        dtype = torch.float32
        full = _full(rows, cols, dtype)
        source_buffers = {
            location: source.extract(full, location).contiguous()
            for location, process in endpoints.source.items() if process == dist.get_rank()
        }
        target_buffers = _targets(target, endpoints, dtype)

        plan = plan_transfer(target, source, PlanBudget())
        # The lowest holder is read by default, so the row shard serves the rows it holds.
        assert 0 in {segment.source_rank for segment in plan.segments}
        execute_transfer(plan, source_buffers, target_buffers, endpoints, dtype)

        _assert_matches(target_buffers, target, full)


class TestB2TransferCPU:

    def test_packed_regions_are_exactly_the_contiguous_views(self):
        generator = torch.Generator().manual_seed(0)
        storage = torch.empty(4096)
        for _ in range(2000):
            ndim = int(torch.randint(1, 5, (1, ), generator=generator))
            shape = tuple(int(size) for size in torch.randint(1, 4, (ndim, ), generator=generator))
            row_major = [math.prod(shape[axis + 1:]) for axis in range(ndim)]
            # Mostly row-major strides, perturbed on some axes, so both answers occur, including on extent-1 axes.
            strides = tuple(stride if bool(
                torch.rand((), generator=generator) < 0.7) else int(torch.randint(1, 9, (1, ), generator=generator))
                            for stride in row_major)
            view = torch.as_strided(storage, shape, strides, storage_offset=3)
            assert _is_packed(shape, strides) == view.is_contiguous(), (shape, strides)
        for shape, strides in (((0, ), (2, )), ((2, 0, 3), (1, 7, 5))):
            view = torch.as_strided(storage, shape, strides)
            assert _is_packed(shape, strides) == view.is_contiguous(), (shape, strides)

    def test_schedule_of_a_single_process_is_all_local_copies(self):
        source = contiguous_split_map((6, 4), [3, 3], 0)
        target = contiguous_split_map((6, 4), [2, 2], 1)
        plan = plan_transfer(target, source)
        endpoints = TransferEndpoints({0: 0, 1: 0}, {0: 0, 1: 0})
        schedules = schedule_transfers([plan, plan], endpoints, StagingBudget(8), 4)
        assert list(schedules) == [0]
        only = schedules[0]
        assert (only.sends, only.receives, only.rounds) == ({}, {}, 0)
        assert only.local_copy_elements == 2 * 24
        with pytest.raises(AffineTransferExecutionError, match='endpoint binding'):
            schedule_transfers([plan], TransferEndpoints({0: 0}, {0: 0, 1: 0}), StagingBudget(8), 4)
        with pytest.raises(AffineTransferExecutionError, match='element_size'):
            schedule_transfers([plan], endpoints, StagingBudget(8), 0)
        # A budget or process the executor would refuse is refused here too, even for an empty batch.
        for budget in (StagingBudget(0), StagingBudget(3), StagingBudget(8.0)):
            with pytest.raises(AffineTransferExecutionError, match='staging budget'):
                schedule_transfers([], endpoints, budget, 4)
        with pytest.raises(AffineTransferExecutionError, match='not named'):
            schedule_transfers([plan], endpoints, StagingBudget(8), 4, process=1)

    def test_schedule_leaves_window_copies_out_of_the_rounds(self):
        source = contiguous_split_map((6, 4), [3, 3], 0)
        target = contiguous_split_map((6, 4), [2, 2], 1)
        plan = plan_transfer(target, source)
        endpoints = TransferEndpoints({0: 0, 1: 1}, {0: 0, 1: 1})
        staged = schedule_transfers([plan, plan], endpoints, StagingBudget(8), 4)
        assert staged[0].sends and staged[0].receives
        # Only the first parameter's sources lie in a window, so only its cross-process chunks leave the rounds.
        locations = [({0, 1}, set()), (set(), set())]
        windowed = schedule_transfers([plan, plan], endpoints, StagingBudget(8), 4, window_locations=locations)
        single = schedule_transfers([plan], endpoints, StagingBudget(8), 4)
        for process in (0, 1):
            assert windowed[process].sends == single[process].sends
            assert windowed[process].receives == single[process].receives
            assert windowed[process].local_copy_elements == staged[process].local_copy_elements
        with pytest.raises(AffineTransferExecutionError, match='window_locations'):
            schedule_transfers([plan, plan], endpoints, StagingBudget(8), 4, window_locations=locations[:1])

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

    def test_a_band_kept_in_place_is_not_copied(self):
        # A relay rank holds its band as a view of its own destination: that copy moves nothing.
        full = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        target = full.clone()
        target[2:] = -1.0
        source = contiguous_split_map((4, 3), [2, 2], 0)
        stats = execute_transfer(plan_transfer(replicated_map((4, 3), 1), source), {
            0: target[:2],
            1: full[2:].clone()
        }, {0: target}, TransferEndpoints({
            0: 0,
            1: 0
        }, {0: 0}), torch.float32, StagingBudget(8))
        assert torch.equal(target, full)
        assert stats.local_copy_bytes == 2 * 3 * 4

    @pytest.mark.parametrize('case, reason', [('swapped_bands', 'a source and a target overlap'),
                                              ('kept_band_is_read', 'a source and a target overlap'),
                                              ('second_writer', 'two targets overlap')])
    def test_a_view_of_the_target_is_still_refused_where_it_aliases(self, case, reason):
        full = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        target = full.clone()
        before = target.clone()
        source = contiguous_split_map((4, 3), [2, 2], 0)
        plan = plan_transfer(replicated_map((4, 3), 1), source)
        endpoints = TransferEndpoints({0: 0, 1: 0}, {0: 0})
        if case == 'swapped_bands':
            # Each band reads rows that the other band's copy writes.
            requests = [TransferRequest('weight', plan, {0: target[2:], 1: target[:2]}, {0: target})]
        elif case == 'kept_band_is_read':
            # Both bands stay in place, so the first request writes nothing; its bands still count as
            # read, and the second request may not write into one of them.
            rows = replicated_map((2, 3), 1)
            requests = [
                TransferRequest('weight', plan, {
                    0: target[:2],
                    1: target[2:]
                }, {0: target}),
                TransferRequest('other', plan_transfer(rows, rows), {0: torch.zeros(2, 3)}, {0: target[:2]}),
            ]
        else:
            # Two requests write the same rows.
            whole = replicated_map((4, 3), 1)
            requests = [
                TransferRequest('weight', plan, {
                    0: target[:2],
                    1: full[2:].clone()
                }, {0: target}),
                TransferRequest('other', plan_transfer(whole, whole), {0: torch.zeros(4, 3)}, {0: target}),
            ]
        with pytest.raises(AffineTransferExecutionError, match=reason):
            execute_transfers(requests, endpoints, torch.float32, StagingBudget(8))
        assert torch.equal(target, before)

    def test_a_prepared_transfer_writes_where_it_was_prepared(self):
        # Why run checks by default: a re-pointed target keeps receiving into its old storage.
        map_ = replicated_map((2, 3), 1)
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        target = torch.full((2, 3), -1.0)
        prepared = prepare_transfers([TransferRequest('weight', plan_transfer(map_, map_), {0: source}, {0: target})],
                                     TransferEndpoints({0: 0}, {0: 0}), torch.float32, StagingBudget(8))
        old = target.data
        target.data = torch.full((2, 3), -2.0)
        with pytest.raises(AffineTransferExecutionError, match='moved or changed'):
            prepared.run()
        assert torch.equal(old, torch.full((2, 3), -1.0)) and torch.equal(target, torch.full((2, 3), -2.0))
        prepared.run(check=False)
        assert torch.equal(old, source) and torch.equal(target, torch.full((2, 3), -2.0))

    def test_requests_rebuilt_after_a_reload_show_the_move(self):
        # As zero_param_storage_shard: a fresh view of ds_tensor on each call, which ZeRO-3 reload_states
        # re-points with .data = ...; a view taken before that keeps the old storage.
        map_ = replicated_map((2, 3), 1)
        plan = plan_transfer(map_, map_)
        endpoints = TransferEndpoints({0: 0}, {0: 0})
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        ds_tensor = torch.full((6, ), -1.0)

        def requests():
            return [TransferRequest('weight', plan, {0: source}, {0: ds_tensor.data.view(-1).narrow(0, 0, 6)})]

        prepared = prepare_transfers(requests(), endpoints, torch.float32, StagingBudget(8))
        ds_tensor.data = torch.full((6, ), -2.0)
        with pytest.raises(AffineTransferExecutionError, match='moved or changed'):
            prepared.run(requests())
        assert torch.equal(ds_tensor, torch.full((6, ), -2.0))
        # Prepared again after the reload, the batch lands in the new storage.
        prepare_transfers(requests(), endpoints, torch.float32, StagingBudget(8)).run(requests())
        assert torch.equal(ds_tensor, source.view(-1))

    def test_prepared_runs_keep_frozen_parameters_in_place(self):
        map_ = replicated_map((2, 3), 1)
        source = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        target = torch.nn.Parameter(torch.full((2, 3), -1.0))
        identity, pointer = id(target), target.data_ptr()
        prepared = prepare_transfers([TransferRequest('weight', plan_transfer(map_, map_), {0: source}, {0: target})],
                                     TransferEndpoints({0: 0}, {0: 0}), torch.float32, StagingBudget(8))
        for step in range(2):
            with torch.no_grad():
                source.add_(step)
            prepared.run()
            assert id(target) == identity and target.data_ptr() == pointer
            assert target.requires_grad and target.grad is None
            assert torch.equal(target, source)

    @pytest.mark.parametrize('enabled', [True, False])
    def test_preparing_leaves_the_garbage_collector_as_it_was(self, enabled):
        map_ = replicated_map((2, 3), 1)
        request = TransferRequest('weight', plan_transfer(map_, map_), {0: torch.ones(2, 3)}, {0: torch.zeros(2, 3)})
        endpoints = TransferEndpoints({0: 0}, {0: 0})
        was = gc.isenabled()
        try:
            gc.enable() if enabled else gc.disable()
            prepare_transfers([request], endpoints, torch.float32, StagingBudget(8))
            assert gc.isenabled() == enabled
            with pytest.raises(AffineTransferExecutionError, match='at least one element'):
                prepare_transfers([request], endpoints, torch.float32, StagingBudget(2))
            assert gc.isenabled() == enabled
        finally:
            gc.enable() if was else gc.disable()

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


@pytest.mark.parametrize('dtype',
                         [torch.int8, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_fused_copies_match_copy_for_every_region_shape(dtype):
    device = torch.device(get_accelerator().current_device_name())
    values = torch.arange(300 * 520, device=device).reshape(300, 520)
    # Distinct imaginary parts, so a copy of only some of a complex element's bytes shows.
    source = torch.complex(values.double(), -values.double()).to(dtype) if dtype.is_complex else values.to(dtype)
    target = torch.zeros(520, 300, device=device, dtype=dtype)
    arena = torch.zeros(1 << 17, device=device, dtype=dtype)
    pairs = [
        (arena[:300 * 64].view(300, 64), source[:, 128:192]),  # aligned strided rows into an arena
        (target[7:9, :100], source[3, 5:205].view(2, 100)),  # contiguous into strided rows
        (arena[300 * 64 + 3:300 * 64 + 80], source[11, 13:90]),  # unaligned run
        (target[100:400, 17:297], source[:300, 1:281]),  # unaligned strided rows on both sides
        (target[500:510, 0:10].t(), source[0:10, 0:10]),  # transposed: not tileable
    ]
    expected = [destination.clone().copy_(region) for destination, region in pairs]
    fused_copies(pairs)()
    get_accelerator().synchronize()
    for (destination, _), want in zip(pairs, expected):
        assert torch.equal(destination, want)
