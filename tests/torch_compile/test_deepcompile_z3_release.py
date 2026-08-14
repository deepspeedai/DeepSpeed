# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.compile.config import CompileConfig
from deepspeed.compile.util import get_deepcompile_handle, is_deepcompile_supported
from torch._subclasses.fake_tensor import FakeTensorMode
from unit.common import DistributedTest

pytestmark = pytest.mark.skipif(not is_deepcompile_supported(),
                                reason="DeepCompile requires CUDA and supported PyTorch")


class TestDeepCompileZ3ReleaseStorage(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    def _device(self):
        return torch.device(get_accelerator().current_device_name())

    def _init_dc(self, fixed_pool_budget=1 << 20):
        dc = get_deepcompile_handle()
        dc.init(dist.get_world_group(), CompileConfig(deepcompile=True), 1024)
        if fixed_pool_budget is not None:
            dc.set_z3_gather_buffer_pool_fixed_budget_for_test(fixed_pool_budget)
        return dc

    def _register_param(self, dc, graph_id, ds_id, shape, persistent=False, register_graph=True):
        device = self._device()
        world_size = dist.get_world_size()
        true_numel = math.prod(shape)
        shard_numel = math.ceil(true_numel / world_size)
        rank = dist.get_rank()
        values = torch.arange(rank * shard_numel, (rank + 1) * shard_numel, device=device, dtype=torch.float32)
        grad_buffer = torch.zeros_like(values)
        dc.register_z3_param(ds_id, list(shape), values, grad_buffer, persistent, values.dtype)
        if register_graph:
            dc.register_graph_z3(graph_id, [ds_id])
        return values

    def _gather_view_and_storage(self, shard, graph_id, ds_id):
        gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id)
        gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)
        view = gathered.reshape(-1).narrow(0, 0, gathered.numel() - 1)
        assert view.untyped_storage().data_ptr() == gathered.untyped_storage().data_ptr()
        storage = view.untyped_storage()
        assert storage.nbytes() >= gathered.numel() * gathered.element_size()
        return view, storage

    def _release(self, view, graph_id, ds_id, n_users, synchronize=True):
        torch.ops.dc.release_param.default(view, graph_id, ds_id, n_users)
        if synchronize:
            get_accelerator().synchronize()

    def _expected_view_sum(self, shape):
        world_size = dist.get_world_size()
        shard_numel = math.ceil(math.prod(shape) / world_size)
        values = torch.arange(0, world_size * shard_numel, dtype=torch.float32, device=self._device())
        values = values[:math.prod(shape)].reshape(-1)
        return values.narrow(0, 0, values.numel() - 1).sum()

    def _pool_state(self, dc):
        keys = ("budget", "charged", "high_water", "entries", "checked_out", "retries", "enabled", "initialized",
                "idle_pressure_score", "pressure_recovery_complete", "pressure_recovery_budget",
                "pressure_recovery_pending_entries", "pressure_recovery_in_progress", "arena_transition_started",
                "arena_transition_flush_complete")
        return dict(zip(keys, dc.get_z3_gather_buffer_pool_state_for_test()))

    def _arena_state(self, dc, graph_id):
        keys = ("enabled", "phases", "capacity", "planned_max_live", "backing_allocations", "request_bytes",
                "reuse_bytes", "fallback_bytes", "active_bytes", "high_water_bytes", "active_leases",
                "high_water_slices", "event_waits", "overlaps", "pointer_stable", "rank_consistency_checks",
                "rank_plan_mismatches", "relocations", "stale_registry_releases")
        return dict(zip(keys, dc.get_z3_prefetch_arena_state_for_test(graph_id)))

    def _configure_arena(self,
                         dc,
                         graph_id,
                         plan_ids,
                         ds_ids,
                         offsets,
                         request_bytes,
                         capacity=512,
                         phase=0,
                         require_backward=False):
        dc.configure_z3_prefetch_arena(graph_id, phase, require_backward, capacity, capacity, 12345 + phase, plan_ids,
                                       ds_ids, offsets, request_bytes)

    def _prefetch(self, shards, graph_id, ds_ids, plan_id, dtypes=None):
        torch.ops.dc.prefetch_params_fused.default(graph_id, shards, ds_ids, dtypes, plan_id)

    def test_prefetch_arena_distinct_slices_retrieval_and_stable_backing(self):
        graph_id, first_id, second_id = 9200, 9201, 9202
        dc = self._init_dc()
        try:
            first = self._register_param(dc, graph_id, first_id, [3], register_graph=False)
            second = self._register_param(dc, graph_id, second_id, [5], register_graph=False)
            dc.register_graph_z3(graph_id, [first_id, second_id])
            # The optimization pass registers this metadata while Dynamo's
            # FakeTensorMode is active.  Registration must not allocate or
            # inspect the real CUDA backing in that context.
            with FakeTensorMode(allow_non_fake_inputs=True):
                self._configure_arena(dc, graph_id, [10, 10], [first_id, second_id], [0, 256], [16, 32])

            before = self._arena_state(dc, graph_id)
            self._prefetch([first, second], graph_id, [first_id, second_id], 10)
            first_view, first_storage = self._gather_view_and_storage(first, graph_id, first_id)
            second_view, second_storage = self._gather_view_and_storage(second, graph_id, second_id)

            assert first_storage.data_ptr() + 256 == second_view.data_ptr()
            assert first_storage.data_ptr() == second_storage.data_ptr()
            assert torch.allclose(first_view.sum(), self._expected_view_sum([3]))
            assert torch.allclose(second_view.sum(), self._expected_view_sum([5]))
            self._release(first_view, graph_id, first_id, 1)
            self._release(second_view, graph_id, second_id, 1)
            after = self._arena_state(dc, graph_id)
            # Python graph compilation only registers immutable plan metadata;
            # the real CUDA backing is materialized by the first native fused
            # prefetch after cross-rank consensus.
            assert before["backing_allocations"] == 0
            assert after["backing_allocations"] == 1
            assert after["pointer_stable"] == 1
            assert after["fallback_bytes"] == 0
            assert after["active_leases"] == after["overlaps"] == 0
            assert first_storage.nbytes() == second_storage.nbytes() == 512
        finally:
            dc.cleanup()

    def test_prefetch_arena_final_release_reuses_slice_after_consumer_stream(self):
        graph_id, first_id, second_id = 9210, 9211, 9212
        if not hasattr(torch.cuda, "_sleep"):  #ignore-cuda
            pytest.skip("CUDA sleep helper is unavailable")
        dc = self._init_dc()
        try:
            first = self._register_param(dc, graph_id, first_id, [4097], register_graph=False)
            second = self._register_param(dc, graph_id, second_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [first_id, second_id])
            first_bytes = first.numel() * dist.get_world_size() * first.element_size()
            second_bytes = second.numel() * dist.get_world_size() * second.element_size()
            capacity = math.ceil(max(first_bytes, second_bytes) / 256) * 256
            self._configure_arena(dc,
                                  graph_id, [10, 11], [first_id, second_id], [0, 0], [first_bytes, second_bytes],
                                  capacity=capacity)

            self._prefetch([first], graph_id, [first_id], 10)
            first_view, first_storage = self._gather_view_and_storage(first, graph_id, first_id)
            result = torch.empty((), device=self._device(), dtype=first_view.dtype)
            consumer_stream = get_accelerator().Stream()
            with get_accelerator().stream(consumer_stream):
                torch.cuda._sleep(int(1e8))  #ignore-cuda
                result.copy_(first_view.sum())
                self._release(first_view, graph_id, first_id, 1, synchronize=False)

            self._prefetch([second], graph_id, [second_id], 11)
            second_view, second_storage = self._gather_view_and_storage(second, graph_id, second_id)
            get_accelerator().synchronize()

            assert second_storage.data_ptr() == first_storage.data_ptr()
            assert torch.allclose(result, self._expected_view_sum([4097]))
            assert torch.allclose(second_view.sum(), self._expected_view_sum([2049]))
            self._release(second_view, graph_id, second_id, 1)
            state = self._arena_state(dc, graph_id)
            assert state["reuse_bytes"] == second_bytes
            assert state["event_waits"] == 1
            assert state["active_leases"] == 0
            assert first_storage.nbytes() == capacity
        finally:
            dc.cleanup()

    def test_prefetch_arena_shares_one_backing_across_forward_and_backward_phases(self):
        graph_id, forward_id, backward_id = 9213, 9214, 9215
        dc = self._init_dc()
        try:
            forward = self._register_param(dc, graph_id, forward_id, [3], register_graph=False)
            backward = self._register_param(dc, graph_id, backward_id, [5], register_graph=False)
            dc.register_graph_z3(graph_id, [forward_id, backward_id])
            self._configure_arena(dc, graph_id, [10], [forward_id], [0], [16], capacity=512, phase=0)
            self._configure_arena(dc, graph_id, [1_000_010], [backward_id], [0], [32], capacity=256, phase=1)

            self._prefetch([forward], graph_id, [forward_id], 10)
            forward_view, forward_storage = self._gather_view_and_storage(forward, graph_id, forward_id)
            self._release(forward_view, graph_id, forward_id, 1)

            self._prefetch([backward], graph_id, [backward_id], 1_000_010)
            backward_view, backward_storage = self._gather_view_and_storage(backward, graph_id, backward_id)
            self._release(backward_view, graph_id, backward_id, 1)

            state = self._arena_state(dc, graph_id)
            assert state["phases"] == 2
            assert state["capacity"] == 512
            assert state["planned_max_live"] == 512
            assert state["backing_allocations"] == 1
            assert state["fallback_bytes"] == 0
            assert state["active_leases"] == state["overlaps"] == 0
            assert state["pointer_stable"] == 1
            assert forward_storage.data_ptr() == backward_storage.data_ptr()
            assert forward_storage.nbytes() == backward_storage.nbytes() == 512
        finally:
            dc.cleanup()

    def test_training_prefetch_arena_waits_for_backward_phase_before_allocating(self):
        graph_id, forward_id, backward_id, pooled_id = 9233, 9234, 9235, 9236
        dc = self._init_dc()
        try:
            forward = self._register_param(dc, graph_id, forward_id, [3], register_graph=False)
            backward = self._register_param(dc, graph_id, backward_id, [5], register_graph=False)
            pooled = self._register_param(dc, graph_id, pooled_id, [7], register_graph=False)
            dc.register_graph_z3(graph_id, [forward_id, backward_id, pooled_id])

            pooled_view, pooled_storage = self._gather_view_and_storage(pooled, graph_id, pooled_id)
            self._release(pooled_view, graph_id, pooled_id, 1)
            before_transition = self._pool_state(dc)
            assert before_transition["charged"] > 0
            assert pooled_storage.nbytes() > 0

            self._configure_arena(dc,
                                  graph_id, [10], [forward_id], [0], [16],
                                  capacity=512,
                                  phase=0,
                                  require_backward=True)

            self._prefetch([forward], graph_id, [forward_id], 10)
            forward_view, forward_storage = self._gather_view_and_storage(forward, graph_id, forward_id)
            pending = self._arena_state(dc, graph_id)
            assert pending["enabled"] == 0
            assert pending["phases"] == 1
            assert pending["backing_allocations"] == 0
            assert pending["fallback_bytes"] == 0
            self._release(forward_view, graph_id, forward_id, 1)

            self._configure_arena(dc,
                                  graph_id, [1_000_010], [backward_id], [0], [32],
                                  capacity=256,
                                  phase=1,
                                  require_backward=True)
            self._prefetch([backward], graph_id, [backward_id], 1_000_010)
            backward_view, backward_storage = self._gather_view_and_storage(backward, graph_id, backward_id)
            active = self._arena_state(dc, graph_id)
            assert active["enabled"] == 1
            assert active["phases"] == 2
            assert active["backing_allocations"] == 1
            assert forward_storage.data_ptr() != backward_storage.data_ptr()
            after_transition = self._pool_state(dc)
            assert after_transition["charged"] == 0
            assert after_transition["entries"] == 0
            assert after_transition["enabled"] == 0
            assert after_transition["arena_transition_started"] == 1
            assert after_transition["arena_transition_flush_complete"] == 1
            assert pooled_storage.nbytes() == 0
            self._release(backward_view, graph_id, backward_id, 1)
        finally:
            dc.cleanup()

    def test_training_prefetch_arena_session_disable_prevents_forward_only_backing(self):
        graph_id, ds_id = 9237, 9238
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [5])
            self._configure_arena(dc, graph_id, [10], [ds_id], [0], [32], capacity=512, phase=0, require_backward=True)
            dc.disable_z3_prefetch_arena(graph_id, "shared_budget")
            self._prefetch([shard], graph_id, [ds_id], 10)
            gathered, _ = self._gather_view_and_storage(shard, graph_id, ds_id)
            state = self._arena_state(dc, graph_id)
            assert state["enabled"] == 0
            assert state["backing_allocations"] == 0
            assert state["fallback_bytes"] > 0
            self._release(gathered, graph_id, ds_id, 1)
        finally:
            dc.cleanup()

    def test_prefetch_arena_asymmetric_session_disable_reaches_terminal_consensus(self):
        graph_id, first_id, second_id = 9238, 9239, 9240
        dc = self._init_dc()
        try:
            first = self._register_param(dc, graph_id, first_id, [3], register_graph=False)
            second = self._register_param(dc, graph_id, second_id, [5], register_graph=False)
            dc.register_graph_z3(graph_id, [first_id, second_id])
            self._configure_arena(dc, graph_id, [10, 11], [first_id, second_id], [0, 256], [16, 32])
            rank = dist.get_rank()
            if rank == 0:
                dc.disable_z3_prefetch_arena(graph_id, "rank_local_budget")

            self._prefetch([first], graph_id, [first_id], -1 if rank == 0 else 10)
            first_view, first_storage = self._gather_view_and_storage(first, graph_id, first_id)
            first_state = self._arena_state(dc, graph_id)
            assert first_state["enabled"] == 0
            assert first_state["backing_allocations"] == 0
            assert first_state["rank_consistency_checks"] == 1
            assert first_state["active_leases"] == 0
            self._release(first_view, graph_id, first_id, 1)
            assert first_storage.nbytes() == 0

            # The globally disabled decision is terminal, so a later fused
            # prefetch falls back without entering another session collective.
            self._prefetch([second], graph_id, [second_id], -1 if rank == 0 else 11)
            second_view, second_storage = self._gather_view_and_storage(second, graph_id, second_id)
            self._release(second_view, graph_id, second_id, 1)
            final_state = self._arena_state(dc, graph_id)
            assert final_state["enabled"] == 0
            assert final_state["backing_allocations"] == 0
            assert final_state["rank_consistency_checks"] == 1
            assert final_state["active_leases"] == 0
            assert second_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_prefetch_arena_relocates_when_compiler_advances_prefetch(self):
        graph_id, first_id, second_id = 9221, 9222, 9223
        dc = self._init_dc()
        try:
            first = self._register_param(dc, graph_id, first_id, [3], register_graph=False)
            second = self._register_param(dc, graph_id, second_id, [5], register_graph=False)
            dc.register_graph_z3(graph_id, [first_id, second_id])
            self._configure_arena(dc, graph_id, [10, 11], [first_id, second_id], [0, 0], [16, 32], capacity=512)

            self._prefetch([first], graph_id, [first_id], 10)
            first_view, first_storage = self._gather_view_and_storage(first, graph_id, first_id)
            self._prefetch([second], graph_id, [second_id], 11)
            second_view, second_storage = self._gather_view_and_storage(second, graph_id, second_id)

            assert first_storage.data_ptr() + 256 == second_view.data_ptr()
            assert first_storage.data_ptr() == second_storage.data_ptr()
            assert torch.allclose(first_view.sum(), self._expected_view_sum([3]))
            assert torch.allclose(second_view.sum(), self._expected_view_sum([5]))
            self._release(first_view, graph_id, first_id, 1)
            self._release(second_view, graph_id, second_id, 1)
            state = self._arena_state(dc, graph_id)
            assert state["relocations"] == 1
            assert state["overlaps"] == 0
            assert state["fallback_bytes"] == 0
            assert state["active_leases"] == 0
        finally:
            dc.cleanup()

    def test_prefetch_arena_stale_release_preserves_newer_registry_generation(self):
        arena_graph_id, newer_graph_id, ds_id = 9218, 9219, 9220
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, arena_graph_id, ds_id, [5], register_graph=False)
            dc.register_graph_z3(arena_graph_id, [ds_id])
            dc.register_graph_z3(newer_graph_id, [ds_id])
            self._configure_arena(dc, arena_graph_id, [10], [ds_id], [0], [32])

            self._prefetch([shard], arena_graph_id, [ds_id], 10)
            arena_view, arena_storage = self._gather_view_and_storage(shard, arena_graph_id, ds_id)

            # Profiling owns this global invalidation API. A nested/newer graph
            # can then install another gathered tensor before the arena graph's
            # release node executes.
            dc.invalidate_gathered_param(ds_id)
            newer_view, newer_storage = self._gather_view_and_storage(shard, newer_graph_id, ds_id)
            assert newer_storage.data_ptr() != arena_storage.data_ptr()

            self._release(arena_view, arena_graph_id, ds_id, 1)
            latest_view, latest_storage = self._gather_view_and_storage(shard, newer_graph_id, ds_id)
            assert latest_storage.data_ptr() == newer_storage.data_ptr()
            assert torch.allclose(latest_view.sum(), self._expected_view_sum([5]))

            state = self._arena_state(dc, arena_graph_id)
            assert state["active_leases"] == 0
            assert state["stale_registry_releases"] == 1
            assert arena_storage.nbytes() == 512
            self._release(newer_view, newer_graph_id, ds_id, 1)
        finally:
            dc.cleanup()

    def test_prefetch_arena_capacity_exhaustion_falls_back_without_leasing_overlap(self):
        graph_id, first_id, second_id = 9218, 9219, 9220
        dc = self._init_dc()
        try:
            first = self._register_param(dc, graph_id, first_id, [3], register_graph=False)
            second = self._register_param(dc, graph_id, second_id, [5], register_graph=False)
            dc.register_graph_z3(graph_id, [first_id, second_id])
            self._configure_arena(dc, graph_id, [10, 11], [first_id, second_id], [0, 0], [16, 32], capacity=256)

            self._prefetch([first], graph_id, [first_id], 10)
            first_view, first_storage = self._gather_view_and_storage(first, graph_id, first_id)
            self._prefetch([second], graph_id, [second_id], 11)
            second_view, second_storage = self._gather_view_and_storage(second, graph_id, second_id)

            assert first_storage.data_ptr() != second_storage.data_ptr()
            assert torch.allclose(first_view.sum(), self._expected_view_sum([3]))
            assert torch.allclose(second_view.sum(), self._expected_view_sum([5]))
            self._release(first_view, graph_id, first_id, 1)
            self._release(second_view, graph_id, second_id, 1)
            state = self._arena_state(dc, graph_id)
            assert state["relocations"] == 0
            assert state["overlaps"] == 0
            assert state["fallback_bytes"] == second.numel() * dist.get_world_size() * second.element_size()
            assert state["active_leases"] == 0
        finally:
            dc.cleanup()

    def test_prefetch_arena_dtype_mismatch_falls_back_without_resizing_backing(self):
        graph_id, ds_id = 9220, 9221
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [5])
            self._configure_arena(dc, graph_id, [10], [ds_id], [0], [32])

            self._prefetch([shard], graph_id, [ds_id], 10, [torch.float16])
            gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, torch.float16)
            gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)
            fallback_storage = gathered.untyped_storage()
            self._release(gathered, graph_id, ds_id, 1)

            state = self._arena_state(dc, graph_id)
            assert state["fallback_bytes"] == shard.numel() * dist.get_world_size() * 2
            assert state["active_leases"] == 0
            assert state["capacity"] == 512
            assert state["pointer_stable"] == 1
            assert fallback_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_prefetch_arena_capacity_overflow_falls_back_without_resizing_backing(self):
        graph_id, ds_id = 9225, 9226
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [5])
            self._configure_arena(dc, graph_id, [10], [ds_id], [0], [16])

            self._prefetch([shard], graph_id, [ds_id], 10)
            gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id)
            gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)
            fallback_storage = gathered.untyped_storage()
            self._release(gathered, graph_id, ds_id, 1)

            state = self._arena_state(dc, graph_id)
            assert state["fallback_bytes"] == shard.numel() * dist.get_world_size() * shard.element_size()
            assert state["active_leases"] == 0
            assert state["capacity"] == 512
            assert state["pointer_stable"] == 1
            assert fallback_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_prefetch_arena_rank_plan_mismatch_falls_back_on_every_rank(self):
        graph_id, ds_id = 9227, 9228
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [5])
            dc.configure_z3_prefetch_arena(graph_id, 0, False, 512, 512, 12345 + dist.get_rank(), [10], [ds_id], [0],
                                           [32])

            self._prefetch([shard], graph_id, [ds_id], 10)
            gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id)
            gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)
            fallback_storage = gathered.untyped_storage()
            self._release(gathered, graph_id, ds_id, 1)

            state = self._arena_state(dc, graph_id)
            assert state["enabled"] == 0
            assert state["rank_consistency_checks"] == 1
            assert state["rank_plan_mismatches"] == 1
            assert state["active_leases"] == 0
            assert fallback_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_prefetch_arena_missing_plan_on_one_rank_falls_back_on_every_rank(self):
        graph_id, ds_id = 9229, 9230
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [5])
            rank = dist.get_rank()
            if rank == 0:
                self._configure_arena(dc, graph_id, [10], [ds_id], [0], [32])

            # The fused graph carries the same plan ID on every rank even if
            # rank-local metadata prevented one executor from configuring the
            # plan. Both ranks must still enter runtime consensus and disable
            # the arena before the gather.
            self._prefetch([shard], graph_id, [ds_id], 10)
            gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id)
            gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)
            fallback_storage = gathered.untyped_storage()
            self._release(gathered, graph_id, ds_id, 1)

            state = self._arena_state(dc, graph_id)
            assert state["enabled"] == 0
            assert state["rank_consistency_checks"] == 1
            assert state["rank_plan_mismatches"] == 1
            assert state["active_leases"] == 0
            assert fallback_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_prefetch_arena_reset_requires_zero_leases(self):
        graph_id, ds_id = 9230, 9231
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [3])
            self._configure_arena(dc, graph_id, [10], [ds_id], [0], [16])
            self._prefetch([shard], graph_id, [ds_id], 10)
            gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id)
            gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)

            with pytest.raises(RuntimeError, match="active leases"):
                dc.reset()
            assert self._arena_state(dc, graph_id)["active_leases"] == 1

            self._release(gathered, graph_id, ds_id, 1)
            assert self._arena_state(dc, graph_id)["active_leases"] == 0
            dc.reset()
        finally:
            dc.cleanup()

    def test_gather_pool_reclaimable_credit_counts_only_idle_storage(self):
        graph_id, first_id, second_id = 9241, 9242, 9243
        dc = self._init_dc()
        try:
            first = self._register_param(dc, graph_id, first_id, [4097], register_graph=False)
            second = self._register_param(dc, graph_id, second_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [first_id, second_id])

            first_view, first_storage = self._gather_view_and_storage(first, graph_id, first_id)
            self._release(first_view, graph_id, first_id, 1)
            idle_credit = dc.get_z3_gather_buffer_pool_reclaimable_bytes()
            assert idle_credit == first_storage.nbytes() > 0

            second_view, second_storage = self._gather_view_and_storage(second, graph_id, second_id)
            checked_out = self._pool_state(dc)
            assert second_storage.data_ptr() == first_storage.data_ptr()
            assert checked_out["charged"] == idle_credit
            assert checked_out["checked_out"] == 1
            assert dc.get_z3_gather_buffer_pool_reclaimable_bytes() == 0
            assert dc.get_z3_gather_buffer_pool_transition_reclaimable_bytes() == idle_credit

            self._release(second_view, graph_id, second_id, 1)
            returned = self._pool_state(dc)
            assert returned["charged"] == idle_credit
            assert returned["checked_out"] == 0
            assert dc.get_z3_gather_buffer_pool_reclaimable_bytes() == idle_credit
            assert dc.get_z3_gather_buffer_pool_transition_reclaimable_bytes() == idle_credit
        finally:
            dc.cleanup()

    def test_storage_reused_after_release_single_use(self):
        graph_id, ds_id, next_ds_id = 9010, 9011, 9012
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097], register_graph=False)
            next_shard = self._register_param(dc, graph_id, next_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [ds_id, next_ds_id])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_ptr = storage.data_ptr()
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() > 0

            next_view, next_storage = self._gather_view_and_storage(next_shard, graph_id, next_ds_id)
            assert next_storage.data_ptr() == before_ptr
            assert torch.allclose(next_view.sum(), self._expected_view_sum([2049]))
            self._release(next_view, graph_id, next_ds_id, 1)
        finally:
            dc.cleanup()

    def test_storage_nonzero_until_final_release_when_multi_use(self):
        graph_id, ds_id = 9020, 9021
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [3])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_release_nbytes = storage.nbytes()
            self._release(view, graph_id, ds_id, 2)
            assert storage.nbytes() == before_release_nbytes
            self._release(view, graph_id, ds_id, 2)
            assert storage.nbytes() == before_release_nbytes
        finally:
            dc.cleanup()

    def test_persistent_param_storage_unchanged_across_release(self):
        graph_id, ds_id = 9030, 9031
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4], persistent=True)
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_ptr = storage.data_ptr()
            before_nbytes = storage.nbytes()
            self._release(view, graph_id, ds_id, 1)
            assert storage.data_ptr() == before_ptr
            assert storage.nbytes() == before_nbytes
        finally:
            dc.cleanup()

    def test_consumer_stream_can_finish_before_storage_reuse(self):
        graph_id, ds_id, next_ds_id = 9040, 9041, 9042
        if not hasattr(torch.cuda, "_sleep"):  #ignore-cuda
            pytest.skip("CUDA sleep helper is unavailable")
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097], register_graph=False)
            next_shard = self._register_param(dc, graph_id, next_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [ds_id, next_ds_id])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            before_ptr = storage.data_ptr()
            result = torch.empty((), device=self._device(), dtype=view.dtype)
            consumer_stream = get_accelerator().Stream()
            with get_accelerator().stream(consumer_stream):
                torch.cuda._sleep(int(1e8))  #ignore-cuda
                result.copy_(view.sum())
                self._release(view, graph_id, ds_id, 1, synchronize=False)

            next_view, next_storage = self._gather_view_and_storage(next_shard, graph_id, next_ds_id)
            get_accelerator().synchronize()
            assert torch.allclose(result, self._expected_view_sum([4097]))
            assert next_storage.data_ptr() == before_ptr
            assert torch.allclose(next_view.sum(), self._expected_view_sum([2049]))
            assert storage.nbytes() > 0
            self._release(next_view, graph_id, next_ds_id, 1)
        finally:
            dc.cleanup()

    def test_repeated_allocator_pressure_recovers_once_and_readmits_non_aligned_working_set(self):
        graph_id, ds_id = 9103, 9104
        dc = self._init_dc(fixed_pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() > 2 << 20

            before_pressure = self._pool_state(dc)
            assert before_pressure["charged"] % (2 << 20) != 0

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, 8 << 20, 8 * gib)
            after_pressure = self._pool_state(dc)
            assert after_pressure["budget"] == before_pressure["charged"]
            assert after_pressure["charged"] == before_pressure["charged"]
            assert after_pressure["idle_pressure_score"] == 2
            assert storage.nbytes() == before_pressure["charged"]

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(3, 8 << 20, 8 * gib)
            after_threshold = self._pool_state(dc)
            assert after_threshold["budget"] == before_pressure["charged"]
            assert after_threshold["charged"] == 0
            assert after_threshold["entries"] == 0
            assert after_threshold["enabled"] == 1
            assert after_threshold["idle_pressure_score"] == 0
            assert after_threshold["pressure_recovery_complete"] == 1
            assert storage.nbytes() == 0

            recovered_view, recovered_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(recovered_view, graph_id, ds_id, 1)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(100, 8 << 20, 8 * gib)
            after_repeat = self._pool_state(dc)
            assert after_repeat["entries"] == 1
            assert after_repeat["charged"] == before_pressure["charged"]
            assert after_repeat["budget"] == before_pressure["charged"]
            assert after_repeat["pressure_recovery_complete"] == 1
            assert recovered_storage.nbytes() == before_pressure["charged"]
        finally:
            dc.cleanup()

    def test_hard_cap_reclaim_preempts_recovery_below_hot_buffer_size(self):
        graph_id, ds_id = 9135, 9136
        dc = self._init_dc(fixed_pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            hot_capacity = storage.nbytes()
            assert hot_capacity > 2 << 20

            # The 2 MiB hard cap cannot hold this target. Recovery must drain
            # the entry and close its one-shot lifecycle with no target.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 64 << 20)
            capped = self._pool_state(dc)
            assert capped["budget"] == 0
            assert capped["charged"] == 0
            assert capped["entries"] == 0
            assert capped["idle_pressure_score"] == 0
            assert capped["pressure_recovery_complete"] == 1
            assert capped["pressure_recovery_budget"] == 0
            assert capped["pressure_recovery_pending_entries"] == 0
            assert storage.nbytes() == 0

            hot_view, hot_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(hot_view, graph_id, ds_id, 1)
            after_release = self._pool_state(dc)
            assert after_release["entries"] == 0
            assert after_release["charged"] == 0
            assert after_release["pressure_recovery_complete"] == 1
            assert hot_storage.nbytes() == 0
        finally:
            dc.cleanup()
