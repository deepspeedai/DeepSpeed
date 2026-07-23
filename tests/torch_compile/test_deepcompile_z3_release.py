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
from unit.common import DistributedTest

pytestmark = pytest.mark.skipif(not is_deepcompile_supported(),
                                reason="DeepCompile requires CUDA and supported PyTorch")


class TestDeepCompileZ3ReleaseStorage(DistributedTest):
    world_size = 2
    non_daemonic_procs = True

    def _device(self):
        return torch.device(get_accelerator().current_device_name())

    def _init_dc(self, pool_budget=1 << 20):
        dc = get_deepcompile_handle()
        dc.init(dist.get_world_group(), CompileConfig(deepcompile=True), 1024)
        if pool_budget is not None:
            dc.set_z3_gather_buffer_pool_budget_for_test(pool_budget)
        return dc

    def _register_param(self, dc, graph_id, ds_id, shape, persistent=False, register_graph=True, dtype=torch.float32):
        device = self._device()
        world_size = dist.get_world_size()
        true_numel = math.prod(shape)
        shard_numel = math.ceil(true_numel / world_size)
        rank = dist.get_rank()
        values = torch.arange(rank * shard_numel, (rank + 1) * shard_numel, device=device,
                              dtype=torch.float32).to(dtype)
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
                "pressure_recovery_pending_entries", "pressure_recovery_in_progress")
        return dict(zip(keys, dc.get_z3_gather_buffer_pool_state_for_test()))

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

    def test_pool_budget_counts_checked_out_storage(self):
        graph_id = 9050
        first_ds_id, checked_out_ds_id, overlapping_ds_id = 9051, 9052, 9053
        dc = self._init_dc(pool_budget=20_000)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            checked_out_shard = self._register_param(dc, graph_id, checked_out_ds_id, [2049], register_graph=False)
            overlapping_shard = self._register_param(dc, graph_id, overlapping_ds_id, [1025], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, checked_out_ds_id, overlapping_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            pool_ptr = first_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)

            checked_out_view, checked_out_storage = self._gather_view_and_storage(checked_out_shard, graph_id,
                                                                                  checked_out_ds_id)
            assert checked_out_storage.data_ptr() == pool_ptr

            overlapping_view, overlapping_storage = self._gather_view_and_storage(overlapping_shard, graph_id,
                                                                                  overlapping_ds_id)
            assert overlapping_storage.data_ptr() != pool_ptr
            self._release(overlapping_view, graph_id, overlapping_ds_id, 1)
            assert overlapping_storage.nbytes() == 0

            self._release(checked_out_view, graph_id, checked_out_ds_id, 1)
            assert checked_out_storage.data_ptr() == pool_ptr
            assert checked_out_storage.nbytes() == first_storage.nbytes()
        finally:
            dc.cleanup()

    def test_zero_pool_budget_uses_resize_to_zero_fallback(self):
        graph_id, ds_id = 9060, 9061
        dc = self._init_dc(pool_budget=0)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_prefetched_storage_is_not_admitted_to_demand_gather_pool(self):
        graph_id, prefetched_ds_id, demand_ds_id = 9070, 9071, 9072
        dc = self._init_dc()
        try:
            prefetched_shard = self._register_param(dc, graph_id, prefetched_ds_id, [4097], register_graph=False)
            demand_shard = self._register_param(dc, graph_id, demand_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [prefetched_ds_id, demand_ds_id])

            torch.ops.dc.prefetch_params_fused.default(graph_id, [prefetched_shard], [prefetched_ds_id])
            prefetched_view, prefetched_storage = self._gather_view_and_storage(prefetched_shard, graph_id,
                                                                                prefetched_ds_id)
            self._release(prefetched_view, graph_id, prefetched_ds_id, 1)
            assert prefetched_storage.nbytes() == 0

            demand_view, demand_storage = self._gather_view_and_storage(demand_shard, graph_id, demand_ds_id)
            self._release(demand_view, graph_id, demand_ds_id, 1)
            assert demand_storage.nbytes() > 0
        finally:
            dc.cleanup()

    def test_prefetch_excludes_existing_pool_storage_from_demand_reuse(self):
        graph_id = 9090
        first_ds_id, prefetched_ds_id, demand_ds_id = 9091, 9092, 9093
        dc = self._init_dc()
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            prefetched_shard = self._register_param(dc, graph_id, prefetched_ds_id, [2049], register_graph=False)
            demand_shard = self._register_param(dc, graph_id, demand_ds_id, [1025], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, prefetched_ds_id, demand_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            pool_ptr = first_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)

            prefetched_view, prefetched_storage = self._gather_view_and_storage(prefetched_shard, graph_id,
                                                                                prefetched_ds_id)
            assert prefetched_storage.data_ptr() == pool_ptr
            dc.set_z3_param_valid_for_test(prefetched_ds_id, False)
            torch.ops.dc.prefetch_params_fused.default(graph_id, [prefetched_shard], [prefetched_ds_id])

            self._release(prefetched_view, graph_id, prefetched_ds_id, 1)
            assert prefetched_storage.nbytes() == 0

            demand_view, demand_storage = self._gather_view_and_storage(demand_shard, graph_id, demand_ds_id)
            self._release(demand_view, graph_id, demand_ds_id, 1)
            assert demand_storage.nbytes() > 0
        finally:
            dc.cleanup()

    def test_prefetch_preparation_failure_rolls_back_storage_exclusion(self):
        graph_id = 9094
        first_ds_id, prefetched_ds_id, demand_ds_id = 9095, 9096, 9097
        dc = self._init_dc()
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            prefetched_shard = self._register_param(dc, graph_id, prefetched_ds_id, [2049], register_graph=False)
            demand_shard = self._register_param(dc, graph_id, demand_ds_id, [1025], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, prefetched_ds_id, demand_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            pool_ptr = first_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)

            prefetched_view, prefetched_storage = self._gather_view_and_storage(prefetched_shard, graph_id,
                                                                                prefetched_ds_id)
            assert prefetched_storage.data_ptr() == pool_ptr
            dc.set_z3_param_valid_for_test(prefetched_ds_id, False)
            dc.set_z3_prefetch_fail_after_exclusions_for_test(1)
            with pytest.raises(RuntimeError, match="injected prefetch preparation failure"):
                torch.ops.dc.prefetch_params_fused.default(graph_id, [prefetched_shard], [prefetched_ds_id])
            dc.set_z3_prefetch_fail_after_exclusions_for_test(0)

            self._release(prefetched_view, graph_id, prefetched_ds_id, 1)
            assert prefetched_storage.nbytes() > 0
            demand_view, demand_storage = self._gather_view_and_storage(demand_shard, graph_id, demand_ds_id)
            assert demand_storage.data_ptr() == pool_ptr
            self._release(demand_view, graph_id, demand_ds_id, 1)
        finally:
            dc.cleanup()

    def test_pressure_recovery_retires_checked_out_working_set_on_final_returns(self):
        graph_id = 9100
        large_ds_id, small_ds_id, transient_ds_id = 9101, 9102, 9103
        dc = self._init_dc(pool_budget=None)
        try:
            large_shard = self._register_param(dc, graph_id, large_ds_id, [1_048_577], register_graph=False)
            small_shard = self._register_param(dc, graph_id, small_ds_id, [524_289], register_graph=False)
            transient_shard = self._register_param(dc, graph_id, transient_ds_id, [262_145], register_graph=False)
            dc.register_graph_z3(graph_id, [large_ds_id, small_ds_id, transient_ds_id])

            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            # Prime two distinct pool entries, then check both out together so
            # recovery cannot depend on an all-idle observation.
            large_view, large_storage = self._gather_view_and_storage(large_shard, graph_id, large_ds_id)
            small_view, small_storage = self._gather_view_and_storage(small_shard, graph_id, small_ds_id)
            self._release(large_view, graph_id, large_ds_id, 1)
            self._release(small_view, graph_id, small_ds_id, 1)
            large_capacity = large_storage.nbytes()
            small_capacity = small_storage.nbytes()
            assert large_capacity > small_capacity

            checked_out_large, checked_out_large_storage = self._gather_view_and_storage(
                large_shard, graph_id, large_ds_id)
            checked_out_small, checked_out_small_storage = self._gather_view_and_storage(
                small_shard, graph_id, small_ds_id)

            before_pressure = self._pool_state(dc)
            assert before_pressure["entries"] == 2
            assert before_pressure["checked_out"] == 2
            assert before_pressure["charged"] == large_capacity + small_capacity

            # Isolated retries preserve the byte-exact working set. Crossing
            # the sustained-pressure threshold latches recovery without
            # reclaiming either live lease.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, 8 << 20, 8 * gib)
            after_pressure = self._pool_state(dc)
            assert after_pressure["budget"] == before_pressure["charged"]
            assert after_pressure["charged"] == before_pressure["charged"]
            assert after_pressure["idle_pressure_score"] == 2

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(3, 8 << 20, 8 * gib)
            after_threshold = self._pool_state(dc)
            assert after_threshold["entries"] == 2
            assert after_threshold["checked_out"] == 2
            assert after_threshold["charged"] == before_pressure["charged"]
            assert after_threshold["idle_pressure_score"] == 0
            assert after_threshold["pressure_recovery_in_progress"] == 1
            assert after_threshold["pressure_recovery_complete"] == 0
            assert after_threshold["pressure_recovery_budget"] == before_pressure["charged"]

            # New demand during the drain cannot extend pool ownership or the
            # captured recovery multiset.
            transient_view, transient_storage = self._gather_view_and_storage(transient_shard, graph_id,
                                                                              transient_ds_id)
            self._release(transient_view, graph_id, transient_ds_id, 1)
            during_drain = self._pool_state(dc)
            assert during_drain["entries"] == 2
            assert during_drain["checked_out"] == 2
            assert during_drain["pressure_recovery_budget"] == before_pressure["charged"]
            assert transient_storage.nbytes() == 0

            # Each final return makes bounded progress. No new pressure sample
            # or all-idle instant is needed to finish the one-shot drain.
            self._release(checked_out_small, graph_id, small_ds_id, 1)
            partial = self._pool_state(dc)
            assert partial["entries"] == 1
            assert partial["checked_out"] == 1
            assert partial["charged"] == large_capacity
            assert partial["pressure_recovery_in_progress"] == 1
            assert partial["pressure_recovery_complete"] == 0
            assert partial["pressure_recovery_pending_entries"] == 1
            assert checked_out_small_storage.nbytes() == 0

            self._release(checked_out_large, graph_id, large_ds_id, 1)
            drained = self._pool_state(dc)
            assert drained["entries"] == 0
            assert drained["charged"] == 0
            assert drained["budget"] == before_pressure["charged"]
            assert drained["enabled"] == 1
            assert drained["pressure_recovery_in_progress"] == 0
            assert drained["pressure_recovery_complete"] == 1
            assert drained["pressure_recovery_pending_entries"] == 2
            assert checked_out_large_storage.nbytes() == 0

            # The complete typed/capacity/device multiset is admitted again.
            recovered_small, recovered_small_storage = self._gather_view_and_storage(
                small_shard, graph_id, small_ds_id)
            self._release(recovered_small, graph_id, small_ds_id, 1)
            recovered_large, recovered_large_storage = self._gather_view_and_storage(
                large_shard, graph_id, large_ds_id)
            self._release(recovered_large, graph_id, large_ds_id, 1)
            recovered_state = self._pool_state(dc)
            assert recovered_state["entries"] == 2
            assert recovered_state["charged"] == before_pressure["charged"]
            assert recovered_state["pressure_recovery_pending_entries"] == 0
            assert recovered_small_storage.nbytes() == small_capacity
            assert recovered_large_storage.nbytes() == large_capacity
        finally:
            dc.cleanup()

    def test_pressure_recovery_retires_multi_user_lease_only_after_final_return(self):
        graph_id, ds_id = 9170, 9171
        dc = self._init_dc(pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, _ = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(first_view, graph_id, ds_id, 1)
            held_view, held_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            held_capacity = held_storage.nbytes()

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)
            assert self._pool_state(dc)["pressure_recovery_in_progress"] == 1

            self._release(held_view, graph_id, ds_id, 2)
            first_return = self._pool_state(dc)
            assert first_return["entries"] == 1
            assert first_return["checked_out"] == 1
            assert first_return["charged"] == held_capacity
            assert first_return["pressure_recovery_in_progress"] == 1
            assert held_storage.nbytes() == held_capacity

            self._release(held_view, graph_id, ds_id, 2)
            final_return = self._pool_state(dc)
            assert final_return["entries"] == 0
            assert final_return["charged"] == 0
            assert final_return["pressure_recovery_in_progress"] == 0
            assert final_return["pressure_recovery_complete"] == 1
            assert held_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_selective_persistence_detaches_reused_storage_from_pool_accounting(self):
        graph_id, second_graph_id, pooled_ds_id, persistent_ds_id = 91006, 91007, 91008, 91009
        dc = self._init_dc(pool_budget=None)
        try:
            pooled_shard = self._register_param(dc, graph_id, pooled_ds_id, [4097], register_graph=False)
            persistent_shard = self._register_param(dc, graph_id, persistent_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [pooled_ds_id, persistent_ds_id])
            dc.register_graph_z3(second_graph_id, [persistent_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            pooled_view, pooled_storage = self._gather_view_and_storage(pooled_shard, graph_id, pooled_ds_id)
            pooled_ptr = pooled_storage.data_ptr()
            self._release(pooled_view, graph_id, pooled_ds_id, 1)
            assert self._pool_state(dc)["entries"] == 1

            dc.set_persistent(persistent_ds_id)
            persistent_view, persistent_storage = self._gather_view_and_storage(persistent_shard, graph_id,
                                                                                persistent_ds_id)
            detached = self._pool_state(dc)
            assert persistent_storage.data_ptr() == pooled_ptr
            assert persistent_storage.nbytes() > 0
            assert detached["entries"] == 0
            assert detached["checked_out"] == 0
            assert detached["charged"] == 0

            dc.set_persistent(persistent_ds_id)
            assert persistent_storage.nbytes() > 0
            assert self._pool_state(dc)["charged"] == 0

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)
            # Keep a tensor dependency so the dispatcher can materialize the
            # Any argument; an empty Python list has no inferable element type.
            torch.ops.dc.end_backward.default([torch.empty(0, device=persistent_view.device)], graph_id, True)
            assert persistent_storage.nbytes() > 0
            assert torch.allclose(persistent_view.sum(), self._expected_view_sum([2049]))
        finally:
            dc.cleanup()

    def test_pressure_recovery_exclusion_retires_without_phantom_target(self):
        graph_id, ds_id = 9172, 9173
        dc = self._init_dc(pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, _ = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(first_view, graph_id, ds_id, 1)
            held_view, held_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)

            dc.set_z3_param_valid_for_test(ds_id, False)
            torch.ops.dc.prefetch_params_fused.default(graph_id, [shard], [ds_id])
            self._release(held_view, graph_id, ds_id, 1)
            excluded = self._pool_state(dc)
            assert excluded["entries"] == 0
            assert excluded["charged"] == 0
            assert excluded["pressure_recovery_in_progress"] == 0
            assert excluded["pressure_recovery_complete"] == 1
            assert excluded["pressure_recovery_budget"] == 0
            assert excluded["pressure_recovery_pending_entries"] == 0
            assert held_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_pressure_recovery_discard_completes_without_phantom_target(self):
        graph_id, ds_id = 9174, 9175
        dc = self._init_dc(pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, _ = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(first_view, graph_id, ds_id, 1)
            held_view, held_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            held_capacity = held_storage.nbytes()
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)

            dc.invalidate_gathered_param(ds_id)
            discarded = self._pool_state(dc)
            assert discarded["entries"] == 0
            assert discarded["charged"] == 0
            assert discarded["pressure_recovery_in_progress"] == 0
            assert discarded["pressure_recovery_complete"] == 1
            assert discarded["pressure_recovery_budget"] == 0
            assert discarded["pressure_recovery_pending_entries"] == 0
            # Discard drops pool ownership but must not mutate a still-live
            # profiling alias.
            assert held_storage.nbytes() == held_capacity
            del held_view
        finally:
            dc.cleanup()

    def test_repeated_allocator_pressure_recovers_once_and_readmits_non_aligned_working_set(self):
        graph_id, ds_id = 9103, 9104
        dc = self._init_dc(pool_budget=None)
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

    def test_repeated_allocator_pressure_preserves_idle_pool_below_budget(self):
        graph_id, ds_id = 9108, 9109
        dc = self._init_dc(pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            before_pressure = self._pool_state(dc)
            assert before_pressure["charged"] < before_pressure["budget"]

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(3, gib, 8 * gib)
            below_budget = self._pool_state(dc)
            assert below_budget["charged"] == before_pressure["charged"]
            assert below_budget["budget"] == before_pressure["budget"]
            assert below_budget["entries"] == 1
            assert below_budget["enabled"] == 1
            assert below_budget["idle_pressure_score"] == 3
            assert storage.nbytes() == before_pressure["charged"]

            # The same sustained score performs one recovery once pressure
            # lowers the budget to the retained charge.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(4, 8 << 20, 8 * gib)
            at_budget = self._pool_state(dc)
            assert at_budget["charged"] == 0
            assert at_budget["budget"] == before_pressure["charged"]
            assert at_budget["entries"] == 0
            assert at_budget["enabled"] == 1
            assert at_budget["idle_pressure_score"] == 0
            assert at_budget["pressure_recovery_complete"] == 1
            assert storage.nbytes() == 0

            recovered_view, recovered_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(recovered_view, graph_id, ds_id, 1)
            assert self._pool_state(dc)["charged"] == before_pressure["charged"]
            assert recovered_storage.nbytes() == before_pressure["charged"]
        finally:
            dc.cleanup()

    def test_allocator_retry_jump_recovers_once_and_preserves_complete_hot_working_set(self):
        graph_id, large_ds_id, small_ds_id = 9105, 9106, 9107
        dc = self._init_dc(pool_budget=None)
        try:
            large_shard = self._register_param(dc, graph_id, large_ds_id, [1_048_577], register_graph=False)
            small_shard = self._register_param(dc, graph_id, small_ds_id, [524_289], register_graph=False)
            dc.register_graph_z3(graph_id, [large_ds_id, small_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            large_view, large_storage = self._gather_view_and_storage(large_shard, graph_id, large_ds_id)
            small_view, small_storage = self._gather_view_and_storage(small_shard, graph_id, small_ds_id)
            self._release(large_view, graph_id, large_ds_id, 1)
            self._release(small_view, graph_id, small_ds_id, 1)
            before_jump = self._pool_state(dc)
            assert before_jump["entries"] == 2
            assert large_storage.nbytes() > small_storage.nbytes()
            large_capacity = large_storage.nbytes()
            small_capacity = small_storage.nbytes()
            working_set_capacity = before_jump["charged"]

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)
            after_jump = self._pool_state(dc)
            assert after_jump["entries"] == 0
            assert after_jump["charged"] == 0
            assert after_jump["budget"] == working_set_capacity
            assert after_jump["enabled"] == 1
            assert after_jump["idle_pressure_score"] == 0
            assert after_jump["pressure_recovery_complete"] == 1
            assert after_jump["pressure_recovery_budget"] == working_set_capacity
            assert after_jump["pressure_recovery_pending_entries"] == 2
            assert large_storage.nbytes() == 0
            assert small_storage.nbytes() == 0

            # A retry in the empty recovery window cannot shrink the budget
            # below the remembered hot-working-set floor.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(6, 8 << 20, 8 * gib)
            empty_window = self._pool_state(dc)
            assert empty_window["entries"] == 0
            assert empty_window["budget"] == working_set_capacity
            assert empty_window["pressure_recovery_budget"] == working_set_capacity
            assert empty_window["pressure_recovery_pending_entries"] == 2

            # Both exact typed/device storage identities are admitted. The
            # smaller entry can arrive first without consuming the larger
            # target's identity or blocking its later admission.
            recovered_small_view, recovered_small_storage = self._gather_view_and_storage(
                small_shard, graph_id, small_ds_id)
            self._release(recovered_small_view, graph_id, small_ds_id, 1)
            after_small_readmit = self._pool_state(dc)
            assert after_small_readmit["entries"] == 1
            assert after_small_readmit["charged"] == small_capacity
            assert after_small_readmit["pressure_recovery_pending_entries"] == 1

            recovered_view, recovered_storage = self._gather_view_and_storage(large_shard, graph_id, large_ds_id)
            recovered_ptr = recovered_storage.data_ptr()
            self._release(recovered_view, graph_id, large_ds_id, 1)
            after_readmit = self._pool_state(dc)
            assert after_readmit["entries"] == 2
            assert after_readmit["charged"] == working_set_capacity
            assert after_readmit["pressure_recovery_pending_entries"] == 0
            assert recovered_storage.nbytes() == large_capacity

            # Counter regression clears the score but not the lifecycle latch
            # or floor; later retry waves cannot evict the re-admitted working set.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, 8 << 20, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(8, 8 << 20, 8 * gib)
            after_repeat = self._pool_state(dc)
            assert after_repeat["entries"] == 2
            assert after_repeat["charged"] == working_set_capacity
            assert after_repeat["budget"] == working_set_capacity
            assert after_repeat["enabled"] == 1
            assert after_repeat["idle_pressure_score"] == 8
            assert after_repeat["pressure_recovery_complete"] == 1
            assert after_repeat["pressure_recovery_budget"] == working_set_capacity
            assert after_repeat["pressure_recovery_pending_entries"] == 0
            assert recovered_storage.nbytes() == large_capacity

            reused_view, reused_storage = self._gather_view_and_storage(large_shard, graph_id, large_ds_id)
            assert reused_storage.data_ptr() == recovered_ptr
            self._release(reused_view, graph_id, large_ds_id, 1)

            dc.cleanup()
            dc = self._init_dc(pool_budget=None)
            self._register_param(dc, 9108, 9110, [3])
            reset_lifecycle = self._pool_state(dc)
            assert reset_lifecycle["pressure_recovery_complete"] == 0
            assert reset_lifecycle["pressure_recovery_budget"] == 0
            assert reset_lifecycle["pressure_recovery_pending_entries"] == 0
        finally:
            dc.cleanup()

    def test_recovery_multiset_admits_duplicate_targets_once_each(self):
        graph_id = 9113
        first_ds_id, second_ds_id, extra_ds_id = 9114, 9115, 9116
        dc = self._init_dc(pool_budget=None)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [524_289], register_graph=False)
            second_shard = self._register_param(dc, graph_id, second_ds_id, [524_289], register_graph=False)
            extra_shard = self._register_param(dc, graph_id, extra_ds_id, [524_289], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, second_ds_id, extra_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            second_view, second_storage = self._gather_view_and_storage(second_shard, graph_id, second_ds_id)
            assert first_storage.data_ptr() != second_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)
            self._release(second_view, graph_id, second_ds_id, 1)
            before_recovery = self._pool_state(dc)
            assert before_recovery["entries"] == 2
            assert first_storage.nbytes() == second_storage.nbytes()

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)
            recovery = self._pool_state(dc)
            assert recovery["entries"] == 0
            assert recovery["pressure_recovery_budget"] == before_recovery["charged"]
            assert recovery["pressure_recovery_pending_entries"] == 2

            recovered_first, _ = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            recovered_second, _ = self._gather_view_and_storage(second_shard, graph_id, second_ds_id)
            self._release(recovered_first, graph_id, first_ds_id, 1)
            self._release(recovered_second, graph_id, second_ds_id, 1)
            recovered = self._pool_state(dc)
            assert recovered["entries"] == 2
            assert recovered["charged"] == before_recovery["charged"]
            assert recovered["pressure_recovery_pending_entries"] == 0

            held_first, _ = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            held_second, _ = self._gather_view_and_storage(second_shard, graph_id, second_ds_id)
            extra_view, extra_storage = self._gather_view_and_storage(extra_shard, graph_id, extra_ds_id)
            self._release(extra_view, graph_id, extra_ds_id, 1)
            assert extra_storage.nbytes() == 0
            assert self._pool_state(dc)["entries"] == 2
            self._release(held_first, graph_id, first_ds_id, 1)
            self._release(held_second, graph_id, second_ds_id, 1)
            assert self._pool_state(dc)["pressure_recovery_pending_entries"] == 0
        finally:
            dc.cleanup()

    def test_recovery_identity_rejects_equal_byte_incompatible_dtype(self):
        graph_id, hot_ds_id, incompatible_ds_id = 9120, 9121, 9122
        dc = self._init_dc(pool_budget=None)
        try:
            hot_shard = self._register_param(dc,
                                             graph_id,
                                             hot_ds_id, [1_048_576],
                                             register_graph=False,
                                             dtype=torch.float32)
            incompatible_shard = self._register_param(dc,
                                                      graph_id,
                                                      incompatible_ds_id, [2_097_152],
                                                      register_graph=False,
                                                      dtype=torch.bfloat16)
            dc.register_graph_z3(graph_id, [hot_ds_id, incompatible_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            hot_view, hot_storage = self._gather_view_and_storage(hot_shard, graph_id, hot_ds_id)
            self._release(hot_view, graph_id, hot_ds_id, 1)
            hot_capacity = hot_storage.nbytes()
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)
            assert self._pool_state(dc)["pressure_recovery_budget"] == hot_capacity

            incompatible_view, incompatible_storage = self._gather_view_and_storage(
                incompatible_shard, graph_id, incompatible_ds_id)
            assert incompatible_storage.nbytes() == hot_capacity
            self._release(incompatible_view, graph_id, incompatible_ds_id, 1)
            assert incompatible_storage.nbytes() == 0
            assert self._pool_state(dc)["entries"] == 0

            held_incompatible_view, held_incompatible_storage = self._gather_view_and_storage(
                incompatible_shard, graph_id, incompatible_ds_id)
            recovered_hot_view, recovered_hot_storage = self._gather_view_and_storage(hot_shard, graph_id, hot_ds_id)
            recovered_hot_ptr = recovered_hot_storage.data_ptr()
            self._release(recovered_hot_view, graph_id, hot_ds_id, 1)
            recovered = self._pool_state(dc)
            assert recovered["entries"] == 1
            assert recovered["charged"] == hot_capacity

            self._release(held_incompatible_view, graph_id, incompatible_ds_id, 1)
            assert held_incompatible_storage.nbytes() == 0
            assert self._pool_state(dc)["charged"] == hot_capacity

            reused_hot_view, reused_hot_storage = self._gather_view_and_storage(hot_shard, graph_id, hot_ds_id)
            assert reused_hot_storage.data_ptr() == recovered_hot_ptr
            self._release(reused_hot_view, graph_id, hot_ds_id, 1)
        finally:
            dc.cleanup()

    def test_adaptive_hard_cap_discards_checked_out_storage_on_return(self):
        graph_id, first_ds_id, checked_out_ds_id = 9130, 9131, 9132
        dc = self._init_dc(pool_budget=None)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [1_048_577], register_graph=False)
            checked_out_shard = self._register_param(dc, graph_id, checked_out_ds_id, [524_289], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, checked_out_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, _ = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            self._release(first_view, graph_id, first_ds_id, 1)
            checked_out_view, checked_out_storage = self._gather_view_and_storage(checked_out_shard, graph_id,
                                                                                  checked_out_ds_id)

            # total / 32 is a 2 MiB hard cap, below this checked-out lease.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, 8 << 20, 64 << 20)
            state = self._pool_state(dc)
            assert state["budget"] == 2 << 20
            assert state["charged"] > state["budget"]
            assert checked_out_storage.nbytes() > state["budget"]
            self._release(checked_out_view, graph_id, checked_out_ds_id, 1)
            assert checked_out_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_over_cap_recovery_drains_and_selects_largest_fitting_target(self):
        graph_id, large_ds_id, small_ds_id = 9176, 9177, 9178
        dc = self._init_dc(pool_budget=None)
        try:
            large_shard = self._register_param(dc, graph_id, large_ds_id, [1_048_577], register_graph=False)
            small_shard = self._register_param(dc, graph_id, small_ds_id, [524_289], register_graph=False)
            dc.register_graph_z3(graph_id, [large_ds_id, small_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            large_view, large_storage = self._gather_view_and_storage(large_shard, graph_id, large_ds_id)
            small_view, small_storage = self._gather_view_and_storage(small_shard, graph_id, small_ds_id)
            self._release(large_view, graph_id, large_ds_id, 1)
            self._release(small_view, graph_id, small_ds_id, 1)
            large_capacity = large_storage.nbytes()
            small_capacity = small_storage.nbytes()
            assert large_capacity > small_capacity

            # total / 32 is a 6 MiB cap. Both non-aligned targets exceed it
            # together, while the larger target fits by itself.
            hard_cap = 6 << 20
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 192 << 20)
            recovered = self._pool_state(dc)
            assert large_capacity <= hard_cap
            assert large_capacity + small_capacity > hard_cap
            assert recovered["entries"] == 0
            assert recovered["charged"] == 0
            assert recovered["pressure_recovery_complete"] == 1
            assert recovered["pressure_recovery_budget"] == large_capacity
            assert recovered["pressure_recovery_budget"] <= hard_cap
            assert recovered["pressure_recovery_pending_entries"] == 1

            rejected_small, rejected_small_storage = self._gather_view_and_storage(small_shard, graph_id, small_ds_id)
            self._release(rejected_small, graph_id, small_ds_id, 1)
            assert rejected_small_storage.nbytes() == 0
            assert self._pool_state(dc)["entries"] == 0

            admitted_large, admitted_large_storage = self._gather_view_and_storage(large_shard, graph_id, large_ds_id)
            self._release(admitted_large, graph_id, large_ds_id, 1)
            admitted = self._pool_state(dc)
            assert admitted["entries"] == 1
            assert admitted["charged"] == large_capacity
            assert admitted["pressure_recovery_pending_entries"] == 0
            assert admitted_large_storage.nbytes() == large_capacity
        finally:
            dc.cleanup()

    def test_recovery_floor_yields_to_hard_cap_before_hot_readmission(self):
        graph_id, ds_id = 9133, 9134
        dc = self._init_dc(pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            hot_capacity = storage.nbytes()
            assert hot_capacity > 2 << 20

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)
            recovered = self._pool_state(dc)
            assert recovered["pressure_recovery_complete"] == 1
            assert recovered["pressure_recovery_budget"] == hot_capacity
            assert recovered["entries"] == 0

            # total / 32 is now 2 MiB. A remembered target larger than the new
            # cap is removed so the target multiset remains satisfiable.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(6, 8 << 20, 64 << 20)
            capped = self._pool_state(dc)
            assert capped["budget"] == 2 << 20
            assert capped["pressure_recovery_budget"] == 0
            assert capped["pressure_recovery_pending_entries"] == 0

            hot_view, hot_storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(hot_view, graph_id, ds_id, 1)
            after_release = self._pool_state(dc)
            assert after_release["entries"] == 0
            assert after_release["charged"] == 0
            assert hot_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_hard_cap_reclaim_preempts_recovery_below_hot_buffer_size(self):
        graph_id, ds_id = 9135, 9136
        dc = self._init_dc(pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            hot_capacity = storage.nbytes()
            assert hot_capacity > 2 << 20

            # The 2 MiB hard cap cannot hold this target. Recovery must still
            # drain the entry and close its one-shot lifecycle with no target,
            # rather than waiting forever for a budget-fitting all-idle state.
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

    def test_new_demand_evicts_oldest_idle_entry_after_pressure(self):
        graph_id = 9140
        first_ds_id, second_ds_id, demand_ds_id = 9141, 9142, 9143
        dc = self._init_dc(pool_budget=None)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [524_289], register_graph=False)
            second_shard = self._register_param(dc, graph_id, second_ds_id, [262_145], register_graph=False)
            demand_shard = self._register_param(dc, graph_id, demand_ds_id, [786_433], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, second_ds_id, demand_ds_id])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            second_view, second_storage = self._gather_view_and_storage(second_shard, graph_id, second_ds_id)
            self._release(first_view, graph_id, first_ds_id, 1)
            self._release(second_view, graph_id, second_ds_id, 1)

            # Preserve 6 MiB of budget after pressure, then let admission-time
            # LRU reclaim the older first entry for larger new demand.
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(2, 24 << 20, 8 * gib)
            demand_view, _ = self._gather_view_and_storage(demand_shard, graph_id, demand_ds_id)
            self._release(demand_view, graph_id, demand_ds_id, 1)
            assert first_storage.nbytes() == 0
            assert second_storage.nbytes() > 0
        finally:
            dc.cleanup()

    def test_allocator_retry_reset_and_jump_keep_budget_state_consistent(self):
        dc = self._init_dc(pool_budget=None)
        try:
            # Registering a graph keeps the process-global weak pool alive
            # between the individual pressure-seam calls below.
            self._register_param(dc, 9150, 9151, [3])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(10, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(11, gib, 8 * gib)
            initial_state = self._pool_state(dc)
            assert initial_state["budget"] == 256 << 20
            assert initial_state["idle_pressure_score"] == 1

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, 8 << 20, 8 * gib)
            reset_state = self._pool_state(dc)
            assert reset_state["retries"] == 0
            assert reset_state["budget"] == 256 << 20
            assert reset_state["idle_pressure_score"] == 0

            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 16 << 20, 8 * gib)
            jump_state = self._pool_state(dc)
            assert jump_state["retries"] == 5
            assert jump_state["budget"] == 4 << 20
            assert jump_state["idle_pressure_score"] == 5
        finally:
            dc.cleanup()

    def test_budget_lowering_discards_checked_out_storage_on_return(self):
        graph_id, first_ds_id, checked_out_ds_id = 9110, 9111, 9112
        dc = self._init_dc(pool_budget=20_000)
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            checked_out_shard = self._register_param(dc, graph_id, checked_out_ds_id, [2049], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, checked_out_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            self._release(first_view, graph_id, first_ds_id, 1)
            checked_out_view, checked_out_storage = self._gather_view_and_storage(checked_out_shard, graph_id,
                                                                                  checked_out_ds_id)
            assert checked_out_storage.data_ptr() == first_storage.data_ptr()

            dc.set_z3_gather_buffer_pool_budget_for_test(0)
            self._release(checked_out_view, graph_id, checked_out_ds_id, 1)
            assert checked_out_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_cleanup_clears_process_global_test_override(self):
        first_graph_id, first_ds_id = 9120, 9121
        dc = self._init_dc()
        first_shard = self._register_param(dc, first_graph_id, first_ds_id, [4097])
        first_view, first_storage = self._gather_view_and_storage(first_shard, first_graph_id, first_ds_id)
        self._release(first_view, first_graph_id, first_ds_id, 1)
        assert first_storage.nbytes() > 0
        dc.cleanup()

        second_graph_id, second_ds_id = 9122, 9123
        dc = self._init_dc(pool_budget=None)
        try:
            second_shard = self._register_param(dc, second_graph_id, second_ds_id, [4097])
            second_view, second_storage = self._gather_view_and_storage(second_shard, second_graph_id, second_ds_id)
            self._release(second_view, second_graph_id, second_ds_id, 1)
            assert second_storage.nbytes() == 0
        finally:
            dc.cleanup()

    def test_cleanup_is_idempotent_and_releases_registry_state(self):
        graph_id, ds_id = 9160, 9161
        dc = self._init_dc()
        first_shard = self._register_param(dc, graph_id, ds_id, [4097])
        first_view, _ = self._gather_view_and_storage(first_shard, graph_id, ds_id)
        self._release(first_view, graph_id, ds_id, 1)

        dc.cleanup()
        dc.cleanup()

        dc = self._init_dc()
        try:
            second_shard = self._register_param(dc, graph_id, ds_id, [2049])
            second_view, _ = self._gather_view_and_storage(second_shard, graph_id, ds_id)
            assert torch.allclose(second_view.sum(), self._expected_view_sum([2049]))
            self._release(second_view, graph_id, ds_id, 1)
        finally:
            dc.cleanup()

    def test_profile_invalidation_discards_checked_out_pool_storage_immediately(self):
        graph_id = 9080
        first_ds_id, invalidated_ds_id, next_ds_id, reused_ds_id = 9081, 9082, 9083, 9084
        dc = self._init_dc()
        try:
            first_shard = self._register_param(dc, graph_id, first_ds_id, [4097], register_graph=False)
            invalidated_shard = self._register_param(dc, graph_id, invalidated_ds_id, [2049], register_graph=False)
            next_shard = self._register_param(dc, graph_id, next_ds_id, [1025], register_graph=False)
            reused_shard = self._register_param(dc, graph_id, reused_ds_id, [513], register_graph=False)
            dc.register_graph_z3(graph_id, [first_ds_id, invalidated_ds_id, next_ds_id, reused_ds_id])

            first_view, first_storage = self._gather_view_and_storage(first_shard, graph_id, first_ds_id)
            pool_ptr = first_storage.data_ptr()
            self._release(first_view, graph_id, first_ds_id, 1)

            invalidated_view, invalidated_storage = self._gather_view_and_storage(invalidated_shard, graph_id,
                                                                                  invalidated_ds_id)
            assert invalidated_storage.data_ptr() == pool_ptr
            dc.invalidate_gathered_param(invalidated_ds_id)

            # The invalidated view deliberately remains alive here. A new gather must
            # not acquire the storage that profiling removed from pool ownership.
            next_view, next_storage = self._gather_view_and_storage(next_shard, graph_id, next_ds_id)
            assert next_storage.data_ptr() != pool_ptr

            self._release(next_view, graph_id, next_ds_id, 1)
            dc.clear_all_gathered_params()
            del invalidated_view
            get_accelerator().synchronize()

            reused_view, reused_storage = self._gather_view_and_storage(reused_shard, graph_id, reused_ds_id)
            assert reused_storage.data_ptr() == next_storage.data_ptr()
            self._release(reused_view, graph_id, reused_ds_id, 1)
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

    def test_pressure_recovery_waits_for_recorded_consumer_stream_before_reallocation(self):
        graph_id, ds_id = 9179, 9180
        if not hasattr(torch.cuda, "_sleep"):  #ignore-cuda
            pytest.skip("CUDA sleep helper is unavailable")
        dc = self._init_dc(pool_budget=None)
        try:
            shard = self._register_param(dc, graph_id, ds_id, [1_048_577])
            gib = 1 << 30
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(0, gib, 8 * gib)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(1, gib, 8 * gib)

            first_view, _ = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(first_view, graph_id, ds_id, 1)
            held_view, _ = self._gather_view_and_storage(shard, graph_id, ds_id)
            dc.update_z3_gather_buffer_pool_allocator_pressure_for_test(5, 8 << 20, 8 * gib)

            result = torch.empty((), device=self._device(), dtype=held_view.dtype)
            consumer_stream = get_accelerator().Stream()
            with get_accelerator().stream(consumer_stream):
                torch.cuda._sleep(int(1e8))  #ignore-cuda
                result.copy_(held_view.sum())
                self._release(held_view, graph_id, ds_id, 1, synchronize=False)

            # The next acquire performs the one-shot allocator flush. The
            # recorded consumer stream must finish before its retired block can
            # participate in the new gather allocation.
            next_view, _ = self._gather_view_and_storage(shard, graph_id, ds_id)
            get_accelerator().synchronize()
            assert torch.allclose(result, self._expected_view_sum([1_048_577]))
            assert torch.allclose(next_view.sum(), self._expected_view_sum([1_048_577]))
            self._release(next_view, graph_id, ds_id, 1)
        finally:
            dc.cleanup()
