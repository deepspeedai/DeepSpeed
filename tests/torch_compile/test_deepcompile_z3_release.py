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
