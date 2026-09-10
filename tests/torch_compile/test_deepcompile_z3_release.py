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

    def _init_dc(self):
        dc = get_deepcompile_handle()
        dc.init(dist.get_world_group(), CompileConfig(deepcompile=True), 1024)
        return dc

    def _register_param(self, dc, graph_id, ds_id, shape, persistent=False):
        device = self._device()
        world_size = dist.get_world_size()
        true_numel = math.prod(shape)
        shard_numel = math.ceil(true_numel / world_size)
        rank = dist.get_rank()
        values = torch.arange(rank * shard_numel, (rank + 1) * shard_numel, device=device, dtype=torch.float32)
        grad_buffer = torch.zeros_like(values)
        dc.register_z3_param(ds_id, list(shape), values, grad_buffer, persistent, values.dtype)
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

    def _configure_arena(self, dc, graph_id, ds_id, shape, occurrence_count, *, bwd=False, offset=0):
        world_size = dist.get_world_size()
        padded_numel = world_size * math.ceil(math.prod(shape) / world_size)
        nbytes = padded_numel * torch.empty((), dtype=torch.float32).element_size()
        capacity = math.ceil((offset + nbytes) / 256) * 256
        dc.configure_z3_gather_arena(graph_id, bwd, capacity, 256, [ds_id] * occurrence_count,
                                     list(range(occurrence_count)), [offset] * occurrence_count,
                                     [nbytes] * occurrence_count, [torch.float32] * occurrence_count,
                                     "test-executor-arena")
        return capacity

    def test_storage_resized_to_zero_after_release_single_use(self):
        graph_id, ds_id = 9010, 9011
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            self._release(view, graph_id, ds_id, 1)
            assert storage.nbytes() == 0
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
            assert storage.nbytes() == 0
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
        graph_id, ds_id = 9040, 9041
        if not hasattr(torch.cuda, "_sleep"):  #ignore-cuda
            pytest.skip("CUDA sleep helper is unavailable")
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, [4097])
            view, storage = self._gather_view_and_storage(shard, graph_id, ds_id)
            padded_bytes = storage.nbytes()
            result = torch.empty((), device=self._device(), dtype=view.dtype)
            consumer_stream = get_accelerator().Stream()
            with get_accelerator().stream(consumer_stream):
                torch.cuda._sleep(int(1e8))  #ignore-cuda
                result.copy_(view.sum())
                self._release(view, graph_id, ds_id, 1, synchronize=False)

            scratch = torch.empty((padded_bytes // view.element_size()) + 1024,
                                  device=self._device(),
                                  dtype=view.dtype)
            scratch.fill_(17)
            get_accelerator().synchronize()
            assert torch.allclose(result, self._expected_view_sum([4097]))
            assert storage.nbytes() == 0
            del scratch
        finally:
            dc.cleanup()

    def test_executor_arena_reuses_backing_across_fused_and_demand_occurrences(self):
        graph_id, ds_id = 9050, 9051
        shape = [4097]
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, shape)
            capacity = self._configure_arena(dc, graph_id, ds_id, shape, occurrence_count=2)
            dc.start_forward()

            prefetched = torch.ops.dc.prefetch_params_fused.default(graph_id, [shard], [ds_id], [torch.float32])
            first = torch.ops.dc.wait_allgather.default(prefetched[0], graph_id, ds_id)
            first_storage = first.untyped_storage()
            first_ptr = first_storage.data_ptr()
            self._release(first, graph_id, ds_id, 1)
            assert first_storage.nbytes() == capacity

            second = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, dtype=torch.float32)
            second = torch.ops.dc.wait_allgather.default(second, graph_id, ds_id)
            second_storage = second.untyped_storage()
            assert second_storage.data_ptr() == first_ptr
            self._release(second, graph_id, ds_id, 1)
            assert second_storage.nbytes() == capacity

            unplanned = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, dtype=torch.float32)
            unplanned = torch.ops.dc.wait_allgather.default(unplanned, graph_id, ds_id)
            unplanned_storage = unplanned.untyped_storage()
            assert unplanned_storage.data_ptr() != first_ptr
            self._release(unplanned, graph_id, ds_id, 1)
            assert unplanned_storage.nbytes() == 0
            assert first_storage.nbytes() == capacity
            dc.end_forward()
        finally:
            dc.cleanup()

    def test_executor_arena_keeps_phase_plans_and_tears_down_each_phase_backing(self):
        graph_id, ds_id = 9060, 9061
        shape = [4097]
        backward_offset = 256
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, shape)
            forward_capacity = self._configure_arena(dc, graph_id, ds_id, shape, occurrence_count=1, bwd=False)
            backward_capacity = self._configure_arena(dc,
                                                      graph_id,
                                                      ds_id,
                                                      shape,
                                                      occurrence_count=1,
                                                      bwd=True,
                                                      offset=backward_offset)

            dc.start_forward()
            forward = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, dtype=torch.float32)
            forward = torch.ops.dc.wait_allgather.default(forward, graph_id, ds_id)
            forward_storage = forward.untyped_storage()
            assert forward_storage.nbytes() == forward_capacity
            assert forward.data_ptr() == forward_storage.data_ptr()
            with pytest.raises(RuntimeError, match="active leases"):
                dc.end_forward()
            self._release(forward, graph_id, ds_id, 1)
            dc.end_forward()
            assert forward_storage.nbytes() == forward_capacity

            dc.start_backward(False)
            backward = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, dtype=torch.float32)
            backward = torch.ops.dc.wait_allgather.default(backward, graph_id, ds_id)
            backward_storage = backward.untyped_storage()
            assert backward_storage.nbytes() == backward_capacity
            assert backward.data_ptr() == backward_storage.data_ptr() + backward_offset
            assert backward_storage.data_ptr() != forward_storage.data_ptr()
            with pytest.raises(RuntimeError, match="active leases"):
                dc.end_backward_phase()
            self._release(backward, graph_id, ds_id, 1)
            dc.end_backward_phase()

            dc.start_forward()
            next_forward = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, dtype=torch.float32)
            next_forward = torch.ops.dc.wait_allgather.default(next_forward, graph_id, ds_id)
            next_forward_storage = next_forward.untyped_storage()
            assert next_forward_storage.nbytes() == forward_capacity
            assert next_forward_storage.data_ptr() not in {
                forward_storage.data_ptr(),
                backward_storage.data_ptr(),
            }
            self._release(next_forward, graph_id, ds_id, 1)
            dc.end_forward()
        finally:
            dc.cleanup()

    def test_disabled_executor_arena_phase_uses_independent_storage(self):
        graph_id, ds_id = 9070, 9071
        shape = [4097]
        dc = self._init_dc()
        try:
            shard = self._register_param(dc, graph_id, ds_id, shape)
            dc.configure_z3_gather_arena(graph_id, False, 0, 256, [], [], [], [], [], "disabled-test")
            dc.start_forward()

            gathered = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, dtype=torch.float32)
            gathered = torch.ops.dc.wait_allgather.default(gathered, graph_id, ds_id)
            storage = gathered.untyped_storage()
            assert storage.nbytes() > 0
            self._release(gathered, graph_id, ds_id, 1)
            assert storage.nbytes() == 0
            dc.end_forward()
        finally:
            dc.cleanup()
