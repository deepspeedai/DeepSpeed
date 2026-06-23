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
