# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed.ops.aio import AsyncIOBuilder
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from unit.common import DistributedTest


class _TinyModel(torch.nn.Module):

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.a = torch.nn.Linear(hidden_dim, hidden_dim)
        self.b = torch.nn.Linear(hidden_dim, hidden_dim)
        self.c = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.c(self.b(self.a(x)))


def _config(max_reuse_distance, prefetch_bucket_size=5e8):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {
            "stage": 3,
            "stage3_max_reuse_distance": max_reuse_distance,
            "stage3_prefetch_bucket_size": prefetch_bucket_size,
            # Keep every parameter partitioned so the test observes release
            # behavior instead of the persistence policy.
            "stage3_param_persistence_threshold": 0,
        },
        "zero_allow_untested_optimizer": True,
    }


class TestZero3ForwardParameterRelease(DistributedTest):
    world_size = 1

    def _initialize(self, max_reuse_distance, prefetch_bucket_size=5e8):
        model = _TinyModel()
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1),
            model_parameters=model.parameters(),
            config=_config(max_reuse_distance, prefetch_bucket_size),
        )
        return engine

    @staticmethod
    def _run_to_complete_trace(engine, x):
        # The first forward records the trace and the second one completes it.
        for _ in range(2):
            engine.backward(engine(x).sum())
            engine.step()

    def test_forward_params_are_released_when_reuse_distance_is_zero(self):
        engine = self._initialize(max_reuse_distance=0)
        x = torch.randn(2, 16)

        self._run_to_complete_trace(engine, x)

        # A completed trace would previously retain every forward-gathered
        # parameter until its backward use because the prefetcher scanned far
        # enough ahead to mark the reuse. `stage3_max_reuse_distance=0` must
        # release them as soon as the forward of each module finishes.
        engine(x)
        for name, param in engine.module.named_parameters():
            assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE, \
                f"{name} was not released after its forward use"

    def test_forward_params_are_retained_with_large_reuse_distance(self):
        engine = self._initialize(max_reuse_distance=1e9)
        x = torch.randn(2, 16)

        self._run_to_complete_trace(engine, x)

        # With the default reuse distance the parameters are expected to stay
        # resident through the backward pass instead of being re-gathered.
        engine(x)
        for name, param in engine.module.named_parameters():
            assert param.ds_status == ZeroParamStatus.AVAILABLE, \
                f"{name} was released despite a large reuse distance"

    def test_forward_params_are_released_without_prefetch(self):
        engine = self._initialize(max_reuse_distance=0, prefetch_bucket_size=0)
        x = torch.randn(2, 16)

        self._run_to_complete_trace(engine, x)

        engine(x)
        for name, param in engine.module.named_parameters():
            assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE, \
                f"{name} was not released after its forward use"

    def test_nvme_offload_trains_with_zero_reuse_distance(self, tmpdir):
        # Releasing a prefetch-skipped NVMe parameter also returns its swap-in
        # buffer. Do not regress the ordering that keeps the buffer count in
        # sync with the trace, otherwise training fails intermittently with a
        # "Not enough buffers" assertion in the partition swapper.
        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip("async-io is not available")

        config = _config(max_reuse_distance=0)
        config["zero_optimization"]["offload_param"] = {
            "device": "nvme",
            "nvme_path": str(tmpdir),
        }

        model = _TinyModel()
        engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1),
            model_parameters=model.parameters(),
            config=config,
        )
        x = torch.randn(2, 16, device=engine.device)

        self._run_to_complete_trace(engine, x)
        engine.backward(engine(x).sum())
        engine.step()
