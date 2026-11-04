# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed
from deepspeed.runtime.zero.stage_1_and_2 import get_zero_flat_partition_metadata
from unit.common import DistributedTest


class _RecordingOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.add_(param.grad, alpha=-group["lr"])
        return loss


def _config():
    return {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {
            "stage": 1,
        },
        "zero_allow_untested_optimizer": True,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
            },
        },
    }


class TestZero1CustomOptimizerMetadata(DistributedTest):
    world_size = 1

    def test_flat_partition_exposes_original_layout(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        optimizer = _RecordingOptimizer(model.parameters())

        engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                       optimizer=optimizer,
                                                       model_parameters=model.parameters(),
                                                       config=_config())

        flat_partition = optimizer.param_groups[0]["params"][0]
        metadata = get_zero_flat_partition_metadata(flat_partition)

        assert flat_partition.ds_zero_flat_partition
        assert metadata is not None
        assert metadata.group_index == 0
        assert metadata.dp_world_size == 1
        assert metadata.partition_start == 0
        assert metadata.partition_numel == flat_partition.numel()
        assert metadata.global_numel == flat_partition.numel()

        expected = [(param.shape, param.numel()) for param in model.parameters()]
        actual = [(item.shape, item.numel) for item in sorted(metadata.params, key=lambda item: item.original_index)]
        assert actual == expected

        total_numel = 0
        previous_end = 0
        for item in metadata.params:
            assert item.flat_offset >= previous_end
            assert item.padding >= 0
            assert item.local_offset == item.flat_offset
            assert item.local_numel == item.numel
            assert item.requires_grad
            total_numel += item.numel
            previous_end = item.flat_offset + item.numel + item.padding

        assert total_numel == sum(param.numel() for param in model.parameters())
        assert previous_end <= metadata.global_numel

        inputs = torch.randn(2, 4, device=engine.device)
        engine.backward(engine(inputs).sum())
        engine.step()

        assert get_zero_flat_partition_metadata(optimizer.param_groups[0]["params"][0]) is metadata


class TestZero1CustomOptimizerMetadataWorldSize2(DistributedTest):
    world_size = 2

    def test_metadata_describes_local_partition_overlap(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        optimizer = _RecordingOptimizer(model.parameters())

        engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                       optimizer=optimizer,
                                                       model_parameters=model.parameters(),
                                                       config=_config())

        flat_partition = optimizer.param_groups[0]["params"][0]
        metadata = get_zero_flat_partition_metadata(flat_partition)
        rank = deepspeed.comm.get_rank()

        assert metadata is not None
        assert metadata.dp_world_size == 2
        assert metadata.global_numel == 2 * flat_partition.numel()
        assert metadata.partition_start == rank * flat_partition.numel()
        assert metadata.partition_numel == flat_partition.numel()

        for item in metadata.params:
            if item.local_numel:
                assert 0 <= item.local_offset
                assert item.local_offset + item.local_numel <= flat_partition.numel()
            else:
                assert item.local_offset == -1

        if rank == 0:
            first = metadata.params[0]
            assert (first.local_offset, first.local_numel) == (0, first.numel)
            assert all(item.local_numel == 0 for item in metadata.params[1:])
        else:
            assert metadata.params[0].local_numel == 0
            second = metadata.params[1]
            assert (second.local_offset, second.local_numel) == (0, second.numel)
