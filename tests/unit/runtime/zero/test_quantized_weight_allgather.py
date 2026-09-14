# Copyright (c) DeepSpeed Team
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime import utils
from deepspeed.runtime.zero import stage_1_and_2
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader


class FakeQuantizer:

    def __init__(self):
        self.quantize_calls = []
        self.quantize_inputs = []
        self.dequantize_calls = []

    def quantize(self, tensor, groups=None):
        self.quantize_calls.append((tensor.numel(), groups))
        self.quantize_inputs.append(tensor.clone())
        return tensor.to(torch.int8), torch.ones((groups, 1), dtype=torch.float32)

    def dequantize(self, tensor, scales):
        self.dequantize_calls.append((tensor.numel(), scales.numel()))
        return tensor.to(torch.float16)


def test_quantized_weight_allgather_chunks_and_removes_padding(monkeypatch):
    group_flat = torch.zeros(20, dtype=torch.float16)
    partitions = [group_flat[:10], group_flat[10:]]
    partitions[0].copy_(torch.arange(10, dtype=torch.float16))
    quantizer = FakeQuantizer()
    monkeypatch.setattr(utils.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 2)

    def fake_all_gather(output, source, group=None):
        output[:source.numel()].copy_(source)
        output[source.numel():2 * source.numel()].copy_(source)

    monkeypatch.setattr(utils.dist, "all_gather_into_tensor", fake_all_gather)
    utils.all_gather_quantized_dp_groups([group_flat], [partitions], [object()], 6, quantizer, 8)

    expected = torch.arange(10, dtype=torch.float16)
    assert torch.equal(partitions[0], expected)
    assert torch.equal(partitions[1], expected)
    assert quantizer.quantize_calls == [(8, 1), (8, 1)]
    assert quantizer.dequantize_calls == [(8, 1)] * 4


def test_quantized_weight_allgather_skips_single_rank(monkeypatch):
    group_flat = torch.arange(10, dtype=torch.float16)
    quantizer = FakeQuantizer()
    monkeypatch.setattr(utils.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 1)

    utils.all_gather_quantized_dp_groups([group_flat], [[group_flat]], [object()], 6, quantizer, 8)

    assert torch.equal(group_flat, torch.arange(10, dtype=torch.float16))
    assert quantizer.quantize_calls == []
    assert quantizer.dequantize_calls == []


def test_quantized_weight_allgather_validates_configuration():
    quantizer = FakeQuantizer()
    try:
        utils.all_gather_quantized_dp_groups([], [], [], 0, quantizer)
    except ValueError as error:
        assert "allgather_bucket_size" in str(error)
    else:
        raise AssertionError("non-positive allgather_bucket_size must fail")


def test_partition_preserved_ranges_handles_partition_boundaries():
    layouts, max_count = utils._partition_preserved_ranges([(2, 4), (9, 12), (17, 20)], 10, 2)

    assert layouts == [[(2, 2, 0), (9, 1, 2)], [(0, 2, 0), (7, 3, 2)]]
    assert max_count == 5


def test_partition_preserved_ranges_rejects_empty_ranges():
    try:
        utils._partition_preserved_ranges([(3, 3)], 10, 2)
    except ValueError as error:
        assert "sorted, disjoint" in str(error)
    else:
        raise AssertionError("empty preserved ranges must fail")


def test_quantized_weight_allgather_repairs_preserved_ranges(monkeypatch):
    group_flat = torch.cat((torch.arange(10), torch.arange(100, 110))).to(torch.float16)
    partitions = [group_flat[:10], group_flat[10:]]
    quantizer = FakeQuantizer()
    monkeypatch.setattr(utils.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 2)

    def fake_all_gather(output, source, group=None):
        if source.dtype == torch.float16:
            output[:2].copy_(source)
            output[2:4].copy_(torch.tensor([102, 103], dtype=torch.float16))
        else:
            output.zero_()

    monkeypatch.setattr(utils.dist, "all_gather_into_tensor", fake_all_gather)
    utils.all_gather_quantized_dp_groups([group_flat], [partitions], [object()],
                                         8,
                                         quantizer,
                                         8,
                                         preserved_param_ranges=[[(2, 4), (12, 14)]])

    assert torch.equal(partitions[0][2:4], torch.tensor([2, 3], dtype=torch.float16))
    assert torch.equal(partitions[1][2:4], torch.tensor([102, 103], dtype=torch.float16))
    assert torch.count_nonzero(partitions[0][:2]) == 0
    assert torch.equal(quantizer.quantize_inputs[0], torch.tensor([0, 1, 0, 0, 4, 5, 6, 7], dtype=torch.float16))
    assert torch.equal(quantizer.quantize_inputs[1], torch.tensor([8, 9, 0, 0, 0, 0, 0, 0], dtype=torch.float16))


def test_quantized_weight_allgather_bounds_preserved_side_channel(monkeypatch):
    group_flat = torch.arange(32, dtype=torch.float16)
    partitions = [group_flat[:16], group_flat[16:]]
    quantizer = FakeQuantizer()
    preserved_collective_sizes = []
    monkeypatch.setattr(utils.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 2)

    def fake_all_gather(output, source, group=None):
        if source.dtype == torch.float16:
            preserved_collective_sizes.append(source.numel())
        output[:source.numel()].copy_(source)
        output[source.numel():2 * source.numel()].copy_(source)

    monkeypatch.setattr(utils.dist, "all_gather_into_tensor", fake_all_gather)
    utils.all_gather_quantized_dp_groups([group_flat], [partitions], [object()],
                                         8,
                                         quantizer,
                                         8,
                                         preserved_param_ranges=[[(0, 32)]])

    assert preserved_collective_sizes == [8, 8]
    assert all(torch.count_nonzero(quantize_input) == 0 for quantize_input in quantizer.quantize_inputs)
    assert torch.equal(partitions[0], torch.arange(16, dtype=torch.float16))
    assert torch.equal(partitions[1], torch.arange(16, dtype=torch.float16))


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestQuantizedWeightAllGatherTraining(DistributedTest):
    world_size = 2

    def test(self, zero_stage):
        if not get_accelerator().is_available():
            pytest.skip("test requires an accelerator")

        hidden_dim = 16
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "zero_quantized_weights": True,
                "allgather_bucket_size": 64,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3,
                    "torch_adam": True,
                },
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.0,
            },
        }

        torch.manual_seed(42)
        model = SimpleModel(hidden_dim, nlayers=2)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        data_loader = random_dataloader(model=model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)

        for batch in data_loader:
            loss = model(batch[0], batch[1])
            assert torch.isfinite(loss)
            model.backward(loss)
            model.step()
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

        model.destroy()


def test_zero_optimizer_forwards_quantized_weight_group_size(monkeypatch):
    captured = {}
    monkeypatch.setattr(stage_1_and_2, "all_gather_quantized_dp_groups", lambda **kwargs: captured.update(kwargs))
    optimizer = object.__new__(stage_1_and_2.DeepSpeedZeroOptimizer)
    optimizer.zero_quantized_weights = True
    optimizer.zero_quantized_weights_group_size = 8192
    optimizer.bit16_groups_flat = []
    optimizer.parallel_partitioned_bit16_groups = []
    optimizer.real_dp_process_group = []
    optimizer.allgather_bucket_size = 50_000_000
    optimizer.weight_quantizer = object()
    optimizer.quantized_weight_preserved_ranges = []

    optimizer._all_gather_weights()

    assert captured["quantization_group_size"] == 8192
