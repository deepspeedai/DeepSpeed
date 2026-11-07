# Copyright (c) DeepSpeed Team
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime import utils
from deepspeed.runtime.zero.partition_parameters import CUDAQuantizer
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zenflow.zenflow_config import ZenFlowConfig
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


def test_quantized_weight_allgather_rejects_empty_preserved_ranges(monkeypatch):
    group_flat = torch.zeros(20, dtype=torch.float16)
    monkeypatch.setattr(utils.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 2)
    with pytest.raises(ValueError, match="sorted, disjoint"):
        utils.all_gather_quantized_dp_groups([group_flat], [[group_flat[:10], group_flat[10:]]], [object()],
                                             16,
                                             FakeQuantizer(),
                                             8,
                                             preserved_param_ranges=[[(3, 3)]])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantized_weight_allgather_repairs_preserved_ranges(monkeypatch, dtype):
    group_flat = torch.zeros(20, dtype=dtype)
    partitions = [group_flat[:10], group_flat[10:]]
    partitions[0].copy_(torch.arange(10, dtype=dtype) + 0.125)
    quantizer = FakeQuantizer()
    monkeypatch.setattr(utils.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 2)

    def fake_all_gather(output, source, group=None):
        for received in output.chunk(2):
            received.copy_(source)

    monkeypatch.setattr(utils.dist, "all_gather_into_tensor", fake_all_gather)
    utils.all_gather_quantized_dp_groups([group_flat], [partitions], [object()],
                                         16,
                                         quantizer,
                                         8,
                                         preserved_param_ranges=[[(0, 2), (8, 12), (18, 20)]])

    # The fake quantizer truncates to integers; only preserved values retain their fractional parts.
    # The middle preserved range crosses a partition boundary, and the final chunk needs padding.
    expected = torch.tensor([0.125, 1.125, 2, 3, 4, 5, 6, 7, 8.125, 9.125], dtype=dtype)
    assert all(torch.equal(partition, expected) for partition in partitions)


@pytest.mark.parametrize("world_size", [2, 4, 8, 64])
@pytest.mark.parametrize("bucket_size", [32, 70, 256])
def test_quantized_weight_allgather_bounds_gathered_buffers(monkeypatch, world_size, bucket_size):
    expected_partition = torch.arange(65, dtype=torch.float16)
    group_flat = expected_partition.repeat(world_size)
    partitions = list(group_flat.chunk(world_size))
    quantizer = FakeQuantizer()
    collective_dtypes = set()
    monkeypatch.setattr(utils.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: world_size)

    def fake_all_gather(output, source, group=None):
        collective_dtypes.add(source.dtype)
        # Each receive allocation, including the preserved side channel, obeys the aggregate bucket.
        # Quantization requires at least one complete eight-element group from every rank.
        assert output.untyped_storage().nbytes() <= max(bucket_size, world_size * 8) * output.element_size()
        for received in output.chunk(world_size):
            received.copy_(source)

    monkeypatch.setattr(utils.dist, "all_gather_into_tensor", fake_all_gather)
    utils.all_gather_quantized_dp_groups([group_flat], [partitions], [object()],
                                         bucket_size,
                                         quantizer,
                                         8,
                                         preserved_param_ranges=[[(0, group_flat.numel())]])

    assert collective_dtypes == {torch.int8, torch.float32, torch.float16}
    assert all(torch.equal(partition, expected_partition) for partition in partitions)


@pytest.mark.parametrize("partition_grads", [False, True])
@pytest.mark.parametrize("dtype,zenflow_config,error", [(torch.float32, None, TypeError),
                                                        (torch.float16, ZenFlowConfig(), ValueError)])
def test_quantized_weight_allgather_rejects_unsupported_optimizer_initialization(partition_grads, dtype,
                                                                                 zenflow_config, error):
    parameters = [
        torch.nn.Parameter(torch.ones(4, dtype=torch.float16)),
        torch.nn.Parameter(torch.ones(4, dtype=dtype))
    ]
    optimizer = torch.optim.SGD([{"params": [parameter]} for parameter in parameters], lr=0.1)
    originals = [parameter.detach().clone() for parameter in parameters]

    with pytest.raises(error, match="zero_quantized_weights"):
        DeepSpeedZeroOptimizer(optimizer, {
            parameter: str(index)
            for index, parameter in enumerate(parameters)
        },
                               timers=None,
                               optimizer_params={},
                               partition_grads=partition_grads,
                               zenflow_config=zenflow_config,
                               zero_quantized_weights=True)

    assert not optimizer.state
    for group, parameter, original in zip(optimizer.param_groups, parameters, originals):
        assert group["params"][0] is parameter
        assert torch.equal(parameter, original)


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestQuantizedWeightAllGatherTraining(DistributedTest):
    world_size = 2

    def test(self, zero_stage, dtype):
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
                "enabled": dtype == torch.float16,
                "loss_scale": 1.0,
            },
            "bf16": {
                "enabled": dtype == torch.bfloat16,
            },
        }

        torch.manual_seed(42)
        model = SimpleModel(hidden_dim, nlayers=2)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
        data_loader = random_dataloader(model=model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)

        for batch in data_loader:
            loss = model(batch[0], batch[1])
            assert torch.isfinite(loss)
            model.backward(loss)
            model.step()
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

        model.destroy()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestQuantizedWeightAllGatherReconstruction(DistributedTest):
    world_size = 2

    def test(self, dtype):
        if not get_accelerator().is_available():
            pytest.skip("test requires an accelerator")

        expected = (torch.arange(140, device=get_accelerator().current_device_name()) / 17).to(dtype)
        expected[2:4] = 0.0001
        expected[69:72] = 20000
        expected[137:140] = 0.0002
        group_flat = torch.zeros_like(expected)
        partitions = list(group_flat.chunk(self.world_size))
        rank = dist.get_rank()
        partitions[rank].copy_(expected.chunk(self.world_size)[rank])

        utils.all_gather_quantized_dp_groups([group_flat], [partitions], [dist.get_world_group()],
                                             128,
                                             CUDAQuantizer(),
                                             preserved_param_ranges=[[(2, 4), (69, 72), (137, 140)]])

        torch.testing.assert_close(group_flat, expected, atol=0.07, rtol=0)
        for start, end in [(2, 4), (69, 72), (137, 140)]:
            assert torch.equal(group_flat[start:end], expected[start:end])
