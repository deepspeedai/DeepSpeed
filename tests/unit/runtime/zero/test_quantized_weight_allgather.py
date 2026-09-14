# Copyright (c) DeepSpeed Team
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.runtime import utils


class FakeQuantizer:

    def __init__(self):
        self.quantize_calls = []
        self.dequantize_calls = []

    def quantize(self, tensor, groups=None):
        self.quantize_calls.append((tensor.numel(), groups))
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
