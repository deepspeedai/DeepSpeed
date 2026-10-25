# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.layers import ColumnParallel, LmHeadLinearAllreduce, RowParallel, ScatterToTensorParallelRegion

from unit.common import DistributedTest


def _tp_group(kind):
    """Return this rank's group: a real 2-rank group, a single-rank group, or None (no TP)."""
    if kind == "none":
        return None
    if kind == "pair":
        return dist.new_group([0, 1])
    singles = [dist.new_group([r]) for r in range(dist.get_world_size())]
    return singles[dist.get_rank()]


class TestTensorParallelBackwardReduce(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("kind", ["pair", "single", "none"])
    def test_row_parallel_forward(self, kind):
        group = _tp_group(kind)
        rank = dist.get_rank()
        x = torch.full((3, ), rank + 1.0, device=get_accelerator().device_name())

        out = RowParallel.apply(group, x, False)

        # Only a multi-rank TP group sums the partial outputs (1 + 2); None must not reduce over the world.
        expected = 3.0 if kind == "pair" else rank + 1.0
        assert torch.all(out == expected)

    @pytest.mark.parametrize("kind", ["pair", "single", "none"])
    def test_column_parallel(self, kind):
        group = _tp_group(kind)
        rank = dist.get_rank()
        x = torch.ones(3, device=get_accelerator().device_name(), requires_grad=True)

        (ColumnParallel.apply(group, x) * (rank + 1)).sum().backward()

        # Only a multi-rank TP group sums the replicated input gradient (1 + 2); without TP each
        # rank keeps its own, and None must not fall back to reducing over the world group.
        expected = 3.0 if kind == "pair" else rank + 1.0
        assert torch.all(x.grad == expected)

    @pytest.mark.parametrize("kind", ["pair", "single", "none"])
    def test_scatter_to_tensor_parallel_region(self, kind):
        group = _tp_group(kind)
        rank = dist.get_rank()
        tp_size = 2 if kind == "pair" else 1
        partition_sizes = [4 // tp_size] * tp_size
        tp_index = rank if kind == "pair" else 0
        x = torch.ones(4, device=get_accelerator().device_name(), requires_grad=True)

        out = ScatterToTensorParallelRegion.apply(group, x, partition_sizes, tp_index)
        (out * (rank + 1)).sum().backward()

        # With TP each rank owns one slice, so the full gradient is the concatenation of every rank's
        # slice gradient; without TP the rank owns the whole input.
        if kind == "pair":
            expected = torch.tensor([1.0, 1.0, 2.0, 2.0])
        else:
            expected = torch.full((4, ), rank + 1.0)
        assert torch.equal(x.grad.cpu(), expected)

    @pytest.mark.parametrize("kind", ["pair", "single", "none"])
    def test_lm_head_linear_allreduce(self, kind):
        group = _tp_group(kind)
        device = get_accelerator().device_name()
        torch.manual_seed(0)
        linear = torch.nn.Linear(4, 3, bias=False).to(device)
        x = torch.randn(1, 2, 4, device=device)
        expected = linear(x).detach()

        with torch.no_grad():
            out = LmHeadLinearAllreduce(linear, group)(x)

        # Every layout must reproduce the unsharded projection: a 2-rank group sums the partial products of
        # its input shards, and without TP the single rank holds the full weight and must not reduce again.
        assert torch.allclose(out, expected, atol=1e-5)
