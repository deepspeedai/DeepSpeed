# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.layers import ColumnParallel, RowParallel

from unit.common import DistributedTest


class TestTensorParallelNonContiguousReduce(DistributedTest):
    """The TP all-reduce must reach the returned tensor even when it is not contiguous.

    Reducing a temporary from ``x.contiguous()`` silently drops the result for non-contiguous ``x``.
    """
    world_size = 2

    def test_row_parallel_forward(self):
        rank = dist.get_rank()
        x = torch.full((3, 2), rank + 1.0, device=get_accelerator().device_name()).t()
        assert not x.is_contiguous()

        out = RowParallel.apply(dist.new_group([0, 1]), x, False)

        assert torch.all(out == 3.0)

    def test_column_parallel_backward(self):
        group = dist.new_group([0, 1])
        rank = dist.get_rank()
        x = torch.ones(3, 2, device=get_accelerator().device_name(), requires_grad=True)

        # The transpose makes the incoming gradient non-contiguous.
        out = ColumnParallel.apply(group, x).t()
        (out * (rank + 1)).sum().backward()

        assert torch.all(x.grad == 3.0)
