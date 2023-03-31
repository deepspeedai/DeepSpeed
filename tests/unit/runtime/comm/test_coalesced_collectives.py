# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
unit tests for coalesced collectives
"""

import torch
import deepspeed.comm as dist
from deepspeed.runtime.comm.coalesced_collectives import reduce_scatter_coalesced
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest


class TestReduceScatterCoalesced(DistributedTest):
    world_size = 2

    def test_single_input(self):
        input = torch.full((6, ), dist.get_rank(), dtype=torch.half, device=get_accelerator().current_device_name())

        (output, ) = reduce_scatter_coalesced([input], dist.get_world_group())

        assert output.shape == (3, )
        assert torch.allclose(output, torch.full_like(output, 0.5))

    def test_two_inputs(self):
        tensor_kwargs = {"device": get_accelerator().current_device_name(), "dtype": torch.half}
        inputs = [
            dist.get_rank() * torch.arange(0, 6, **tensor_kwargs),
            dist.get_rank() * torch.arange(6, 9, **tensor_kwargs),
        ]

        output1, output2 = reduce_scatter_coalesced(inputs, dist.get_world_group())

        if dist.get_rank() == 0:
            assert output1.shape == (3, )
            assert torch.allclose(output1, torch.arange(0, 3, **tensor_kwargs) / 2)
            assert output2.shape == (2, )
            assert torch.allclose(output2, torch.arange(6, 8, **tensor_kwargs) / 2)
        elif dist.get_rank() == 1:
            assert output1.shape == (3, )
            assert torch.allclose(output1, torch.arange(3, 6, **tensor_kwargs) / 2)
            assert output2.shape == (1, )
            assert torch.allclose(output2, torch.arange(8, 9, **tensor_kwargs) / 2)


class TestReduceScatterCoalescedTensorSmallerThanWorldSize(DistributedTest):
    world_size = 2

    def test(self):
        input = torch.zeros((1, ), dtype=torch.half, device=get_accelerator().current_device_name())

        (output, ) = reduce_scatter_coalesced([input], dist.get_world_group())

        if dist.get_rank() == 0:
            assert output.shape == (1, )
            assert torch.allclose(output, torch.zeros_like(output))
        elif dist.get_rank() == 1:
            assert output.shape == (0, )
