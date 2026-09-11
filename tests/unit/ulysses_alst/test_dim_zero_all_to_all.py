# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
UlyssesPlus: the dimension-zero all-to-all that trades sequence shards for head shards
"""

from deepspeed.accelerator import get_accelerator
from deepspeed.sequence.layer import _dim_zero_all_to_all
from unit.common import DistributedTest
from unit.util import torch_assert_close, torch_assert_equal
import deepspeed.comm as dist
import pytest
import torch


@pytest.mark.parametrize("compile_exchange", [False, True])
class TestDimZeroAllToAll(DistributedTest):
    world_size = 2

    def test_gradient_reaches_the_input(self, compile_exchange):
        """The exchange is a permutation, so d/dx of ``sum(exchange(x) ** 2)`` is ``2 * x`` on every rank.

        A collective that fills a caller-allocated buffer passes this eagerly and returns a zero gradient once
        the call is inside a compiled region, because the traced graph has no data dependence to differentiate.
        """
        group = dist.new_group(ranks=list(range(self.world_size)))
        device = get_accelerator().current_device_name()

        torch.manual_seed(1234)
        x = torch.randn(self.world_size, 4, 8, device=device, dtype=torch.float32, requires_grad=True)

        def exchange_and_square(tensor):
            return (_dim_zero_all_to_all(group, tensor)**2).sum()

        fn = torch.compile(exchange_and_square) if compile_exchange else exchange_and_square
        fn(x).backward()

        assert x.grad is not None, "the exchange produced no gradient at all"
        torch_assert_close(x.grad, 2 * x.detach())

    def test_forward_moves_the_expected_slices(self, compile_exchange):
        """Slot ``i`` of this rank's output holds the slot this rank owns in rank ``i``'s input."""
        group = dist.new_group(ranks=list(range(self.world_size)))
        device = get_accelerator().current_device_name()
        rank = dist.get_rank(group)

        slots = torch.arange(self.world_size, device=device, dtype=torch.float32).unsqueeze(1)
        x = rank * 10 + slots.expand(self.world_size, 3).contiguous()

        fn = torch.compile(_dim_zero_all_to_all) if compile_exchange else _dim_zero_all_to_all
        output = fn(group, x)

        expected = torch.arange(self.world_size, device=device, dtype=torch.float32).unsqueeze(1) * 10 + rank
        torch_assert_equal(output, expected.expand(self.world_size, 3))
