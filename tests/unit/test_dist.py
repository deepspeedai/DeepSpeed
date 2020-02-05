import torch
import torch.distributed as dist

from common import distributed_test

import pytest


@distributed_test(world_size=3)
def test_init():
    assert dist.is_initialized()
    assert dist.get_world_size() == 3
    assert dist.get_rank() < 3


# Demonstration of pytest's paramaterization
@pytest.mark.parametrize('number,color', [(1138, 'purple')])
def test_dist_args(number, color):
    """Outer test function with inputs from pytest.mark.parametrize(). Uses a distributed
    helper function.
    """
    @distributed_test(world_size=2)
    def _test_dist_args_helper(x, color='red'):
        assert dist.get_world_size() == 2
        assert x == 1138
        assert color == 'purple'

    """Ensure that we can parse args to distributed_test decorated functions. """
    _test_dist_args_helper(number, color=color)


@distributed_test(world_size=[1, 2, 4])
def test_dist_allreduce():
    x = torch.ones(1, 3).cuda() * (dist.get_rank() + 1)
    sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
    result = torch.ones(1, 3).cuda() * sum_of_ranks
    dist.all_reduce(x)
    assert torch.all(x == result)
