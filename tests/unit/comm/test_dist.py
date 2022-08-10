import torch
import deepspeed.comm as dist

from tests.unit.common import DistributedTest

import pytest


class TestInit(DistributedTest):
    world_size = 3

    def test(self):
        assert dist.is_initialized()
        assert dist.get_world_size() == 3
        assert dist.get_rank() < 3


# Demonstration of pytest's parameterization and fixtures
@pytest.fixture(params=["hello"])
def greeting(request):
    return request.param


@pytest.mark.parametrize("number,color", [(1138, "purple")])
class TestDistArgs(DistributedTest):
    world_size = 2
    """ Classes that use DistributedTest class must define a test* method """
    @pytest.mark.parametrize("shape", ["icosahedron"])
    def test(self, number, color, shape, greeting):
        """Ensure that we can parse args to DistributedTest methods. """
        assert dist.get_world_size() == 2
        assert number == 1138
        assert color == "purple"
        assert shape == "icosahedron"
        assert greeting == "hello"


# Demonstration of distributed tests grouped in single class
@pytest.mark.parametrize("number", [1138])
class TestGroupedDistTest(DistributedTest):
    world_size = 2

    def test_one(self, number):
        assert dist.get_world_size() == 2
        assert number == 1138

    def test_two(self, number, color="purple"):
        assert dist.get_world_size() == 2
        assert number == 1138
        assert color == "purple"


# Demonstration of world_size override
class TestWorldSizeOverrideDistTest(DistributedTest):
    world_size = 2

    def test_world_size_2(self):
        assert dist.get_world_size() == 2

    @pytest.mark.world_size(1)
    def test_world_size_1(self):
        assert dist.get_world_size() == 1


class TestDistAllReduce(DistributedTest):
    world_size = [1, 2, 4]

    def test(self):
        x = torch.ones(1, 3).cuda() * (dist.get_rank() + 1)
        sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
        result = torch.ones(1, 3).cuda() * sum_of_ranks
        dist.all_reduce(x)
        assert torch.all(x == result)
