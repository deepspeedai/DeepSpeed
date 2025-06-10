from unit.common import DistributedTest
import pytest
@pytest.mark.parametrize("tp_size", [2, 4])
class TestEmpty(DistributedTest):
    world_size = 4
    reuse_dist_env = True # wsz=4 and reuse_dist_env=True will hang.

    def test(self, tp_size: int):
        print("finished test")
        return
