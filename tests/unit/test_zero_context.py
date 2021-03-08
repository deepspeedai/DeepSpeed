import os
import torch
import pytest

import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from common import distributed_test


def setup_serial_env():
    # Setup for a serial run
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29503'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'


def test_scattered_init_dist():
    setup_serial_env()
    assert not torch.distributed.is_initialized()
    with deepspeed.zero.Init():
        assert torch.distributed.is_initialized()


@distributed_test(world_size=2)
def test_scatter_gather():
    with deepspeed.zero.Init():
        l = torch.nn.Linear(6, 3)
    assert l.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert l.weight.numel() == 1

    # Ensure there is no impact outside the context
    l2 = torch.nn.Linear(6, 3)
    assert not hasattr(l2.weight, 'ds_status')
    assert l2.weight.numel() == l2.in_features * l2.out_features

    with deepspeed.zero.GatheredParameters(l.weight):
        assert l.weight.ds_status == ZeroParamStatus.AVAILABLE
        assert l.weight.numel() == l.in_features * l.out_features


@distributed_test(world_size=2)
def test_gather_update():
    with deepspeed.zero.Init():
        l = torch.nn.Linear(4, 2)
    assert l.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE

    # Gather and make a change
    with deepspeed.zero.GatheredParameters(l.weight, modifier_rank=1):
        assert l.weight.ds_status == ZeroParamStatus.AVAILABLE
        if torch.distributed.get_rank() == 1:
            with torch.no_grad():
                l.weight.zero_()

    # should now be scattered again

    # Now gather again and ensure the change is global
    with deepspeed.zero.GatheredParameters(l.weight):
        # all ranks compare
        assert torch.equal(l.weight, torch.zeros_like(l.weight))


@pytest.mark.skip('WIP')
def test_external_param():
    setup_serial_env()

    print()

    class ExtLinear(torch.nn.Module):
        def __init__(self, dim=10, copycat=None):
            super().__init__()
            self.dim = dim
            self.linear = torch.nn.Linear(dim, dim)
            if copycat is not None:
                with deepspeed.zero.GatheredParameters(self.linear.weight,
                                                  modifier_rank=0), \
                     torch.no_grad():
                    self.linear.weight.copy_(copycat.linear.weight)

            if hasattr(self.linear.weight, 'ds_id'):
                print('registering')
                super().ds_register_external_parameter('samyam', self.linear.weight)

        def forward(self, input):
            yamsam = self.linear(input)
            if hasattr(self.linear.weight, 'ds_status'):
                assert self.linear.weight.ds_status == ZeroParamStatus.AVAILABLE
            jeff = torch.nn.functional.linear(yamsam, self.linear.weight)
            return jeff

    l1_base = ExtLinear().half().cuda()
    l2_base = ExtLinear().half().cuda()

    input = torch.rand(10).half().cuda()

    l1_base_out = l1_base(input.clone().detach())
    l2_base_out = l2_base(input.clone().detach())

    with deepspeed.zero.Init():
        l1_test = ExtLinear(copycat=l1_base).cuda()
        #l2_test = ExtLinear(copycat=l2_base).cuda()
        assert l1_test.linear.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE

    # XXX l1 and l2 share their external parameter (l2.linear.weight)

    assert l1_test.linear.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
    l1_test_out = l1_test(input.clone().detach())
    #assert torch.allclose(l1_base_out, l1_test_out)

    #l2_test_out = l2_test(input.clone().detach())
    #assert torch.allclose(l2_base_out, l2_test_out)


def test_scatter_halftype():
    setup_serial_env()

    with deepspeed.zero.Init():
        l = torch.nn.Linear(10, 10)
        assert l.weight.ds_tensor.dtype == torch.float16

        y = torch.LongTensor([3, 3])
        assert y.dtype == torch.long
