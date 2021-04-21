import os
import sys
from types import SimpleNamespace

import torch
import pytest

import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus, partitioned_param_data_shape

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
    assert l.weight.shape == torch.Size(partitioned_param_data_shape)

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


config_dict = {
    "train_batch_size": 1,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 138.
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1,
    }
}


def test_ext_param_getattr():
    setup_serial_env()

    class ExtLinear(torch.nn.Module):
        def __init__(self, dim=16):
            super().__init__()
            self.dim = dim
            self.linear1 = torch.nn.Linear(dim, dim)
            self.linear2 = torch.nn.Linear(dim, dim)

        def forward(self, input):
            A = self.linear1(input)
            B = self.linear2(A)

            # external use of self.linear1.weight
            C = torch.nn.functional.linear(B, self.linear1.weight)
            return C.sum()

    net = ExtLinear()

    args = SimpleNamespace(local_rank=0)
    engine, optim, _, _ = deepspeed.initialize(args=args,
                                               model=net,
                                               model_parameters=net.parameters(),
                                               config_params=config_dict)

    with deepspeed.zero.GatheredParameters(net.linear1.weight):
        assert net.linear1.weight.numel() == net.dim**2

    input = torch.rand(net.dim).to(engine.device).half()
    loss = engine(input)
    engine.backward(loss)
    engine.step()


def test_scatter_halftype():
    setup_serial_env()

    with deepspeed.zero.Init():
        l = torch.nn.Linear(10, 10)
        assert l.weight.ds_tensor.dtype == torch.float16

        y = torch.LongTensor([3, 3])
        assert y.dtype == torch.long


class DanglingBias(torch.nn.Linear):
    def forward(self, *inputs):
        out = super().forward(*inputs)
        # return the bias to trigger a dangling external param
        return out, self.bias


class DataClass:
    """Just wraps data in an object. """
    def __init__(self, out=None, bias=None):
        self.out = out
        self.bias = bias


class DanglingBiasClass(DanglingBias):
    def forward(self, *inputs):
        out, bias = super().forward(*inputs)
        return DataClass(out=out, bias=bias)


class DanglingAttention(torch.nn.Linear):
    def __init__(self, dim=16, return_obj=False):
        super().__init__(dim, dim)
        self.dim = dim
        self.return_obj = return_obj
        if return_obj:
            self.d_linear = DanglingBiasClass(dim, dim)
        else:
            self.d_linear = DanglingBias(dim, dim)

    def forward(self, input):
        out = super().forward(input)
        if self.return_obj:
            out_obj = self.d_linear(out)
            assert out_obj.bias.ds_status == ZeroParamStatus.AVAILABLE
            # forward the external param
            return out_obj.out, out_obj.bias
        else:
            out, bias = self.d_linear(out)
            assert bias.ds_status == ZeroParamStatus.AVAILABLE
            return out, bias


class ModelContainer(torch.nn.Module):
    def __init__(self, dim=16, return_obj=False):
        super().__init__()
        self.dim = dim
        self.linear1 = torch.nn.Linear(dim, dim)
        self.dangler = DanglingAttention(dim, return_obj=return_obj)

    def forward(self, input):
        act1 = self.linear1(input)
        # bias is actually dangler.d_linear1.bias
        act2, bias = self.dangler(act1)
        assert bias.ds_status == ZeroParamStatus.AVAILABLE
        return (act2 + bias).sum()


class DanglingExt(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim
        self.container = ModelContainer(dim)

    def forward(self, input):
        out = self.container(input)

        # Make sure it's at the right level of the stack
        assert len(self._external_params) == 0
        assert len(self.container._external_params) == 1
        assert len(self.container.dangler._external_params) == 0
        return out


def test_ext_param_return():
    setup_serial_env()

    net = DanglingExt()

    args = SimpleNamespace(local_rank=0)
    engine, optim, _, _ = deepspeed.initialize(args=args,
                                               model=net,
                                               model_parameters=net.parameters(),
                                               config_params=config_dict)

    for _ in range(5):
        input = torch.rand(net.dim).to(engine.device).half()
        loss = engine(input)
        engine.backward(loss)
        engine.step()


@pytest.mark.skip('WIP')
def test_ext_param_returnobj():
    setup_serial_env()
    print()

    net = ModelContainer(return_obj=True)

    args = SimpleNamespace(local_rank=0)
    engine, optim, _, _ = deepspeed.initialize(args=args,
                                               model=net,
                                               model_parameters=net.parameters(),
                                               config_params=config_dict)

    for _ in range(5):
        input = torch.rand(net.dim).to(engine.device).half()
        loss = engine(input)
        assert len(net._external_params) == 1
        assert len(net.dangler._external_params) == 0
        engine.backward(loss)
        engine.step()
