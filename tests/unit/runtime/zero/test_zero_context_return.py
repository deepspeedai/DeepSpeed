# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace
import torch
import pytest
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from utils import setup_serial_env
from unit.common import DistributedTest, preferred_dtype


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
            assert hasattr(bias, 'ds_status') or hasattr(bias, 'ds_param_alias')
            z3_bias = bias if hasattr(bias, 'ds_status') else bias.ds_param_alias
            assert z3_bias.ds_status == ZeroParamStatus.AVAILABLE
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


class ModelContainerVariableOutputType(ModelContainer):

    def __init__(self, dim=16, output_type=dict):
        super().__init__()
        self.output_type = output_type
        self.dim = dim
        self.linear1 = torch.nn.Linear(dim, dim)

    def forward(self, input):
        act1 = self.linear1(input)
        if self.output_type is dict:
            return {'loss': act1.sum()}
        if self.output_type is torch.tensor:
            return act1.sum()


config = {
    "train_batch_size": 1,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015
        }
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1,
    }
}

if get_accelerator().is_fp16_supported():
    config["fp16"] = {"enabled": True, "loss_scale": 138.}
elif get_accelerator().is_bf16_supported():
    config["bf16"] = {"enabled": True}


class TestReturnParam(DistributedTest):
    world_size = 1

    def test_ext_param_return(self):
        setup_serial_env()

        net = DanglingExt()

        args = SimpleNamespace(local_rank=0)
        engine, _, _, _ = deepspeed.initialize(args=args, model=net, model_parameters=net.parameters(), config=config)

        for _ in range(5):
            input = torch.rand(net.dim).to(engine.device).to(preferred_dtype())
            loss = engine(input)
            engine.backward(loss)
            engine.step()

    @pytest.mark.skip('WIP')
    def test_ext_param_returnobj(self):
        setup_serial_env()
        print()

        net = ModelContainer(return_obj=True)

        args = SimpleNamespace(local_rank=0)
        engine, _, _, _ = deepspeed.initialize(args=args, model=net, model_parameters=net.parameters(), config=config)

        for _ in range(5):
            input = torch.rand(net.dim).to(engine.device).to(preferred_dtype())
            loss = engine(input)
            assert len(net._external_params) == 1
            assert len(net.dangler._external_params) == 0
            engine.backward(loss)
            engine.step()

    @pytest.mark.parametrize('output_type', [torch.tensor, dict, None])
    def test_stage_3_output_type(self, output_type):
        setup_serial_env()
        print()

        net = ModelContainerVariableOutputType(output_type=output_type)

        args = SimpleNamespace(local_rank=0)
        engine, _, _, _ = deepspeed.initialize(args=args, model=net, model_parameters=net.parameters(), config=config)

        for _ in range(1):
            input = torch.rand(net.dim).to(engine.device).to(preferred_dtype())
            loss = engine(input)
            if loss is not None:
                if isinstance(loss, dict):
                    loss = loss['loss']
                engine.backward(loss)
                engine.step()
