import os
from types import SimpleNamespace

import torch
import pytest

import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus, partitioned_param_data_shape
import deepspeed.comm as dist

from unit.common import DistributedTest, get_master_port
from unit.simple_model import SimpleModel


def setup_serial_env():
    # Setup for a serial run
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = get_master_port()
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'


def test_scattered_init_dist():
    setup_serial_env()
    assert not dist.is_initialized()
    with deepspeed.zero.Init():
        assert dist.is_initialized()


class TestScatterGather(DistributedTest):
    world_size = 2

    def test(self):
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


class TestGatherUpdate(DistributedTest):
    world_size = 2

    def test(self):
        with deepspeed.zero.Init():
            l = torch.nn.Linear(4, 2)
        assert l.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE

        # Gather and make a change
        with deepspeed.zero.GatheredParameters(l.weight, modifier_rank=1):
            assert l.weight.ds_status == ZeroParamStatus.AVAILABLE
            if dist.get_rank() == 1:
                with torch.no_grad():
                    l.weight.zero_()

        # should now be scattered again

        # Now gather again and ensure the change is global
        with deepspeed.zero.GatheredParameters(l.weight):
            # all ranks compare
            assert torch.equal(l.weight, torch.zeros_like(l.weight))


config = {
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


def test_throughput_calculation():
    setup_serial_env()

    train_micro_batch_size_per_gpu = 7
    gradient_accumulation_steps = 6
    config_dict = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
            }
        },
        "zero_optimization": {
            "stage": 0
        },
    }

    args = SimpleNamespace(local_rank=0)
    net = SimpleModel(hidden_dim=4)
    engine, _, _, _ = deepspeed.initialize(args=args,
                                           config=config_dict,
                                           model=net,
                                           model_parameters=net.parameters())
    assert engine.tput_timer.batch_size == train_micro_batch_size_per_gpu * gradient_accumulation_steps

    assert not engine.tput_timer.initialized
    assert not engine.tput_timer.started
    assert engine.tput_timer.start_step == 2
    assert engine.tput_timer.start_time == 0
    assert engine.tput_timer.micro_step_count == 0
    assert engine.tput_timer.global_step_count == 0
    assert engine.tput_timer.total_elapsed_time == 0

    # calling stop() while uninitialized - has no effect
    engine.tput_timer.stop()
    assert not engine.tput_timer.initialized
    assert not engine.tput_timer.started
    assert engine.tput_timer.start_time == 0
    assert engine.tput_timer.micro_step_count == 0
    assert engine.tput_timer.global_step_count == 0
    assert engine.tput_timer.total_elapsed_time == 0

    # any call to start() (from dataloader or not) initializes the timer
    engine.tput_timer.start()
    assert engine.tput_timer.initialized
    assert engine.tput_timer.started
    assert engine.tput_timer.start_time == 0
    assert engine.tput_timer.micro_step_count == 0
    assert engine.tput_timer.global_step_count == 0
    assert engine.tput_timer.total_elapsed_time == 0

    # calling stop() after initialized - increments the local micro step counter
    engine.tput_timer.stop()
    assert engine.tput_timer.initialized
    assert not engine.tput_timer.started
    assert engine.tput_timer.start_time == 0
    assert engine.tput_timer.micro_step_count == 1
    assert engine.tput_timer.global_step_count == 0
    assert engine.tput_timer.total_elapsed_time == 0

    # calling start()/stop() to increment the step counter until start_step
    while engine.tput_timer.micro_step_count < (gradient_accumulation_steps *
                                                engine.tput_timer.start_step):
        engine.tput_timer.start()
        global_step = (engine.tput_timer.micro_step_count +
                       1) % gradient_accumulation_steps == 0
        engine.tput_timer.stop(global_step=global_step)
    assert engine.tput_timer.global_step_count == engine.tput_timer.start_step
    assert engine.tput_timer.total_elapsed_time == 0

    # calling start()/stop() accumulates duration during gradient accumulation
    while engine.tput_timer.global_step_count == engine.tput_timer.start_step:
        engine.tput_timer.start()
        current_duration = engine.tput_timer.step_elapsed_time
        total_duration = engine.tput_timer.total_elapsed_time

        global_step = (engine.tput_timer.micro_step_count +
                       1) % gradient_accumulation_steps == 0
        engine.tput_timer.stop(global_step=global_step)
        duration = engine.tput_timer.end_time - engine.tput_timer.start_time
        # step elapsed time is reset after gradient accumulation steps
        assert engine.tput_timer.step_elapsed_time == (
            0 if engine.tput_timer.global_step_count != engine.tput_timer.start_step else
            current_duration + duration)
        assert engine.tput_timer.total_elapsed_time == total_duration + duration


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
                                               config=config)

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
        # Ensure we have registered the original unmodified bias parameter as an ext param
        assert id(self.container.dangler.d_linear.bias
                  ) in self.container._external_params.keys()
        return out


def test_ext_param_return():
    setup_serial_env()

    net = DanglingExt()

    args = SimpleNamespace(local_rank=0)
    engine, optim, _, _ = deepspeed.initialize(args=args,
                                               model=net,
                                               model_parameters=net.parameters(),
                                               config=config)

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
                                               config=config)

    for _ in range(5):
        input = torch.rand(net.dim).to(engine.device).half()
        loss = engine(input)
        assert len(net._external_params) == 1
        assert len(net.dangler._external_params) == 0
        engine.backward(loss)
        engine.step()


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


@pytest.mark.parametrize('output_type', [torch.tensor, dict, None])
def test_stage_3_output_type(output_type):
    setup_serial_env()
    print()

    net = ModelContainerVariableOutputType(output_type=output_type)

    args = SimpleNamespace(local_rank=0)
    engine, optim, _, _ = deepspeed.initialize(args=args,
                                               model=net,
                                               model_parameters=net.parameters(),
                                               config=config)

    for _ in range(1):
        input = torch.rand(net.dim).to(engine.device).half()
        loss = engine(input)
        if loss is not None:
            if isinstance(loss, dict):
                loss = loss['loss']
            engine.backward(loss)
            engine.step()


# Test that no sub-class or super-class is missed
class ConvX(torch.nn.Conv1d):
    def __init__(self, *args):
        super().__init__(*args)
        # This would not be partitioned before bugfix 5ca8167
        self.param_in = torch.nn.Parameter(torch.FloatTensor(5).uniform_())

    def forward(self, x):
        return x


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvX(1, 3, 4)
        self.param = torch.nn.Parameter(torch.FloatTensor(5).uniform_())

    def forward(self, x):
        return x


def test_subclass_param():
    setup_serial_env()
    with deepspeed.zero.Init(config=config):
        model = ConvNet()

    assert model.param.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert model.conv1.param_in.ds_status == ZeroParamStatus.NOT_AVAILABLE


# test that sub-classes get params that aren't prematurely partitioned and thus requiring gathering
# fixed by https://github.com/microsoft/DeepSpeed/pull/1202
class GrandPa(torch.nn.Module):
    def __init__(self, *args):
        super().__init__(*args)
        self.param_grandpa = torch.nn.Parameter(torch.ones(5))
        self.param_grandpa.data = (self.param_grandpa.data +
                                   1).data  # test param is not yet partitioned


class Pa(GrandPa):
    def __init__(self, *args):
        super().__init__(*args)
        self.param_pa = torch.nn.Parameter(torch.ones(5))
        self.param_pa.data = (self.param_pa.data +
                              1).data  # test param is not yet partitioned
        self.param_grandpa.data = (self.param_grandpa.data +
                                   1).data  # test param is not yet partitioned


class Son(Pa):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(5))
        self.param.data = (self.param.data + 1).data  # test param is not yet partitioned
        self.param_pa.data = (self.param_pa.data +
                              1).data  # test param is not yet partitioned
        self.param_grandpa.data = (self.param_grandpa.data +
                                   1).data  # test param is not yet partitioned


def test_subclass_param_init():
    setup_serial_env()
    with deepspeed.zero.Init(config=config):
        model = Son().cpu()

    # test that all params have been partitioned
    assert model.param_grandpa.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert model.param_pa.ds_status == ZeroParamStatus.NOT_AVAILABLE
    assert model.param.ds_status == ZeroParamStatus.NOT_AVAILABLE

    # test that the weights manipulation during each __init__ worked in all w/o needing gathering
    ones = torch.ones(5).half().cuda()
    with deepspeed.zero.GatheredParameters(list(model.parameters(recurse=False))):
        assert torch.equal(model.param, ones + 1)
        assert torch.equal(model.param_pa, ones + 2)
        assert torch.equal(model.param_grandpa, ones + 3)


class TestDSInitWZinit(DistributedTest):
    world_size = 2

    def test(self):
        ds_config = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            }
        }

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = torch.nn.Linear(4, 4)

            def magic(self):
                return 42

        with deepspeed.zero.Init():
            model = Model()
            engine, *_ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())
        assert engine.magic() == 42
