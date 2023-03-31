# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import torch
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus, partitioned_param_data_shape
import deepspeed.comm as dist

from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from utils import setup_serial_env


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


class TestZeroGatheredParametersFree(DistributedTest):
    world_size = 1

    def test(self):
        config_dict = {"train_batch_size": 1, "zero_optimization": {"stage": 3}}
        hidden_dim = 10

        class MyModel(torch.nn.Module):

            def __init__(self, hidden_dim):
                super(MyModel, self).__init__()
                self.l1 = torch.nn.Linear(hidden_dim, hidden_dim)

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = MyModel(hidden_dim)

        with deepspeed.zero.GatheredParameters(list(model.parameters())):
            assert model.l1.weight.numel() != 0, "GatheredParameters should give a non-0-sized tensor"

        # on exit from `GatheredParameters` the gathered params should be freed and not leak memory
        assert model.l1.weight.numel() == 0, "outside of GatheredParameters the param should go back to be 0-sized"


class TestSerialContext(DistributedTest):
    world_size = 1
    init_distributed = False
    set_dist_env = False

    def test_subclass_param(self):
        setup_serial_env()
        with deepspeed.zero.Init(config=config):
            model = ConvNet()

        assert model.param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        assert model.conv1.param_in.ds_status == ZeroParamStatus.NOT_AVAILABLE

    def test_scattered_init_dist(self):
        setup_serial_env()
        assert not dist.is_initialized()
        with deepspeed.zero.Init():
            assert dist.is_initialized()

    def test_scatter_halftype(self):
        setup_serial_env()

        with deepspeed.zero.Init():
            l = torch.nn.Linear(10, 10)
            assert l.weight.ds_tensor.dtype == torch.float16

            y = torch.LongTensor([3, 3])
            assert y.dtype == torch.long

    def test_throughput_calculation(self):
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
        while engine.tput_timer.micro_step_count < (gradient_accumulation_steps * engine.tput_timer.start_step):
            engine.tput_timer.start()
            global_step = (engine.tput_timer.micro_step_count + 1) % gradient_accumulation_steps == 0
            engine.tput_timer.stop(global_step=global_step)
        assert engine.tput_timer.global_step_count == engine.tput_timer.start_step
        assert engine.tput_timer.total_elapsed_time == 0

        # calling start()/stop() accumulates duration during gradient accumulation
        while engine.tput_timer.global_step_count == engine.tput_timer.start_step:
            engine.tput_timer.start()
            current_duration = engine.tput_timer.step_elapsed_time
            total_duration = engine.tput_timer.total_elapsed_time

            global_step = (engine.tput_timer.micro_step_count + 1) % gradient_accumulation_steps == 0
            engine.tput_timer.stop(global_step=global_step)
            duration = engine.tput_timer.end_time - engine.tput_timer.start_time
            # step elapsed time is reset after gradient accumulation steps
            assert engine.tput_timer.step_elapsed_time == (
                0 if engine.tput_timer.global_step_count != engine.tput_timer.start_step else current_duration +
                duration)
            assert engine.tput_timer.total_elapsed_time == total_duration + duration

    def test_ext_param_getattr(self):
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
