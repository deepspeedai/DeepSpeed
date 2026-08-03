# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import torch
import torch.nn as nn
import deepspeed.comm as dist

import pytest

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.utils import RepeatingLoader
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest

HIDDEN_DIM = 32
LAYERS = 8


@pytest.fixture
def sequential_model():
    model = torch.nn.Sequential(
        *[nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(LAYERS)],
        nn.Linear(HIDDEN_DIM, 1),
    )
    return model


@pytest.fixture
def simple_config():
    config_dict = {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "pipeline": {
            "activation_checkpoint_interval": 1
        }
    }
    return config_dict


@pytest.fixture
def batch_input():
    return torch.randn(1, HIDDEN_DIM)


@pytest.fixture
def mixed_param_model():
    # ReLU carries no parameters and Linear does, so _is_checkpointable differs between blocks
    # and a misaligned result list is visible, not just a wrongly sized one.
    return torch.nn.Sequential(
        *[nn.ReLU() for _ in range(LAYERS // 2)],
        *[nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(LAYERS // 2)],
    )


class TestPipeModuleSequential(DistributedTest):
    world_size = 2
    # needs to be set for torch.compile: running torch.compile with daemonic process causes an error
    non_daemonic_procs = True

    @pytest.mark.parametrize("activation_checkpoints", [False, True])
    @pytest.mark.parametrize("use_compile", [False, True])
    def test(self, sequential_model, simple_config, batch_input, activation_checkpoints, use_compile):
        base_model = copy.deepcopy(sequential_model)
        base_input = batch_input.clone().detach()
        base_output = base_model(base_input)
        base_output = base_output
        base_params = sum(p.numel() for p in base_model.parameters())

        pipe_model = copy.deepcopy(sequential_model)
        pipe_model = PipelineModule(layers=pipe_model, num_stages=2)
        if (use_compile):
            pipe_model.compile()
        # Ensure all parameters are accounted for.
        my_params = sum(p.numel() for p in pipe_model.parameters())
        total_pipe_params = torch.LongTensor([my_params]).to(get_accelerator().device_name())
        dist.all_reduce(total_pipe_params)
        total_pipe_params = total_pipe_params.item()
        assert total_pipe_params == base_params

        pipe_model, _, _, _ = deepspeed.initialize(config=simple_config,
                                                   model=pipe_model,
                                                   model_parameters=[p for p in pipe_model.parameters()])

        if activation_checkpoints:
            deepspeed.checkpointing.configure(None,
                                              deepspeed_config=pipe_model.config,
                                              partition_activations=True,
                                              contiguous_checkpointing=True,
                                              num_checkpoints=9)

        if pipe_model.is_first_stage or pipe_model.is_last_stage:
            pipe_input = base_input.clone().detach().to(get_accelerator().device_name())
            # label 0 is meaningless
            dataset = [(pipe_input, 0)]
            loader = RepeatingLoader(dataset)
            data_iter = iter(loader)
        else:
            data_iter = None

        pipe_output = pipe_model.eval_batch(data_iter=data_iter)

        base_output = base_output.to('cpu')
        pipe_output = pipe_output.to('cpu')

        assert torch.allclose(base_output, pipe_output, atol=1e-4)


class TestPipeModuleCheckpointInterval(DistributedTest):
    world_size = 1

    def test_set_checkpoint_interval(self, mixed_param_model):
        model = PipelineModule(layers=copy.deepcopy(mixed_param_model), num_stages=1, activation_checkpoint_interval=4)
        model._precompute_checkpointable_values()
        assert model.is_checkpointable_results == [False, True]

        model.set_checkpoint_interval(1)

        # the setter has to update the interval forward() reads, not a separate attribute
        assert model.activation_checkpoint_interval == 1

        # and the cached results have to be rebuilt for the new interval rather than appended to,
        # otherwise forward() zips the layer blocks against results computed for the old interval
        reference = PipelineModule(layers=copy.deepcopy(mixed_param_model),
                                   num_stages=1,
                                   activation_checkpoint_interval=1)
        reference._precompute_checkpointable_values()
        assert model.is_checkpointable_results == reference.is_checkpointable_results

    def test_setter_keeps_reentrant_input_grads(self, mixed_param_model, simple_config):
        # The engine decides whether to mark the first stage's inputs as requiring grad, which
        # reentrant checkpointing needs, or its first checkpointed segment is cut off from
        # autograd. Start from a config that disables checkpointing, so the only thing that
        # turns it on is the setter this PR fixes.
        config = copy.deepcopy(simple_config)
        config["pipeline"]["activation_checkpoint_interval"] = 0

        model = PipelineModule(layers=copy.deepcopy(mixed_param_model), num_stages=1)
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               model_parameters=[p for p in model.parameters()])
        assert not engine._reentrant_activation_checkpointing()

        engine.module.set_checkpoint_interval(2)

        # forward() now checkpoints, so the inputs have to require grad to match
        assert engine._reentrant_activation_checkpointing()

    def test_non_reentrant_checkpointing_does_not_need_input_grads(self, mixed_param_model, simple_config):
        # use_reentrant=False swaps in non_reentrant_checkpoint, which does not need the inputs
        # to require grad, so the decision has to track the function and not just the interval.
        config = copy.deepcopy(simple_config)
        config["pipeline"]["use_reentrant"] = False

        model = PipelineModule(layers=copy.deepcopy(mixed_param_model), num_stages=1)
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               model_parameters=[p for p in model.parameters()])
        assert engine.module.activation_checkpoint_interval > 0
        assert not engine._reentrant_activation_checkpointing()

    def test_setter_honors_non_reentrant_when_the_config_disables_checkpointing(self, mixed_param_model,
                                                                                simple_config):
        # use_reentrant only reached the module inside the positive-interval branch, so a config
        # that disables checkpointing left activation_checkpoint_func at the reentrant default.
        # set_checkpoint_interval() then enabled checkpointing with the wrong function, silently
        # ignoring use_reentrant=False, and _is_checkpointable() reads the same function.
        config = copy.deepcopy(simple_config)
        config["pipeline"]["activation_checkpoint_interval"] = 0
        config["pipeline"]["use_reentrant"] = False

        model = PipelineModule(layers=copy.deepcopy(mixed_param_model), num_stages=1)
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               model_parameters=[p for p in model.parameters()])
        assert engine.module.activation_checkpoint_interval == 0
        assert engine.module.activation_checkpoint_func is ds_checkpointing.non_reentrant_checkpoint

        engine.module.set_checkpoint_interval(1)

        assert engine.module.activation_checkpoint_func is ds_checkpointing.non_reentrant_checkpoint
        assert not engine._reentrant_activation_checkpointing()
