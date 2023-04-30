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


class TestPipeModuleSequential(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("activation_checkpoints", [False, True])
    def test(self, sequential_model, simple_config, batch_input, activation_checkpoints):
        base_model = copy.deepcopy(sequential_model)
        base_input = batch_input.clone().detach()
        base_output = base_model(base_input)
        base_output = base_output
        base_params = sum(p.numel() for p in base_model.parameters())

        pipe_model = copy.deepcopy(sequential_model)
        pipe_model = PipelineModule(layers=pipe_model, num_stages=2)

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
