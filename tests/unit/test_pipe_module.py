import copy

import torch
import torch.nn as nn
import torch.distributed as dist

import pytest

import deepspeed

from deepspeed.runtime.pipe.topology import PipeDataParallelTopology, PipeModelDataParallelTopology
PipeTopo = PipeDataParallelTopology

from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.utils import RepeatingLoader

from common import distributed_test
from simple_model import args_from_dict

HIDDEN_DIM = 32
LAYERS = 8


@pytest.fixture
def sequential_model():
    model = torch.nn.Sequential(
        *[nn.Linear(HIDDEN_DIM,
                    HIDDEN_DIM) for _ in range(LAYERS)],
        nn.Linear(HIDDEN_DIM,
                  1),
    )
    return model


@pytest.fixture
def simple_args(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.9,
                          0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "pipeline": {
            "activation_checkpoint_interval": 1
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    return args


def test_pipe_module_sequential(sequential_model, simple_args):
    batch_input = torch.randn(1, HIDDEN_DIM)

    @distributed_test(world_size=4)
    def _helper():
        base_model = copy.deepcopy(sequential_model)
        base_input = batch_input.clone().detach()
        base_output = base_model(base_input)
        base_output = base_output
        base_params = sum(p.numel() for p in base_model.parameters())

        pipe_model = copy.deepcopy(sequential_model)
        pipe_model = PipelineModule(layers=pipe_model, num_stages=4)

        # Ensure all parameters are accounted for.
        my_params = sum(p.numel() for p in pipe_model.parameters())
        total_pipe_params = torch.LongTensor([my_params]).to('cuda')
        dist.all_reduce(total_pipe_params)
        total_pipe_params = total_pipe_params.item()
        assert total_pipe_params == base_params

        pipe_model, _, _, _ = deepspeed.initialize(
            args=simple_args,
            model=pipe_model,
            model_parameters=[p for p in pipe_model.parameters()])

        if pipe_model.is_first_stage or pipe_model.is_last_stage:
            pipe_input = base_input.clone().detach().to('cuda')
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

    _helper()
