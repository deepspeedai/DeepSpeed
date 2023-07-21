# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import pytest
import deepspeed
from deepspeed.profiling.flops_profiler import get_model_profile
from unit.simple_model import SimpleModel, random_dataloader
from unit.common import DistributedTest
from deepspeed.runtime.utils import required_torch_version
from deepspeed.accelerator import get_accelerator

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=1.3),
                                reason='requires Pytorch version 1.3 or above')


def within_range(val, target, tolerance):
    return abs(val - target) / target < tolerance


TOLERANCE = 0.05


class LeNet5(torch.nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            torch.nn.Tanh(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return logits, probs


class TestFlopsProfiler(DistributedTest):
    world_size = 1

    def test(self):
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001,
                }
            },
            "zero_optimization": {
                "stage": 0
            },
            "fp16": {
                "enabled": True,
            },
            "flops_profiler": {
                "enabled": True,
                "step": 1,
                "module_depth": -1,
                "top_modules": 3,
            },
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim, empty_grad=False)

        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.half)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            if n == 3: break
        assert within_range(model.flops_profiler.flops, 200, tolerance=TOLERANCE)
        assert model.flops_profiler.params == 110

    def test_flops_profiler_in_inference(self):
        mod = LeNet5(10)
        batch_size = 1024
        input = torch.randn(batch_size, 1, 32, 32)
        flops, macs, params = get_model_profile(
            mod,
            tuple(input.shape),
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=3,
            warm_up=1,
            as_string=False,
            ignore_modules=None,
        )
        print(flops, macs, params)
        assert within_range(flops, 866076672, TOLERANCE)
        assert within_range(macs, 426516480, TOLERANCE)
        assert params == 61706
