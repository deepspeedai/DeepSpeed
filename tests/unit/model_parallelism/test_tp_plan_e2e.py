# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest, preferred_dtype


def skip_on_device():
    if get_accelerator().device_name() == 'xpu':
        pytest.skip("XPU requires a higher version for test")


class TestTPPlanEndToEnd(DistributedTest):
    world_size = 2

    def test_tp_plan_basic_training(self):
        skip_on_device()

        class SimpleHFModel(torch.nn.Module):

            def __init__(self, hidden_size=64):
                super().__init__()
                self.config = type(
                    'Config', (), {'base_model_tp_plan': {
                        'layers.*.q_proj': 'colwise',
                        'layers.*.o_proj': 'rowwise'
                    }})()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(hidden_size, hidden_size * 2),
                     torch.nn.Linear(hidden_size * 2, hidden_size)])

            def forward(self, x):
                return self.layers[1](self.layers[0](x))

        model = SimpleHFModel()

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 0
            },
            "steps_per_print": 1,
        }

        if preferred_dtype() == torch.float16:
            ds_config["fp16"] = {"enabled": True}
        elif preferred_dtype() == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2

        input_tensor = torch.randn(2, 4, 64).to(get_accelerator().current_device_name())
        output = engine(input_tensor)
        loss = output.mean()
        engine.backward(loss)
        engine.step()

    def test_tp_plan_with_zero1(self):
        skip_on_device()

        class SimpleHFModel(torch.nn.Module):

            def __init__(self, hidden_size=64):
                super().__init__()
                self.config = type(
                    'Config', (), {'base_model_tp_plan': {
                        'layers.*.q_proj': 'colwise',
                        'layers.*.o_proj': 'rowwise'
                    }})()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(hidden_size, hidden_size * 2),
                     torch.nn.Linear(hidden_size * 2, hidden_size)])

            def forward(self, x):
                return self.layers[1](self.layers[0](x))

        model = SimpleHFModel()

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 1
            },
            "steps_per_print": 1,
        }

        if preferred_dtype() == torch.float16:
            ds_config["fp16"] = {"enabled": True}
        elif preferred_dtype() == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2

        for _ in range(3):
            input_tensor = torch.randn(2, 4, 64).to(get_accelerator().current_device_name())
            output = engine(input_tensor)
            loss = output.mean()
            engine.backward(loss)
            engine.step()

            for p in engine.parameters():
                assert not torch.isnan(p).any()

    def test_tp_plan_with_zero2(self):
        skip_on_device()

        class SimpleHFModel(torch.nn.Module):

            def __init__(self, hidden_size=64):
                super().__init__()
                self.config = type(
                    'Config', (), {'base_model_tp_plan': {
                        'layers.*.q_proj': 'colwise',
                        'layers.*.o_proj': 'rowwise'
                    }})()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(hidden_size, hidden_size * 2),
                     torch.nn.Linear(hidden_size * 2, hidden_size)])

            def forward(self, x):
                return self.layers[1](self.layers[0](x))

        model = SimpleHFModel()

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 2
            },
            "steps_per_print": 1,
        }

        if preferred_dtype() == torch.float16:
            ds_config["fp16"] = {"enabled": True}
        elif preferred_dtype() == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2

        input_tensor = torch.randn(2, 4, 64).to(get_accelerator().current_device_name())
        output = engine(input_tensor)
        loss = output.mean()
        engine.backward(loss)
        engine.step()


class TestTPPlanCorrectness(DistributedTest):
    world_size = 2

    def test_tp_plan_correctness_basic(self):
        skip_on_device()

        class SimpleHFModel(torch.nn.Module):

            def __init__(self, hidden_size=64):
                super().__init__()
                self.config = type(
                    'Config', (), {'base_model_tp_plan': {
                        'layers.*.q_proj': 'colwise',
                        'layers.*.o_proj': 'rowwise'
                    }})()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(hidden_size, hidden_size * 2),
                     torch.nn.Linear(hidden_size * 2, hidden_size)])

            def forward(self, x):
                return self.layers[1](self.layers[0](x))

        torch.manual_seed(42)
        model = SimpleHFModel()

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 0
            },
        }

        if preferred_dtype() == torch.float16:
            ds_config["fp16"] = {"enabled": True}
        elif preferred_dtype() == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        torch.manual_seed(123)
        input_tensor = torch.randn(2, 4, 64).to(get_accelerator().current_device_name())
        output = engine(input_tensor)

        assert output.shape == (2, 4, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
