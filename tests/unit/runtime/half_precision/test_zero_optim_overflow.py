# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import pytest
import numpy as np
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.utils import safe_set_full_grad


def has_inf_or_nan(x):
    float_x = x.float()
    nan = float_x.isnan()
    inf = float_x.isinf()
    inf_or_nan = nan.logical_or(inf)
    return inf_or_nan.float().max()


def run_model_step(model, x_sample, y_label, grad_value):
    loss = model(x_sample, y_label)
    model.backward(loss)
    for p in model.parameters():
        grad = torch.empty_like(p, dtype=p.dtype)
        grad.fill_(grad_value)
        safe_set_full_grad(p, grad)
    model.step()


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("offload_optimizer", [False, True])
class TestZeROFloat16(DistributedTest):
    world_size = 2

    def test_no_overflow(self, zero_stage, offload_optimizer):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 8,
                "loss_scale_window": 2
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }

        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**8
        expected_scale_window = 2
        # Ensure the dynamic loss scaler is correctly configured.
        loss_scaler = optim.loss_scaler

        assert optim.dynamic_loss_scale == True
        assert loss_scaler.cur_scale == expected_loss_scale
        assert loss_scaler.scale_window == expected_scale_window

        num_iterations = 10
        grad_values = np.random.uniform(-0.1, 0.1, num_iterations)
        data_loader = random_dataloader(model=model,
                                        total_samples=num_iterations,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for i, (batch, grad_value) in enumerate(zip(data_loader, grad_values)):
            run_model_step(model, batch[0], batch[1], grad_value)
            assert loss_scaler.cur_scale == expected_loss_scale
            assert loss_scaler.cur_iter == (i + 1)

            if loss_scaler.cur_iter % expected_scale_window == 0:
                expected_loss_scale *= 2

    def test_all_overflow(self, zero_stage, offload_optimizer):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

        overflow_gradients = [float('inf'), float('-inf')] + [float('nan')] * 6
        initial_scale_power = len(overflow_gradients)
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": initial_scale_power,
                "loss_scale_window": 2,
                "hysteresis": 1,
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }

        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**initial_scale_power
        expected_scale_window = 2
        # Ensure the dynamic loss scaler is correctly configured.
        loss_scaler = optim.loss_scaler

        assert optim.dynamic_loss_scale == True
        assert loss_scaler.cur_scale == expected_loss_scale
        assert loss_scaler.scale_window == expected_scale_window

        data_loader = random_dataloader(model=model,
                                        total_samples=len(overflow_gradients),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for i, (batch, grad_value) in enumerate(zip(data_loader, overflow_gradients)):
            run_model_step(model, batch[0], batch[1], grad_value)
            expected_loss_scale = max(expected_loss_scale / 2, 1)
            assert loss_scaler.cur_scale == expected_loss_scale
            assert loss_scaler.cur_iter == (i + 1)

    def test_some_overflow(self, zero_stage, offload_optimizer):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        initial_scale_power = 8
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": initial_scale_power,
                "loss_scale_window": 2,
                "hysteresis": 1,
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }

        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**initial_scale_power
        expected_scale_window = 2
        # Ensure the dynamic loss scaler is correctly configured.
        loss_scaler = optim.loss_scaler

        assert optim.dynamic_loss_scale == True
        assert loss_scaler.cur_scale == expected_loss_scale
        assert loss_scaler.scale_window == expected_scale_window

        expected_iteration = 0

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf'), float('nan')]
        expected_iteration += len(overflow_gradients)
        data_loader = random_dataloader(model=model,
                                        total_samples=len(overflow_gradients),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for batch, grad_value in zip(data_loader, overflow_gradients):
            run_model_step(model, batch[0], batch[1], grad_value)

        expected_loss_scale /= (2**len(overflow_gradients))
        assert loss_scaler.cur_scale == expected_loss_scale
        assert loss_scaler.cur_iter == expected_iteration

        # Run model scale_window + 1 times to increase scale once
        normal_gradients = np.random.uniform(-0.1, 0.1, expected_scale_window + 1)
        expected_iteration += len(normal_gradients)
        data_loader = random_dataloader(model=model,
                                        total_samples=len(normal_gradients),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for batch, grad_value in zip(data_loader, normal_gradients):
            run_model_step(model, batch[0], batch[1], grad_value)

        expected_loss_scale *= 2
        assert loss_scaler.cur_scale == expected_loss_scale
        assert loss_scaler.cur_iter == expected_iteration

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf')]
        expected_iteration += len(overflow_gradients)
        data_loader = random_dataloader(model=model,
                                        total_samples=len(overflow_gradients),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for batch, grad_value in zip(data_loader, overflow_gradients):
            run_model_step(model, batch[0], batch[1], grad_value)

        expected_loss_scale /= (2**len(overflow_gradients))
        assert loss_scaler.cur_scale == expected_loss_scale
        assert loss_scaler.cur_iter == expected_iteration


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("offload_optimizer", [False, True])
class TestZeROBFloat16(DistributedTest):
    world_size = 2

    def test_no_overflow(self, zero_stage, offload_optimizer):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "bf16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }

        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        num_iterations = 10
        grad_values = np.random.uniform(-0.1, 0.1, num_iterations)
        data_loader = random_dataloader(model=model,
                                        total_samples=num_iterations,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for i, (batch, grad_value) in enumerate(zip(data_loader, grad_values)):
            run_model_step(model, batch[0], batch[1], grad_value)

        assert model.skipped_steps == 0
        assert all([not has_inf_or_nan(p) for p in model.parameters()])

    def test_detect_grad_overflow(self, zero_stage, offload_optimizer):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "bf16": {
                "enabled": True,
                "check_grad_overflow": True
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }

        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        overflow_gradients = [float('inf'), float('-inf')] + [float('nan')] * 6
        data_loader = random_dataloader(model=model,
                                        total_samples=len(overflow_gradients),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)

        for i, (batch, grad_value) in enumerate(zip(data_loader, overflow_gradients)):
            run_model_step(model, batch[0], batch[1], grad_value)
            assert model.skipped_steps == (i + 1)

        assert all([not has_inf_or_nan(p) for p in model.parameters()])

    def test_ignore_grad_overflow(self, zero_stage, offload_optimizer):
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "bf16": {
                "enabled": True,
                "check_grad_overflow": False
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }

        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        overflow_gradients = [float('inf'), float('-inf')] + [float('nan')] * 6
        data_loader = random_dataloader(model=model,
                                        total_samples=len(overflow_gradients),
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)

        for i, (batch, grad_value) in enumerate(zip(data_loader, overflow_gradients)):
            run_model_step(model, batch[0], batch[1], grad_value)

        assert model.skipped_steps == 0
        assert all([has_inf_or_nan(p) for p in model.parameters()])
