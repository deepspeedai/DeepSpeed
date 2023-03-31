# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import numpy as np
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def run_model_step(model, gradient_list):
    for value in gradient_list:
        for p in model.parameters():
            p.grad = torch.empty_like(p, dtype=p.dtype)
            p.grad.fill_(value)
        model.step()


class TestFused(DistributedTest):
    world_size = 1

    def test_no_overflow(self):
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
                "loss_scale": 0,
                "initial_scale_power": 8,
                "loss_scale_window": 2
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**8
        expected_scale_window = 2
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.dynamic_loss_scale == True
        assert optim.cur_scale == expected_loss_scale
        assert optim.scale_window == expected_scale_window

        for i, value in enumerate(np.random.uniform(-0.1, 0.1, 10)):
            run_model_step(model, [value])
            assert optim.cur_scale == expected_loss_scale
            assert optim.cur_iter == (i + 1)
            if optim.cur_iter % expected_scale_window == 0:
                expected_loss_scale *= 2

    def test_all_overflow(self):
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
                "loss_scale": 0,
                "initial_scale_power": 4,
                "loss_scale_window": 2
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**4
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.dynamic_loss_scale == True
        assert optim.cur_scale == expected_loss_scale

        overflow_gradients = [float('inf'), float('-inf')] + [float('nan')] * 6
        for i, value in enumerate(overflow_gradients):
            run_model_step(model, [value])
            expected_loss_scale = max(expected_loss_scale / 2, 1)
            assert optim.cur_scale == expected_loss_scale
            assert optim.cur_iter == (i + 1)

    def test_some_overflow(self):
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
                "loss_scale": 0,
                "initial_scale_power": 8,
                "loss_scale_window": 2
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**8
        expected_scale_window = 2
        expected_iteration = 0
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.dynamic_loss_scale == True
        assert optim.cur_scale == expected_loss_scale
        assert optim.scale_window == expected_scale_window

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf'), float('nan')]
        expected_iteration += len(overflow_gradients)
        run_model_step(model, overflow_gradients)
        expected_loss_scale /= (2**len(overflow_gradients))
        assert optim.cur_scale == expected_loss_scale
        assert optim.cur_iter == expected_iteration

        # Run model scale_window + 1 times to increase scale once
        normal_gradients = np.random.uniform(-0.1, 0.1, expected_scale_window + 1)
        expected_iteration += len(normal_gradients)
        run_model_step(model, normal_gradients)
        expected_loss_scale *= 2
        assert optim.cur_scale == expected_loss_scale
        assert optim.cur_iter == expected_iteration

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf')]
        expected_iteration += len(overflow_gradients)
        run_model_step(model, overflow_gradients)
        expected_loss_scale /= (2**len(overflow_gradients))
        assert optim.cur_scale == expected_loss_scale
        assert optim.cur_iter == expected_iteration


class TestUnfused(DistributedTest):
    world_size = 1

    def test_no_overflow(self):
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 8,
                "loss_scale_window": 2
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        expected_loss_scale = 2**8
        expected_scale_window = 2
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.dynamic_loss_scale == True
        assert optim.cur_scale == expected_loss_scale
        assert optim.scale_window == expected_scale_window

        for i, value in enumerate(np.random.uniform(-0.1, 0.1, 10)):
            run_model_step(model, [value])
            assert optim.cur_scale == expected_loss_scale
            assert optim.cur_iter == (i + 1)
            if optim.cur_iter % expected_scale_window == 0:
                expected_loss_scale *= 2

    def test_all_overflow(self):
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 4,
                "loss_scale_window": 2,
                "min_loss_scale": 0.25
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**4
        expected_min_loss_scale = 0.25
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.dynamic_loss_scale == True
        assert optim.cur_scale == expected_loss_scale
        assert optim.min_loss_scale == expected_min_loss_scale

        overflow_gradients = [float('inf'), float('-inf')] + [float('nan')] * 6
        for i, value in enumerate(overflow_gradients):
            run_model_step(model, [value])
            expected_loss_scale = max(expected_loss_scale / 2, expected_min_loss_scale)
            assert optim.cur_scale == expected_loss_scale
            assert optim.cur_iter == (i + 1)

    def test_some_overflow(self):
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 8,
                "loss_scale_window": 2
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**8
        expected_scale_window = 2
        expected_iteration = 0
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.dynamic_loss_scale == True
        assert optim.cur_scale == expected_loss_scale
        assert optim.scale_window == expected_scale_window

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf'), float('nan')]
        expected_iteration += len(overflow_gradients)
        run_model_step(model, overflow_gradients)
        expected_loss_scale /= (2**len(overflow_gradients))
        assert optim.cur_scale == expected_loss_scale
        assert optim.cur_iter == expected_iteration

        # Run model scale_window + 1 times to increase scale once
        normal_gradients = np.random.uniform(-0.1, 0.1, expected_scale_window + 1)
        expected_iteration += len(normal_gradients)
        run_model_step(model, normal_gradients)
        expected_loss_scale *= 2
        assert optim.cur_scale == expected_loss_scale
        assert optim.cur_iter == expected_iteration

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf')]
        expected_iteration += len(overflow_gradients)
        run_model_step(model, overflow_gradients)
        expected_loss_scale /= (2**len(overflow_gradients))
        assert optim.cur_scale == expected_loss_scale
        assert optim.cur_iter == expected_iteration
