# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import pytest
import numpy as np
from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.ops.op_builder import FusedLambBuilder
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.loss_scaler import LossScaleConfig, LossScaleProfile
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.precision_config import DeepSpeedFP16Config


def run_model_step(model, gradient_list):
    for value in gradient_list:
        for p in model.parameters():
            p.grad = torch.empty_like(p, dtype=p.dtype)
            p.grad.fill_(value)
        model.step()


class TestFused(DistributedTest):
    world_size = 1

    def test_no_overflow(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

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
        assert optim.loss_scale_config.dynamic_loss_scale == True
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.scale_window == expected_scale_window

        for i, value in enumerate(np.random.uniform(-0.1, 0.1, 10)):
            run_model_step(model, [value])
            assert optim.loss_scale_config.cur_scale == expected_loss_scale
            assert optim.loss_scale_config.cur_iter == (i + 1)
            if optim.loss_scale_config.cur_iter % expected_scale_window == 0:
                expected_loss_scale *= 2

    def test_all_overflow(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
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
                "loss_scale_window": 2,
                "hysteresis": 1
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**4
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.loss_scale_config.dynamic_loss_scale == True
        assert optim.loss_scale_config.cur_scale == expected_loss_scale

        overflow_gradients = [float('inf'), float('-inf')] + [float('nan')] * 6
        for i, value in enumerate(overflow_gradients):
            run_model_step(model, [value])
            expected_loss_scale = max(expected_loss_scale / 2, 1)
            assert optim.loss_scale_config.cur_scale == expected_loss_scale
            assert optim.loss_scale_config.cur_iter == (i + 1)

    def test_some_overflow(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
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
                "loss_scale_window": 2,
                "hysteresis": 1
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**8
        expected_scale_window = 2
        expected_iteration = 0
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.loss_scale_config.dynamic_loss_scale == True
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.scale_window == expected_scale_window

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf'), float('nan')]
        expected_iteration += len(overflow_gradients)
        run_model_step(model, overflow_gradients)
        expected_loss_scale /= (2**len(overflow_gradients))
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.cur_iter == expected_iteration

        # Run model scale_window + 1 times to increase scale once
        normal_gradients = np.random.uniform(-0.1, 0.1, expected_scale_window + 1)
        expected_iteration += len(normal_gradients)
        run_model_step(model, normal_gradients)
        expected_loss_scale *= 2
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.cur_iter == expected_iteration

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf')]
        expected_iteration += len(overflow_gradients)
        run_model_step(model, overflow_gradients)
        expected_loss_scale /= (2**len(overflow_gradients))
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.cur_iter == expected_iteration


@pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                    reason="FusedLambBuilder has not been implemented on this system.")
class TestUnfused(DistributedTest):
    world_size = 1

    def test_no_overflow(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
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
        assert optim.loss_scale_config.dynamic_loss_scale == True
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.scale_window == expected_scale_window

        for i, value in enumerate(np.random.uniform(-0.1, 0.1, 10)):
            run_model_step(model, [value])
            assert optim.loss_scale_config.cur_scale == expected_loss_scale
            assert optim.loss_scale_config.cur_iter == (i + 1)
            if optim.loss_scale_config.cur_iter % expected_scale_window == 0:
                expected_loss_scale *= 2

    def test_all_overflow(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

        min_loss_scale_value = 2.0

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
                "min_loss_scale": min_loss_scale_value,
                "hysteresis": 1
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**4
        expected_min_loss_scale = min_loss_scale_value
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.loss_scale_config.dynamic_loss_scale == True
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.min_loss_scale == expected_min_loss_scale

        overflow_gradients = [float('inf'), float('-inf')] + [float('nan')] * 6
        for i, value in enumerate(overflow_gradients):
            run_model_step(model, [value])
            expected_loss_scale = max(expected_loss_scale / 2, expected_min_loss_scale)
            assert optim.loss_scale_config.cur_scale == expected_loss_scale
            assert optim.loss_scale_config.cur_iter == (i + 1)

    def test_some_overflow(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
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
                "loss_scale_window": 2,
                "hysteresis": 1
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        expected_loss_scale = 2**8
        expected_scale_window = 2
        expected_iteration = 0
        # Ensure the dynamic loss scaler is correctly configured.
        assert optim.loss_scale_config.dynamic_loss_scale == True
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.scale_window == expected_scale_window

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf'), float('nan')]
        expected_iteration += len(overflow_gradients)
        run_model_step(model, overflow_gradients)
        expected_loss_scale /= (2**len(overflow_gradients))
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.cur_iter == expected_iteration

        # Run model scale_window + 1 times to increase scale once
        normal_gradients = np.random.uniform(-0.1, 0.1, expected_scale_window + 1)
        expected_iteration += len(normal_gradients)
        run_model_step(model, normal_gradients)
        expected_loss_scale *= 2
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.cur_iter == expected_iteration

        # Run model with overflows to decrease scale
        overflow_gradients = [float('inf')]
        expected_iteration += len(overflow_gradients)
        run_model_step(model, overflow_gradients)
        expected_loss_scale /= (2**len(overflow_gradients))
        assert optim.loss_scale_config.cur_scale == expected_loss_scale
        assert optim.loss_scale_config.cur_iter == expected_iteration


@pytest.mark.parametrize("optimizer_class", [FP16_Optimizer, FP16_UnfusedOptimizer])
class TestZeroStage0Hysteresis(DistributedTest):
    world_size = 1

    def _optimizer(self, optimizer_class, **fp16_config):
        fp16 = DeepSpeedFP16Config(enabled=True, loss_scale=0, initial_scale_power=8, **fp16_config)
        profile = LossScaleProfile.FUSED if optimizer_class is FP16_Optimizer else LossScaleProfile.UNFUSED
        loss_scale_config = LossScaleConfig(low_precision_dtype=torch.float16,
                                            dynamic_loss_scale=True,
                                            static_loss_scale=0,
                                            dynamic_loss_args=fp16.dynamic_loss_scale_args(),
                                            profile=profile)
        param = torch.nn.Parameter(torch.zeros(4, dtype=torch.float16))
        return optimizer_class(torch.optim.SGD([param], lr=0.1), loss_scale_config=loss_scale_config)

    def test_hysteresis(self, optimizer_class):
        optimizer = self._optimizer(optimizer_class, hysteresis=3)
        scales = []
        for _ in range(4):
            optimizer._update_scale(True)
            scales.append(optimizer.loss_scale_config.cur_scale)
        # The first hysteresis - 1 overflows are absorbed, then each overflow halves the scale.
        assert scales == [2**8, 2**8, 2**7, 2**6]

    def test_consecutive_hysteresis(self, optimizer_class):
        optimizer = self._optimizer(optimizer_class, hysteresis=2, consecutive_hysteresis=True)
        for _ in range(4):
            optimizer._update_scale(True)
            optimizer._update_scale(False)
        # A clean step refills the hysteresis, so isolated overflows never lower the scale.
        assert optimizer.loss_scale_config.cur_scale == 2**8

    def test_hysteresis_survives_a_checkpoint(self, optimizer_class):
        optimizer = self._optimizer(optimizer_class, hysteresis=3)
        optimizer._update_scale(True)
        restored = self._optimizer(optimizer_class, hysteresis=3)
        restored.load_state_dict(optimizer.state_dict())
        # One overflow is already spent, so the next one is the last one absorbed.
        assert restored.loss_scale_config.cur_hysteresis == 2
