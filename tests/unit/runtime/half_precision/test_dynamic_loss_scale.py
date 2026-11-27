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


class TestLossScaleGrowthUpdate(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("optimizer_type", [FP16_Optimizer, FP16_UnfusedOptimizer, 1, 2, 3])
    def test_parameter_updates(self, optimizer_type):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

        model = torch.nn.Linear(1, 1, bias=False).to(get_accelerator().current_device_name(), dtype=torch.float16)
        with torch.no_grad():
            model.weight.fill_(4.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        if isinstance(optimizer_type, int):
            config = dict(train_micro_batch_size_per_gpu=1,
                          gradient_clipping=0.0,
                          zero_allow_untested_optimizer=True,
                          zero_optimization=dict(stage=optimizer_type),
                          fp16=dict(enabled=True, initial_scale_power=3, loss_scale_window=2))
            model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            runner = model
        else:
            optimizer = optimizer_type(optimizer,
                                       dynamic_loss_scale=True,
                                       dynamic_loss_args=dict(init_scale=8, scale_window=2, min_scale=1))
            runner = optimizer

        inputs = torch.full((1, 1), 2.0, device=get_accelerator().current_device_name(), dtype=torch.float16)
        # The third step grows the scale from 8 to 16; every update must remain 0.5.
        for step, expected_scale in enumerate([8, 8, 16], start=1):
            runner.zero_grad()
            runner.backward(model(inputs).sum())
            runner.step()
            assert optimizer.loss_scale == expected_scale
            with deepspeed.zero.GatheredParameters(model.parameters()):
                assert next(model.parameters()).item() == pytest.approx(4.0 - 0.5 * step, abs=1e-3)

        if isinstance(optimizer_type, int):
            model.destroy()


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

    @pytest.mark.skipif(not get_accelerator().is_fp16_supported(), reason="fp16 is not supported")
    def test_hysteresis_through_the_engine(self, optimizer_class):
        # SGD gets FP16_UnfusedOptimizer from the engine, Adam gets FP16_Optimizer.
        optimizer_type = "Adam" if optimizer_class is FP16_Optimizer else "SGD"
        config_dict = {
            "train_batch_size": 1,
            "optimizer": {
                "type": optimizer_type,
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 8,
                "loss_scale_window": 100,
                "hysteresis": 2
            }
        }

        def engine():
            model = SimpleModel(hidden_dim=1)
            model, optim, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      model_parameters=model.parameters())
            assert isinstance(optim, optimizer_class)
            return model, optim

        model, optim = engine()
        run_model_step(model, [float('inf')])
        assert optim.loss_scale_config.cur_scale == 2**8, "the first overflow is absorbed"

        # Resume from the optimizer state an engine checkpoint stores: one overflow is already spent.
        restored_model, restored_optim = engine()
        restored_optim.load_state_dict(optim.state_dict())
        run_model_step(restored_model, [float('inf')])
        assert restored_optim.loss_scale_config.cur_scale == 2**7
