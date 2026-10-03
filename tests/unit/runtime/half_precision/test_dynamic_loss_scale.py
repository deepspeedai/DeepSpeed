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
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer


def run_model_step(model, gradient_list):
    for value in gradient_list:
        for p in model.parameters():
            p.grad = torch.empty_like(p, dtype=p.dtype)
            p.grad.fill_(value)
        model.step()


class TestLossScaleGrowthUpdate(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("optimizer_type", ["fused", "unfused", 1, 2, 3])
    @pytest.mark.parametrize("clip_grad", [0.0, 1.5, 0.5])
    def test_parameter_updates(self, optimizer_type, clip_grad):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

        model = torch.nn.Linear(1, 1, bias=False).to(get_accelerator().current_device_name(), dtype=torch.float16)
        with torch.no_grad():
            model.weight.fill_(4.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        if isinstance(optimizer_type, str):
            # Isolate the wrappers from the engine's separate ZeRO-0 backward scaling issue.
            wrapper = FP16_Optimizer if optimizer_type == "fused" else FP16_UnfusedOptimizer
            optimizer = wrapper(optimizer,
                                dynamic_loss_scale=True,
                                dynamic_loss_args={
                                    "init_scale": 8,
                                    "scale_window": 2,
                                    "min_scale": 1
                                },
                                clip_grad=clip_grad)
            runner = optimizer
        else:
            config = {
                "train_micro_batch_size_per_gpu": 1,
                "gradient_clipping": clip_grad,
                "zero_allow_untested_optimizer": True,
                "zero_optimization": {
                    "stage": optimizer_type
                },
                "fp16": {
                    "enabled": True,
                    "initial_scale_power": 3,
                    "loss_scale_window": 2,
                    "hysteresis": 1
                }
            }
            model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
            runner = model

        expected_weight = 4.0
        expected_grad = min(2.0, clip_grad) if clip_grad else 2.0
        # A constant derivative exposes undersized updates on growth steps; overflow must skip the update.
        for value, expected_scale in zip([2.0, 2.0, 2.0, float("inf"), 2.0, 2.0, 2.0], [8, 8, 16, 8, 8, 8, 16]):
            runner.zero_grad()
            inputs = torch.tensor([[value]], device=get_accelerator().current_device_name(), dtype=torch.float16)
            runner.backward(model(inputs).sum())
            runner.step()
            if value != float("inf"):
                expected_weight -= 0.25 * expected_grad
            assert optimizer.loss_scale == expected_scale
            with deepspeed.zero.GatheredParameters(model.parameters()):
                assert next(model.parameters()).item() == pytest.approx(expected_weight, abs=1e-3)

        if not isinstance(optimizer_type, str):
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
                "loss_scale_window": 2
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
                "min_loss_scale": min_loss_scale_value
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
