# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging

import torch
import deepspeed
import deepspeed.comm as dist
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


class _LogRecorder:
    # Minimal handler collecting DeepSpeed-logger INFO records, mirroring
    # tests/unit/utils/test_pin_memory_tracker.py's caplog idiom, needed here
    # because the test drives two independent optimizer instances and wants
    # a fresh record list for each rather than one caplog fixture's history.
    def __init__(self, logger):
        self.logger = logger
        self.records = []
        self._old_level = logger.level

    def __enter__(self):
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.records.append(record.getMessage())
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        return self

    def __exit__(self, *exc_info):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._old_level)


@pytest.mark.parametrize("optimizer_cls", [FP16_Optimizer, FP16_UnfusedOptimizer])
def test_overflow_logs_only_on_rank_zero(optimizer_cls, monkeypatch):
    """fp16 dynamic-loss-scale overflow/rescale messages must not be logged on
    every rank (issue #1533): on a many-rank run every rank printed the same
    "Grad overflow"/"Reducing dynamic loss scale" lines, drowning real signal.
    Directly building the optimizer and faking dist.get_rank() is the only
    way to exercise multi-rank logging behavior without a real distributed
    job or a GPU.
    """
    ds_logger = logging.getLogger("DeepSpeed")
    old_propagate = ds_logger.propagate
    ds_logger.propagate = True
    try:
        monkeypatch.setattr(dist, "is_initialized", lambda: True)

        monkeypatch.setattr(dist, "get_rank", lambda *args, **kwargs: 0)
        param0 = torch.nn.Parameter(torch.zeros(4))
        optimizer0 = optimizer_cls(torch.optim.SGD([param0], lr=0.1), dynamic_loss_scale=True, verbose=True)
        param0.grad = torch.full_like(param0, float('nan'))
        with _LogRecorder(ds_logger) as rank0:
            overflow = optimizer0.step()
        assert overflow is True
        rank0_records = [m for m in rank0.records if 'verflow' in m or 'oss scale' in m]
        # Sanity check the harness can observe the message at all, otherwise
        # an empty result on rank 1 below would prove nothing.
        assert rank0_records, "expected an overflow/loss-scale message to be logged on rank 0"

        monkeypatch.setattr(dist, "get_rank", lambda *args, **kwargs: 1)
        param1 = torch.nn.Parameter(torch.zeros(4))
        optimizer1 = optimizer_cls(torch.optim.SGD([param1], lr=0.1), dynamic_loss_scale=True, verbose=True)
        param1.grad = torch.full_like(param1, float('nan'))
        with _LogRecorder(ds_logger) as rank1:
            optimizer1.step()
        rank1_records = [m for m in rank1.records if 'verflow' in m or 'oss scale' in m]
        assert not rank1_records, \
            f"fp16 overflow/loss-scale messages must not be logged on non-zero ranks, got: {rank1_records}"
    finally:
        ds_logger.propagate = old_propagate
