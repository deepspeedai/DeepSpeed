# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import copy
from types import SimpleNamespace

import pytest
import torch

import deepspeed
from deepspeed.checkpoint.constants import CLIP_GRAD, DS_VERSION
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def _config(*, compute_grad_norm, gradient_clipping=0.0, stage=1, offload_optimizer=None, zenflow=None):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {
            "enabled": True,
            "check_grad_overflow": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
            },
        },
        "zero_optimization": {
            "stage": stage,
            "compute_grad_norm": compute_grad_norm,
            "offload_optimizer": offload_optimizer,
            "zenflow": zenflow,
        },
        "gradient_clipping": gradient_clipping,
    }


def test_identity_unscale_is_skipped():
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.clip_grad = 0.0
    optimizer.custom_loss_scaler = False
    optimizer.loss_scaler = SimpleNamespace(cur_scale=1.0)
    gradient = torch.ones(4)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        optimizer.unscale_and_clip_grads([gradient], total_norm=None)

    assert "aten::mul_" not in {event.key for event in profiler.key_averages()}


def test_non_identity_scale_still_unscales():
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.clip_grad = 0.0
    optimizer.custom_loss_scaler = False
    optimizer.loss_scaler = SimpleNamespace(cur_scale=2.0)
    gradient = torch.ones(4)

    optimizer.unscale_and_clip_grads([gradient], total_norm=None)

    torch.testing.assert_close(gradient, torch.full_like(gradient, 0.5))


def test_checkpoint_clipping_rejects_disabled_norm():
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.compute_grad_norm = False
    optimizer.loss_scaler = SimpleNamespace()
    optimizer.dynamic_loss_scale = False
    optimizer.overflow = False
    optimizer.clip_grad = 0.0

    with pytest.raises(ValueError, match="checkpoint with gradient clipping"):
        optimizer._load_global_state({
            CLIP_GRAD: 1.0,
            DS_VERSION: "0.18.0",
        })


class TestZero1OptimizerFastPath(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize("stage", [1, 2])
    def test_fast_path_matches_default_update(self, stage):
        torch.manual_seed(123)
        baseline_model = SimpleModel(hidden_dim=4)
        fast_model = copy.deepcopy(baseline_model)
        baseline_engine, baseline_optimizer, _, _ = deepspeed.initialize(
            model=baseline_model,
            model_parameters=baseline_model.parameters(),
            config=_config(compute_grad_norm=True, stage=stage))
        fast_engine, fast_optimizer, _, _ = deepspeed.initialize(
            model=fast_model,
            model_parameters=fast_model.parameters(),
            config=_config(compute_grad_norm=False, stage=stage))
        inputs = torch.randn(1, 4, device=baseline_engine.device, dtype=torch.bfloat16)
        targets = torch.randn(1, 4, device=baseline_engine.device, dtype=torch.bfloat16)

        baseline_loss = baseline_engine(inputs, targets)
        fast_loss = fast_engine(inputs, targets)
        baseline_engine.backward(baseline_loss)
        fast_engine.backward(fast_loss)
        baseline_engine.step()
        fast_engine.step()

        torch.testing.assert_close(fast_loss, baseline_loss)
        assert baseline_optimizer._global_grad_norm is not None
        assert fast_optimizer._global_grad_norm is None
        for baseline_parameter, fast_parameter in zip(baseline_engine.module.parameters(),
                                                      fast_engine.module.parameters()):
            torch.testing.assert_close(fast_parameter, baseline_parameter)

    @pytest.mark.parametrize("stage", [1, 2])
    def test_finite_step_skips_norm_but_updates_parameters(self, stage):
        model = SimpleModel(hidden_dim=4)
        engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                       model_parameters=model.parameters(),
                                                       config=_config(compute_grad_norm=False, stage=stage))
        inputs = torch.randn(1, 4, device=engine.device, dtype=torch.bfloat16)
        targets = torch.randn(1, 4, device=engine.device, dtype=torch.bfloat16)
        before = [parameter.detach().clone() for parameter in engine.module.parameters()]

        engine.backward(engine(inputs, targets))
        engine.step()

        assert optimizer.check_grad_overflow
        assert optimizer._global_grad_norm is None
        assert engine.get_global_grad_norm() is None
        assert any(not torch.equal(previous, current) for previous, current in zip(before, engine.module.parameters()))

    def test_overflow_check_still_skips_the_step(self):
        model = SimpleModel(hidden_dim=4)
        engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                       model_parameters=model.parameters(),
                                                       config=_config(compute_grad_norm=False))
        inputs = torch.randn(1, 4, device=engine.device, dtype=torch.bfloat16)
        targets = torch.randn(1, 4, device=engine.device, dtype=torch.bfloat16)
        engine.backward(engine(inputs, targets))
        gradient = next(gradient for gradients in optimizer.averaged_gradients.values() for gradient in gradients
                        if gradient is not None)
        gradient.view(-1)[0] = float("nan")
        before = [parameter.detach().clone() for parameter in engine.module.parameters()]

        engine.step()

        assert optimizer.overflow
        assert optimizer._global_grad_norm is None
        assert all(torch.equal(previous, current) for previous, current in zip(before, engine.module.parameters()))

    def test_gradient_clipping_rejects_disabled_norm(self):
        model = SimpleModel(hidden_dim=4)
        with pytest.raises(ValueError, match="requires gradient_clipping=0"):
            deepspeed.initialize(model=model,
                                 model_parameters=model.parameters(),
                                 config=_config(compute_grad_norm=False, gradient_clipping=1.0))

    def test_optimizer_offload_rejects_disabled_norm(self):
        model = SimpleModel(hidden_dim=4)
        with pytest.raises(ValueError, match="does not support optimizer offload"):
            deepspeed.initialize(model=model,
                                 model_parameters=model.parameters(),
                                 config=_config(compute_grad_norm=False, offload_optimizer={"device": "cpu"}))

    def test_zenflow_rejects_disabled_norm(self):
        model = SimpleModel(hidden_dim=4)
        with pytest.raises(ValueError, match="does not support ZenFlow"):
            deepspeed.initialize(model=model,
                                 model_parameters=model.parameters(),
                                 config=_config(compute_grad_norm=False, zenflow={}))
