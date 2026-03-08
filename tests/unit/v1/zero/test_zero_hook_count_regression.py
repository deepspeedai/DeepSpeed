# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression tests for count_used_parameters_in_backward() call count.

Verifies fix for https://github.com/deepspeedai/DeepSpeed/issues/7885:
count_used_parameters_in_backward() was called once per gradient hook
(O(n) calls per backward) instead of once per backward phase (O(1)
for non-reentrant, O(p) for reentrant with p phases).
"""

import pytest
import torch
from unittest.mock import patch

import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader


def get_config_dict(zero_stage):
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
    }

    if zero_stage == 3:
        config_dict["zero_optimization"]["stage3_param_persistence_threshold"] = 0

    if get_accelerator().is_bf16_supported():
        config_dict["bf16"] = {"enabled": True}
    elif get_accelerator().is_fp16_supported():
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}

    return config_dict


class TestHookCountRegression(DistributedTest):
    """Test that count_used_parameters_in_backward is not called per-hook."""
    world_size = 2

    @pytest.mark.parametrize("zero_stage", [2, 3])
    def test_non_reentrant_single_count_call(self, zero_stage):
        """Non-reentrant backward should call count_used_parameters_in_backward exactly once."""
        hidden_dim = 16
        model = SimpleModel(hidden_dim)
        config = get_config_dict(zero_stage)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

        data_loader = random_dataloader(model=engine, total_samples=4, hidden_dim=hidden_dim, device=engine.device)

        # Determine the correct module path to patch based on stage
        if zero_stage == 2:
            patch_target = "deepspeed.runtime.zero.stage_1_and_2.count_used_parameters_in_backward"
        else:
            patch_target = "deepspeed.runtime.zero.stage3.count_used_parameters_in_backward"

        call_counts = []

        for batch in data_loader:
            with patch(patch_target, wraps=deepspeed.runtime.utils.count_used_parameters_in_backward) as mock_count:
                loss = engine(batch[0], batch[1])
                engine.backward(loss)
                call_counts.append(mock_count.call_count)
            engine.step()
            break

        # Non-reentrant: exactly 1 call per backward
        assert call_counts[0] == 1, (f"Expected exactly 1 call to count_used_parameters_in_backward "
                                     f"per backward, got {call_counts[0]}")

    @pytest.mark.parametrize("zero_stage", [2, 3])
    def test_training_step_succeeds_after_fix(self, zero_stage):
        """Verify a full training step produces a finite loss after the caching fix."""
        hidden_dim = 16
        model = SimpleModel(hidden_dim)
        config = get_config_dict(zero_stage)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

        data_loader = random_dataloader(model=engine, total_samples=8, hidden_dim=hidden_dim, device=engine.device)

        losses = []
        for i, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            assert torch.isfinite(loss), f"Loss is not finite at step {i}: {loss.item()}"
            losses.append(loss.item())
            engine.backward(loss)
            engine.step()
            if i >= 1:
                break

        assert len(losses) >= 2, "Expected at least 2 training steps"
