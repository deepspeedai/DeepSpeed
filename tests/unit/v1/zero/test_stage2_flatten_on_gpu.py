# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Test that ZeRO Stage 1 and 2 use the GPU flatten path when VRAM is sufficient.
Parametrized over zero_stage (1, 2) and dtype (fp32, fp16, bf16).
"""

import pytest
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import set_log_level_from_string
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


def _apply_dtype_to_config(config_dict, dtype):
    """Set bf16/fp16 in config_dict based on dtype; skip if not supported."""
    if dtype == "bf16":
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")
        config_dict["bf16"] = {"enabled": True}
    elif dtype == "fp16":
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator")
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
    # fp32: no half-precision block


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"], ids=["fp32", "fp16", "bf16"])
class TestStage2FlattenOnGPU(DistributedTest):
    """ZeRO-1 and ZeRO-2 with small model should flatten on GPU (sufficient VRAM)."""

    world_size = 2  # Run on 2 GPUs when available

    def test_flatten_on_gpu_path_taken(self, monkeypatch, zero_stage, dtype):
        """Assert the GPU flatten path was used (not CPU flatten + move)."""
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        _apply_dtype_to_config(config_dict, dtype)

        set_log_level_from_string("info")
        log_messages = []

        def mock_logger_info(msg, *args, **kwargs):
            log_messages.append(msg if isinstance(msg, str) else str(msg))

        monkeypatch.setattr("deepspeed.utils.logger.info", mock_logger_info)

        hidden_dim = 64
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )

        # Small model + no CPU offload => GPU path; that path logs "on GPU"
        gpu_path_logs = [m for m in log_messages if "Flattening param group" in m and "on GPU" in m]
        assert gpu_path_logs, (
            f"Expected GPU flatten path (logger.info should be called with 'Flattening param group' and 'on GPU'). "
            f"Captured messages: {log_messages}")

    def test_flat_buffers_on_accelerator(self, zero_stage, dtype):
        """Regression: flat buffers must end up on the accelerator (not left on CPU)."""
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        _apply_dtype_to_config(config_dict, dtype)

        hidden_dim = 64
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
        engine, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )
        opt = engine.optimizer
        assert hasattr(opt, "bit16_groups_flat"), "ZeRO-1/2 optimizer should have bit16_groups_flat"
        device_type = get_accelerator().device_name()
        for i, flat in enumerate(opt.bit16_groups_flat):
            assert flat.device.type == device_type, (f"Flat buffer {i} must be on {device_type}, got {flat.device}")
