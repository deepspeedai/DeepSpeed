# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Test that ZeRO Stage 1 and 2 use the GPU flatten path when VRAM is sufficient.
Parametrized over zero_stage (1, 2) and dtype (fp32, fp16, bf16).
"""

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import set_log_level_from_string
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader

_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class _MisalignedParamModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.ones(1))
        self.weight = torch.nn.Parameter(torch.ones(8, 8))

    def forward(self, x):
        return (x @ self.weight).sum() + self.offset.sum()


def _init_alignment_engine(zero_stage):
    if not get_accelerator().is_available():
        pytest.skip("Accelerator not available")
    if not get_accelerator().is_bf16_supported():
        pytest.skip("bf16 is not supported on this accelerator")
    model = _MisalignedParamModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    config = {"train_micro_batch_size_per_gpu": 1, "bf16": {"enabled": True}, "zero_optimization": {"stage": zero_stage}}
    return deepspeed.initialize(config=config,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model.parameters())[0]


def _flat_weight(engine):
    opt = engine.optimizer
    index = next(i for i, param in enumerate(opt.round_robin_bit16_groups[0]) if param is engine.module.weight)
    offset = opt.round_robin_bit16_offsets[0][index]
    return opt.bit16_groups_flat[0].narrow(0, offset, engine.module.weight.numel()).view_as(engine.module.weight), offset


def _alignment_step(engine, lr=None):
    if lr is not None:
        engine.optimizer.optimizer.param_groups[0]["lr"] = lr
    data = torch.ones(1, 8, device=engine.device, dtype=torch.bfloat16)
    engine.backward(engine(data))
    engine.step()


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestStage12ParamAlignment(DistributedTest):
    world_size = 2

    def test_model_params_remain_16_byte_aligned(self, tmpdir, zero_stage):
        engine = _init_alignment_engine(zero_stage)
        flat_weight, offset = _flat_weight(engine)
        weight = engine.module.weight
        assert offset * weight.element_size() % 16 == 0
        assert weight.data_ptr() % 16 == 0
        assert weight.data_ptr() == flat_weight.data_ptr()

        before = weight.detach().clone()
        _alignment_step(engine)
        assert weight.data_ptr() == flat_weight.data_ptr()
        assert not torch.equal(weight, before)

        expected = weight.detach().clone()
        checkpoint_dir = str(tmpdir)
        engine.save_checkpoint(checkpoint_dir, tag="alignment")
        for load_kwargs in ({"load_module_only": True}, {"load_optimizer_states": False}):
            loaded = _init_alignment_engine(zero_stage)
            loaded.load_checkpoint(checkpoint_dir, tag="alignment", **load_kwargs)
            loaded_flat_weight, _ = _flat_weight(loaded)
            assert loaded.module.weight.data_ptr() % 16 == 0
            assert loaded.module.weight.data_ptr() == loaded_flat_weight.data_ptr()
            assert torch.equal(loaded.module.weight, expected)

            _alignment_step(loaded, lr=0.0)
            assert loaded.module.weight.data_ptr() == loaded_flat_weight.data_ptr()
            assert torch.equal(loaded.module.weight, expected)


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

        # Small model + no CPU offload => accelerator path logs "Flattening param group ... (sufficient memory)"
        accel_path_logs = [m for m in log_messages if "Flattening param group" in m and "(sufficient memory)" in m]
        assert accel_path_logs, (
            f"Expected accelerator flatten path (log should contain 'Flattening param group' and '(sufficient memory)'). "
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

    @pytest.mark.world_size(1)
    def test_flatten_on_accelerator_training_step(self, zero_stage, dtype):
        """Regression: flat buffer must be detached so inplace ops during step don't crash."""
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
        for flat in engine.optimizer.bit16_groups_flat:
            assert flat.grad_fn is None, ("Flat buffer must be detached from autograd graph"
                                          " to prevent inplace-modification errors during optimizer step")

        data_loader = random_dataloader(model=engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=engine.device,
                                        dtype=_DTYPE_MAP[dtype])
        for batch in data_loader:
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()