# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Test that ZeRO Stage 1 and 2 use the GPU flatten path when VRAM is sufficient.
Parametrized over zero_stage (1, 2) and dtype (fp32, fp16, bf16).
"""

import glob
import os

import pytest
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.constants import (BASE_OPTIMIZER_STATE, BASE_OPTIMIZER_STATE_STEP, GROUP_PADDINGS,
                                            SINGLE_PARTITION_OF_FP32_GROUPS)
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, PARAM_ALIGNMENT_PADDINGS
from deepspeed.utils import safe_get_full_grad, safe_set_full_grad, set_log_level_from_string
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
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


class _FallbackLayoutModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.ones(1))
        self.first = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64)
        self.second = torch.nn.Parameter((torch.arange(64, dtype=torch.float32) + 128).reshape(8, 8) / 64)

    def forward(self, x):
        return (x @ self.first).sum() + (x @ self.second).sum() + self.offset.sum()


def _init_alignment_engine(zero_stage, parameter_alignment=True, offload_optimizer=False):
    if not get_accelerator().is_available():
        pytest.skip("Accelerator not available")
    if not get_accelerator().is_bf16_supported():
        pytest.skip("bf16 is not supported on this accelerator")
    model = _MisalignedParamModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    zero_config = {"stage": zero_stage}
    if parameter_alignment is not None:
        zero_config["parameter_alignment"] = parameter_alignment
    if offload_optimizer:
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": False}
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": zero_config,
    }
    if offload_optimizer:
        config["zero_allow_untested_optimizer"] = True
        config["zero_force_ds_cpu_optimizer"] = False
    return deepspeed.initialize(config=config, model=model, optimizer=optimizer,
                                model_parameters=model.parameters())[0]


def _flat_weight(engine):
    opt = engine.optimizer
    index = next(i for i, param in enumerate(opt.round_robin_bit16_groups[0]) if param is engine.module.weight)
    offset = opt.round_robin_bit16_offsets[0][index]
    return opt.bit16_groups_flat[0].narrow(0, offset,
                                           engine.module.weight.numel()).view_as(engine.module.weight), offset


def _alignment_step(engine, lr=None, input_value=1):
    if lr is not None:
        engine.optimizer.optimizer.param_groups[0]["lr"] = lr
    data = torch.full((1, 8), input_value, device=engine.device, dtype=torch.bfloat16)
    engine.backward(engine(data))
    engine.step()


@pytest.mark.parametrize("zero_stage", [1, 2])
class TestStage12ParamAlignment(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("parameter_alignment", [None, False], ids=["unset", "explicit-false"])
    def test_parameter_alignment_is_opt_in(self, zero_stage, parameter_alignment):
        engine = _init_alignment_engine(zero_stage, parameter_alignment=parameter_alignment)

        flat_weight, offset = _flat_weight(engine)
        assert offset == engine.module.offset.numel()
        assert engine.module.weight.data_ptr() == flat_weight.data_ptr()
        assert engine.module.weight.data_ptr() % 16 != 0

    @pytest.mark.world_size(1)
    def test_default_layout_optimizer_offload_uses_unpadded_gradients(self, zero_stage):
        engine = _init_alignment_engine(zero_stage, parameter_alignment=None, offload_optimizer=True)
        data = torch.ones(1, 8, device=engine.device, dtype=torch.bfloat16)
        engine.backward(engine(data))

        full_grad = safe_get_full_grad(engine.module.weight)
        assert torch.equal(full_grad, torch.ones_like(full_grad))
        before = engine.module.weight.detach().clone()
        engine.step()
        assert not torch.equal(engine.module.weight, before)

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

    def test_current_optimizer_checkpoint_continues_training(self, tmpdir, zero_stage):
        engine = _init_alignment_engine(zero_stage)
        _alignment_step(engine)
        checkpoint_dir = str(tmpdir)
        engine.save_checkpoint(checkpoint_dir, tag="optimizer-continuation")

        _alignment_step(engine, input_value=2)
        expected = {name: param.detach().clone() for name, param in engine.module.named_parameters()}

        loaded = _init_alignment_engine(zero_stage)
        loaded.load_checkpoint(checkpoint_dir, tag="optimizer-continuation")
        _alignment_step(loaded, input_value=2)
        for name, param in loaded.module.named_parameters():
            assert torch.equal(param, expected[name])

    def test_checkpoint_layout_mismatch_is_safe(self, tmpdir, zero_stage):
        engine = _init_alignment_engine(zero_stage)
        _alignment_step(engine)
        checkpoint_dir = str(tmpdir)
        engine.save_checkpoint(checkpoint_dir, tag="padded-layout")
        expected = {name: param.detach().clone() for name, param in engine.module.named_parameters()}

        for load_kwargs in ({"load_module_only": True}, {"load_optimizer_states": False}):
            loaded = _init_alignment_engine(zero_stage, parameter_alignment=False)
            loaded.load_checkpoint(checkpoint_dir, tag="padded-layout", **load_kwargs)
            for name, param in loaded.module.named_parameters():
                assert torch.equal(param, expected[name])

        loaded = _init_alignment_engine(zero_stage, parameter_alignment=False)
        with pytest.raises(RuntimeError, match="parameter-alignment layout"):
            loaded.load_checkpoint(checkpoint_dir, tag="padded-layout")

    @pytest.mark.parametrize("load_parameter_alignment", [False, True], ids=["unpadded", "aligned"])
    def test_legacy_unpadded_checkpoint_migrates_across_layouts(self, tmpdir, zero_stage, load_parameter_alignment):
        engine = _init_alignment_engine(zero_stage, parameter_alignment=False)
        _alignment_step(engine)
        checkpoint_dir = str(tmpdir)
        tag = "legacy-unpadded"
        engine.save_checkpoint(checkpoint_dir, tag=tag)

        if engine.global_rank == 0:
            pattern = os.path.join(checkpoint_dir, tag, "*_optim_states.pt")
            for checkpoint_file in glob.glob(pattern):
                checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
                checkpoint[OPTIMIZER_STATE_DICT].pop(PARAM_ALIGNMENT_PADDINGS)
                torch.save(checkpoint, checkpoint_file)
        dist.barrier()

        fp32_state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=tag)
        for name, param in engine.module.named_parameters():
            assert torch.equal(fp32_state[name].to(param.dtype), param.detach().cpu())

        _alignment_step(engine, input_value=2)
        expected = {name: param.detach().clone() for name, param in engine.module.named_parameters()}

        loaded = _init_alignment_engine(zero_stage, parameter_alignment=load_parameter_alignment)
        loaded.load_checkpoint(checkpoint_dir, tag=tag)
        _alignment_step(loaded, input_value=2)
        for name, param in loaded.module.named_parameters():
            assert torch.equal(param, expected[name])

    @pytest.mark.world_size(1)
    def test_cpu_flatten_fallback_preserves_layout_and_trains(self, monkeypatch, zero_stage):
        if not get_accelerator().is_available():
            pytest.skip("Accelerator not available")
        if not get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")

        monkeypatch.setattr(get_accelerator(), "available_memory", lambda *args, **kwargs: 0)
        model = _FallbackLayoutModel()
        expected_first = model.first.detach().to(torch.bfloat16)
        expected_second = model.second.detach().to(torch.bfloat16)
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "parameter_alignment": True,
            },
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               optimizer=optimizer,
                                               model_parameters=model.parameters())

        opt = engine.optimizer
        offsets = {
            id(param): offset
            for param, offset in zip(opt.round_robin_bit16_groups[0], opt.round_robin_bit16_offsets[0])
        }
        assert offsets[id(engine.module.offset)] == 0
        assert offsets[id(engine.module.first)] == 8
        assert offsets[id(engine.module.second)] == 8 + engine.module.first.numel()
        assert engine.module.first.data_ptr() % 16 == 0
        assert engine.module.second.data_ptr() % 16 == 0
        assert torch.equal(engine.module.first.detach().cpu(), expected_first)
        assert torch.equal(engine.module.second.detach().cpu(), expected_second)

        before = engine.module.second.detach().clone()
        data = torch.ones(1, 8, device=engine.device, dtype=torch.bfloat16)
        for _ in range(2):
            engine.backward(engine(data))
            engine.step()
        assert not torch.equal(engine.module.second, before)

    @pytest.mark.world_size(1)
    def test_safe_full_grad_accounts_for_alignment_padding(self, zero_stage):
        engine = _init_alignment_engine(zero_stage)
        data = torch.ones(1, 8, device=engine.device, dtype=torch.bfloat16)
        engine.backward(engine(data))

        weight = engine.module.weight
        full_grad = safe_get_full_grad(weight)
        assert torch.equal(full_grad, torch.ones_like(full_grad))

        replacement = torch.full_like(full_grad, 3)
        safe_set_full_grad(weight, replacement)
        assert torch.equal(safe_get_full_grad(weight), replacement)

    def test_pre_padding_checkpoint_preserves_tensor_metadata(self, zero_stage):
        engine = _init_alignment_engine(zero_stage, parameter_alignment=True)
        opt = engine.optimizer
        group_id = 0
        world_size = dist.get_world_size(group=opt.real_dp_process_group[group_id])
        rank = dist.get_rank(group=opt.real_dp_process_group[group_id])
        alignment = opt.nccl_start_alignment_factor * world_size
        unpadded_numel = sum(param.numel() for param in opt.round_robin_bit16_groups[group_id])
        old_group_numel = ((unpadded_numel + alignment - 1) // alignment) * alignment
        old_partition_size = old_group_numel // world_size
        partition_start = rank * old_partition_size
        old_group_padding = max(0, partition_start + old_partition_size - unpadded_numel)
        old_group_padding = min(old_group_padding, old_partition_size)

        param_id = 0
        tensor_step = torch.tensor([17], dtype=torch.int64)
        current_rank_sd = {
            BASE_OPTIMIZER_STATE: {
                "state": {
                    param_id: {
                        "exp_avg": torch.arange(old_partition_size, dtype=torch.float32),
                        "tensor_step": tensor_step,
                    }
                },
                "param_groups": [{
                    "params": [param_id]
                }],
            },
            SINGLE_PARTITION_OF_FP32_GROUPS:
            [torch.zeros(old_partition_size - old_group_padding, dtype=torch.float32)],
            GROUP_PADDINGS: [old_group_padding],
        }

        converted = opt._convert_unpadded_rigid_optimizer_state(current_rank_sd)
        converted_state = converted["state"][param_id]
        assert converted_state["exp_avg"].numel() == opt.single_partition_of_fp32_groups[group_id].numel()
        assert torch.equal(converted_state["tensor_step"], tensor_step)

    def test_pre_padding_elastic_checkpoint_preserves_tensor_metadata(self, zero_stage):
        engine = _init_alignment_engine(zero_stage, parameter_alignment=True)
        _alignment_step(engine)
        opt = engine.optimizer
        group_id = 0
        world_size = dist.get_world_size(group=opt.real_dp_process_group[group_id])
        alignment = opt.nccl_start_alignment_factor * world_size
        unpadded_numel = sum(param.numel() for param in opt.round_robin_bit16_groups[group_id])
        old_group_numel = ((unpadded_numel + alignment - 1) // alignment) * alignment
        old_partition_size = old_group_numel // world_size
        tensor_step = torch.tensor([17], dtype=torch.int64)
        all_state_dict = []

        for rank in range(world_size):
            partition_start = rank * old_partition_size
            group_padding = max(0, partition_start + old_partition_size - unpadded_numel)
            group_padding = min(group_padding, old_partition_size)
            lean_partition_size = old_partition_size - group_padding
            all_state_dict.append({
                BASE_OPTIMIZER_STATE: [{
                    "exp_avg": torch.arange(lean_partition_size, dtype=torch.float32),
                    "tensor_step": tensor_step.clone(),
                }],
                BASE_OPTIMIZER_STATE_STEP:
                17,
                SINGLE_PARTITION_OF_FP32_GROUPS: [torch.zeros(lean_partition_size, dtype=torch.float32)],
                GROUP_PADDINGS: [group_padding],
            })

        opt._restore_elastic_base_optimizer_state(all_state_dict, unpadded_layout=True)
        flat_param = opt.optimizer.param_groups[group_id]["params"][0]
        restored_state = opt.optimizer.state[flat_param]
        assert restored_state["exp_avg"].numel() == flat_param.numel()
        assert restored_state["tensor_step"].numel() == 1
        assert restored_state["tensor_step"].item() == tensor_step.item()


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
