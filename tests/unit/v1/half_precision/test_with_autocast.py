# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
from unit.common import DistributedTest, allclose_on_all_ranks
from deepspeed.ops.op_builder import CPUAdamBuilder
from unit.simple_model import SimpleModel, random_dataloader
from unit.util import bf16_required_version_check
from deepspeed.accelerator import get_accelerator
from unit.v1.zero.test_zero_user_backward import (initialize_distributed, create_ddp_model, collect_ddp_gradients,
                                                  collect_gradients_safe, compare_gradients)


class TestTorchAutocastWithPrecisionModes(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("precision_mode,zero_stage", [
        pytest.param("bf16_full", 1, id="z1_bf16_full_autocast"),
        pytest.param("bf16_full", 2, id="z2_bf16_full_autocast"),
        pytest.param("bf16_full", 3, id="z3_bf16_full_autocast"),
    ])
    def test_gradients_match_ddp_with_autocast(self, precision_mode, zero_stage):
        """Test BF16 with torch_autocast by comparing gradients with DDP baseline."""
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        hidden_dim = 6
        lr = 1e-3
        seed = 123

        device, rank, dtype = initialize_distributed()

        # Create DDP baseline with torch.autocast
        model_ddp, optimizer_ddp = create_ddp_model(SimpleModel,
                                                    device,
                                                    rank,
                                                    dtype,
                                                    seed=seed,
                                                    lr=lr,
                                                    hidden_dim=hidden_dim,
                                                    nlayers=2)

        torch.manual_seed(seed)
        ds_model = SimpleModel(hidden_dim, nlayers=2)

        # BF16 configuration
        autocast_dtype = torch.bfloat16
        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": lr
                }
            },
            "torch_autocast": {
                "enabled": True,
                "dtype": str(autocast_dtype)
            },
            "bf16": {
                "enabled": True,
                "bf16_master_weights_and_grads": True,
                "bf16_optimizer_states": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }

        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=ds_model,
                                               model_parameters=ds_model.parameters())

        data_loader = random_dataloader(model=engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        dtype=torch.bfloat16)
        batch = next(iter(data_loader))

        # DDP with torch.autocast
        optimizer_ddp.zero_grad()
        with torch.autocast(device_type=get_accelerator().device_name(), dtype=autocast_dtype, enabled=True):
            loss_ddp = model_ddp(batch[0], batch[1])
        loss_ddp.backward()
        grads_ddp = collect_ddp_gradients(model_ddp)

        # DeepSpeed with torch_autocast config
        loss_ds = engine(batch[0], batch[1])
        engine.backward(loss_ds)
        grads_ds = collect_gradients_safe(engine)

        compare_gradients(grads_ddp, grads_ds, step_info=f"precision_mode={precision_mode}, zero_stage={zero_stage}")

        # Verify parameters have correct comm_dtype attribute for autocast
        from deepspeed.runtime.torch_autocast import has_comm_dtype, get_comm_dtype
        for name, param in engine.module.named_parameters():
            if "weight" in name:
                # Linear layer weights should have comm_dtype set
                assert has_comm_dtype(param), f"Parameter {name} should have comm_dtype attribute"
                assert get_comm_dtype(param) == autocast_dtype, \
                    f"Parameter {name} comm_dtype should be {autocast_dtype}, got {get_comm_dtype(param)}"

        optimizer_ddp.step()
        engine.step()

        optimizer_ddp.zero_grad()
        engine.zero_grad()
        engine.destroy()

    @pytest.mark.parametrize("precision_mode,zero_stage", [
        pytest.param("fp16_master_wg", 2, id="z2_fp16_master_wg_autocast"),
        pytest.param("fp16_master_wg", 3, id="z3_fp16_master_wg_autocast"),
    ])
    def test_parameters_match_ddp_after_step(self, precision_mode, zero_stage):
        """Test that parameters match DDP after a training step.
        Note: This test is for FP16 where gradients are scaled and hard to compare.
        """
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        # FP16 mode requires CPU offload
        if precision_mode == "fp16_master_wg" and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        # FP16 mode requires FP16 support
        if precision_mode == "fp16_master_wg" and not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")

        hidden_dim = 6
        lr = 1e-3
        seed = 123

        device, rank, dtype = initialize_distributed()

        # For fp16 mode with autocast, use float32 model parameters
        # For bf16 mode, use bfloat16 model parameters
        model_dtype = torch.float32 if precision_mode == "fp16_master_wg" else dtype

        # Create DDP baseline with torch.autocast
        model_ddp, optimizer_ddp = create_ddp_model(SimpleModel,
                                                    device,
                                                    rank,
                                                    model_dtype,
                                                    seed=seed,
                                                    lr=lr,
                                                    hidden_dim=hidden_dim,
                                                    nlayers=2)

        torch.manual_seed(seed)
        ds_model = SimpleModel(hidden_dim, nlayers=2)

        # Configure based on precision mode
        if precision_mode == "bf16_full":
            autocast_dtype = torch.bfloat16
            precision_config = {
                "bf16": {
                    "enabled": True,
                    "bf16_master_weights_and_grads": True,
                    "bf16_optimizer_states": True
                }
            }
            zero_config = {"stage": zero_stage}
            data_dtype = torch.bfloat16
            use_grad_scaler = False
        else:  # fp16_master_wg
            autocast_dtype = torch.float16
            precision_config = {"fp16": {"enabled": True, "fp16_master_weights_and_grads": True}}
            zero_config = {"stage": zero_stage, "offload_optimizer": {"device": "cpu"}}
            data_dtype = torch.float16
            use_grad_scaler = True

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": lr
                }
            },
            "torch_autocast": {
                "enabled": True,
                "dtype": str(autocast_dtype)
            },
            "zero_optimization": zero_config,
            **precision_config
        }

        engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                               model=ds_model,
                                               model_parameters=ds_model.parameters())

        data_loader = random_dataloader(model=engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        dtype=data_dtype)
        batch = next(iter(data_loader))

        # DDP with torch.autocast and optional GradScaler for fp16
        if use_grad_scaler:
            scaler = torch.amp.GradScaler()

        optimizer_ddp.zero_grad()
        with torch.autocast(device_type=get_accelerator().device_name(), dtype=autocast_dtype, enabled=True):
            loss_ddp = model_ddp(batch[0], batch[1])

        if use_grad_scaler:
            scaler.scale(loss_ddp).backward()
            scaler.step(optimizer_ddp)
            scaler.update()
        else:
            loss_ddp.backward()
            optimizer_ddp.step()

        # DeepSpeed with torch_autocast config
        loss_ds = engine(batch[0], batch[1])
        engine.backward(loss_ds)
        engine.step()

        # Compare parameters after the optimizer step
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        for (name_ddp, param_ddp), (name_ds, param_ds) in zip(model_ddp.named_parameters(),
                                                              engine.module.named_parameters()):
            # Remove 'module.' prefix from both for comparison
            name_ddp_clean = name_ddp.replace('module.', '')
            name_ds_clean = name_ds.replace('module.', '')
            assert name_ddp_clean == name_ds_clean, f"Parameter name mismatch: {name_ddp_clean} vs {name_ds_clean}"

            # Get full parameter for ZeRO stage 3
            if hasattr(param_ds, 'ds_status') and param_ds.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                with deepspeed.zero.GatheredParameters([param_ds], modifier_rank=0):
                    param_ds_full = param_ds.detach().clone().cpu().float()
            else:
                param_ds_full = param_ds.detach().clone().cpu().float()

            param_ddp_full = param_ddp.detach().clone().cpu().float()

            # Use allclose_on_all_ranks for comparison
            allclose_on_all_ranks(
                param_ddp_full,
                param_ds_full,
                rtol=1e-3,
                atol=1e-3,
                assert_message=
                f"Parameters differ for {name_ddp_clean} at precision_mode={precision_mode}, zero_stage={zero_stage}")

        # Verify parameters have correct comm_dtype attribute for autocast
        from deepspeed.runtime.torch_autocast import has_comm_dtype, get_comm_dtype
        for name, param in engine.module.named_parameters():
            if "weight" in name:
                # Linear layer weights should have comm_dtype set
                assert has_comm_dtype(param), f"Parameter {name} should have comm_dtype attribute"
                assert get_comm_dtype(param) == autocast_dtype, \
                    f"Parameter {name} comm_dtype should be {autocast_dtype}, got {get_comm_dtype(param)}"

        engine.destroy()
