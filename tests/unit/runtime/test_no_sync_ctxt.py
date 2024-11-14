# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from contextlib import nullcontext
import torch

from unit.simple_model import SimpleModel, random_dataloader
from unit.common import DistributedTest

import deepspeed
import deepspeed.comm as dist
from deepspeed.utils import safe_get_full_grad


class TestNoSyncCtxt(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
    def test_zero_stage(self, zero_stage, dtype):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
        }

        invalid_cfg = zero_stage > 1
        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}

        hidden_dim = 64
        total_samples = 32
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model,
                                        total_samples=total_samples,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)
        dist.barrier()

        with pytest.raises(AssertionError) if invalid_cfg else nullcontext() as assertinfo:
            with model.no_sync():
                for _, batch in enumerate(data_loader):
                    loss = model(batch[0], batch[1])
                    model.backward(loss)
        if invalid_cfg:
            assert ("no_sync context manager is incompatible" in str(assertinfo))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("zero_stage", [0, 1])
    def test_engine_step(self, zero_stage, dtype):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
        }

        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}

        hidden_dim = 64
        total_samples = 32
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model,
                                        total_samples=total_samples,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)
        dist.barrier()

        with model.no_sync():
            for _, batch in enumerate(data_loader):
                loss = model(batch[0], batch[1])
                model.backward(loss)
                with pytest.raises(AssertionError) as assertinfo:
                    model.step()
                assert ("It is illegal to call Engine.step() inside no_sync context manager" in str(assertinfo))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("zero_stage", [0, 1])
    def test_multiple_ctxts(self, zero_stage, dtype):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
        }

        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}

        hidden_dim = 64
        total_samples = 32
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model,
                                        total_samples=total_samples,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)
        dist.barrier()

        param_list = list(model.parameters())
        first_losses = []
        first_grad_norms = []
        with model.no_sync():
            for _, batch in enumerate(data_loader):
                loss = model(batch[0], batch[1])
                first_losses.append(loss.item())
                model.backward(loss)
                grad_norm = sum([safe_get_full_grad(p).norm() for p in param_list])
                first_grad_norms.append(grad_norm.item())

        second_losses = []
        second_grad_norms = []

        model.zero_grad()
        with model.no_sync():
            for _, batch in enumerate(data_loader):
                loss = model(batch[0], batch[1])
                second_losses.append(loss.item())
                model.backward(loss)
                grad_norm = sum([safe_get_full_grad(p).norm() for p in param_list])
                second_grad_norms.append(grad_norm.item())

        assert len(first_losses) == len(second_losses)
        for x, y in zip(first_losses, second_losses):
            assert x == y

        assert len(first_grad_norms) == len(second_grad_norms)
        for x, y in zip(first_grad_norms, second_grad_norms):
            assert x == y

    def test_reentry(self):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "zero_optimization": {
                "stage": 1,
            },
        }

        hidden_dim = 64
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        dist.barrier()

        with model.no_sync():
            with pytest.raises(AssertionError) as assertinfo:
                with model.no_sync():
                    pass
            assert ("no_sync context manager reentry is unsupported" in str(assertinfo))
