# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed
import deepspeed.comm as dist
import torch
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader


def create_model(config_dict):
    hidden_dim = 64
    model = SimpleModel(hidden_dim)
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    return model


def train_shared_loss(num_models, config_dict, dtype):
    hidden_dim = 64

    models = [create_model(config_dict) for _ in range(num_models)]
    data_loader = random_dataloader(model=models[0],
                                    total_samples=4,
                                    hidden_dim=hidden_dim,
                                    device=models[0].device,
                                    dtype=dtype)
    dist.barrier()
    for _, batch in enumerate(data_loader):
        losses = [m.module(batch[0], batch[1]) for m in models]
        loss = sum(l / (i + 1) for i, l in enumerate(losses))
        loss.backward()

        for m in models:
            m._backward_epilogue()

        for m in models:
            m.step()

        for m in models:
            m.optimizer.zero_grad()

    for m in models:
        m.destroy()


def train_independent_loss(num_models, config_dict, dtype):
    hidden_dim = 64

    models = [create_model(config_dict) for _ in range(num_models)]
    data_loader = random_dataloader(model=models[0],
                                    total_samples=4,
                                    hidden_dim=hidden_dim,
                                    device=models[0].device,
                                    dtype=dtype)
    dist.barrier()
    for _, batch in enumerate(data_loader):
        losses = [m.module(batch[0], batch[1]) for m in models]
        for m, loss in zip(models, losses):
            m.backward(loss)
            m.step()

    for m in models:
        m.destroy()


@pytest.mark.parametrize('num_models', [1, 2, 3])
class TestMultipleModels(DistributedTest):
    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize('shared_loss', [False, True])
    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('fp32_grad_accum', [False, True])
    @pytest.mark.parametrize('contiguous_gradients', [False, True])
    @pytest.mark.parametrize('overlap_comm', [False, True])
    def test_zero_optimizer(self, num_models, shared_loss, zero_stage, fp32_grad_accum, contiguous_gradients,
                            overlap_comm):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
                "contiguous_gradients": contiguous_gradients,
                "overlap_comm": overlap_comm,
            },
            "fp16": {
                "initial_scale_power": 8,
                "enabled": True
            },
        }
        if fp32_grad_accum:
            config_dict["data_types"] = {"grad_accum_dtype": "fp32"}

        if shared_loss:
            train_shared_loss(num_models=num_models, config_dict=config_dict, dtype=torch.float16)
        else:
            train_independent_loss(num_models=num_models, config_dict=config_dict, dtype=torch.float16)

    # TODO: Combination of shared_loss==True and bf16.immediate_grad_update==False is currently broken
    @pytest.mark.parametrize('shared_loss', [False, True])
    def test_bf16_optimizer(self, num_models, shared_loss):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": 1,
            },
            "bf16": {
                "enabled": True,
                "immediate_grad_update": True,
            },
            "data_types": {
                "grad_accum_dtype": "fp32"
            }
        }

        if shared_loss:
            train_shared_loss(num_models=num_models, config_dict=config_dict, dtype=torch.bfloat16)
        else:
            train_independent_loss(num_models=num_models, config_dict=config_dict, dtype=torch.bfloat16)
