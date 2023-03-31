# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed.comm as dist
import torch

from unit.common import DistributedTest
from unit.simple_model import random_dataloader
from unit.util import bf16_required_version_check

import deepspeed
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.ops.aio import AsyncIOBuilder


def validate_full_tensors(model):
    for _, lp in model.named_parameters():
        hp = safe_get_full_fp32_param(lp)
        exp_avg = safe_get_full_optimizer_state(lp, 'exp_avg')
        exp_avg_sq = safe_get_full_optimizer_state(lp, 'exp_avg_sq')
        hp_grad = safe_get_full_grad(lp)
        param_list = [hp, hp_grad, exp_avg, exp_avg_sq]
        if lp.requires_grad:
            assert all([p is not None for p in param_list])
        else:
            assert all([p is None for p in param_list])


class MyModel(torch.nn.Module):

    def __init__(self, hidden_dim, frozen_weights):
        super(MyModel, self).__init__()
        self.act = torch.nn.ReLU()
        self.cel = torch.nn.CrossEntropyLoss()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, 1),
             torch.nn.Linear(1, 1),
             torch.nn.Linear(1, hidden_dim)])
        if frozen_weights:
            self.linears[0].weight.requires_grad = False
            self.linears[0].bias.requires_grad = False

    def forward(self, x, y):
        for l in self.linears:
            x = l(x)
            x = self.act(x)
        loss = self.cel(x, y)
        val = (x, loss)
        return val


def run_fragmented_model(model, config_dict, hidden_dim, dtype):
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
    data_loader = random_dataloader(model=model,
                                    total_samples=10,
                                    hidden_dim=hidden_dim,
                                    device=model.device,
                                    dtype=dtype)
    dist.barrier()
    for n, batch in enumerate(data_loader):
        loss = model(batch[0], batch[1])
        loss = loss[1]
        model.backward(loss)
        validate_full_tensors(model)
        model.step()


@pytest.mark.parametrize('frozen_weights', [True, False])
class TestTensorFragment(DistributedTest):
    # Need multiple gpus to test possible hanging
    world_size = 2

    @pytest.mark.parametrize('zero_stage', [1, 2, 3])
    @pytest.mark.parametrize('offload_device', [OffloadDeviceEnum.none, OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme])
    def test_zero_fragments(self, tmpdir, zero_stage, offload_device, frozen_weights):
        if offload_device == OffloadDeviceEnum.nvme:
            if zero_stage != 3:
                pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")
            if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
                pytest.skip('Skip tests since async-io is not compatible')

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 2
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }

        if offload_device == OffloadDeviceEnum.cpu:
            config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
        elif offload_device == OffloadDeviceEnum.nvme:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": offload_device,
                "nvme_path": str(tmpdir)
            }

        hidden_dim = 128
        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=config_dict):
                model = MyModel(hidden_dim, frozen_weights)
        else:
            model = MyModel(hidden_dim, frozen_weights)

        run_fragmented_model(model, config_dict, hidden_dim, torch.float16)

    def test_bf16_fragments(self, frozen_weights):
        if frozen_weights:
            pytest.skip("TODO: Frozen weights not currently supported by BF16 Optimizer")

        if not bf16_required_version_check(accelerator_check=False):
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 0,
            }
        }

        hidden_dim = 128
        model = MyModel(hidden_dim, frozen_weights)
        run_fragmented_model(model, config_dict, hidden_dim, torch.bfloat16)
