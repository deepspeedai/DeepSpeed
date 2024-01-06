# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pytest
import deepspeed.comm as dist
import torch

from unit.common import DistributedTest
from unit.simple_model import random_dataloader, SimpleModel

import deepspeed
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import Init
from deepspeed.ops.aio import AsyncIOBuilder


class TestNVMeCheckpointing(DistributedTest):
    world_size = 1

    @pytest.mark.parametrize('param_offload_device, optim_offload_device',
                             [(OffloadDeviceEnum.cpu, OffloadDeviceEnum.cpu),
                              (OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme),
                              (OffloadDeviceEnum.nvme, OffloadDeviceEnum.nvme)])
    def test_nvme_checkpointing(self, tmpdir, param_offload_device, optim_offload_device):
        zero_dir, ckpt_dir = os.path.join(tmpdir, "zero"), os.path.join(tmpdir, "checkpoint")

        first_stage_steps, second_stage_steps = 2, 2

        if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
            pytest.skip('Skip tests since async-io is not compatible')

        torch.manual_seed(123)

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015,
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": param_offload_device,
                    "nvme_path": str(zero_dir)
                },
                "offload_optimizer": {
                    "device": optim_offload_device,
                    "nvme_path": str(zero_dir)
                },
                "sub_group_size": 100,
                "stage3_max_live_parameters": 100,
                "stage3_param_persistence_threshold": 0,
            },
            "aio": {
                "block_size": 1048576  # Minimum AIO bytes, anything smaller than this will not be offloaded
            }
        }

        hidden_dim, nlayers = 2048, 2
        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim, nlayers=nlayers, empty_grad=False)

        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        model.empty_partition_cache()

        assert first_stage_steps > 0

        data_loader = random_dataloader(model=model,
                                        total_samples=first_stage_steps,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        dist.barrier()
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        dist.barrier()
        model.save_checkpoint(ckpt_dir)

        if second_stage_steps > 0:
            second_stage_batches = list(
                random_dataloader(model=model,
                                  total_samples=second_stage_steps,
                                  hidden_dim=hidden_dim,
                                  device=model.device,
                                  dtype=torch.float16))
            dist.barrier()
            for n, batch in enumerate(second_stage_batches):
                loss = model(batch[0], batch[1])
                model.backward(loss)
                model.step()
            dist.barrier()

        final_batch = next(
            iter(
                random_dataloader(model=model,
                                  total_samples=1,
                                  hidden_dim=hidden_dim,
                                  device=model.device,
                                  dtype=torch.float16)))
        dist.barrier()
        loss_before = float(model(final_batch[0], final_batch[1]))

        # Needed in ZeRO 3. Not doing so can give memory leak
        model.destroy()

        # TODO: This should be on the engine? There needs to be a better way.
        Init.param_id = 0

        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim, nlayers=nlayers, empty_grad=False)

        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        model.load_checkpoint(ckpt_dir)

        if second_stage_steps > 0:
            dist.barrier()
            for n, batch in enumerate(second_stage_batches):
                loss = model(batch[0], batch[1])
                model.backward(loss)
                model.step()
            dist.barrier()

        dist.barrier()
        loss_after = float(model(final_batch[0], final_batch[1]))

        assert loss_before == loss_after
