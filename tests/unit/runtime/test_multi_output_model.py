# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
from pytest import approx
from unit.common import DistributedTest
from unit.multi_output_model import MultiOutputModel, multi_output_dataloader


class TestTwoOutputModel(DistributedTest):
    world_size = 1

    def test(self, tmpdir):
        grad_accumulation_steps = 2
        micro_batch_size = 1
        world_size = self.world_size
        config_dict = {
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": grad_accumulation_steps,
            "train_batch_size": micro_batch_size * grad_accumulation_steps * world_size,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True
            }
        }

        hidden_dim = 10
        weight_value = 0.1

        model = MultiOutputModel(hidden_dim, weight_value)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        total_samples = 4
        data_loader = multi_output_dataloader(model=model,
                                              total_samples=total_samples,
                                              hidden_dim=hidden_dim,
                                              device=model.device,
                                              inputs=[1.0, 2.0],
                                              targets=[1, 2])
        for n, batch in enumerate(data_loader):
            assert len(batch) % 2 == 0, \
                 f"multi_output_dataloader failed to return even number of data samples (input+target)"

            midpoint = len(batch) // 2
            inputs, targets = batch[:midpoint], batch[midpoint:]
            loss_tuple = model(inputs, targets)

            expected_loss = torch.tensor(2.302734375, dtype=torch.half, device=model.device)
            for loss in loss_tuple:
                assert loss.shape == torch.Size([])
                assert loss.item() == approx(expected_loss.item())

            summed_loss = sum(loss_tuple)
            scaled_loss = model.backward(summed_loss)
            expected_scaled_loss = summed_loss.float() / grad_accumulation_steps
            assert scaled_loss.item() == approx(expected_scaled_loss.item())

            model.step()


class TestThreeOutputModel(DistributedTest):
    world_size = 1

    def test(self, tmpdir):
        grad_accumulation_steps = 3
        micro_batch_size = 1
        world_size = 1
        config_dict = {
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": grad_accumulation_steps,
            "train_batch_size": micro_batch_size * grad_accumulation_steps * world_size,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True
            }
        }

        hidden_dim = 10
        weight_value = 0.1

        model = MultiOutputModel(hidden_dim, weight_value)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        total_samples = grad_accumulation_steps * micro_batch_size * 2
        data_loader = multi_output_dataloader(model=model,
                                              total_samples=total_samples,
                                              hidden_dim=hidden_dim,
                                              device=model.device,
                                              inputs=[1.0, 2.0, 3.0],
                                              targets=[1, 2, 3])
        for n, batch in enumerate(data_loader):
            assert len(batch) % 2 == 0, \
                 f"multi_output_dataloader failed to return even number of data samples (input+target)"

            midpoint = len(batch) // 2
            inputs, targets = batch[:midpoint], batch[midpoint:]
            loss_tuple = model(inputs, targets)
            assert len(loss_tuple) == 3

            expected_loss = torch.tensor(2.302734375, dtype=torch.half, device=model.device)

            for loss in loss_tuple:
                assert loss.shape == torch.Size([])
                assert loss.item() == approx(expected_loss.item())

            summed_loss = sum(loss_tuple)
            scaled_loss = model.backward(summed_loss)
            expected_scaled_loss = summed_loss.float() / grad_accumulation_steps
            assert scaled_loss.item() == approx(expected_scaled_loss.item())

            model.step()
