# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
from unit.common import DistributedTest, preferred_dtype
from unit.simple_model import SimpleModel, random_dataloader


class TestZ3MultipleModelCall(DistributedTest):
    world_size = 1

    def test_z3_multiple_model_call(self):
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": 3
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        hidden_dim, nlayers = 2048, 3
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayers)
        model_engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                     model=model,
                                                     model_parameters=model.parameters())
        data_loader = iter(
            random_dataloader(model=model_engine, total_samples=10, hidden_dim=hidden_dim, device=model_engine.device))

        for n, batch in enumerate(data_loader):
            loss1 = model_engine(batch[0], batch[1])
            with torch.no_grad():
                loss2 = model_engine(batch[0], batch[1])
            loss = loss1 + loss2
            model_engine.backward(loss)
            for name, submodule in model_engine.module.linears._modules.items():
                assert hasattr(submodule, "ds_grads_remaining"), \
                  f"linears.{name} does not have variable ds_grads_remaining"
                assert submodule.ds_grads_remaining == 0, \
                  f"ds_grads_remaining of linears.{name} is not 0 ({submodule.ds_grads_remaining})"
            model_engine.step()
