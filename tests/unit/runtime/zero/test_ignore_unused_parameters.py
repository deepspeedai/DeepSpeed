# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from unit.common import DistributedTest
from unit.simple_model import UnusedParametersModel, random_dataloader
from deepspeed.ops.op_builder import CPUAdamBuilder

import deepspeed
from deepspeed.accelerator import get_accelerator


@pytest.mark.parametrize('ignore_unused_parameters', [False, True])
class TestStage2IgnoreUnusedParameters(DistributedTest):
    world_size = 1

    def test(self, ignore_unused_parameters):
        use_cpu_offload = True

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": 2,
                "cpu_offload": use_cpu_offload,
                "ignore_unused_parameters": ignore_unused_parameters
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
        }
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        else:
            config_dict["bf16"] = {"enabled": True}
        hidden_dim = 4

        model = UnusedParametersModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        data_loader = random_dataloader(model=model, total_samples=10, hidden_dim=hidden_dim, device=model.device)

        def _loop():
            for n, batch in enumerate(data_loader):
                loss = model(batch[0], batch[1])
                model.backward(loss)
                model.step()

        if ignore_unused_parameters:
            _loop()
        else:
            with pytest.raises(AssertionError) as e:
                _loop()
            assert e.value.args and 'ignore_unused_parameters' in e.value.args[0]
