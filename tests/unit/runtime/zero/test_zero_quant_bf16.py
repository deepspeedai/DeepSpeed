# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader


class TestZeroQuantBF16(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("zero_quantized_weights", [True])
    def test_bf16_quantized_weights(self, zero_quantized_weights):
        if not deepspeed.get_accelerator().is_bf16_supported():
            pytest.skip("bf16 is not supported by this accelerator")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "zero_quantized_weights": zero_quantized_weights,
            },
            "bf16": {
                "enabled": True
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            }
        }

        hidden_dim = 128
        model = SimpleModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(model=model, config=config_dict)

        # Ensure model is in bf16
        for param in model.parameters():
            assert param.dtype == torch.bfloat16

        data_loader = random_dataloader(model=model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)

        for n, batch in enumerate(data_loader):
            # This triggers all_gather and dequantization
            loss = model(batch[0], batch[1])

            # Verify that param.data is indeed bfloat16 after all_gather
            for name, param in model.named_parameters():
                assert param.data.dtype == torch.bfloat16, f"Parameter {name} data dtype is {param.data.dtype}, expected torch.bfloat16"

            model.backward(loss)
            model.step()
            break
