# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest

from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataloader
from mup.shape import set_base_shapes
from deepspeed.accelerator import get_accelerator


@pytest.mark.parametrize("optimizer, expected_opt_class", [("MuAdam", torch.optim.Adam),
                                                           ("MuAdamW", torch.optim.AdamW), ("MuSGD", torch.optim.SGD)]) # yapf: disable
@pytest.mark.parametrize("zero_offload", [True, False]) # yapf: disable
class TestMuPOptimizers(DistributedTest):
    world_size = 1
    reuse_dist_env = True

    def test(self, optimizer, expected_opt_class, zero_offload):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "zero_allow_untested_optimizer": True,
            "optimizer": {
                "type": optimizer,
                "params": {
                    "lr": 0.00015,
                }
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "cpu_offload": zero_offload
            }
        }
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True}
        elif get_accelerator().is_bf16_supported():
            config_dict["bf16"] = {"enabled": True}
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        set_base_shapes(model, None)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model, total_samples=50, hidden_dim=hidden_dim, device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        ds_optimizer = model.optimizer.optimizer
        assert isinstance(ds_optimizer, expected_opt_class)
