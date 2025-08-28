# Copyright (c) 2025 Peng Du and Zhipeng Wang
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest

from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.accelerator import get_accelerator
if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)

# 'optimizer_type, zero_stage, lr, hidden_dim, nlayer'

muon_configs = []
for optimizer_name in ['muon', 'adam']:
    for stage in [1, 2]:
        for lr in [0.01, 0.05]:
            for model_dim in [32, 128]:
                for nlayer in [5, 10]:
                    muon_configs.append([optimizer_name, stage, lr, model_dim, nlayer])


def set_muon_flag(params):
    for p in params:
        if p.ndim >= 2:
            setattr(p, "use_muon", True)
        else:
            setattr(p, "use_muon", False)


@pytest.mark.parametrize('optimizer_type, zero_stage, lr, hidden_dim, nlayer', muon_configs)
class TestMuonConfigs(DistributedTest):

    def test(self, optimizer_type, zero_stage, lr, hidden_dim, nlayer):
        optimizer_params = {"lr": lr}
        batch_size = 8
        config_dict = {
            "train_batch_size": batch_size,
            "optimizer": {
                "type": optimizer_type,
                "params": optimizer_params
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
            }
        }
        # Perform a few training steps to ensure the optimizer works correctly

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayer)
        if 'muon' in optimizer_type:
            set_muon_flag(model.parameters())
        initial_params = [p.clone().cpu() for p in model.parameters()]
        engine, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )
        assert optimizer_type in optimizer.optimizer.__class__.__name__.lower(
        ), f"Expected optimizer type {optimizer_type}, got {optimizer.optimizer.__class__.__name__}"
        steps = 5
        for _ in range(steps):
            # Random inputs: (batch_size, hidden_dim)
            x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
            # Random class labels: (batch_size,)
            y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
            # Forward + loss
            loss = engine(x, y)
            # Backward
            engine.backward(loss)
            engine.step()

        # Verify that parameters have been updated
        after_training = [p.clone().cpu() for p in model.parameters()]
        for initial, final in zip(initial_params, after_training):
            assert not torch.equal(initial.cpu(), final.cpu()), "Parameters should have been updated during training"
