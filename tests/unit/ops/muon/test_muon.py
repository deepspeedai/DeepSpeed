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

# 'optimizer_type, zero_stage, lr, hidden_dim, nlayer, offload_optimizer, save_muon_momentum_buffer_in_memory'

muon_configs = []
for optimizer_name in ['muon', 'adam']:
    for stage in [1, 2, 3]:
        for lr in [0.01, 0.05]:
            for model_dim in [32, 128]:
                for nlayer in [5, 10]:
                    for offload_optimizer in [True, False]:
                        for save_in_mem in ([True, False] if stage == 3 else [False]):
                            muon_configs.append(
                                [optimizer_name, stage, lr, model_dim, nlayer, offload_optimizer, save_in_mem])


@pytest.mark.parametrize(
    'optimizer_type, zero_stage, lr, hidden_dim, nlayer, offload_optimizer, save_muon_momentum_buffer_in_memory',
    muon_configs)
class TestMuonConfigs(DistributedTest):

    def test(self, optimizer_type, zero_stage, lr, hidden_dim, nlayer, offload_optimizer,
             save_muon_momentum_buffer_in_memory):
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
                "reduce_scatter": False,
                "save_muon_momentum_buffer_in_memory": save_muon_momentum_buffer_in_memory,
            },
        }
        if offload_optimizer:
            config_dict["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        # Perform a few training steps to ensure the optimizer works correctly

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayer)
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


class TestGramNewtonSchulz(DistributedTest):
    """Test Gram Newton-Schulz integration with Muon optimizer."""

    world_size = 2
    reuse_dist_env = True

    @pytest.mark.parametrize('ns_method', ['gram', 'standard'])
    @pytest.mark.parametrize('zero_stage', [1, 2])
    def test_ns_method_training(self, ns_method, zero_stage):
        """Verify both ns_method values work end-to-end with DeepSpeed."""
        hidden_dim = 64
        batch_size = 8
        config_dict = {
            "train_batch_size": batch_size,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": 0.01,
                    "ns_method": ns_method,
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_scatter": False,
            },
        }

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=3)
        initial_params = [p.clone().cpu() for p in model.parameters()]
        engine, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )

        for _ in range(3):
            x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
            y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()

        after_training = [p.clone().cpu() for p in model.parameters()]
        for initial, final in zip(initial_params, after_training):
            assert not torch.equal(initial, final), "Parameters should have been updated"

    @pytest.mark.parametrize('ns_method', ['gram', 'standard'])
    def test_ns_method_stage3(self, ns_method):
        """Verify ns_method works with ZeRO Stage 3."""
        hidden_dim = 64
        batch_size = 8
        config_dict = {
            "train_batch_size": batch_size,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": 0.01,
                    "ns_method": ns_method,
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 3,
                "reduce_scatter": False,
            },
        }

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=3)
        engine, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )

        for _ in range(3):
            x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
            y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
            loss = engine(x, y)
            engine.backward(loss)
            engine.step()
