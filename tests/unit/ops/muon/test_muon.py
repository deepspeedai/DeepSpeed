# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest

from muon import MuonWithAuxAdam
from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.accelerator import get_accelerator

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)

# 'optimizer_type, zero_offload, zero_stage, lr, extra_params'
muon_configs = [
    ["muon", False, 1, 0.00015, {}],
    ["muon", False, 2, 0.00015, {}],
    ["muon", False, 3, 0.00015, {}],
    ["muon", True, 2, 0.00015, {}],
    ["muon", True, 3, 0.00015, {}],
    ["muon", False, 2, 0.001, {}],
    ["muon", True, 2, 0.001, {}],
    ["muon", False, 2, 0.00015, {
        "momentum": 0.95
    }],
    ["muon", False, 2, 0.00015, {
        "backend_steps": 5
    }],
    ["muon", True, 2, 0.00015, {
        "weight_decay": 0.01
    }],
]


def _mark_use_muon(params):
    """Flag parameters that should be handled by Muon (ndim >= 2)."""
    for p in params:
        if p.ndim >= 2:
            setattr(p, "_use_muon", True)


@pytest.mark.parametrize('optimizer_type, zero_offload, zero_stage, lr, extra_params', muon_configs)
class TestMuonConfigs(DistributedTest):
    world_size = 1
    reuse_dist_env = True

    def test(self, optimizer_type, zero_offload, zero_stage, lr, extra_params, tmp_path):
        model = SimpleModel(10)
        _mark_use_muon(model.parameters())

        optimizer_params = {"lr": lr}

        optimizer_params.update(extra_params)

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
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
                "cpu_offload": zero_offload
            }
        }

        engine, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )

        ds_optimizer = getattr(engine.optimizer, "optimizer", engine.optimizer)
        assert isinstance(ds_optimizer, MuonWithAuxAdam)

        # Verify that param groups carry the use_muon flag correctly
        for group in ds_optimizer.param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                assert all(getattr(p, "_use_muon", False) for p in group["params"])
            else:
                assert all(not getattr(p, "_use_muon", False) for p in group["params"])

        # Perform a few training steps to ensure the optimizer works correctly
        initial_params = [p.clone() for p in model.parameters()]

        for step in range(3):
            # Create some dummy data for training
            input_data = torch.randn(2, 10, device=engine.device, dtype=torch.half)
            target = torch.randint(0, 2, (2, ), device=engine.device)

            # Forward pass
            output = engine(input_data)
            loss = torch.nn.functional.cross_entropy(output, target)

            # Backward pass and optimization step
            engine.backward(loss)
            engine.step()

        # Verify that parameters have been updated
        final_params = [p.clone() for p in model.parameters()]
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final), "Parameters should have been updated during training"
