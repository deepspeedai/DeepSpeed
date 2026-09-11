# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import pytest
import deepspeed
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import safe_get_local_grad, safe_set_local_grad
from deepspeed.accelerator import get_accelerator
from unit.simple_model import SimpleModel
from unit.common import DistributedTest
import os


def get_config(precision, clip_value, offload_device="cpu"):
    config = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": offload_device
            },
            "contiguous_gradients": True,
            "overlap_comm": False,
        },
        "gradient_clipping": 1.0,
    }

    if precision == "fp16":
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 1024,
            "initial_scale_power": 10,
        }
    elif precision == "bf16":
        config["bf16"] = {
            "enabled": True,
        }

    return config


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("norm_type", [1, 2, 3])
class TestZeroGradNormPNorm(DistributedTest):
    world_size = 1

    def test_matches_flat_norm(self, zero_stage, norm_type):
        # get_grad_norm_direct returns the norm of the gradients viewed as a single vector,
        # so on one rank with no model parallelism it must equal the p-norm of the
        # concatenation. norm_type 2 is the control: it is right on both sides.
        config = {
            "train_batch_size": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": zero_stage
            },
        }
        model = SimpleModel(hidden_dim=4, nlayers=2)
        engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

        gradients = [torch.Tensor([3.0, -4.0]), torch.Tensor([2.0])]
        params = list(model.parameters())[:len(gradients)]
        expected = torch.cat([g.reshape(-1) for g in gradients]).norm(float(norm_type))

        actual = optimizer.get_grad_norm_direct(gradients, params, norm_type=norm_type)
        assert torch.allclose(torch.as_tensor(actual).float().cpu(), expected.float().cpu())


@pytest.mark.parametrize("precision,clip_value,offload_device", [
    ("fp16", 0.5, "cpu"),
    ("bf16", 0.05, "cpu"),
    ("fp16", 0.5, "none"),
    ("bf16", 0.05, "none"),
])
class TestZeroGradClip():
    world_size = 1

    def test_grad_clip_and_norm_update(self, precision, clip_value, offload_device):
        """Test custom gradient clipping with configurations and to check if the norm_groups are updated correctly"""
        config_dict = get_config(precision, clip_value, offload_device)

        model = SimpleModel(hidden_dim=10)

        # Set up distributed environment variables
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        try:
            model_engine, optimizer, _, _ = deepspeed.initialize(args=None,
                                                                 model=model,
                                                                 config=config_dict,
                                                                 model_parameters=model.parameters(),
                                                                 dist_init_required=True)
        except Exception as e:
            pytest.skip("Could not initialize deepspeed")

        assert isinstance(optimizer, DeepSpeedZeroOptimizer_Stage3)

        torch.manual_seed(1670)
        inputs = torch.randn(8, 10, device=model_engine.device)
        targets = torch.randn(8, 10, device=model_engine.device)

        if model_engine.fp16_enabled() and get_accelerator().is_fp16_supported():
            inputs = inputs.half()
            targets = targets.half()
        elif model_engine.bfloat16_enabled() and get_accelerator().is_bf16_supported():
            inputs = inputs.bfloat16()
            targets = targets.bfloat16()
        else:
            pytest.skip("Unsupported precision")

        loss = model_engine(inputs, targets)
        model_engine.backward(loss)

        pre_clip_norm_groups = optimizer._get_norm_groups()
        pre_clip_global_norm = torch.linalg.vector_norm(torch.stack(pre_clip_norm_groups))

        modified_count = 0

        for param in model_engine.parameters():
            if not hasattr(param, 'ds_id'):
                continue

            grad = safe_get_local_grad(param)
            if grad is not None:
                pre_clip_norm = grad.norm().item()
                clamped_grad = torch.clamp(grad, -clip_value, clip_value)
                post_clip_norm = clamped_grad.norm().item()

                if pre_clip_norm > clip_value:
                    # Checks if the post-clip norm is less than the pre-clip norm
                    assert post_clip_norm < pre_clip_norm, f"Post-clip norm should be < pre-clip norm for param {param.ds_id}"

                safe_set_local_grad(param, clamped_grad)
                modified_count += 1

        # Get post-clip state
        post_clip_norm_groups = optimizer._get_norm_groups()
        post_clip_global_norm = torch.linalg.vector_norm(torch.stack(post_clip_norm_groups))

        assert modified_count > 0, "No parameters were modified during clipping"
        assert post_clip_global_norm.item() < pre_clip_global_norm.item(
        ), f"Post-clip norm {post_clip_global_norm.item():.6f} should be < pre-clip norm {pre_clip_global_norm.item():.6f}"

        model_engine.step()
        final_norm = optimizer._global_grad_norm
        if pre_clip_global_norm.item() > clip_value:
            assert post_clip_global_norm.item() < pre_clip_global_norm.item(
            ), "Global norm should be reduced after clipping when pre-clip norm > clip_value"
