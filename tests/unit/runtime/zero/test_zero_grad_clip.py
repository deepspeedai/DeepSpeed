import torch
import pytest
import deepspeed
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import safe_get_local_grad, safe_set_local_grad
from unit.common import DistributedTest
from unit.simple_model import SimpleModel

def get_config(precision, clip_value, offload_device="cpu"):
    """Helper function to create DeepSpeed config."""
    config = {
        "train_batch_size": 8,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-4}
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": offload_device},
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

@pytest.mark.parametrize("precision,clip_value,offload_device", [
    ("fp16", 0.5, "cpu"),
    ("bf16", 0.05, "cpu"),
    ("fp16", 0.5, "none"),
    ("bf16", 0.05, "none"),
])
class TestZeroGradClip():
    world_size = 1 

    def test_grad_clip_and_norm_update(self, precision, clip_value, offload_device):
        """Test custom gradient clipping with different precisions and configurations."""
        print(f"\nDEBUG: Starting test with {precision}, clip_value={clip_value}, offload={offload_device}")
        
        config_dict = get_config(precision, clip_value, offload_device)
        print(f"DEBUG: Using config: {config_dict}")

        model = SimpleModel(hidden_dim=10)
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config_params=config_dict
        )

        assert isinstance(optimizer, DeepSpeedZeroOptimizer_Stage3)

        torch.manual_seed(1670)
        inputs = torch.randn(8, 10, device=model_engine.device)
        targets = torch.randn(8, 10, device=model_engine.device)

        if model_engine.fp16_enabled():
            inputs = inputs.half()
            targets = targets.half()
        elif model_engine.bfloat16_enabled():
            inputs = inputs.bfloat16()
            targets = targets.bfloat16()

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
                # Apply clipping
                clamped_grad = torch.clamp(grad, -clip_value, clip_value)
                post_clip_norm = clamped_grad.norm().item()
                
                if pre_clip_norm > clip_value:
                    print(f"DEBUG: Param {param.ds_id} - Pre-clip norm: {pre_clip_norm:.6f}, Post-clip norm: {post_clip_norm:.6f}")
                    assert post_clip_norm <= pre_clip_norm, f"Post-clip norm should be <= pre-clip norm for param {param.ds_id}"
                
                safe_set_local_grad(param, clamped_grad)
                modified_count += 1

        # Get post-clip state
        post_clip_norm_groups = optimizer._get_norm_groups()
        post_clip_global_norm = torch.linalg.vector_norm(torch.stack(post_clip_norm_groups))

        print(f"DEBUG: Pre-clip global norm: {pre_clip_global_norm.item():.6f}")
        print(f"DEBUG: Post-clip global norm: {post_clip_global_norm.item():.6f}")
        assert modified_count > 0, "No parameters were modified during clipping"
        assert post_clip_global_norm.item() < pre_clip_global_norm.item(), \
            f"Post-clip norm {post_clip_global_norm.item():.6f} should be < pre-clip norm {pre_clip_global_norm.item():.6f}"

        model_engine.step()
        
        final_norm = optimizer._global_grad_norm
        if pre_clip_global_norm.item() > clip_value:
            assert post_clip_global_norm.item() < pre_clip_global_norm.item(), \
                "Global norm should be reduced after clipping when pre-clip norm > clip_value"