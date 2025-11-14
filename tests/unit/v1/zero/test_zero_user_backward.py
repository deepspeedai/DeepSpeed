# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from torch.nn.parallel import DistributedDataParallel as DDP

from unit.common import DistributedTest, preferred_dtype, allclose_on_all_ranks
from unit.simple_model import SimpleModel, random_dataloader
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import safe_get_full_grad


class SimpleNonScalarModel(torch.nn.Module):
    """Model that returns non-scalar output for testing tensor.backward(grad)"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Returns non-scalar output
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def get_config_dict(zero_stage, gradient_accumulation_steps=1):
    """Helper to create config dict with common settings"""
    config_dict = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 1,
        "zero_optimization": {
            "stage": zero_stage,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
    }

    if zero_stage == 3:
        # For ZeRO-3, force partitioning of all parameters
        config_dict["zero_optimization"]["stage3_param_persistence_threshold"] = 0

    if get_accelerator().is_bf16_supported():
        config_dict["bf16"] = {"enabled": True}
    elif get_accelerator().is_fp16_supported():
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}

    return config_dict


def collect_gradients_safe(model):
    """Collect gradients from model parameters using safe_get_full_grad API"""
    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad = safe_get_full_grad(param)
            if grad is not None:
                # Remove 'module.' prefix if present (DeepSpeed wraps the model)
                clean_name = name.replace('module.', '')
                grads[clean_name] = grad.detach().clone().cpu()
    return grads


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardBasic(DistributedTest):
    """Test basic functionality of user backward (loss.backward()) by comparing with PyTorch DDP"""
    world_size = 2

    def test_loss_backward_matches_ddp(self, zero_stage):
        """Test that DeepSpeed loss.backward() produces same gradients as PyTorch DDP"""
        hidden_dim = 4
        lr = 1e-3

        # Create two identical models
        torch.manual_seed(42)
        model_ddp = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        torch.manual_seed(42)
        model_deepspeed = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        # Initialize DDP baseline
        deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
        device = get_accelerator().current_device_name()
        rank = get_accelerator().current_device()
        dtype = preferred_dtype()

        model_ddp = model_ddp.to(device=device, dtype=dtype)
        model_ddp = DDP(model_ddp, device_ids=[rank], output_device=rank)
        optimizer_ddp = torch.optim.Adam(model_ddp.parameters(), lr=lr)

        # Initialize DeepSpeed
        config = get_config_dict(zero_stage)
        model_engine, _, _, _ = deepspeed.initialize(config=config,
                                                     model=model_deepspeed,
                                                     model_parameters=model_deepspeed.parameters())

        data_loader = random_dataloader(model=model_engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=model_engine.device)

        # Run one training step with DDP
        batch = next(iter(data_loader))
        optimizer_ddp.zero_grad()
        loss_ddp = model_ddp(batch[0], batch[1])
        loss_ddp.backward()

        # Collect DDP gradients
        grads_ddp = {}
        for name, param in model_ddp.named_parameters():
            if param.grad is not None:
                # Remove 'module.' prefix from DDP
                clean_name = name.replace('module.', '')
                grads_ddp[clean_name] = param.grad.detach().clone().cpu()

        # Run one training step with DeepSpeed using loss.backward()
        loss_ds = model_engine(batch[0], batch[1])
        loss_ds.backward()
        grads_ds = collect_gradients_safe(model_engine)

        # Compare gradients across all ranks
        assert len(grads_ddp) == len(grads_ds), \
            f"Different number of parameters with gradients: DDP={len(grads_ddp)}, DeepSpeed={len(grads_ds)}"
        for name in grads_ddp.keys():
            assert name in grads_ds, f"Parameter {name} missing in DeepSpeed gradients"
            # Convert both to fp32 for comparison in case of dtype mismatch
            grads_ddp_fp32 = grads_ddp[name].float()
            grads_ds_fp32 = grads_ds[name].float()
            allclose_on_all_ranks(grads_ddp_fp32,
                                  grads_ds_fp32,
                                  assert_message=f"Gradients differ for parameter {name} between DDP and DeepSpeed")

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardNonScalar(DistributedTest):
    """Test non-scalar backward support"""
    world_size = 2

    def test_non_scalar_backward(self, zero_stage):
        """Test that tensor.backward(grad) works correctly by comparing with PyTorch DDP"""
        hidden_dim = 4
        batch_size = 2
        lr = 1e-3

        # Create two identical models - one for PyTorch DDP, one for DeepSpeed
        torch.manual_seed(42)
        model_ddp = SimpleNonScalarModel(hidden_dim=hidden_dim)

        torch.manual_seed(42)
        model_deepspeed = SimpleNonScalarModel(hidden_dim=hidden_dim)

        # Initialize DDP baseline
        deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
        device = get_accelerator().current_device_name()
        rank = get_accelerator().current_device()
        dtype = preferred_dtype()

        model_ddp = model_ddp.to(device=device, dtype=dtype)
        model_ddp = DDP(model_ddp, device_ids=[rank], output_device=rank)
        optimizer_ddp = torch.optim.Adam(model_ddp.parameters(), lr=lr)

        # Initialize DeepSpeed
        config = get_config_dict(zero_stage)
        model_engine, _, _, _ = deepspeed.initialize(config=config,
                                                     model=model_deepspeed,
                                                     model_parameters=model_deepspeed.parameters())

        # Create same input for both models
        torch.manual_seed(123)
        x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)

        # DDP baseline: forward, backward with gradient
        optimizer_ddp.zero_grad()
        output_ddp = model_ddp(x)
        grad_output = torch.ones_like(output_ddp)
        output_ddp.backward(grad_output)

        # Collect DDP gradients after backward, before step
        ddp_grads = {}
        for name, param in model_ddp.named_parameters():
            if param.grad is not None:
                # Remove 'module.' prefix from DDP
                clean_name = name.replace('module.', '')
                ddp_grads[clean_name] = param.grad.detach().clone().cpu()

        # DeepSpeed: forward, backward with gradient
        output_deepspeed = model_engine(x)
        grad_output_ds = torch.ones_like(output_deepspeed)
        output_deepspeed.backward(grad_output_ds)

        # Collect DeepSpeed gradients after backward, before step using safe API
        deepspeed_grads = {}
        for name, param in model_engine.named_parameters():
            # Remove 'module.' prefix if present (DeepSpeed wraps the model)
            clean_name = name.replace('module.', '')
            grad = safe_get_full_grad(param)
            if grad is not None:
                deepspeed_grads[clean_name] = grad.detach().clone().cpu()

        # Compare gradients across all ranks
        assert len(ddp_grads) == len(deepspeed_grads), \
            f"Gradient count mismatch: DDP has {len(ddp_grads)}, DeepSpeed has {len(deepspeed_grads)}"

        for name in ddp_grads.keys():
            assert name in deepspeed_grads, f"Gradient for parameter {name} missing in DeepSpeed model"

            # Gradients should match between DDP and DeepSpeed
            # Note: DeepSpeed may accumulate gradients in fp32 even when model is bf16,
            # so we convert both to fp32 for comparison
            ddp_grad_fp32 = ddp_grads[name].float()
            deepspeed_grad_fp32 = deepspeed_grads[name].float()
            allclose_on_all_ranks(
                ddp_grad_fp32,
                deepspeed_grad_fp32,
                assert_message=
                f"Gradient for parameter {name} mismatch between DDP and DeepSpeed after non-scalar backward")

        # Now run optimizer step
        optimizer_ddp.step()
        model_engine.step()

        # Collect DDP parameters after step
        ddp_params = {}
        for name, param in model_ddp.named_parameters():
            # Remove 'module.' prefix from DDP
            clean_name = name.replace('module.', '')
            ddp_params[clean_name] = param.detach().clone().cpu()

        # Collect DeepSpeed parameters after step
        deepspeed_params = {}
        for name, param in model_engine.named_parameters():
            # Remove 'module.' prefix if present (DeepSpeed wraps the model)
            clean_name = name.replace('module.', '')
            if zero_stage == 3:
                with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                    deepspeed_params[clean_name] = param.detach().clone().cpu()
            else:
                deepspeed_params[clean_name] = param.detach().clone().cpu()

        # Compare parameters across all ranks
        assert len(ddp_params) == len(deepspeed_params), \
            f"Parameter count mismatch: DDP has {len(ddp_params)}, DeepSpeed has {len(deepspeed_params)}"

        for name in ddp_params.keys():
            assert name in deepspeed_params, f"Parameter {name} missing in DeepSpeed model"

            # Parameters should match between DDP and DeepSpeed
            # Convert to fp32 for comparison in case of dtype mismatch
            ddp_param_fp32 = ddp_params[name].float()
            deepspeed_param_fp32 = deepspeed_params[name].float()
            allclose_on_all_ranks(
                ddp_param_fp32,
                deepspeed_param_fp32,
                assert_message=f"Parameter {name} mismatch between DDP and DeepSpeed after non-scalar backward")

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardGradAccumulation(DistributedTest):
    """Test gradient accumulation with user backward"""
    world_size = 2

    def test_grad_accumulation(self, zero_stage):
        """Test that gradient accumulation works correctly with loss.backward() by comparing with DDP"""
        hidden_dim = 4
        gradient_accumulation_steps = 4
        lr = 1e-3

        # Create two identical models
        torch.manual_seed(42)
        model_ddp = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        torch.manual_seed(42)
        model_deepspeed = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        # Initialize DDP baseline
        deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
        device = get_accelerator().current_device_name()
        rank = get_accelerator().current_device()
        dtype = preferred_dtype()

        model_ddp = model_ddp.to(device=device, dtype=dtype)
        model_ddp = DDP(model_ddp, device_ids=[rank], output_device=rank)
        optimizer_ddp = torch.optim.Adam(model_ddp.parameters(), lr=lr)

        # Initialize DeepSpeed with gradient accumulation
        config = get_config_dict(zero_stage, gradient_accumulation_steps=gradient_accumulation_steps)
        model_engine, _, _, _ = deepspeed.initialize(config=config,
                                                     model=model_deepspeed,
                                                     model_parameters=model_deepspeed.parameters())

        data_loader = random_dataloader(model=model_engine,
                                        total_samples=16,
                                        hidden_dim=hidden_dim,
                                        device=model_engine.device)

        # Run training with gradient accumulation
        for i, batch in enumerate(data_loader):
            # DDP: Manual gradient accumulation
            loss_ddp = model_ddp(batch[0], batch[1])
            # Scale loss for gradient accumulation
            (loss_ddp / gradient_accumulation_steps).backward()

            # DeepSpeed: Built-in gradient accumulation
            loss_ds = model_engine(batch[0], batch[1])
            loss_ds.backward()

            # Compare gradients at accumulation boundary
            if model_engine.is_gradient_accumulation_boundary():
                # Collect DDP gradients
                grads_ddp = {}
                for name, param in model_ddp.named_parameters():
                    if param.grad is not None:
                        clean_name = name.replace('module.', '')
                        grads_ddp[clean_name] = param.grad.detach().clone().cpu()

                # Collect DeepSpeed gradients
                grads_ds = collect_gradients_safe(model_engine)

                # Compare gradients
                assert len(grads_ddp) == len(grads_ds), \
                    f"Different number of parameters with gradients at step {i}: DDP={len(grads_ddp)}, DS={len(grads_ds)}"
                for name in grads_ddp.keys():
                    assert name in grads_ds, f"Parameter {name} missing in DeepSpeed gradients at step {i}"
                    # Convert both to fp32 for comparison in case of dtype mismatch
                    grads_ddp_fp32 = grads_ddp[name].float()
                    grads_ds_fp32 = grads_ds[name].float()
                    allclose_on_all_ranks(grads_ddp_fp32,
                                          grads_ds_fp32,
                                          assert_message=f"Gradients differ for {name} at step {i}")

                # Step both optimizers
                optimizer_ddp.step()
                optimizer_ddp.zero_grad()

            # Step DeepSpeed (handles gradient accumulation internally)
            model_engine.step()

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardMultipleEngines(DistributedTest):
    """Test multiple DeepSpeed engines with combined loss without manual _backward_epilogue()"""
    world_size = 2

    def test_multiple_engines_combined_loss(self, zero_stage):
        """Test that multiple engines work with combined loss.backward() without manual _backward_epilogue()

        This test compares the behavior with PyTorch DDP baseline to ensure correctness.
        """
        hidden_dim = 4
        batch_size = 2
        num_models = 3
        lr = 1e-3

        # Initialize distributed
        deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
        device = get_accelerator().current_device_name()
        rank = get_accelerator().current_device()
        dtype = preferred_dtype()

        # Create DDP baseline models
        ddp_models = []
        ddp_optimizers = []
        for i in range(num_models):
            torch.manual_seed(42 + i)
            model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
            model = model.to(device=device, dtype=dtype)
            model = DDP(model, device_ids=[rank], output_device=rank)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            ddp_models.append(model)
            ddp_optimizers.append(optimizer)

        # Create multiple DeepSpeed engines with identical initialization
        model_engines = []
        for i in range(num_models):
            torch.manual_seed(42 + i)  # Same seed as DDP models
            model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)
            config = get_config_dict(zero_stage)
            engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
            model_engines.append(engine)

        # Create same input for all models
        torch.manual_seed(123)
        x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
        y = torch.randint(0, hidden_dim, (batch_size, ), device=device)

        # DDP baseline: compute losses and combined backward
        for optimizer in ddp_optimizers:
            optimizer.zero_grad()

        ddp_losses = []
        for model in ddp_models:
            loss = model(x, y)
            ddp_losses.append(loss)

        ddp_combined_loss = sum(l / (i + 1) for i, l in enumerate(ddp_losses))
        ddp_combined_loss.backward()

        # Collect DDP gradients for each model
        ddp_grads_per_model = []
        for model in ddp_models:
            grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    clean_name = name.replace('module.', '')
                    grads[clean_name] = param.grad.detach().clone().cpu()
            ddp_grads_per_model.append(grads)

        # DeepSpeed: compute losses and combined backward WITHOUT manual _backward_epilogue()
        ds_losses = []
        for engine in model_engines:
            loss = engine(x, y)
            ds_losses.append(loss)

        # Combine losses with weighted sum (as in the documentation example)
        ds_combined_loss = sum(l / (i + 1) for i, l in enumerate(ds_losses))

        # Call backward on combined loss WITHOUT manual _backward_epilogue()
        # This should now work automatically
        ds_combined_loss.backward()

        # Collect DeepSpeed gradients for each engine and compare with DDP
        for engine_idx, engine in enumerate(model_engines):
            ds_grads = collect_gradients_safe(engine)
            ddp_grads = ddp_grads_per_model[engine_idx]

            assert len(ds_grads) > 0, f"Engine {engine_idx} has no gradients after combined_loss.backward()"
            assert len(ddp_grads) == len(ds_grads), \
                f"Engine {engine_idx}: DDP has {len(ddp_grads)} gradients, DeepSpeed has {len(ds_grads)}"

            # Compare gradients between DDP and DeepSpeed for this model
            for name in ddp_grads.keys():
                assert name in ds_grads, f"Engine {engine_idx}: Parameter {name} missing in DeepSpeed gradients"

                # Convert both to fp32 for comparison in case of dtype mismatch
                ddp_grad_fp32 = ddp_grads[name].float()
                ds_grad_fp32 = ds_grads[name].float()
                allclose_on_all_ranks(
                    ddp_grad_fp32,
                    ds_grad_fp32,
                    assert_message=
                    f"Engine {engine_idx}: Gradients differ for parameter {name} between DDP and DeepSpeed")

        # Step all DDP models
        for optimizer in ddp_optimizers:
            optimizer.step()
            optimizer.zero_grad()

        # Step all DeepSpeed engines
        for engine in model_engines:
            engine.step()
            engine.optimizer.zero_grad()

        # Run another iteration to ensure everything still works
        torch.manual_seed(456)
        x2 = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
        y2 = torch.randint(0, hidden_dim, (batch_size, ), device=device)

        # DDP second iteration
        ddp_losses2 = [model(x2, y2) for model in ddp_models]
        ddp_combined_loss2 = sum(l / (i + 1) for i, l in enumerate(ddp_losses2))
        ddp_combined_loss2.backward()

        ddp_grads_per_model2 = []
        for model in ddp_models:
            grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    clean_name = name.replace('module.', '')
                    grads[clean_name] = param.grad.detach().clone().cpu()
            ddp_grads_per_model2.append(grads)

        # DeepSpeed second iteration
        ds_losses2 = [engine(x2, y2) for engine in model_engines]
        ds_combined_loss2 = sum(l / (i + 1) for i, l in enumerate(ds_losses2))
        ds_combined_loss2.backward()

        # Verify gradients again and compare with DDP
        for engine_idx, engine in enumerate(model_engines):
            ds_grads = collect_gradients_safe(engine)
            ddp_grads = ddp_grads_per_model2[engine_idx]

            assert len(ds_grads) > 0, f"Engine {engine_idx} has no gradients in second iteration"
            assert len(ddp_grads) == len(ds_grads), \
                f"Engine {engine_idx} (iter 2): DDP has {len(ddp_grads)} gradients, DeepSpeed has {len(ds_grads)}"

            for name in ddp_grads.keys():
                assert name in ds_grads, f"Engine {engine_idx} (iter 2): Parameter {name} missing in DeepSpeed gradients"

                ddp_grad_fp32 = ddp_grads[name].float()
                ds_grad_fp32 = ds_grads[name].float()
                allclose_on_all_ranks(
                    ddp_grad_fp32,
                    ds_grad_fp32,
                    assert_message=f"Engine {engine_idx} (iter 2): Gradients differ for parameter {name}")

        # Step both
        for optimizer in ddp_optimizers:
            optimizer.step()

        for engine in model_engines:
            engine.step()

        # Cleanup
        for engine in model_engines:
            engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardSeparateLoss(DistributedTest):
    """Test using separate loss functions"""
    world_size = 2

    def test_separate_loss_function(self, zero_stage):
        """Test that separate loss function works correctly by comparing with PyTorch DDP"""
        hidden_dim = 4
        lr = 1e-3

        class SimpleOutputModel(torch.nn.Module):
            """Model that returns output without computing loss"""

            def __init__(self, hidden_dim):
                super().__init__()
                self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        # Create two identical models
        torch.manual_seed(42)
        model_ddp = SimpleOutputModel(hidden_dim=hidden_dim)

        torch.manual_seed(42)
        model_deepspeed = SimpleOutputModel(hidden_dim=hidden_dim)

        # Initialize DDP baseline
        deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
        device = get_accelerator().current_device_name()
        rank = get_accelerator().current_device()
        dtype = preferred_dtype()

        model_ddp = model_ddp.to(device=device, dtype=dtype)
        model_ddp = DDP(model_ddp, device_ids=[rank], output_device=rank)
        optimizer_ddp = torch.optim.Adam(model_ddp.parameters(), lr=lr)

        # Initialize DeepSpeed
        config = get_config_dict(zero_stage)
        model_engine, _, _, _ = deepspeed.initialize(config=config,
                                                     model=model_deepspeed,
                                                     model_parameters=model_deepspeed.parameters())

        # Define loss function separately
        loss_fn = torch.nn.CrossEntropyLoss()

        # Create data (use same seed for reproducibility)
        batch_size = 2
        torch.manual_seed(456)
        x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
        y = torch.randint(0, hidden_dim, (batch_size, ), device=device)

        # DDP: forward, loss, backward
        optimizer_ddp.zero_grad()
        output_ddp = model_ddp(x)
        loss_ddp = loss_fn(output_ddp, y)
        loss_ddp.backward()

        # Collect DDP gradients
        grads_ddp = {}
        for name, param in model_ddp.named_parameters():
            if param.grad is not None:
                clean_name = name.replace('module.', '')
                grads_ddp[clean_name] = param.grad.detach().clone().cpu()

        # DeepSpeed: forward, loss, backward
        output_ds = model_engine(x)
        loss_ds = loss_fn(output_ds, y)
        loss_ds.backward()
        grads_ds = collect_gradients_safe(model_engine)

        # Compare gradients across all ranks
        assert len(grads_ddp) == len(grads_ds), \
            f"Different number of parameters with gradients: DDP={len(grads_ddp)}, DeepSpeed={len(grads_ds)}"
        for name in grads_ddp.keys():
            assert name in grads_ds, f"Parameter {name} missing in DeepSpeed gradients"
            # Convert both to fp32 for comparison in case of dtype mismatch
            grads_ddp_fp32 = grads_ddp[name].float()
            grads_ds_fp32 = grads_ds[name].float()
            allclose_on_all_ranks(grads_ddp_fp32,
                                  grads_ds_fp32,
                                  assert_message=f"Gradients differ for parameter {name} between DDP and DeepSpeed")

        model_engine.destroy()
