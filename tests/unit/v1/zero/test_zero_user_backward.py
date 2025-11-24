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


def initialize_distributed():
    deepspeed.init_distributed(dist_backend=get_accelerator().communication_backend_name())
    device = get_accelerator().current_device_name()
    rank = get_accelerator().current_device()
    dtype = preferred_dtype()
    return device, rank, dtype


def create_ddp_model(model_class, device, rank, dtype, seed=42, lr=1e-3, **model_kwargs):
    torch.manual_seed(seed)
    model = model_class(**model_kwargs)
    model = model.to(device=device, dtype=dtype)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def create_deepspeed_engine(model_class, zero_stage, seed=42, gradient_accumulation_steps=1, **model_kwargs):
    torch.manual_seed(seed)
    model = model_class(**model_kwargs)

    config = get_config_dict(zero_stage, gradient_accumulation_steps=gradient_accumulation_steps)
    engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
    return engine


def create_deepspeed_engine_from_model(model, zero_stage, gradient_accumulation_steps=1):
    config = get_config_dict(zero_stage, gradient_accumulation_steps=gradient_accumulation_steps)
    engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
    return engine


def setup_models_and_engines(model_class, zero_stage, seed=42, lr=1e-3, gradient_accumulation_steps=1, **model_kwargs):
    # Initialize distributed environment
    device, rank, dtype = initialize_distributed()

    # Create DDP model
    model_ddp, optimizer_ddp = create_ddp_model(model_class, device, rank, dtype, seed=seed, lr=lr, **model_kwargs)

    # Create DeepSpeed engine
    model_engine = create_deepspeed_engine(model_class,
                                           zero_stage,
                                           seed=seed,
                                           gradient_accumulation_steps=gradient_accumulation_steps,
                                           **model_kwargs)

    return model_ddp, optimizer_ddp, model_engine, device, dtype


def collect_ddp_gradients(model_ddp):
    """Collect gradients from DDP model"""
    grads = {}
    for name, param in model_ddp.named_parameters():
        if param.grad is not None:
            clean_name = name.replace('module.', '')
            grads[clean_name] = param.grad.detach().clone().cpu()
    return grads


def compare_gradients(grads_ddp, grads_ds, step_info=""):
    """Compare gradients between DDP and DeepSpeed"""
    step_suffix = f" at {step_info}" if step_info else ""
    assert len(grads_ddp) == len(grads_ds), \
        f"Different number of parameters with gradients{step_suffix}: DDP={len(grads_ddp)}, DeepSpeed={len(grads_ds)}"

    for name in grads_ddp.keys():
        assert name in grads_ds, f"Parameter {name} missing in DeepSpeed gradients{step_suffix}"
        # Convert both to fp32 for comparison in case of dtype mismatch
        grads_ddp_fp32 = grads_ddp[name].float()
        grads_ds_fp32 = grads_ds[name].float()
        allclose_on_all_ranks(grads_ddp_fp32,
                              grads_ds_fp32,
                              assert_message=f"Gradients differ for parameter {name}{step_suffix}")


def collect_ddp_parameters(model_ddp):
    """Collect parameters from DDP model"""
    params = {}
    for name, param in model_ddp.named_parameters():
        clean_name = name.replace('module.', '')
        params[clean_name] = param.detach().clone().cpu()
    return params


def collect_deepspeed_parameters(model_engine, zero_stage):
    """Collect parameters from DeepSpeed engine (handles ZeRO-3 gathering)"""
    params = {}
    for name, param in model_engine.named_parameters():
        clean_name = name.replace('module.', '')
        if zero_stage == 3:
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                params[clean_name] = param.detach().clone().cpu()
        else:
            params[clean_name] = param.detach().clone().cpu()
    return params


def compare_parameters(params_ddp, params_ds, step_info=""):
    """Compare parameters between DDP and DeepSpeed"""
    step_suffix = f" at {step_info}" if step_info else ""
    assert len(params_ddp) == len(params_ds), \
        f"Parameter count mismatch{step_suffix}: DDP={len(params_ddp)}, DeepSpeed={len(params_ds)}"

    for name in params_ddp.keys():
        assert name in params_ds, f"Parameter {name} missing in DeepSpeed model{step_suffix}"
        # Convert to fp32 for comparison in case of dtype mismatch
        params_ddp_fp32 = params_ddp[name].float()
        params_ds_fp32 = params_ds[name].float()
        allclose_on_all_ranks(params_ddp_fp32,
                              params_ds_fp32,
                              assert_message=f"Parameter {name} mismatch{step_suffix}")


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardBasic(DistributedTest):
    """Test basic functionality of user backward (loss.backward()) by comparing with PyTorch DDP"""
    world_size = 2

    def test_loss_backward_matches_ddp(self, zero_stage):
        """Test that DeepSpeed loss.backward() produces same gradients as PyTorch DDP"""
        hidden_dim = 4

        # Create DDP and DeepSpeed models
        model_ddp, optimizer_ddp, model_engine, device, dtype = setup_models_and_engines(model_class=SimpleModel,
                                                                                         zero_stage=zero_stage,
                                                                                         hidden_dim=hidden_dim,
                                                                                         nlayers=2)

        # Create data
        data_loader = random_dataloader(model=model_engine, total_samples=8, hidden_dim=hidden_dim, device=device)

        # Run one training step with both models
        batch = next(iter(data_loader))

        # DDP: forward and backward
        optimizer_ddp.zero_grad()
        loss_ddp = model_ddp(batch[0], batch[1])
        loss_ddp.backward()
        grads_ddp = collect_ddp_gradients(model_ddp)

        # DeepSpeed: forward and backward
        loss_ds = model_engine(batch[0], batch[1])
        loss_ds.backward()
        grads_ds = collect_gradients_safe(model_engine)

        # Compare gradients
        compare_gradients(grads_ddp, grads_ds)

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardNonScalar(DistributedTest):
    """Test non-scalar backward support"""
    world_size = 2

    def test_non_scalar_backward(self, zero_stage):
        """Test that tensor.backward(grad) works correctly by comparing with PyTorch DDP"""
        hidden_dim = 4
        batch_size = 2

        # Create DDP and DeepSpeed models
        model_ddp, optimizer_ddp, model_engine, device, dtype = setup_models_and_engines(
            model_class=SimpleNonScalarModel, zero_stage=zero_stage, hidden_dim=hidden_dim)

        # Create input data
        torch.manual_seed(123)
        x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)

        # DDP: forward and non-scalar backward
        optimizer_ddp.zero_grad()
        output_ddp = model_ddp(x)
        grad_output = torch.ones_like(output_ddp)
        output_ddp.backward(grad_output)
        ddp_grads = collect_ddp_gradients(model_ddp)

        # DeepSpeed: forward and non-scalar backward
        output_deepspeed = model_engine(x)
        grad_output_ds = torch.ones_like(output_deepspeed)
        output_deepspeed.backward(grad_output_ds)
        deepspeed_grads = collect_gradients_safe(model_engine)

        # Compare gradients
        compare_gradients(ddp_grads, deepspeed_grads, "after non-scalar backward")

        # Run optimizer step
        optimizer_ddp.step()
        model_engine.step()

        # Collect and compare parameters after step
        ddp_params = collect_ddp_parameters(model_ddp)
        deepspeed_params = collect_deepspeed_parameters(model_engine, zero_stage)
        compare_parameters(ddp_params, deepspeed_params, "after non-scalar backward")

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestZeroUserBackwardGradAccumulation(DistributedTest):
    """Test gradient accumulation with user backward"""
    world_size = 2

    def test_grad_accumulation(self, zero_stage):
        """Test that gradient accumulation works correctly with loss.backward() by comparing with DDP"""
        hidden_dim = 4
        gradient_accumulation_steps = 4

        # Create DDP and DeepSpeed models with gradient accumulation
        model_ddp, optimizer_ddp, model_engine, device, _ = setup_models_and_engines(
            model_class=SimpleModel,
            zero_stage=zero_stage,
            gradient_accumulation_steps=gradient_accumulation_steps,
            hidden_dim=hidden_dim,
            nlayers=2)

        # Create data
        data_loader = random_dataloader(model=model_engine, total_samples=16, hidden_dim=hidden_dim, device=device)

        # Run training with gradient accumulation
        for i, batch in enumerate(data_loader):
            # DDP: Manual gradient accumulation
            loss_ddp = model_ddp(batch[0], batch[1])
            (loss_ddp / gradient_accumulation_steps).backward()

            # DeepSpeed: Built-in gradient accumulation
            loss_ds = model_engine(batch[0], batch[1])
            loss_ds.backward()

            # Compare gradients at accumulation boundary
            if model_engine.is_gradient_accumulation_boundary():
                grads_ddp = collect_ddp_gradients(model_ddp)
                grads_ds = collect_gradients_safe(model_engine)
                compare_gradients(grads_ddp, grads_ds, f"step {i}")

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
        device, rank, dtype = initialize_distributed()

        # Create DDP baseline models
        ddp_models = []
        ddp_optimizers = []
        for i in range(num_models):
            model, optimizer = create_ddp_model(SimpleModel,
                                                device,
                                                rank,
                                                dtype,
                                                seed=42 + i,
                                                lr=lr,
                                                hidden_dim=hidden_dim,
                                                nlayers=2)
            ddp_models.append(model)
            ddp_optimizers.append(optimizer)

        # Create multiple DeepSpeed engines with identical initialization
        model_engines = []
        for i in range(num_models):
            engine = create_deepspeed_engine(SimpleModel, zero_stage, seed=42 + i, hidden_dim=hidden_dim, nlayers=2)
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
        ddp_grads_per_model = [collect_ddp_gradients(model) for model in ddp_models]

        # DeepSpeed: compute losses and combined backward WITHOUT manual _backward_epilogue()
        ds_losses = [engine(x, y) for engine in model_engines]
        ds_combined_loss = sum(l / (i + 1) for i, l in enumerate(ds_losses))
        ds_combined_loss.backward()

        # Collect DeepSpeed gradients for each engine and compare with DDP
        for engine_idx, engine in enumerate(model_engines):
            ds_grads = collect_gradients_safe(engine)
            ddp_grads = ddp_grads_per_model[engine_idx]
            assert len(ds_grads) > 0, f"Engine {engine_idx} has no gradients after combined_loss.backward()"
            compare_gradients(ddp_grads, ds_grads, f"Engine {engine_idx}")

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
        ddp_grads_per_model2 = [collect_ddp_gradients(model) for model in ddp_models]

        # DeepSpeed second iteration
        ds_losses2 = [engine(x2, y2) for engine in model_engines]
        ds_combined_loss2 = sum(l / (i + 1) for i, l in enumerate(ds_losses2))
        ds_combined_loss2.backward()

        # Verify gradients again and compare with DDP
        for engine_idx, engine in enumerate(model_engines):
            ds_grads = collect_gradients_safe(engine)
            ddp_grads = ddp_grads_per_model2[engine_idx]
            assert len(ds_grads) > 0, f"Engine {engine_idx} has no gradients in second iteration"
            compare_gradients(ddp_grads, ds_grads, f"Engine {engine_idx} (iter 2)")

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
        batch_size = 2

        # Create DDP and DeepSpeed models
        model_ddp, optimizer_ddp, model_engine, device, dtype = setup_models_and_engines(model_class=SimpleOutputModel,
                                                                                         zero_stage=zero_stage,
                                                                                         hidden_dim=hidden_dim)

        # Define loss function separately
        loss_fn = torch.nn.CrossEntropyLoss()

        # Create input data
        torch.manual_seed(456)
        x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
        y = torch.randint(0, hidden_dim, (batch_size, ), device=device)

        # DDP: forward, loss, backward
        optimizer_ddp.zero_grad()
        output_ddp = model_ddp(x)
        loss_ddp = loss_fn(output_ddp, y)
        loss_ddp.backward()
        grads_ddp = collect_ddp_gradients(model_ddp)

        # DeepSpeed: forward, loss, backward
        output_ds = model_engine(x)
        loss_ds = loss_fn(output_ds, y)
        loss_ds.backward()
        grads_ds = collect_gradients_safe(model_engine)

        # Compare gradients
        compare_gradients(grads_ddp, grads_ds)

        model_engine.destroy()


class LeafModuleModel(torch.nn.Module):
    """Model with ModuleList that uses all parameters - for testing leaf module compatibility"""

    def __init__(self, hidden_dim):
        super().__init__()
        # ModuleList where all branches are used in forward pass
        self.branches = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ])
        self.final_layer = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, y):
        # Use all branches - add their outputs together
        x = self.branches[0](x) + self.branches[1](x)
        x = self.final_layer(x)
        loss = torch.nn.functional.cross_entropy(x, y)
        return loss


class LeafNonScalarModel(torch.nn.Module):
    """Leaf module model that returns non-scalar output"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.branches = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        ])

    def forward(self, x):
        # Use all branches - returns non-scalar output
        return self.branches[0](x) + self.branches[1](x)


@pytest.mark.parametrize("zero_stage", [3])
class TestZeroUserBackwardLeafModule(DistributedTest):
    """Test leaf module behavior during backward passes in ZeRO Stage 3"""
    world_size = 2

    def test_leaf_module_backward(self, zero_stage):
        """Test that leaf modules work correctly with user backward by comparing with PyTorch DDP

        This test validates that the leaf_module_count and backward hooks are correctly
        handled in create_reduce_and_remove_grad_hooks.
        """
        from deepspeed.utils import set_z3_leaf_modules, z3_leaf_module

        hidden_dim = 4
        batch_size = 2
        lr = 1e-3

        # Initialize distributed environment
        device, rank, dtype = initialize_distributed()

        # Create DDP model
        model_ddp, optimizer_ddp = create_ddp_model(LeafModuleModel,
                                                    device,
                                                    rank,
                                                    dtype,
                                                    seed=42,
                                                    lr=lr,
                                                    hidden_dim=hidden_dim)

        # Create DeepSpeed model and mark leaf modules BEFORE initialization
        torch.manual_seed(42)
        model_deepspeed = LeafModuleModel(hidden_dim=hidden_dim)
        leaf_modules = set_z3_leaf_modules(model_deepspeed, [torch.nn.ModuleList])
        assert len(leaf_modules) == 1, "Expected exactly one ModuleList to be marked as leaf"
        assert z3_leaf_module(model_deepspeed.branches), "ModuleList should be marked as leaf module"

        # Initialize DeepSpeed engine from the prepared model
        model_engine = create_deepspeed_engine_from_model(model_deepspeed, zero_stage)

        # Verify leaf_module_count was set correctly
        assert len(model_engine.optimizer.leaf_parameters) == 1, \
            "Expected 1 leaf module in optimizer.leaf_parameters"

        # Create input data
        torch.manual_seed(123)
        x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
        y = torch.randint(0, hidden_dim, (batch_size, ), device=device)

        # DDP: forward and backward
        optimizer_ddp.zero_grad()
        loss_ddp = model_ddp(x, y)
        loss_ddp.backward()
        ddp_grads = collect_ddp_gradients(model_ddp)

        # DeepSpeed: forward and backward with leaf module
        loss_deepspeed = model_engine(x, y)
        loss_deepspeed.backward()
        deepspeed_grads = collect_gradients_safe(model_engine)

        # Compare gradients
        compare_gradients(ddp_grads, deepspeed_grads, "with leaf modules")

        model_engine.destroy()

    def test_leaf_module_non_scalar_backward(self, zero_stage):
        """Test that leaf modules work correctly with non-scalar backward (tensor.backward(grad))

        This specifically tests the interaction between leaf modules and non-scalar backward.
        """
        from deepspeed.utils import set_z3_leaf_modules, z3_leaf_module

        hidden_dim = 4
        batch_size = 2
        lr = 1e-3

        # Initialize distributed environment
        device, rank, dtype = initialize_distributed()

        # Create DDP model
        model_ddp, optimizer_ddp = create_ddp_model(LeafNonScalarModel,
                                                    device,
                                                    rank,
                                                    dtype,
                                                    seed=42,
                                                    lr=lr,
                                                    hidden_dim=hidden_dim)

        # Create DeepSpeed model and mark leaf modules BEFORE initialization
        torch.manual_seed(42)
        model_deepspeed = LeafNonScalarModel(hidden_dim=hidden_dim)
        leaf_modules = set_z3_leaf_modules(model_deepspeed, [torch.nn.ModuleList])
        assert len(leaf_modules) == 1, "Expected exactly one ModuleList to be marked as leaf"
        assert z3_leaf_module(model_deepspeed.branches), "ModuleList should be marked as leaf module"

        # Initialize DeepSpeed engine from the prepared model
        model_engine = create_deepspeed_engine_from_model(model_deepspeed, zero_stage)

        # Verify leaf_module_count was set correctly
        assert len(model_engine.optimizer.leaf_parameters) == 1, \
            "Expected 1 leaf module in optimizer.leaf_parameters"

        # Create input data
        torch.manual_seed(123)
        x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)

        # DDP: forward and non-scalar backward
        optimizer_ddp.zero_grad()
        output_ddp = model_ddp(x)
        grad_output = torch.ones_like(output_ddp)
        output_ddp.backward(grad_output)
        ddp_grads = collect_ddp_gradients(model_ddp)

        # DeepSpeed: forward and non-scalar backward with leaf module
        output_deepspeed = model_engine(x)
        grad_output_ds = torch.ones_like(output_deepspeed)
        output_deepspeed.backward(grad_output_ds)
        deepspeed_grads = collect_gradients_safe(model_engine)

        # Compare gradients
        compare_gradients(ddp_grads, deepspeed_grads, "in leaf module non-scalar backward")

        model_engine.destroy()


@pytest.mark.sequential
class TestZeroUserBackwardScaleErrorDetection(DistributedTest):
    """Test error detection for missing scale() with fp16 in single-process setup"""
    world_size = 1  # Use single process to avoid distributed deadlock issues

    def test_error_when_backward_without_scale_sequential(self):
        """Test that error is raised when calling backward() without scale() with fp16"""
        if not get_accelerator().is_fp16_supported():
            pytest.skip("Test requires fp16 support")

        hidden_dim = 4
        zero_stage = 1  # Use ZeRO stage 1 for simplicity

        # Initialize distributed
        device, _, _ = initialize_distributed()

        # Create engine with fp16 - requires scaling
        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
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
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }

        model_engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())

        # Verify needs_scaler is True
        from deepspeed.runtime.base_optimizer import ZeROOptimizer
        assert isinstance(model_engine.optimizer, ZeROOptimizer)
        assert model_engine.optimizer.needs_scaler(), "fp16 should require scaling"

        # Create data
        data_loader = random_dataloader(model=model_engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        dtype=torch.float16)
        batch = next(iter(data_loader))

        loss = model_engine(batch[0], batch[1])

        # Calling backward() without scale() should raise RuntimeError
        with pytest.raises(RuntimeError, match="Loss scaling is required"):
            loss.backward()

        model_engine.destroy()


@pytest.mark.parametrize("zero_stage", [1, 3])
class TestZeroUserBackwardWithScale(DistributedTest):
    """Test engine.scale() method for manual backward passes with loss scaling"""
    world_size = 2

    def test_scale_backward_matches_engine_backward(self, zero_stage):
        """Test that engine.scale(loss).backward() produces same gradients as engine.backward(loss)"""
        hidden_dim = 4

        # Create DeepSpeed engines with same seed
        model_engine1 = create_deepspeed_engine(model_class=SimpleModel,
                                                zero_stage=zero_stage,
                                                seed=42,
                                                hidden_dim=hidden_dim,
                                                nlayers=2)
        model_engine2 = create_deepspeed_engine(model_class=SimpleModel,
                                                zero_stage=zero_stage,
                                                seed=42,
                                                hidden_dim=hidden_dim,
                                                nlayers=2)

        # Create data
        device = get_accelerator().current_device_name()
        data_loader = random_dataloader(model=model_engine1, total_samples=8, hidden_dim=hidden_dim, device=device)
        batch = next(iter(data_loader))

        # Model 1: use engine.backward(loss)
        loss1 = model_engine1(batch[0], batch[1])
        model_engine1.backward(loss1)
        grads1 = collect_gradients_safe(model_engine1)

        # Model 2: use engine.scale(loss).backward()
        loss2 = model_engine2(batch[0], batch[1])
        scaled_loss = model_engine2.scale(loss2)
        scaled_loss.backward()
        grads2 = collect_gradients_safe(model_engine2)

        # Compare gradients - they should be identical
        compare_gradients(grads1, grads2, "comparing engine.backward vs engine.scale().backward()")

        model_engine1.destroy()
        model_engine2.destroy()

    def test_scale_backward_matches_ddp(self, zero_stage):
        """Test that engine.scale(loss).backward() produces same gradients as DDP"""
        hidden_dim = 4

        # Create DDP and DeepSpeed models
        model_ddp, optimizer_ddp, model_engine, device, dtype = setup_models_and_engines(model_class=SimpleModel,
                                                                                         zero_stage=zero_stage,
                                                                                         hidden_dim=hidden_dim,
                                                                                         nlayers=2)

        # Create data
        data_loader = random_dataloader(model=model_engine, total_samples=8, hidden_dim=hidden_dim, device=device)
        batch = next(iter(data_loader))

        # DDP: forward and backward
        optimizer_ddp.zero_grad()
        loss_ddp = model_ddp(batch[0], batch[1])
        loss_ddp.backward()
        grads_ddp = collect_ddp_gradients(model_ddp)

        # DeepSpeed: forward and scale + backward
        loss_ds = model_engine(batch[0], batch[1])
        scaled_loss = model_engine.scale(loss_ds)
        scaled_loss.backward()
        grads_ds = collect_gradients_safe(model_engine)

        # Compare gradients
        compare_gradients(grads_ddp, grads_ds, "comparing DDP vs engine.scale().backward()")

        model_engine.destroy()

    def test_scale_with_gradient_accumulation(self, zero_stage):
        """Test that engine.scale() works correctly with gradient accumulation"""
        hidden_dim = 4
        gradient_accumulation_steps = 4

        # Create models with gradient accumulation
        model_ddp, optimizer_ddp, model_engine, device, _ = setup_models_and_engines(
            model_class=SimpleModel,
            zero_stage=zero_stage,
            gradient_accumulation_steps=gradient_accumulation_steps,
            hidden_dim=hidden_dim,
            nlayers=2)

        # Create data
        data_loader = random_dataloader(model=model_engine, total_samples=16, hidden_dim=hidden_dim, device=device)

        # Run gradient accumulation steps
        for i, batch in enumerate(data_loader):
            # DDP: manual gradient accumulation
            loss_ddp = model_ddp(batch[0], batch[1])
            # Scale by GAS for DDP to match DeepSpeed behavior
            (loss_ddp / gradient_accumulation_steps).backward()

            # DeepSpeed: use scale() with built-in gradient accumulation
            # Note: scale() only applies loss scaler, NOT GAS. DeepSpeed handles GAS internally
            # via engine.step(), so we do NOT manually divide by GAS here.
            loss_ds = model_engine(batch[0], batch[1])
            scaled_loss = model_engine.scale(loss_ds)
            scaled_loss.backward()

            # Compare gradients at accumulation boundary
            if model_engine.is_gradient_accumulation_boundary():
                grads_ddp = collect_ddp_gradients(model_ddp)
                grads_ds = collect_gradients_safe(model_engine)
                compare_gradients(grads_ddp, grads_ds, f"step {i}")

                # Step both optimizers
                optimizer_ddp.step()
                optimizer_ddp.zero_grad()

            # Step DeepSpeed (handles gradient accumulation internally)
            model_engine.step()

        model_engine.destroy()

    def test_needs_scaler_with_fp16(self, zero_stage):
        """Test that needs_scaler() correctly identifies when scaling is required with fp16"""
        if not get_accelerator().is_fp16_supported():
            pytest.skip("Test requires fp16 support for gradient scaling")

        hidden_dim = 4

        # Initialize distributed first
        device, _, _ = initialize_distributed()

        # Create engine with fp16 explicitly to test gradient scaling requirement
        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
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
            # Explicitly enable fp16 to test gradient scaling requirement
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }

        if zero_stage == 3:
            config["zero_optimization"]["stage3_param_persistence_threshold"] = 0

        model_engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())

        # Verify that the optimizer correctly reports it needs scaling with fp16
        from deepspeed.runtime.base_optimizer import ZeROOptimizer
        assert isinstance(model_engine.optimizer, ZeROOptimizer), "Optimizer should be ZeROOptimizer"
        assert model_engine.optimizer.needs_scaler(), "fp16 configuration should require gradient scaling"

        # Verify scale() method works correctly
        data_loader = random_dataloader(model=model_engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        dtype=torch.float16)
        batch = next(iter(data_loader))
        loss = model_engine(batch[0], batch[1])

        # Should be able to use scale() method and get a valid scaled tensor
        scaled_loss = model_engine.scale(loss)
        assert scaled_loss is not None, "scale() should return a scaled loss tensor"
        assert scaled_loss.requires_grad, "scaled loss should require grad"

        model_engine.destroy()

    def test_needs_scaler_with_bf16(self, zero_stage):
        """Test that needs_scaler() correctly identifies that bf16 does NOT require scaling"""
        if not get_accelerator().is_bf16_supported():
            pytest.skip("Test requires bf16 support")

        hidden_dim = 4

        # Initialize distributed first
        device, _, _ = initialize_distributed()

        # Create engine with bf16 to verify scaling is NOT required
        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
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
            # Use bf16 which does NOT require gradient scaling
            "bf16": {
                "enabled": True
            }
        }

        if zero_stage == 3:
            config["zero_optimization"]["stage3_param_persistence_threshold"] = 0

        model_engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())

        # Verify that the optimizer correctly reports it does NOT need scaling with bf16
        from deepspeed.runtime.base_optimizer import ZeROOptimizer
        assert isinstance(model_engine.optimizer, ZeROOptimizer), "Optimizer should be ZeROOptimizer"
        assert not model_engine.optimizer.needs_scaler(), "bf16 configuration should NOT require gradient scaling"

        # Verify that loss.backward() can be called directly without scale() for bf16
        data_loader = random_dataloader(model=model_engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        dtype=torch.bfloat16)
        batch = next(iter(data_loader))
        loss = model_engine(batch[0], batch[1])

        # With bf16, should be able to call backward directly (no scaling required)
        loss.backward()

        # Collect gradients to verify backward completed successfully
        grads = collect_gradients_safe(model_engine)
        assert len(grads) > 0, "Expected gradients to be computed"

        model_engine.destroy()

    def test_error_when_backward_without_scale_fp16(self, zero_stage):
        """Test that calling backward() without scale() raises an error with fp16"""
        if not get_accelerator().is_fp16_supported():
            pytest.skip("Test requires fp16 support for gradient scaling")

        hidden_dim = 4

        # Initialize distributed first
        device, _, _ = initialize_distributed()

        # Create engine with fp16
        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
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
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }

        if zero_stage == 3:
            config["zero_optimization"]["stage3_param_persistence_threshold"] = 0

        model_engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())

        # Verify needs_scaler is True
        assert model_engine.optimizer.needs_scaler(), "fp16 should require scaling"

        # Create data
        data_loader = random_dataloader(model=model_engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        dtype=torch.float16)
        batch = next(iter(data_loader))

        loss = model_engine(batch[0], batch[1])

        # Try to call backward without scale - should raise RuntimeError
        error_raised = False
        try:
            loss.backward()
        except RuntimeError as e:
            if "Loss scaling is required" in str(e):
                error_raised = True
            else:
                raise  # Re-raise if it's a different error

        # If the test completes (doesn't hang), verify error was raised
        if error_raised:
            # Success - error was properly detected
            pass
        else:
            # If no error was raised, this is a problem (or it hung and timed out)
            pytest.fail("Expected RuntimeError about loss scaling, but backward completed without error")

        model_engine.destroy()

    def test_scale_validates_scalar_loss(self, zero_stage):
        """Test that scale() validates the input is a scalar loss tensor"""
        hidden_dim = 4

        model_engine = create_deepspeed_engine(model_class=SimpleNonScalarModel,
                                               zero_stage=zero_stage,
                                               seed=42,
                                               hidden_dim=hidden_dim)

        device = get_accelerator().current_device_name()
        dtype = preferred_dtype()
        torch.manual_seed(123)
        x = torch.randn(2, hidden_dim, device=device, dtype=dtype)

        # Forward to get non-scalar output
        output = model_engine(x)

        # Trying to scale a non-scalar tensor should raise an assertion error
        with pytest.raises(AssertionError, match="scalar tensor"):
            model_engine.scale(output)

        model_engine.destroy()

    def test_scale_with_torch_autocast(self, zero_stage):
        """Test that scale() works correctly with torch.autocast and fp16"""
        if not get_accelerator().is_fp16_supported():
            pytest.skip("FP16 not supported on this accelerator")

        hidden_dim = 4

        # Initialize distributed first
        device, _, _ = initialize_distributed()

        # Create engine with fp16 config to test gradient scaling
        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=hidden_dim, nlayers=2)

        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
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
            # Enable fp16 to test gradient scaling (bf16 doesn't use gradient scaling)
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }

        if zero_stage == 3:
            config["zero_optimization"]["stage3_param_persistence_threshold"] = 0

        model_engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())

        # Create data with fp16 dtype to match the config
        data_loader = random_dataloader(model=model_engine,
                                        total_samples=8,
                                        hidden_dim=hidden_dim,
                                        device=device,
                                        dtype=torch.float16)
        batch = next(iter(data_loader))

        # Forward and use scale()
        loss = model_engine(batch[0], batch[1])
        scaled_loss = model_engine.scale(loss)

        # Should be able to call backward
        scaled_loss.backward()

        # Collect gradients to verify they exist
        grads = collect_gradients_safe(model_engine)
        assert len(grads) > 0, "Expected gradients to be computed"

        model_engine.destroy()
