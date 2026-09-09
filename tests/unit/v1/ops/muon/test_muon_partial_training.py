# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Test for PR #7869: Fix Muon optimizer with partial model training

This test verifies that the fix for Muon optimizer parameter grouping works
correctly when only part of the model parameters are trainable.

The bug occurred when:
1. Some parameters use Muon optimizer (p.use_muon = True)
2. Other parameters use AdamW optimizer (p.use_muon = False)
3. All trainable parameters happen to use the same optimizer type

This caused one of the parameter groups to be empty, leading to:
ValueError: torch.cat(): expected a non-empty list of Tensors

The fix filters parameters to only include those with requires_grad=True,
ensuring empty parameter groups are properly handled.
"""

import torch
import torch.nn as nn
import deepspeed
import pytest
from unit.common import DistributedTest
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam, ZenFlowCPUAdam
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam


class PartialTrainableModel(nn.Module):
    """
    A model where some parameters use Muon and some use AdamW.

    This simulates the scenario where:
    - Hidden layers use Muon (ndim >= 2)
    - Embeddings and biases use AdamW (ndim < 2)
    """

    def __init__(self, vocab_size=100, hidden_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Set use_muon attribute for parameters
        # Muon should be used for ndim >= 2 (matrices)
        # AdamW should be used for ndim < 2 (embeddings, biases)
        for name, param in self.named_parameters():
            if param.ndim >= 2:
                param.use_muon = True
            else:
                param.use_muon = False


class TestMuonPartialModelTraining(DistributedTest):
    """Test Muon optimizer with partial model training scenarios."""

    world_size = 2
    reuse_dist_env = True
    requires_cuda_env = False

    def test_muon_with_all_trainable_params(self):
        """
        Test when all parameters are trainable.

        This should work fine as both Muon and AdamW parameter groups
        will be non-empty.
        """
        model = PartialTrainableModel()

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 0.02,
                    "weight_decay": 0.01
                }
            },
            "zero_optimization": {
                "stage": 2
            },
        }

        # This should not raise ValueError
        model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.parameters(),
                                                     config=ds_config)

        # Verify the model was initialized successfully
        assert model_engine is not None

    def test_muon_with_partial_trainable_params_same_optimizer(self):
        """
        Test the bug scenario: all trainable params use the same optimizer.

        This is the bug case where:
        - All trainable parameters have use_muon=True (or all False)
        - This causes one parameter group to be empty
        - Without the fix, this raises: ValueError: torch.cat(): expected a non-empty list of Tensors

        The fix filters by requires_grad, so empty groups are properly handled.
        """
        model = PartialTrainableModel()

        # Freeze all Linear layers (which have use_muon=True)
        # Keep only embeddings and biases trainable (use_muon=False)
        for name, param in model.named_parameters():
            if "layers" in name or "output" in name:
                param.requires_grad = False

        # Now all trainable parameters have use_muon=False
        # This would cause muon_params to be empty without the fix

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 0.02,
                    "weight_decay": 0.01
                }
            },
            "zero_optimization": {
                "stage": 2
            },
        }

        # This would raise ValueError without the fix
        # With the fix, it should initialize successfully
        model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.parameters(),
                                                     config=ds_config)

        # Verify the model was initialized successfully
        assert model_engine is not None

    def test_muon_with_mixed_trainable_params(self):
        """
        Test when trainable parameters use both optimizers.

        This is the normal case where:
        - Some trainable params have use_muon=True
        - Some trainable params have use_muon=False
        - Both parameter groups are non-empty

        This should work fine even without the fix.
        """
        model = PartialTrainableModel()

        # Freeze only the first Linear layer
        # This leaves both Muon and AdamW parameters trainable
        for name, param in model.named_parameters():
            if "layers.0" in name:
                param.requires_grad = False

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 0.02,
                    "weight_decay": 0.01
                }
            },
            "zero_optimization": {
                "stage": 2
            },
        }

        # This should work fine
        model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.parameters(),
                                                     config=ds_config)

        # Verify the model was initialized successfully
        assert model_engine is not None

    @pytest.mark.parametrize(
        "optimizer_params, expected_muon_lr, expected_adam_lr",
        [
            ({
                "lr": 0.02,
                "weight_decay": 0.01
            }, 0.02, 0.02),
            ({
                "lr": 0.02,
                "muon_lr": 0.04,
                "adam_lr": 0.001,
                "weight_decay": 0.01
            }, 0.04, 0.001),
        ],
    )
    def test_muon_adam_learning_rate_overrides(self, optimizer_params, expected_muon_lr, expected_adam_lr):
        model = PartialTrainableModel()

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Muon",
                "params": optimizer_params
            },
            "zero_optimization": {
                "stage": 2
            },
        }

        model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.parameters(),
                                                     config=ds_config)

        group_lrs = {
            param_group["use_muon"]: param_group["lr"]
            for param_group in model_engine.basic_optimizer.param_groups
        }
        assert group_lrs[True] == expected_muon_lr
        assert group_lrs[False] == expected_adam_lr

    @pytest.mark.parametrize("adam_w_mode, expected_optimizer", [(True, torch.optim.AdamW), (False, torch.optim.Adam)])
    def test_muon_aux_adam_backend_dispatch(self, adam_w_mode, expected_optimizer):
        model = PartialTrainableModel()
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 0.02,
                    "torch_adam": True,
                    "adam_w_mode": adam_w_mode,
                },
            },
            "zero_optimization": {
                "stage": 2
            },
        }

        model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.parameters(),
                                                     config=ds_config)

        assert isinstance(model_engine.basic_optimizer.aux_optimizer, expected_optimizer)
        assert model_engine.basic_optimizer.aux_optimizer.state is model_engine.basic_optimizer.state


@pytest.mark.parametrize("optimizer_class", [torch.optim.Adam, torch.optim.AdamW])
def test_muon_aux_adam_matches_torch(optimizer_class):
    actual_param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    expected_param = torch.nn.Parameter(actual_param.detach().clone())
    group_options = {
        "lr": 0.01,
        "betas": (0.8, 0.9),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    optimizer = MuonWithAuxAdam([dict(params=[actual_param], use_muon=False, **group_options)],
                                adam_optimizer=optimizer_class)
    reference = optimizer_class([dict(params=[expected_param], **group_options)])

    for grad in (torch.tensor([0.25, -0.5]), torch.tensor([-0.1, 0.2]), torch.tensor([0.3, 0.4])):
        actual_param.grad = grad.clone()
        expected_param.grad = grad.clone()
        optimizer.step()
        reference.step()

    torch.testing.assert_close(actual_param, expected_param)
    assert optimizer.aux_optimizer.state is optimizer.state
    assert optimizer.state[actual_param].keys() == reference.state[expected_param].keys()


@pytest.mark.parametrize("adam_w_mode, optimizer_class", [(True, torch.optim.AdamW), (False, torch.optim.Adam)])
def test_muon_aux_adam_falls_back_to_inline_update(adam_w_mode, optimizer_class):
    actual_param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    expected_param = torch.nn.Parameter(actual_param.detach().clone())
    group_options = {
        "lr": 0.01,
        "betas": (0.8, 0.9),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    optimizer = MuonWithAuxAdam([dict(params=[actual_param], use_muon=False, **group_options)],
                                adam_w_mode=adam_w_mode)
    reference = optimizer_class([expected_param], **group_options)

    for grad in (torch.tensor([0.25, -0.5]), torch.tensor([-0.1, 0.2]), torch.tensor([0.3, 0.4])):
        actual_param.grad = grad.clone()
        expected_param.grad = grad.clone()
        optimizer.step()
        reference.step()

    torch.testing.assert_close(actual_param, expected_param)


def test_muon_aux_adam_falls_back_when_backend_initialization_fails():

    class UnavailableAdam:

        def __init__(self, *args, **kwargs):
            raise RuntimeError("backend unavailable")

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MuonWithAuxAdam([dict(params=[param], use_muon=False)],
                                adam_optimizer=UnavailableAdam,
                                fallback_to_inline=True)

    assert optimizer.aux_optimizer is None


def test_muon_aux_adam_passes_step_id_to_sequential_zenflow():

    class SequentialAdam(torch.optim.AdamW):

        def __init__(self, params):
            super().__init__(params)
            self.overlap_step = False
            self.received_step_id = None

        def step(self, step_id, closure=None):
            self.received_step_id = step_id
            return super().step(closure)

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = MuonWithAuxAdam([dict(params=[param], use_muon=False)], adam_optimizer=SequentialAdam)
    param.grad = torch.ones_like(param)
    optimizer.step(step_id=3)

    assert optimizer.aux_optimizer.received_step_id == 3


def test_muon_aux_adam_rebinds_zero_parameter_groups():
    original_param = torch.nn.Parameter(torch.tensor([1.0]))
    replacement_param = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = MuonWithAuxAdam([dict(params=[original_param], use_muon=False, lr=0.1)],
                                adam_optimizer=torch.optim.AdamW)

    optimizer.param_groups[0]["params"] = [replacement_param]
    replacement_param.grad = torch.ones_like(replacement_param)
    optimizer.step()

    assert original_param.item() == 1.0
    assert replacement_param.item() < 2.0
    assert original_param not in optimizer.state
    assert replacement_param in optimizer.state
    assert optimizer.aux_optimizer.param_groups[0] is optimizer.param_groups[0]


class _StrictAdamBackend(torch.optim.Optimizer):

    def __init__(self, params, **kwargs):
        defaults = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, bias_correction=True, amsgrad=False)
        defaults.update(kwargs)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            assert group["bias_correction"]
            assert group["amsgrad"] is False
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["step"] += 1


def test_muon_aux_adam_loads_legacy_inline_state_with_backend_defaults():
    legacy_param = torch.nn.Parameter(torch.tensor([1.0]))
    legacy_optimizer = MuonWithAuxAdam([dict(params=[legacy_param], use_muon=False, lr=0.1)])
    legacy_param.grad = torch.ones_like(legacy_param)
    legacy_optimizer.step()
    legacy_state_dict = legacy_optimizer.state_dict()

    restored_param = torch.nn.Parameter(torch.tensor([1.0]))
    restored_optimizer = MuonWithAuxAdam([dict(params=[restored_param], use_muon=False, lr=0.1)],
                                         adam_optimizer=_StrictAdamBackend)
    restored_optimizer.load_state_dict(legacy_state_dict)
    restored_param.grad = torch.ones_like(restored_param)
    restored_optimizer.step()

    assert restored_optimizer.state[restored_param]["step"] == 2


class _AdamSelectionEngine:

    def __init__(self, cpu_offload=False, bf16_states=False, zenflow=False):
        self.cpu_offload = cpu_offload
        self.bf16_states = bf16_states
        self.zenflow = zenflow
        self.overlap_step = True

    def zero_use_cpu_optimizer(self):
        return self.cpu_offload

    def bf16_optimizer_states(self):
        return self.bf16_states


@pytest.mark.parametrize(
    "engine, parameters, adam_w_mode, expected_class, expected_kwargs",
    [
        (_AdamSelectionEngine(), {
            "torch_adam": True
        }, True, torch.optim.AdamW, {
            "foreach": None
        }),
        (_AdamSelectionEngine(cpu_offload=True), {
            "torch_adam": True
        }, False, torch.optim.Adam, {
            "foreach": None
        }),
        (_AdamSelectionEngine(), {}, True, FusedAdam, {
            "adam_w_mode": True
        }),
        (_AdamSelectionEngine(cpu_offload=True), {}, False, DeepSpeedCPUAdam, {
            "adamw_mode": False,
            "fp32_optimizer_states": True
        }),
        (_AdamSelectionEngine(cpu_offload=True, bf16_states=True), {
            "fp32_optimizer_states": True
        }, True, DeepSpeedCPUAdam, {
            "adamw_mode": True,
            "fp32_optimizer_states": False
        }),
        (_AdamSelectionEngine(cpu_offload=True, zenflow=True), {}, True, ZenFlowCPUAdam, {
            "adamw_mode": True,
            "fp32_optimizer_states": True,
            "overlap_step": True
        }),
    ],
)
def test_adam_backend_selection(engine, parameters, adam_w_mode, expected_class, expected_kwargs):
    optimizer_class, optimizer_kwargs = DeepSpeedEngine.get_optimizer_configuration(engine, parameters, adam_w_mode)

    assert optimizer_class is expected_class
    assert optimizer_kwargs == expected_kwargs
