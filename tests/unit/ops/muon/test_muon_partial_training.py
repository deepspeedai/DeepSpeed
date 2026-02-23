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

import torch.nn as nn
import deepspeed


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


class TestMuonPartialModelTraining:
    """Test Muon optimizer with partial model training scenarios."""

    def test_muon_with_all_trainable_params(self):
        """
        Test when all parameters are trainable.

        This should work fine as both Muon and AdamW parameter groups
        will be non-empty.
        """
        model = PartialTrainableModel()

        ds_config = {
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
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
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
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
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
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
