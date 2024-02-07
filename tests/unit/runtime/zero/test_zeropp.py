# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import deepspeed.comm as dist
from torch.nn import Module

from unit.common import DistributedTest
from unit.simple_model import random_dataloader

import deepspeed

from deepspeed.runtime.zero.config import DeepSpeedZeroConfig

import torch.nn as nn
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

import numpy as np


class NNModel(nn.Module):

    def __init__(self, h_dim=1024, n_layers=2):
        super(NNModel, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(h_dim, h_dim) for i in range(n_layers)])
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
        return self.cross_entropy_loss(x, y)


def test_zero_hpz_partition_size_config():
    config = DeepSpeedZeroConfig(**{"zero_hpz_partition_size": 4})
    assert config.zero_hpz_partition_size == 4


def _assert_no_secondary_tensor_group(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_secondary_tensor is None
        assert param.ds_zero_param_process_group is None


def _check_secondary_tensor_existence(model: Module) -> None:
    for _, param in model.named_parameters():
        if param.ds_secondary_tensor is not None:
            return True
    return False


def _assert_secondary_tensor_size(model: Module) -> None:
    for name, param in model.named_parameters():
        assert param.ds_secondary_tensor is not None, f"param {param.ds_id}:{name} does not have secondary tensor"
        assert param.ds_secondary_tensor.size()[0] % param.ds_tensor.size()[0] == 0


#Large sweep along hidden dim, num_layers, and zpg of different sizes
#Assert when zpg=1 that secondary group and tensors are invalid
@pytest.mark.sequential
@pytest.mark.parametrize("h_dim", [1024])
@pytest.mark.parametrize("n_layers", [9])
@pytest.mark.parametrize("zpg", [1, 2, 4])
class TestZeroPPConfigSweep(DistributedTest):
    world_size = 4

    def test(self, h_dim: int, n_layers: int, zpg: int) -> None:
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "zero_hpz_partition_size": zpg,
                "zero_quantized_weights": True,
                "zero_quantized_gradients": True,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        model = NNModel(h_dim, n_layers)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model, total_samples=20, hidden_dim=h_dim, device=model.device)
        dist.barrier()
        if zpg == 1:
            _assert_no_secondary_tensor_group(model)

        for n, batch in enumerate(data_loader):
            if n == 0 and zpg != 1:
                _assert_secondary_tensor_size(model)
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    def test_eval(self, h_dim: int, n_layers: int, zpg: int) -> None:
        # in this test case, we are testing that hpz should be enabled when eval mode is on
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "zero_hpz_partition_size": zpg,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        model = NNModel(h_dim, n_layers)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model, total_samples=20, hidden_dim=h_dim, device=model.device)
        dist.barrier()
        if zpg == 1:
            _assert_no_secondary_tensor_group(model)

        for n, batch in enumerate(data_loader):
            if zpg != 1:
                # here we check that the hpz is enabled when the previous iteration does not update the model
                _assert_secondary_tensor_size(model)
            with torch.no_grad():
                loss = model(batch[0], batch[1])

    def test_gradient_accumulation(self, h_dim: int, n_layers: int, zpg: int) -> None:
        # in this test case, we are testing that hpz should be enabled for the intermediate gradient accumulation steps
        # In this test, we should disable loss_scale
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 3,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "zero_hpz_partition_size": zpg,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0.,
            }
        }

        model = NNModel(h_dim, n_layers)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model, total_samples=20, hidden_dim=h_dim, device=model.device)
        dist.barrier()
        if zpg == 1:
            _assert_no_secondary_tensor_group(model)

        for n, batch in enumerate(data_loader):
            if n == 0 and zpg != 1:
                _assert_secondary_tensor_size(model)
            # here we cannot assert that secondary tensor does not exist because the gradient is likely overflowed as we use random data
            if n > 0 and n % 3 != 0 and zpg != 1:
                # if the previous iteration does not update the model, then the hpz should be enabled
                assert _check_secondary_tensor_existence(model), f"n={n}"
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.nightly
@pytest.mark.parametrize("model_name", ["gpt2"])
class TestZeroPPConvergence(DistributedTest):
    world_size = 4

    def load_and_prepare_data(self, model_name):
        """Load model, tokenizer and dataset, and prepare data loader."""
        from datasets import load_dataset

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Load and tokenize dataset
        dataset = load_dataset("wikitext", 'wikitext-103-raw-v1', split='train[:1%]').filter(lambda x: x["text"])

        def tokenize_function(examples):
            # Tokenize and ensure 'labels' are the same as 'input_ids'
            tokenized_output = tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors='pt')
            tokenized_output["labels"] = tokenized_output["input_ids"].clone()
            return tokenized_output

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Create data loader
        data_loader = DataLoader(tokenized_dataset, batch_size=1, shuffle=False)
        return model, data_loader

    def get_loss(self, model, data_loader, config_dict, step=500):
        """Train the model and calculate average loss."""
        # Initialize DeepSpeed
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        dist.barrier()
        model.train()

        # Training loop
        losses = []
        for n, batch in enumerate(data_loader):
            if n >= step:
                break
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            losses.append(loss.item())

        return np.nanmean(losses[-100:])

    def get_config_dict(self, use_quantized_weights=False, use_hpz=False):
        """Generate the configuration dictionary for DeepSpeed."""
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5
                }
            },
            "fp16": {
                "enabled": True
            }
        }
        if use_quantized_weights:
            config["zero_optimization"]["zero_quantized_weights"] = True
        if use_hpz:
            config["zero_optimization"]["zero_hpz_partition_size"] = self.world_size // 2
        return config

    def test(self, model_name):
        torch.manual_seed(0)
        model, data_loader = self.load_and_prepare_data(model_name)
        zeropp_loss = self.get_loss(model, data_loader, self.get_config_dict(use_quantized_weights=True, use_hpz=True))
        model, data_loader = self.load_and_prepare_data(model_name)
        baseline_loss = self.get_loss(model, data_loader, self.get_config_dict())

        # Output and assert
        print(f"zeropp_loss={zeropp_loss}, baseline_loss={baseline_loss}")
        assert zeropp_loss < baseline_loss * 1.1, f"zeropp_loss={zeropp_loss}, baseline_loss={baseline_loss}"
