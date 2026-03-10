# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


def skip_on_device():
    if get_accelerator().device_name() == 'xpu':
        pytest.skip("XPU requires a higher version for test")


class TestTPPlanRealHFModels(DistributedTest):
    """End-to-end tests using real HuggingFace models"""
    world_size = 2

    def test_llama_tp_plan_with_zero1(self):
        """Test Llama model + tp_plan + ZeRO1"""
        skip_on_device()

        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            pytest.skip("transformers not installed")

        # Create small Llama config (to save memory)
        config = AutoConfig.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        # Initialize model from config (no pretrained weights)
        model = AutoModelForCausalLM.from_config(config)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 1
            },
            "bf16": {
                "enabled": True
            },
            "steps_per_print": 1,
        }

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        # Verify tp_plan is auto-extracted and applied
        assert engine.autotp_size() == 2

        # Simple training step
        input_ids = torch.randint(0, 1000, (1, 16)).to(get_accelerator().current_device_name())
        outputs = engine(input_ids, labels=input_ids)
        engine.backward(outputs.loss)
        engine.step()

        assert not torch.isnan(outputs.loss)

    def test_qwen2_tp_plan_with_zero2(self):
        """Test Qwen2 model + tp_plan + ZeRO2"""
        skip_on_device()

        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            pytest.skip("transformers not installed")

        # Create small Qwen2 config
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen2-7B",
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        model = AutoModelForCausalLM.from_config(config)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 2
            },
            "bf16": {
                "enabled": True
            },
            "steps_per_print": 1,
        }

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2

        # Train for a few steps
        for _ in range(3):
            input_ids = torch.randint(0, 1000, (1, 16)).to(get_accelerator().current_device_name())
            outputs = engine(input_ids, labels=input_ids)
            engine.backward(outputs.loss)
            engine.step()

            assert not torch.isnan(outputs.loss)

    def test_custom_model_with_custom_tp_plan(self):
        """Test custom model + custom tp_plan"""
        skip_on_device()

        class CustomTransformerModel(torch.nn.Module):

            def __init__(self, hidden_size=64):
                super().__init__()
                self.config = type(
                    'Config', (), {
                        'base_model_tp_plan': {
                            'encoder.*.attention.query': 'colwise',
                            'encoder.*.attention.key': 'colwise',
                            'encoder.*.attention.value': 'colwise',
                            'encoder.*.attention.output': 'rowwise',
                            'encoder.*.ffn.intermediate': 'colwise',
                            'encoder.*.ffn.output': 'rowwise',
                        }
                    })()

                # Simple encoder layers
                self.encoder = torch.nn.ModuleList([
                    torch.nn.ModuleDict({
                        'attention':
                        torch.nn.ModuleDict({
                            'query': torch.nn.Linear(hidden_size, hidden_size),
                            'key': torch.nn.Linear(hidden_size, hidden_size),
                            'value': torch.nn.Linear(hidden_size, hidden_size),
                            'output': torch.nn.Linear(hidden_size, hidden_size),
                        }),
                        'ffn':
                        torch.nn.ModuleDict({
                            'intermediate': torch.nn.Linear(hidden_size, hidden_size * 4),
                            'output': torch.nn.Linear(hidden_size * 4, hidden_size),
                        })
                    }) for _ in range(2)
                ])

            def forward(self, x):
                for layer in self.encoder:
                    # Simplified attention
                    q = layer.attention.query(x)
                    k = layer.attention.key(x)
                    v = layer.attention.value(x)
                    attn_out = layer.attention.output(q + k + v)

                    # FFN
                    intermediate = torch.relu(layer.ffn.intermediate(attn_out))
                    x = layer.ffn.output(intermediate)
                return x

        model = CustomTransformerModel(hidden_size=64)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 0
            },
            "bf16": {
                "enabled": True
            },
        }

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2

        # Training step
        input_tensor = torch.randn(2, 4, 64).to(get_accelerator().current_device_name())
        output = engine(input_tensor)
        loss = output.mean()
        engine.backward(loss)
        engine.step()
