# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed.comm as dist
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.layers import LinearLayer
from deepspeed.runtime.tensor_parallel.config import _get_hf_tp_plan
from deepspeed.utils import groups
from unit.common import DistributedTest


def skip_on_device():
    if get_accelerator().device_name() == "xpu":
        pytest.skip("XPU requires a higher version for test")


class TestTPPlanRealHFModels(DistributedTest):
    """End-to-end tests using real HuggingFace models"""

    world_size = 2

    def test_qwen2_tp_plan_with_zero2(self):
        """Test an untied Qwen2 LM head with gathered column output and ZeRO2."""
        skip_on_device()

        try:
            from transformers import AutoModelForCausalLM, Qwen2Config
        except ImportError:
            pytest.skip("transformers not installed")

        # Construct locally so the test does not download model configuration or weights.
        config = Qwen2Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            tie_word_embeddings=False,
        )

        model = AutoModelForCausalLM.from_config(config)
        assert model.lm_head.weight is not model.model.embed_tokens.weight

        tp_plan = _get_hf_tp_plan(model)
        assert tp_plan["lm_head"] in ("colwise_rep", "colwise_gather_output")

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4
                }
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
        assert isinstance(model.lm_head, LinearLayer)
        assert model.lm_head.gather_output
        assert model.lm_head.weight.shape == (500, 128)
        assert model.model.embed_tokens.weight.shape == (1000, 128)

        # Train for a few steps
        for _ in range(3):
            input_ids = torch.randint(0, 1000, (1, 16)).to(get_accelerator().current_device_name())
            dist.broadcast(
                input_ids,
                src=groups.get_tensor_model_parallel_src_rank(),
                group=groups.get_tensor_model_parallel_group(),
            )
            outputs = engine(input_ids, labels=input_ids)
            assert outputs.logits.shape == (1, 16, 1000)
            engine.backward(outputs.loss)
            engine.step()

            assert not torch.isnan(outputs.loss)

    def test_qwen2_tp_plan_with_uneven_vocab(self):
        """Test an untied Qwen2 LM head with an uneven gathered vocabulary size."""
        skip_on_device()

        try:
            from transformers import AutoModelForCausalLM, Qwen2Config
        except ImportError:
            pytest.skip("transformers not installed")

        config = Qwen2Config(
            vocab_size=1001,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            tie_word_embeddings=False,
        )

        model = AutoModelForCausalLM.from_config(config)
        assert model.lm_head.weight is not model.model.embed_tokens.weight

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 0
            },
        }

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        tp_rank = groups.get_tensor_model_parallel_rank()
        output_partition_sizes = list(model.lm_head._partition_sizes)
        assert engine.autotp_size() == 2
        assert isinstance(model.lm_head, LinearLayer)
        assert model.lm_head.gather_output
        assert model.lm_head.weight.shape == (output_partition_sizes[tp_rank], config.hidden_size)
        assert model.model.embed_tokens.weight.shape == (config.vocab_size, config.hidden_size)

        input_ids = torch.randint(0, config.vocab_size, (1, 16)).to(get_accelerator().current_device_name())
        dist.broadcast(
            input_ids,
            src=groups.get_tensor_model_parallel_src_rank(),
            group=groups.get_tensor_model_parallel_group(),
        )
        outputs = engine(input_ids)
        assert outputs.logits.shape == (1, 16, config.vocab_size)

    def test_qwen2_tied_lm_head_falls_back_to_replicated(self):
        """Test that an actual Qwen2 Parameter tie remains replicated."""
        skip_on_device()

        try:
            from transformers import AutoModelForCausalLM, Qwen2Config
        except ImportError:
            pytest.skip("transformers not installed")

        config = Qwen2Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            tie_word_embeddings=True,
        )
        model = AutoModelForCausalLM.from_config(config)
        assert model.lm_head.weight is model.model.embed_tokens.weight

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 0
            },
        }

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2
        assert isinstance(model.model.layers[0].self_attn.q_proj, LinearLayer)
        assert isinstance(model.model.embed_tokens, torch.nn.Embedding)
        assert isinstance(model.lm_head, torch.nn.Linear)
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert model.lm_head.weight.shape == (1000, 128)

        input_ids = torch.randint(0, 1000, (1, 16)).to(get_accelerator().current_device_name())
        dist.broadcast(
            input_ids,
            src=groups.get_tensor_model_parallel_src_rank(),
            group=groups.get_tensor_model_parallel_group(),
        )
        outputs = engine(input_ids)
        assert outputs.logits.shape == (1, 16, 1000)

    def test_custom_model_with_custom_tp_plan(self):
        """Test custom model + custom tp_plan"""
        skip_on_device()

        class CustomTransformerModel(torch.nn.Module):

            def __init__(self, hidden_size=64):
                super().__init__()
                self.config = type(
                    "Config",
                    (),
                    {
                        "base_model_tp_plan": {
                            "encoder.*.attention.query": "colwise",
                            "encoder.*.attention.key": "colwise",
                            "encoder.*.attention.value": "colwise",
                            "encoder.*.attention.output": "rowwise",
                            "encoder.*.ffn.intermediate": "colwise",
                            "encoder.*.ffn.output": "rowwise",
                        }
                    },
                )()

                # Simple encoder layers
                self.encoder = torch.nn.ModuleList([
                    torch.nn.ModuleDict({
                        "attention":
                        torch.nn.ModuleDict({
                            "query": torch.nn.Linear(hidden_size, hidden_size),
                            "key": torch.nn.Linear(hidden_size, hidden_size),
                            "value": torch.nn.Linear(hidden_size, hidden_size),
                            "output": torch.nn.Linear(hidden_size, hidden_size),
                        }),
                        "ffn":
                        torch.nn.ModuleDict({
                            "intermediate": torch.nn.Linear(hidden_size, hidden_size * 4),
                            "output": torch.nn.Linear(hidden_size * 4, hidden_size),
                        }),
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
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4
                }
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
        input_tensor = torch.randn(2, 4, 64, dtype=torch.bfloat16).to(get_accelerator().current_device_name())
        dist.broadcast(
            input_tensor,
            src=groups.get_tensor_model_parallel_src_rank(),
            group=groups.get_tensor_model_parallel_group(),
        )
        output = engine(input_tensor)
        loss = output.mean()
        engine.backward(loss)
        engine.step()


def build_qwen3_config(num_key_value_heads):
    from transformers import Qwen3Config

    # Construct locally so the test does not download model configuration or weights.
    return Qwen3Config(
        vocab_size=1000,
        hidden_size=1024,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=num_key_value_heads,
        head_dim=128,
        max_position_embeddings=128,
    )


def build_autotp_config(autotp_size):
    return {
        "train_micro_batch_size_per_gpu": 1,
        "tensor_parallel": {
            "autotp_size": autotp_size
        },
        "zero_optimization": {
            "stage": 0
        },
    }


class TestQwen3UnevenTPPlan(DistributedTest):
    """Regression coverage for Qwen3 GQA with a non-divisible TP width."""

    world_size = 3

    def test_qwen3_tp3_keeps_attention_heads_aligned(self):
        skip_on_device()

        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            pytest.skip("transformers not installed")

        config = build_qwen3_config(num_key_value_heads=8)
        model = AutoModelForCausalLM.from_config(config)

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_autotp_config(3))

        # 8 KV heads over 3 ranks splits as 3/3/2 whole heads of 128 elements each.
        tp_rank = groups.get_tensor_model_parallel_rank()
        expected_q_sizes = (768, 768, 512)
        expected_kv_sizes = (384, 384, 256)
        attention = model.model.layers[0].self_attn
        assert attention.q_proj.weight.shape[0] == expected_q_sizes[tp_rank]
        assert attention.k_proj.weight.shape[0] == expected_kv_sizes[tp_rank]
        assert attention.v_proj.weight.shape[0] == expected_kv_sizes[tp_rank]

        input_ids = torch.randint(0, config.vocab_size, (1, 8), device=get_accelerator().current_device_name())
        dist.broadcast(
            input_ids,
            src=groups.get_tensor_model_parallel_src_rank(),
            group=groups.get_tensor_model_parallel_group(),
        )
        outputs = engine(input_ids=input_ids, labels=input_ids)
        assert outputs.logits.shape == (1, 8, config.vocab_size)

    def test_autotp_size_above_kv_head_count_leaves_trailing_ranks_empty(self):
        """Ranks beyond the KV head count hold no attention shard but still compute correctly."""
        skip_on_device()

        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            pytest.skip("transformers not installed")

        config = build_qwen3_config(num_key_value_heads=2)
        model = AutoModelForCausalLM.from_config(config)

        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_autotp_config(3))

        # 2 KV heads over 3 ranks: the third rank gets nothing, and the row-parallel
        # all-reduce sums its empty contribution as zero.
        tp_rank = groups.get_tensor_model_parallel_rank()
        expected_q_sizes = (1024, 1024, 0)
        expected_kv_sizes = (128, 128, 0)
        attention = model.model.layers[0].self_attn
        assert attention.q_proj.weight.shape[0] == expected_q_sizes[tp_rank]
        assert attention.k_proj.weight.shape[0] == expected_kv_sizes[tp_rank]

        input_ids = torch.randint(0, config.vocab_size, (1, 8), device=get_accelerator().current_device_name())
        dist.broadcast(
            input_ids,
            src=groups.get_tensor_model_parallel_src_rank(),
            group=groups.get_tensor_model_parallel_group(),
        )
        outputs = engine(input_ids=input_ids, labels=input_ids)
        assert outputs.logits.shape == (1, 8, config.vocab_size)
