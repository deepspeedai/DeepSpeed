# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
from unit.common import DistributedTest

import deepspeed.comm as dist
from deepspeed.linear import LoRAConfig, init_lora
from deepspeed.linear.optimized_linear import LoRAOptimizedLinear
from unit.simple_model import random_dataloader, SimpleModel

try:
    import transformers
except ImportError:
    transformers = None

if transformers is None:
    pytest.skip("transformers is required for this test", allow_module_level=True)


def injection_assert(model):
    # pick out random linear that should have been replaced and initialized
    q_proj = model.model.layers[1].self_attn.q_proj

    assert isinstance(q_proj, LoRAOptimizedLinear), "injection did not happen"
    assert q_proj._initialized, "lora was not initialized properly"
    assert isinstance(q_proj.lora_weight_1, torch.nn.Linear)
    assert isinstance(q_proj.lora_weight_2, torch.nn.Linear)


class TestEngine(DistributedTest):
    world_size = 2

    def test_model(self):
        lora_config = LoRAConfig(lora_r=16, lora_alpha=16, base_weight_sharding=2)
        quant_config = None
        hidden_dim = 64
        nlayers = 4

        with deepspeed.linear.Init(lora_config=lora_config, quant_config=quant_config):
            model = SimpleModel(hidden_dim=hidden_dim, nlayers=nlayers)

        init_lora(model)

        model_norms = [model.linears[i].weight.norm().item() for i in range(nlayers)]

        ds_config = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "bf16": {
                "enabled": True
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": 1
            }
        }
        model, *_ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())

        engine_norms = [model.module.linears[i].weight.norm().item() for i in range(nlayers)]

        # Ensure that sharded weights are not broadcast during engine init
        assert engine_norms == model_norms, f"{dist.get_rank()=} base weight norms are not the same after engine init, {engine_norms=} != {model_norms=}"

        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestInitTransformers(DistributedTest):
    world_size = 2

    def test_pretrained_init(self):
        lora_config = LoRAConfig(lora_r=16, lora_alpha=16, base_weight_sharding=2)
        quant_config = None

        with deepspeed.linear.Init(lora_config=lora_config, quant_config=quant_config):
            model = transformers.AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-Llama-3")

        injection_assert(model)

    def test_config_init(self):
        lora_config = LoRAConfig(lora_r=16, lora_alpha=16, base_weight_sharding=2)
        quant_config = None

        config = transformers.AutoConfig.from_pretrained("llamafactory/tiny-random-Llama-3")

        with deepspeed.linear.Init(lora_config=lora_config, quant_config=quant_config):
            model = transformers.AutoModelForCausalLM.from_config(config)

        injection_assert(model)
