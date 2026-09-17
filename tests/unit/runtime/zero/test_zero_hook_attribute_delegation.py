# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest


class AttributeDelegatingWrapper(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.wrapped = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = super().__getattr__("wrapped")
            return getattr(wrapped, name)

    def forward(self, inputs):
        return self.wrapped(inputs)


class DelegatingWrapperModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.input = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output = AttributeDelegatingWrapper(torch.nn.Linear(hidden_dim, hidden_dim))

    def forward(self, inputs):
        return self.output(self.input(inputs)).sum()


class TestHookAttributeDelegation(DistributedTest):
    world_size = 1

    def test(self):
        config = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 0,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-3,
                    "torch_adam": True,
                },
            },
        }
        if get_accelerator().is_bf16_supported():
            config["bf16"] = {"enabled": True}
        else:
            config["fp16"] = {"enabled": True, "initial_scale_power": 8}

        hidden_dim = 4
        model = DelegatingWrapperModel(hidden_dim)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)

        wrapper = engine.module.output
        wrapped = wrapper.wrapped
        assert "pre_bwd_fn" in wrapper.__dict__
        assert "post_bwd_fn" in wrapper.__dict__
        assert wrapper.pre_bwd_fn is not wrapped.pre_bwd_fn
        assert wrapper.post_bwd_fn is not wrapped.post_bwd_fn

        for _ in range(2):
            inputs = torch.randn(1, hidden_dim, device=engine.device, dtype=engine.module.input.weight.dtype)
            loss = engine(inputs)
            engine.backward(loss)
            assert wrapper.__dict__["applied_pre_backward_ref_cnt"] == 0
            assert wrapped.__dict__["applied_pre_backward_ref_cnt"] == 0
            engine.step()
