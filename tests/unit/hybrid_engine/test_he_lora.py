# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import math
import torch
import torch.nn.functional as F
import pytest
import deepspeed
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.ops.op_builder import OpBuilder
from deepspeed.utils import safe_get_full_grad
import numpy.testing as npt
from unit.common import DistributedTest
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("This op had not been implemented on this system.", allow_module_level=True)

from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM)

rocm_version = OpBuilder.installed_rocm_version()
if rocm_version != (0, 0):
    pytest.skip("skip inference tests on rocm for now", allow_module_level=True)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1, lora_droppout=0):
    from deepspeed.compression.helper import recursive_getattr, recursive_setattr

    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and part_module_name in name:
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(module.weight, lora_dim, lora_scaling, lora_droppout,
                               module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


class LinearLayer_LoRA(torch.nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self, weight, lora_dim=0, lora_scaling=1, lora_droppout=0, bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError("You are training to use LoRA, whose reduced dim should be larger than 1")

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = torch.nn.Parameter(torch.zeros(
            columns, lora_dim))  # apply transpose so in forward we do not need to transpose again
        self.lora_left_weight = torch.nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = torch.nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = torch.nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_left_weight)

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias) + (
                self.lora_dropout(input) @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling


def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


@pytest.mark.seq_inference
@pytest.mark.parametrize("batch_size", [1], ids=["bsz=1"])
@pytest.mark.parametrize("zero_stage", [2, 3], ids=["zero_stage=2", "zero_stage=3"])
@pytest.mark.parametrize("model_name", ["EleutherAI/gpt-neo-125m", "facebook/opt-350m", "bigscience/bloom-560m"])
@pytest.mark.parametrize("offload_device", ["none", "cpu"])
class TestHybridEngineLoRA(DistributedTest):
    world_size = 1

    def get_model(self, model_name):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.dropout = 0.0
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
        model = model.half()
        model = model.to(f'cuda:{local_rank}')
        return model

    def get_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_train_sentences(self, batch_size):
        sentences = [
            r"\n\nHuman: I am trying to write a fairy tale. What is the most popular plot?\n\n"
            r"Assistant: The most popular plot might be a princess goes to a faraway land, falls in love",
            r"\n\nHuman: What flowers should I grow to attract bees?\n\nAssistant: The reason you want bees "
            r"in your garden is to attract pollinators and get more fruit or vegetable production."
        ]
        if batch_size <= 2:
            return sentences[:batch_size]
        else:
            raise NotImplementedError(f"batch_size {batch_size} not implemented")

    def test_lora(self, batch_size, model_name, zero_stage, offload_device):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        model = self.get_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        train_sentences = self.get_train_sentences(batch_size)

        # Inject LoRA
        model = convert_linear_layer_to_lora(model, "", 8)
        model = only_optimize_lora_parameters(model)

        ds_config = {
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.0,
                    "betas": [0.9, 0.95]
                }
            },
            "train_batch_size": batch_size,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 12
            },
            "hybrid_engine": {
                "enabled": True,
                "pin_parameters": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": offload_device
                }
            }
        }

        model, *_ = deepspeed.initialize(model=model, config=ds_config)

        # Verify gradient norm is larger than 0
        before_grad_update_layer0_params = [
            ele.detach().cpu().float().numpy() for ele in model.layer_params[0]
            if ele is not None and len(ele.shape) > 1
        ]

        model.train()
        batch = tokenizer(train_sentences, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        batch = to_device(batch, f'cuda:{local_rank}')
        batch["labels"] = batch["input_ids"]
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        model.backward(loss)

        grad_norm_dict = dict()
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                grad_norm_dict[name] = torch.linalg.norm(safe_get_full_grad(param))

        model.step()
        grad_norm = sum([ele.detach().cpu().numpy() for ele in grad_norm_dict.values()])
        assert grad_norm > 1E-5

        # Verify parameter remains the same
        after_grad_update_layer0_params = [
            ele.detach().cpu().float().numpy() for ele in model.layer_params[0]
            if ele is not None and len(ele.shape) > 1
        ]
        for lhs, rhs in zip(before_grad_update_layer0_params, after_grad_update_layer0_params):
            npt.assert_allclose(lhs, rhs, 1E-5, 1E-5)

        # Verify fuse will mutate layer_params
        model.eval()
        with GatheredParameters(model.parameters()):
            model.fuse_lora_weight()

        after_grad_update_layer0_params_lora_fused = [
            ele.detach().cpu().float().numpy() for ele in model.layer_params[0]
            if ele is not None and len(ele.shape) > 1
        ]

        for lhs, rhs in zip(before_grad_update_layer0_params, after_grad_update_layer0_params_lora_fused):
            with pytest.raises(AssertionError):
                npt.assert_allclose(lhs, rhs, 1E-5, 1E-5)

        with GatheredParameters(model.parameters()):
            model.unfuse_lora_weight()
