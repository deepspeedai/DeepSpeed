# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import pytest
import deepspeed
from deepspeed.ops.op_builder import OpBuilder
from unit.common import DistributedTest

from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM)

rocm_version = OpBuilder.installed_rocm_version()
if rocm_version != (0, 0):
    pytest.skip("skip inference tests on rocm for now", allow_module_level=True)


@pytest.mark.seq_inference
@pytest.mark.parametrize("batch_size", [1, 2], ids=["bsz=1", "bsz=2"])
@pytest.mark.parametrize("model_name", ["huggyllama/llama-7b"])
class TestHybridEngineLlama(DistributedTest):
    world_size = 1

    def _generate(self, model, tokenizer, prompt):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        tokens = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
        for t in tokens:
            if torch.is_tensor(tokens[t]):
                tokens[t] = tokens[t].to(f'cuda:{local_rank}')
        #output = model.generate(**tokens, do_sample=False, max_length=100)
        output = model.generate(tokens.input_ids, do_sample=False, max_length=100)
        outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
        return outputs

    def get_model(self, model_name):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.dropout = 0.0
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
        # Make the model smaller so we can run it on a single GPU in CI
        _ = [model.model.layers.pop(-1) for _ in range(8)]
        model = model.half()
        model = model.to(f'cuda:{local_rank}')
        return model

    def get_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_prompt(self, batch_size):
        if batch_size == 1:
            prompt = ["Microsoft is in Washington"]
        elif batch_size == 2:
            prompt = ["DeepSpeed is", "Microsoft is in Washington"]
        else:
            raise NotImplementedError(f"batch_size {batch_size} not implemented")
        return prompt

    def test_correctness(self, batch_size, model_name):
        pytest.skip("skip test for now, will fix in follow-up PR")
        model = self.get_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        prompt = self.get_prompt(batch_size)

        base_out = self._generate(model, tokenizer, prompt)

        ds_config = {"train_batch_size": 1, "fp16": {"enabled": True}, "hybrid_engine": {"enabled": True}}
        model, *_ = deepspeed.initialize(model=model, config=ds_config)

        model.eval()
        ds1_out = self._generate(model, tokenizer, prompt)
        assert base_out == ds1_out, f"base_out: {base_out}, ds1_out: {ds1_out}"

        model.train()
        model.eval()
        ds2_out = self._generate(model, tokenizer, prompt)
        assert base_out == ds2_out

    def test_functionality(self, batch_size, model_name):
        model = self.get_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        prompt = self.get_prompt(batch_size)

        ds_config = {"train_batch_size": 1, "fp16": {"enabled": True}, "hybrid_engine": {"enabled": True}}
        model, *_ = deepspeed.initialize(model=model, config=ds_config)

        model.eval()
        ds1_out = self._generate(model, tokenizer, prompt)

        model.train()
        model.eval()
        ds2_out = self._generate(model, tokenizer, prompt)

        assert ds1_out == ds2_out, f"ds1_out: {ds1_out}, ds2_out: {ds2_out}"
