# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import inspect

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest
from transformers import OPTConfig, OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoderLayer


class TestHybridEngineOPTCache(DistributedTest):
    world_size = 2

    def test_native_generation_after_training(self):
        parameters = inspect.signature(OPTDecoderLayer.forward).parameters
        if not ({'cache_position', 'past_key_values'} & parameters.keys()):
            pytest.skip('Installed OPT uses the supported legacy cache contract')
        torch.manual_seed(8197)
        config = OPTConfig(vocab_size=64,
                           hidden_size=32,
                           ffn_dim=64,
                           num_hidden_layers=2,
                           num_attention_heads=4,
                           max_position_embeddings=128,
                           dropout=0.0,
                           attention_dropout=0.0,
                           pad_token_id=0,
                           bos_token_id=1,
                           eos_token_id=2)
        model = OPTForCausalLM(config).to(get_accelerator().current_device_name())
        oracle = OPTForCausalLM(config).to(get_accelerator().current_device_name())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=optimizer,
                                               config={
                                                   'train_micro_batch_size_per_gpu': 2,
                                                   'zero_optimization': {
                                                       'stage': 0
                                                   },
                                                   'hybrid_engine': {
                                                       'enabled': True,
                                                       'max_out_tokens': 128
                                                   },
                                                   'steps_per_print': 1000,
                                               })
        # Exercise real backward/update and repeated cache creation at two shapes.
        for step in range(10):
            tokens = (torch.arange(8 + step % 2 * 8, device=engine.device) % 60 + 3).repeat(2, 1)
            mask = torch.ones_like(tokens)
            engine.train()
            before = model.get_input_embeddings().weight.detach().clone()
            loss = engine(input_ids=tokens, attention_mask=mask, labels=tokens).loss
            assert torch.isfinite(loss)
            engine.backward(loss)
            engine.step()
            assert not torch.equal(before, model.get_input_embeddings().weight)
            oracle.load_state_dict(model.state_dict())
            oracle.eval()
            engine.eval()
            with torch.no_grad():
                expected = oracle.generate(tokens, attention_mask=mask, max_new_tokens=8, do_sample=False)
                actual = engine.module.generate(tokens, attention_mask=mask, max_new_tokens=8, do_sample=False)
            assert torch.equal(actual, expected)
