# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""End-to-end AutoTPRollout test: a real ZeRO engine, real AutoTP sharding and Hugging Face generate()."""

from types import SimpleNamespace

import pytest
import torch
import transformers

import deepspeed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout import RolloutConfig, RolloutRequest, SamplingConfig, build_rollout
from deepspeed.utils import safe_get_full_fp32_param
from unit.common import DistributedTest

PAD = 0
NEW_TOKENS = 6


def _tiny_config():
    # Every dimension AutoTP splits is several times tp_grain_size (64), so no rank gets an empty shard.
    return transformers.LlamaConfig(vocab_size=512,
                                    hidden_size=256,
                                    intermediate_size=512,
                                    num_hidden_layers=2,
                                    num_attention_heads=4,
                                    num_key_value_heads=4,
                                    max_position_embeddings=64,
                                    tie_word_embeddings=False,
                                    attn_implementation="eager")


def _greedy_by_unsharded_model(config, train_module, prompt_ids):
    """The independent oracle: the same masters in an ordinary model, one prompt at a time."""
    # The gather is a collective, so every rank builds this once however many prompts it has.
    masters = {name: safe_get_full_fp32_param(param) for name, param in train_module.named_parameters()}
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32).to(prompt_ids.device).eval()
    model.load_state_dict(masters)
    rows = []
    for row in prompt_ids.split(1):
        rows.append(
            model.generate(row,
                           attention_mask=(row != PAD).long(),
                           max_new_tokens=NEW_TOKENS,
                           do_sample=False,
                           eos_token_id=None,
                           pad_token_id=PAD))
    return torch.cat(rows)


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestAutoTPRollout(DistributedTest):
    world_size = 2

    def test_generation_follows_the_training_engine(self, zero_stage):
        device = get_accelerator().current_device_name()
        torch.manual_seed(0)
        config = _tiny_config()
        model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=torch.optim.SGD(model.parameters(), lr=0.5),
                                               config={
                                                   "train_micro_batch_size_per_gpu": 1,
                                                   "zero_allow_untested_optimizer": True,
                                                   "zero_optimization": {
                                                       "stage": zero_stage
                                                   },
                                               })
        tokenizer = SimpleNamespace(pad_token_id=PAD, eos_token_id=None)
        rollout = build_rollout(RolloutConfig(engine="autotp"), student_engine=engine, tokenizer=tokenizer)

        # The ranks ask for different numbers of prompts, of different widths.
        if dist.get_rank() == 0:
            prompt_ids = torch.tensor([[PAD, 5, 6], [7, 8, 9]], device=device)
        else:
            prompt_ids = torch.tensor([[3, 4, 5, 6, 7]], device=device)
        request = RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=(prompt_ids != PAD).long())
        greedy = SamplingConfig(max_new_tokens=NEW_TOKENS, temperature=0.0)

        batch = rollout.generate(request, greedy)
        assert batch.response_start_idx.tolist() == [prompt_ids.shape[1]] * prompt_ids.shape[0]
        assert torch.equal(batch.input_ids, _greedy_by_unsharded_model(config, engine.module, prompt_ids))

        # A stale inference copy would keep generating from the weights it had before this step.
        labels = torch.randint(1, config.vocab_size, (2, 8), device=device)
        engine.train()
        engine.backward(engine(input_ids=labels, labels=labels).loss)
        engine.step()
        rollout.sync_weights(step=1)
        batch = rollout.generate(request, greedy)
        assert torch.equal(batch.input_ids, _greedy_by_unsharded_model(config, engine.module, prompt_ids))

        # Sampling needs a shared seed: generate() itself raises if the ranks' tokens differ.
        sampling = SamplingConfig(max_new_tokens=NEW_TOKENS, temperature=1.0, n_samples_per_prompt=2)
        first = rollout.generate(request, sampling)
        second = rollout.generate(request, sampling)
        assert first.input_ids.shape == (prompt_ids.shape[0] * 2, prompt_ids.shape[1] + NEW_TOKENS)
        assert not torch.equal(first.input_ids, second.input_ids)
