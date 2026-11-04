# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Accelerator-backed v1 HybridEngineRollout tests."""

import os
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig


class _LocalKVModel(torch.nn.Module):
    """Emits fewer KV heads than its global config, like an AutoTP shard."""
    _supports_cache_class = True

    def __init__(self, local_heads):
        super().__init__()
        self.local_heads = local_heads
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = PretrainedConfig(max_position_embeddings=32,
                                       num_hidden_layers=1,
                                       num_attention_heads=4,
                                       num_key_value_heads=4,
                                       hidden_size=4,
                                       head_dim=1)

    def forward(self, input_ids, attention_mask, past_key_values=None, cache_position=None, **kwargs):
        heads = torch.arange(1, self.local_heads + 1, device=input_ids.device)
        states = input_ids[:, None, :, None].float() * heads[None, :, None, None]
        _, values = past_key_values.update(states,
                                           states,
                                           layer_idx=0,
                                           cache_kwargs={"cache_position": cache_position})
        if attention_mask.dim() == 4:
            valid = attention_mask[:, 0, 0, :] == 0
        else:
            valid = torch.zeros((values.shape[0], values.shape[2]), dtype=torch.bool, device=input_ids.device)
            valid[:, :attention_mask.shape[1]] = attention_mask.bool()
        cache_sums = (values * valid[:, None, :, None]).sum(dim=(1, 2, 3)).long()
        next_tokens = cache_sums.remainder(29) + 3
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 32), device=input_ids.device)
        logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
        return SimpleNamespace(logits=logits)


def _local_kv_tokens(prompts, new_tokens, local_heads):
    head_sum = sum(range(1, local_heads + 1))
    expected = []
    for prompt in prompts:
        tokens = list(prompt)
        for _ in range(new_tokens):
            tokens.append(sum(tokens) * head_sum % 29 + 3)
        expected.append(tokens)
    return expected


def _count_replays(monkeypatch, accelerator):
    counter = {"replays": 0}
    original_replay = accelerator.replay_graph

    def replay(graph):
        counter["replays"] += 1
        return original_replay(graph)

    monkeypatch.setattr(accelerator, "replay_graph", replay)
    return counter


@pytest.mark.parametrize("local_heads", [4, 2, 1])
def test_static_graph_uses_prefill_kv_shape(local_heads, monkeypatch):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("requires graph capture")

    counter = _count_replays(monkeypatch, accelerator)
    device = torch.device(accelerator.device_name())
    model = _LocalKVModel(local_heads).to(device).eval()
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    prompts = [[1, 3, 4], [3, 4, 5]]
    prompt_ids = torch.tensor(prompts, device=device)
    new_tokens = 4
    output = rollout.generate(RolloutRequest(prompt_ids, torch.ones_like(prompt_ids)),
                              SamplingConfig(max_new_tokens=new_tokens, temperature=0))

    assert output.input_ids.tolist() == _local_kv_tokens(prompts, new_tokens, local_heads)
    assert counter["replays"] == new_tokens - 1
    assert model.config.num_key_value_heads == 4


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("local_heads", [4, 2, 1])
def test_continuous_generation_uses_local_kv_shape(local_heads, use_graph, monkeypatch):
    accelerator = get_accelerator()
    if use_graph and (not accelerator.is_available() or accelerator.create_graph() is None):
        pytest.skip("requires graph capture")

    counter = _count_replays(monkeypatch, accelerator)
    device = torch.device(accelerator.device_name())
    model = _LocalKVModel(local_heads).to(device).eval()
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph))
    # Three requests on two slots force a refill into an already allocated cache row.
    prompts = [[1, 3, 4], [3, 4, 5], [2, 2, 6]]
    prompt_ids = torch.tensor(prompts, device=device)
    new_tokens = 4
    output = rollout.generate(RolloutRequest(prompt_ids, torch.ones_like(prompt_ids)),
                              SamplingConfig(max_new_tokens=new_tokens, temperature=0, continuous_batch_size=2))

    assert output.input_ids.tolist() == _local_kv_tokens(prompts, new_tokens, local_heads)
    decode_steps = rollout.get_last_continuous_stats()["decode_steps"]
    assert counter["replays"] == (decode_steps if use_graph else 0)
    assert decode_steps > 0
    assert model.config.num_key_value_heads == 4


def test_continuous_generation_profile_on_accelerator():
    accelerator = get_accelerator()
    if not accelerator.is_available():
        pytest.skip("An accelerator is required for asynchronous profiling coverage")

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = PretrainedConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, values = past_key_values.update(states, states, layer_idx=0, cache_kwargs=kwargs)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16), device=input_ids.device)
            logits[..., 7] = 1
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    device = torch.device(accelerator.device_name())
    model = CacheClassModel().to(device)
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(enable_profiling=True),
    )
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4]], device=device),
        torch.ones((2, 3), dtype=torch.long, device=device),
    )

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=1))

    profile = rollout.get_last_profile()
    assert output.input_ids[:, 3:].cpu().tolist() == [[7, 7], [7, 7]]
    assert profile["num_prefill_forwards"] == 2
    assert profile["num_decode_forwards"] == 2
    assert profile["num_generated_tokens"] == 4
    assert profile["active_batch_size"] == 1
    assert profile["continuous_batch_size"] == 1
    assert profile["total_ms"] >= profile["generation_ms"]


def test_continuous_graph_generation_refills_a_fixed_slot_on_cuda():
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph rollout coverage")

    class CacheConfig(SimpleNamespace):

        def get_text_config(self, **_kwargs):
            return self

        @property
        def per_layer_config(self):
            return [self]

    class CacheClassModel(torch.nn.Module):
        _supports_cache_class = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = CacheConfig(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )

        def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **kwargs):
            states = input_ids[:, None, :, None].to(dtype=torch.float32)
            _, values = past_key_values.update(states, states, layer_idx=0, cache_kwargs=kwargs)
            cache_sums = values[:, 0].sum(dim=(1, 2)).long()
            if input_ids.shape[1] > 1:
                next_tokens = torch.where(cache_sums == 6, 2, 7)
            else:
                next_tokens = torch.where((cache_sums == 14) | (cache_sums == 16), 2, 8)
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 16), device=input_ids.device)
            logits.scatter_(2, next_tokens[:, None, None].expand(-1, input_ids.shape[1], 1), 1)
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    device = torch.device(accelerator.device_name())
    graph_rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel().to(device)),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
        cfg=HybridEngineRolloutConfig(use_graph_capture=True),
    )
    eager_rollout = HybridEngineRollout(
        SimpleNamespace(module=CacheClassModel().to(device)),
        SimpleNamespace(pad_token_id=0, eos_token_id=2),
    )
    request = RolloutRequest(
        torch.tensor([[1, 2, 3], [1, 2, 4], [3, 3, 3]], device=device),
        torch.ones((3, 3), dtype=torch.long, device=device),
    )

    sampling = SamplingConfig(max_new_tokens=2, temperature=0, continuous_batch_size=2)
    eager_output = eager_rollout.generate(request, sampling)
    graph_output = graph_rollout.generate(request, sampling)

    assert torch.equal(graph_output.input_ids, eager_output.input_ids)
    assert torch.equal(graph_output.attention_mask, eager_output.attention_mask)
    assert graph_output.input_ids[:, 3:].cpu().tolist() == [[2, 0], [7, 2], [7, 2]]
    stats = graph_rollout.get_last_continuous_stats()
    assert stats["cache_capacity"] == 5
    assert stats["peak_cache_length"] == 4
    assert stats["decode_steps"] == 2
    assert stats["trim_count"] == 0
    assert stats["trim_frequency"] == 0.0


def test_continuous_graph_matches_pretrained_model_when_enabled():
    model_name = os.getenv("DEEPSPEED_PRETRAINED_ROLLOUT_MODEL")
    if not model_name:
        pytest.skip("set DEEPSPEED_PRETRAINED_ROLLOUT_MODEL to run pretrained rollout coverage")

    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for pretrained CUDA graph rollout coverage")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(accelerator.device_name())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eager_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    graph_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    graph_model.load_state_dict(eager_model.state_dict())

    sampling = SamplingConfig(max_new_tokens=32, temperature=0, continuous_batch_size=1)
    eager_rollout = HybridEngineRollout(SimpleNamespace(module=eager_model), tokenizer)
    graph_rollout = HybridEngineRollout(
        SimpleNamespace(module=graph_model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(use_graph_capture=True),
    )
    original_create_graph = accelerator.create_graph
    capture_count = [0]

    def counted_create_graph():
        capture_count[0] += 1
        return original_create_graph()

    accelerator.create_graph = counted_create_graph
    try:
        prompts = (
            "CUDA Graph replay avoids repeated Python launch overhead during autoregressive decoding.",
            "DeepSpeed continuous batching keeps physical cache slots stable across request refills.",
        )
        for seed, prompt in zip((1234, 1235), prompts):
            torch.manual_seed(seed)
            encoded = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
            request = RolloutRequest(encoded.input_ids.to(device), encoded.attention_mask.to(device))
            eager_output = eager_rollout.generate(request, sampling)
            graph_output = graph_rollout.generate(request, sampling)

            assert torch.equal(graph_output.input_ids, eager_output.input_ids)
            assert torch.equal(graph_output.attention_mask, eager_output.attention_mask)
    finally:
        accelerator.create_graph = original_create_graph

    assert capture_count[0] == 1
