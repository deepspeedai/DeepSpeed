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


def _tiny_llama(device):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=128)
    model = LlamaForCausalLM(config)
    # A random initialization gives almost uniform next-token distributions, under which sampling from the
    # wrong distribution is hard to detect. Scaling the output head makes them peaked.
    with torch.no_grad():
        model.lm_head.weight.mul_(30)
    return model.to(device).eval()


def _rollout(model, use_graph_capture):
    # eos -1 never matches, so every row generates the full length.
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=-1)
    return HybridEngineRollout(SimpleNamespace(module=model),
                               tokenizer,
                               cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph_capture))


def _random_request(batch_size, prompt_len, device):
    generator = torch.Generator().manual_seed(1)
    prompt_ids = torch.randint(1, 64, (batch_size, prompt_len), generator=generator)
    return RolloutRequest(prompt_ids.to(device), torch.ones_like(prompt_ids).to(device))


def _reference_log_probs(logits, temperature, top_k, top_p):
    """Log-probs of the distribution ``SamplingConfig`` describes, derived from the raw logits."""
    scores = logits.float() / temperature
    if top_k > 0:
        kth_score = scores.topk(top_k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < kth_score, float("-inf"))
    if top_p < 1.0:
        sorted_probs, order = scores.softmax(dim=-1).sort(dim=-1, descending=True)
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        # A token is kept while the more likely tokens hold less than top_p of the mass. The small margin keeps
        # tokens on the boundary, where the graph's incremental decode and this full forward can round apart.
        drop_sorted = mass_before >= top_p + 1e-4
        drop = drop_sorted.scatter(-1, order, drop_sorted)
        scores = scores.masked_fill(drop, float("-inf"))
    return scores.log_softmax(dim=-1)


# Plain graph decode, continuous batching with graph capture, and continuous batching without it.
@pytest.mark.parametrize("use_graph_capture, continuous_batch_size", [(True, None), (True, 8), (False, 8)])
@pytest.mark.parametrize("temperature, top_k, top_p", [(1.0, -1, 1.0), (0.7, 5, 1.0), (1.0, -1, 0.8)])
def test_decode_samples_from_the_configured_distribution(temperature, top_k, top_p, use_graph_capture,
                                                         continuous_batch_size):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph decode coverage")

    device = torch.device(accelerator.device_name())
    model = _tiny_llama(device)
    rollout = _rollout(model, use_graph_capture)
    prompt_len, new_tokens = 8, 32
    request = _random_request(batch_size=32, prompt_len=prompt_len, device=device)
    sampling = SamplingConfig(max_new_tokens=new_tokens,
                              temperature=temperature,
                              top_k=top_k,
                              top_p=top_p,
                              continuous_batch_size=continuous_batch_size)

    torch.manual_seed(2)
    output = rollout.generate(request, sampling)

    with torch.no_grad():
        logits = model(output.input_ids).logits
    # Positions prompt_len - 1 .. end - 2 predict the generated tokens.
    log_probs = _reference_log_probs(logits[:, prompt_len - 1:-1], temperature, top_k, top_p)
    tokens = output.input_ids[:, prompt_len:]
    token_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    assert torch.isfinite(token_log_probs).all(), "sampled a token outside the top-k / top-p support"

    # Tokens drawn from q satisfy E[log q(x)] = -H(q) and P[x = argmax q] = mean max q.
    probs = log_probs.exp()
    entropy = -(probs * log_probs.clamp(min=-1e30)).sum(dim=-1)
    gap = (token_log_probs + entropy).flatten()
    gap_z = gap.mean() / (gap.std() / gap.numel()**0.5)
    max_probs, argmax = probs.max(dim=-1)
    hits = (tokens == argmax).float()
    hit_z = (hits.mean() - max_probs.mean()) / ((max_probs * (1 - max_probs)).sum().sqrt() / hits.numel())
    assert abs(gap_z.item()) < 4, f"log-prob gap z-score {gap_z.item():.2f}"
    assert abs(hit_z.item()) < 4, f"argmax hit-rate z-score {hit_z.item():.2f}"
