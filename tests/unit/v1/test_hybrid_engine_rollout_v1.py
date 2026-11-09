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


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_static_graph_matches_native_llama_generation(attn_implementation):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("requires graph capture")

    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1234)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=128)
    config._attn_implementation = attn_implementation
    device = torch.device(accelerator.device_name())
    model = LlamaForCausalLM(config).to(device).eval()
    assert model.config._attn_implementation == attn_implementation
    prompt_ids = torch.randint(1, 64, (4, 8), generator=torch.Generator().manual_seed(1)).to(device)
    attention_mask = torch.ones_like(prompt_ids)
    new_tokens = 16
    with torch.no_grad():
        expected = model.generate(prompt_ids,
                                  attention_mask=attention_mask,
                                  max_new_tokens=new_tokens,
                                  do_sample=False,
                                  eos_token_id=None,
                                  pad_token_id=0)

    # eos -1 never matches, so every row generates the full length.
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    output = rollout.generate(RolloutRequest(prompt_ids, attention_mask),
                              SamplingConfig(max_new_tokens=new_tokens, temperature=0))

    assert output.input_ids.tolist() == expected.tolist()


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


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("left_padding", [False, True])
def test_continuous_rollout_matches_native_llama_generation_on_accelerator(attn_implementation, left_padding,
                                                                           monkeypatch):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("Graph capture is required for native-model rollout coverage")
    if not accelerator.is_fp16_supported():
        pytest.skip("FP16 is required for native-model rollout coverage")

    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1234)
    config = LlamaConfig(vocab_size=32,
                         hidden_size=32,
                         intermediate_size=64,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=64,
                         pad_token_id=0,
                         eos_token_id=None)
    config._attn_implementation = attn_implementation
    device = torch.device(accelerator.device_name())
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    prompt_ids = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 8], [1, 9, 10, 11]], device=device)
    if left_padding:
        prompt_ids[0, :2] = 0
        prompt_ids[1, :1] = 0
    attention_mask = (prompt_ids != 0).long()
    request = RolloutRequest(prompt_ids, attention_mask)
    sampling = SamplingConfig(max_new_tokens=5, temperature=0, continuous_batch_size=2)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None)
    eager_rollout = HybridEngineRollout(SimpleNamespace(module=model), tokenizer)
    graph_rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                        tokenizer,
                                        cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    with torch.no_grad():
        reference = model.generate(input_ids=prompt_ids,
                                   attention_mask=attention_mask,
                                   max_new_tokens=sampling.max_new_tokens,
                                   do_sample=False)
    expected_attention = torch.cat((attention_mask, torch.ones_like(reference[:, 4:])), dim=1)
    expected_response_start = torch.full((3, ), 4, device=device, dtype=torch.long)

    original_capture_to_graph = accelerator.capture_to_graph
    capture_count = 0

    def counted_capture_to_graph(*args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        return original_capture_to_graph(*args, **kwargs)

    monkeypatch.setattr(accelerator, "capture_to_graph", counted_capture_to_graph)
    # Three requests and two slots force refill; the second call also exercises graph reuse.
    for rollout in (eager_rollout, graph_rollout, graph_rollout):
        output = rollout.generate(request, sampling)
        assert torch.equal(output.input_ids, reference)
        assert torch.equal(output.attention_mask, expected_attention)
        assert torch.equal(output.response_start_idx, expected_response_start)
    assert capture_count == 1


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
    # A random initialization gives almost uniform next-token distributions, under which a wrong temperature or
    # cutoff rarely changes the drawn token. Scaling the output head makes them peaked.
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


@pytest.mark.parametrize("use_graph_capture, continuous_batch_size", [(True, None), (True, 8), (False, 8)])
@pytest.mark.parametrize("temperature, top_k, top_p", [(1.0, 0, 1.0), (0.7, 5, 1.0), (1.0, 0, 0.8)])
def test_decode_sampling_matches_hf_generate(temperature, top_k, top_p, use_graph_capture, continuous_batch_size):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph decode coverage")

    device = torch.device(accelerator.device_name())
    model = _tiny_llama(device)
    new_tokens = 32
    # Every prompt gets its own continuous batching slot, so each step draws the tokens of all rows in one call, as
    # generate() does, and the same seed gives both the same random numbers.
    request = _random_request(batch_size=8, prompt_len=8, device=device)
    sampling = SamplingConfig(max_new_tokens=new_tokens,
                              temperature=temperature,
                              top_k=top_k,
                              top_p=top_p,
                              continuous_batch_size=continuous_batch_size)

    torch.manual_seed(2)
    output = _rollout(model, use_graph_capture).generate(request, sampling)

    torch.manual_seed(2)
    with torch.no_grad():
        expected = model.generate(request.prompt_ids,
                                  attention_mask=request.prompt_attention_mask,
                                  max_new_tokens=new_tokens,
                                  do_sample=True,
                                  temperature=temperature,
                                  top_k=top_k,
                                  top_p=top_p,
                                  eos_token_id=None,
                                  pad_token_id=0)

    assert output.input_ids.tolist() == expected.tolist()


@pytest.mark.parametrize("use_graph_capture, continuous_batch_size", [(True, None), (True, 8), (False, 8)])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_decode_repetition_penalty_matches_hf_generate(temperature, use_graph_capture, continuous_batch_size):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph decode coverage")

    device = torch.device(accelerator.device_name())
    model = _tiny_llama(device)
    new_tokens = 32
    request = _random_request(batch_size=8, prompt_len=8, device=device)
    sampling = SamplingConfig(max_new_tokens=new_tokens,
                              temperature=temperature,
                              top_k=0,
                              top_p=1.0,
                              continuous_batch_size=continuous_batch_size)
    hf_sampling = {
        "do_sample": True,
        "temperature": temperature,
        "top_k": 0,
        "top_p": 1.0
    } if temperature > 0 else {
        "do_sample": False
    }

    def hf_generate():
        torch.manual_seed(2)
        with torch.no_grad():
            return model.generate(request.prompt_ids,
                                  attention_mask=request.prompt_attention_mask,
                                  max_new_tokens=new_tokens,
                                  eos_token_id=None,
                                  pad_token_id=0,
                                  **hf_sampling).tolist()

    model.generation_config.repetition_penalty = 1.0
    unpenalized = hf_generate()
    model.generation_config.repetition_penalty = 1.3
    expected = hf_generate()
    assert expected != unpenalized

    torch.manual_seed(2)
    output = _rollout(model, use_graph_capture).generate(request, sampling)

    assert output.input_ids.tolist() == expected
