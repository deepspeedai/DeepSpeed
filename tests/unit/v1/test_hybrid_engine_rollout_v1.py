# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Accelerator-backed v1 HybridEngineRollout tests."""

import asyncio
import copy
import gc
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.config import HybridEngineConfig
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout import Tool, ToolSet
from deepspeed.runtime.rollout.base import RolloutRecord, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from deepspeed.utils.static_cache import DeepSpeedStaticCache
from unit.common import DistributedTest


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
    rollout = HybridEngineRollout(_engine_for(model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1, decode=lambda ids, **_kwargs: ""),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    prompts = [[1, 3, 4], [3, 4, 5]]
    prompt_ids = torch.tensor(prompts, device=device)
    new_tokens = 4
    output = rollout.generate(_records(prompt_ids, torch.ones_like(prompt_ids)),
                              SamplingConfig(max_new_tokens=new_tokens, temperature=0))

    assert [sample.tokens for sample in output.samples] == _local_kv_tokens(prompts, new_tokens, local_heads)
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
    rollout = HybridEngineRollout(_engine_for(model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1, decode=lambda ids, **_kwargs: ""),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph, continuous_batch_size=2))
    # Three requests on two slots force a refill into an already allocated cache row.
    prompts = [[1, 3, 4], [3, 4, 5], [2, 2, 6]]
    prompt_ids = torch.tensor(prompts, device=device)
    new_tokens = 4
    output = rollout.generate(_records(prompt_ids, torch.ones_like(prompt_ids)),
                              SamplingConfig(max_new_tokens=new_tokens, temperature=0))

    assert [sample.tokens for sample in output.samples] == _local_kv_tokens(prompts, new_tokens, local_heads)
    decode_steps = rollout.get_last_continuous_stats()["decode_steps"]
    assert counter["replays"] == (decode_steps if use_graph else 0)
    assert decode_steps > 0
    assert model.config.num_key_value_heads == 4


class TestAutoTPContinuousRollout(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("use_graph", [False, True])
    def test_matches_generate_on_sharded_llama(self, use_graph):
        accelerator = get_accelerator()
        if use_graph and accelerator.create_graph() is None:
            pytest.skip("requires graph capture")
        # Eager continuous decode passes a 2-D mask, so newer Transformers query the cache offset.
        hf_needs_offset = hasattr(Cache, "get_query_offset")
        cache_has_offset = hasattr(DeepSpeedStaticCache, "get_query_offset")
        if not use_graph and hf_needs_offset and not cache_has_offset:
            pytest.skip("DeepSpeedStaticCache lacks get_query_offset for this Transformers version")

        from transformers import AutoModelForCausalLM, LlamaConfig

        config = LlamaConfig(vocab_size=128,
                             hidden_size=64,
                             intermediate_size=128,
                             num_hidden_layers=2,
                             num_attention_heads=4,
                             num_key_value_heads=4,
                             max_position_embeddings=64)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 0
            },
            "hybrid_engine": {
                "max_out_tokens": 64
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
        model.eval()
        head_dim = config.hidden_size // config.num_attention_heads
        assert model.model.layers[0].self_attn.k_proj.weight.shape[0] == config.num_key_value_heads * head_dim // 2

        device = torch.device(accelerator.current_device_name())
        prompt_ids = torch.tensor([[5, 17, 42, 9], [33, 8, 71, 2], [64, 11, 19, 90]], device=device)
        prompt_attn = torch.ones_like(prompt_ids)
        new_tokens = 6
        with torch.no_grad():
            expected = model.generate(prompt_ids,
                                      attention_mask=prompt_attn,
                                      max_new_tokens=new_tokens,
                                      do_sample=False,
                                      eos_token_id=None,
                                      pad_token_id=0)

        rollout = HybridEngineRollout(engine,
                                      SimpleNamespace(pad_token_id=0,
                                                      eos_token_id=-1,
                                                      decode=lambda ids, **_kwargs: ""),
                                      cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph,
                                                                    continuous_batch_size=2))
        try:
            # Three requests on two slots force a refill into an already allocated cache row.
            output = rollout.generate(_records(prompt_ids, prompt_attn),
                                      SamplingConfig(max_new_tokens=new_tokens, temperature=0))
            assert [sample.tokens for sample in output.samples] == expected.tolist()
            assert model.config.num_key_value_heads == 4
        finally:
            # A live graph holding NCCL work hangs process-group teardown.
            del rollout
            gc.collect()
            accelerator.synchronize()


def _engine_for(module, max_out_tokens=None):
    """A stand-in for a DeepSpeed engine wrapping ``module``, without ZeRO-3 or tensor parallelism.

    ``max_out_tokens`` defaults to the model's position limit, so it caps nothing a test generates.
    """
    if max_out_tokens is None:
        max_out_tokens = getattr(getattr(module, "config", None), "max_position_embeddings", None) or 1024
    return SimpleNamespace(module=module,
                           zero_optimization_partition_weights=lambda: False,
                           autotp_size=lambda: 1,
                           _config=SimpleNamespace(hybrid_engine=HybridEngineConfig(max_out_tokens=max_out_tokens)))


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
        _engine_for(model),
        SimpleNamespace(pad_token_id=0, eos_token_id=2, decode=lambda ids, **_kwargs: ""),
        cfg=HybridEngineRolloutConfig(continuous_batch_size=1, enable_profiling=True),
    )
    request = RolloutRequest(samples=[
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1]),
        RolloutRecord(tokens=[1, 2, 4], attention_mask=[1, 1, 1]),
    ])

    output = rollout.generate(request, SamplingConfig(max_new_tokens=2, temperature=0))

    profile = rollout.get_last_profile()
    assert [sample.tokens[3:] for sample in output.samples] == [[7, 7], [7, 7]]
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
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, decode=lambda ids, **_kwargs: "")
    graph_rollout = HybridEngineRollout(
        _engine_for(CacheClassModel().to(device)),
        tokenizer,
        cfg=HybridEngineRolloutConfig(use_graph_capture=True, continuous_batch_size=2),
    )
    eager_rollout = HybridEngineRollout(_engine_for(CacheClassModel().to(device)),
                                        tokenizer,
                                        cfg=HybridEngineRolloutConfig(continuous_batch_size=2))

    def make_request():
        return RolloutRequest(samples=[
            RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1]),
            RolloutRecord(tokens=[1, 2, 4], attention_mask=[1, 1, 1]),
            RolloutRecord(tokens=[3, 3, 3], attention_mask=[1, 1, 1]),
        ])

    sampling = SamplingConfig(max_new_tokens=2, temperature=0)
    eager_output = eager_rollout.generate(make_request(), sampling)
    graph_output = graph_rollout.generate(make_request(), sampling)

    assert [sample.tokens for sample in graph_output.samples] == [sample.tokens for sample in eager_output.samples]
    assert [sample.attention_mask
            for sample in graph_output.samples] == [sample.attention_mask for sample in eager_output.samples]
    # A response stops at its EOS (token 2).
    assert [sample.tokens[3:] for sample in graph_output.samples] == [[2], [7, 2], [7, 2]]
    stats = graph_rollout.get_last_continuous_stats()
    # 2 cache rows x 5 positions: fp32 keys and values with one head of width 1, int64 attention mask,
    # and one int64 write position per row. Both decode paths report the same cache footprint.
    expected_cache_bytes = 2 * 5 * 4 * 2 + 2 * 5 * 8 + 2 * 8
    assert stats["cache_memory_bytes"] == expected_cache_bytes
    assert eager_rollout.get_last_continuous_stats()["cache_memory_bytes"] == expected_cache_bytes
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
    request = _records(prompt_ids, attention_mask)
    sampling = SamplingConfig(max_new_tokens=5, temperature=0)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None, decode=lambda ids, **_kwargs: "")
    eager_rollout = HybridEngineRollout(_engine_for(model),
                                        tokenizer,
                                        cfg=HybridEngineRolloutConfig(continuous_batch_size=2))
    graph_rollout = HybridEngineRollout(_engine_for(model),
                                        tokenizer,
                                        cfg=HybridEngineRolloutConfig(use_graph_capture=True, continuous_batch_size=2))
    with torch.no_grad():
        reference = model.generate(input_ids=prompt_ids,
                                   attention_mask=attention_mask,
                                   max_new_tokens=sampling.max_new_tokens,
                                   do_sample=False)
    expected_attention = torch.cat((attention_mask, torch.ones_like(reference[:, 4:])), dim=1).tolist()

    original_capture_to_graph = accelerator.capture_to_graph
    capture_count = 0

    def counted_capture_to_graph(*args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        return original_capture_to_graph(*args, **kwargs)

    monkeypatch.setattr(accelerator, "capture_to_graph", counted_capture_to_graph)
    # Three requests and two slots force refill; the second call also exercises graph reuse.
    for rollout in (eager_rollout, graph_rollout, graph_rollout):
        output = rollout.generate(request, sampling).samples
        assert [sample.tokens for sample in output] == reference.tolist()
        assert [sample.attention_mask for sample in output] == expected_attention
        assert all(sample.response_length == sampling.max_new_tokens for sample in output)
    assert capture_count == 1


def _records(prompt_ids, attention_mask):
    samples = [
        RolloutRecord(tokens=row, attention_mask=mask)
        for row, mask in zip(prompt_ids.tolist(), attention_mask.tolist())
    ]
    return RolloutRequest(samples=samples)


def _peaked_llama(device):
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


def _decode_rollout(model, use_graph_capture, continuous_batch_size=None):
    # eos -1 never matches, so every row generates the full length.
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=-1, decode=lambda ids, **_kwargs: "")
    return HybridEngineRollout(_engine_for(model),
                               tokenizer,
                               cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph_capture,
                                                             continuous_batch_size=continuous_batch_size))


def _random_prompts(device):
    prompt_ids = torch.randint(1, 64, (8, 8), generator=torch.Generator().manual_seed(1)).to(device)
    return prompt_ids, torch.ones_like(prompt_ids)


@pytest.mark.parametrize("use_graph_capture, continuous_batch_size", [(True, None), (True, 8), (False, 8)])
@pytest.mark.parametrize("temperature, top_k, top_p, repetition_penalty", [(1.0, 0, 1.0, 1.0), (0.7, 5, 1.0, 1.0),
                                                                           (1.0, 0, 0.8, 1.0), (0.7, 0, 1.0, 1.3),
                                                                           (0.0, 0, 1.0, 1.3)])
def test_decode_paths_match_hf_generate(temperature, top_k, top_p, repetition_penalty, use_graph_capture,
                                        continuous_batch_size):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("requires graph capture")

    device = torch.device(accelerator.device_name())
    model = _peaked_llama(device)
    new_tokens = 32
    prompt_ids, attention_mask = _random_prompts(device)
    # Every prompt gets its own continuous batching slot, so each step draws the tokens of all rows in one call, as
    # generate() does, and the same seed gives both the same random numbers.
    sampling = SamplingConfig(max_new_tokens=new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    model.generation_config.repetition_penalty = repetition_penalty
    hf_sampling = {"do_sample": True, "temperature": temperature, "top_k": top_k, "top_p": top_p}
    if temperature <= 0.0:
        hf_sampling = {"do_sample": False}

    torch.manual_seed(2)
    output = _decode_rollout(model, use_graph_capture,
                             continuous_batch_size).generate(_records(prompt_ids, attention_mask), sampling)

    torch.manual_seed(2)
    with torch.no_grad():
        expected = model.generate(prompt_ids,
                                  attention_mask=attention_mask,
                                  max_new_tokens=new_tokens,
                                  eos_token_id=None,
                                  pad_token_id=0,
                                  **hf_sampling)

    assert [sample.tokens for sample in output.samples] == expected.tolist()


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_static_graph_matches_native_llama_generation(attn_implementation):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("requires graph capture")

    device = torch.device(accelerator.device_name())
    model = _peaked_llama(device)
    model.config._attn_implementation = attn_implementation
    prompt_ids, attention_mask = _random_prompts(device)
    new_tokens = 16
    with torch.no_grad():
        expected = model.generate(prompt_ids,
                                  attention_mask=attention_mask,
                                  max_new_tokens=new_tokens,
                                  do_sample=False,
                                  eos_token_id=None,
                                  pad_token_id=0)

    output = _decode_rollout(model, True).generate(_records(prompt_ids, attention_mask),
                                                   SamplingConfig(max_new_tokens=new_tokens, temperature=0))

    assert [sample.tokens for sample in output.samples] == expected.tolist()


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

    sampling = SamplingConfig(max_new_tokens=32, temperature=0)
    eager_rollout = HybridEngineRollout(_engine_for(eager_model),
                                        tokenizer,
                                        cfg=HybridEngineRolloutConfig(continuous_batch_size=1))
    graph_rollout = HybridEngineRollout(
        _engine_for(graph_model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(use_graph_capture=True, continuous_batch_size=1),
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
            encoded = tokenizer(prompt, padding="max_length", max_length=32, truncation=True)

            def make_request():
                return RolloutRequest(
                    samples=[RolloutRecord(tokens=encoded["input_ids"], attention_mask=encoded["attention_mask"])])

            eager_output = eager_rollout.generate(make_request(), sampling)
            graph_output = graph_rollout.generate(make_request(), sampling)

            assert graph_output.samples[0].tokens == eager_output.samples[0].tokens
            assert graph_output.samples[0].attention_mask == eager_output.samples[0].attention_mask
    finally:
        accelerator.create_graph = original_create_graph

    assert capture_count[0] == 1


class _OkTool(Tool):
    name = "ok"
    description = "Returns ok."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return "ok"


def _tiny_qwen3(device):
    from transformers import AutoModelForCausalLM, Qwen3Config

    config = Qwen3Config(vocab_size=64,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         head_dim=16,
                         max_position_embeddings=128)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(device).eval()
    model.generation_config.repetition_penalty = 1.0
    return model


@pytest.mark.parametrize("use_graph_capture", [False, True])
def test_continuous_tool_output_that_would_pass_max_out_tokens_ends_the_trajectory(use_graph_capture):
    accelerator = get_accelerator()
    if not accelerator.is_available():
        pytest.skip("requires an accelerator")
    if use_graph_capture and not accelerator.is_graph_capture_supported():
        pytest.skip("requires graph capture")
    device = accelerator.current_device_name()
    model = _tiny_qwen3(device)
    prompt = [5, 6, 7, 8, 9, 10]
    with torch.no_grad():
        first = model.generate(torch.tensor([prompt], device=device),
                               max_new_tokens=1,
                               do_sample=False,
                               pad_token_id=0)[0, -1].item()
    # The first greedy token ends the turn, and every turn calls the tool.
    call = '<tool_call>{"name": "ok", "arguments": {}}</tool_call>'
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=first, decode=lambda ids, **_kwargs: call)
    rollout = HybridEngineRollout(_engine_for(model, max_out_tokens=len(prompt) + 10),
                                  tokenizer,
                                  HybridEngineRolloutConfig(multi_turn=True,
                                                            max_turns=3,
                                                            use_graph_capture=use_graph_capture,
                                                            continuous_batch_size=2),
                                  tools=ToolSet([_OkTool()]))
    request = RolloutRequest(samples=[RolloutRecord(tokens=prompt, attention_mask=[1] * len(prompt))])
    sampling = SamplingConfig(max_new_tokens=8, temperature=0.0, n_samples_per_prompt=3)

    with patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=[40] * 30):
        records = rollout.generate(request, sampling).samples

    for record in records:
        assert record.tokens == prompt + [first]
        assert record.status == RolloutRecord.Status.TRUNCATED


class _RankTool(Tool):
    """Answers with the rank that runs it, so ranks that each run their tools disagree."""
    name = "rank"
    description = "Returns the rank."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return str(dist.get_rank())


def _force_tool_calls(rollout, turns):
    """Make each trajectory's first ``turns`` turns call the rank tool, whatever the model generated."""
    call = rollout.parse_tool_calls('<tool_call>{"name": "rank", "arguments": {}}</tool_call>', rollout.tools)
    finish_turn = rollout._finish_turn

    def finish(trajectories, tool_queue, i, *args):
        finish_turn(trajectories, tool_queue, i, *args)
        if trajectories[i].num_turns <= turns and i not in tool_queue.in_flight:
            trajectories[i].status = RolloutRecord.Status.PENDING
            tool_queue.submit(i, call)

    rollout._finish_turn = finish


def _rank_tool_tokens(calls, outputs, *_args):
    return [50 + int(output) for output in outputs]


class _SlowRankTool(Tool):
    """Answers with its rank, after a delay that differs per rank and per call, so ranks finish tools at different steps."""
    name = "rank"
    description = "Returns the rank."
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.calls = 0

    async def execute(self, arguments):
        self.calls += 1
        await asyncio.sleep(0.01 * (1 + 3 * dist.get_rank()) * (1 + self.calls % 3))
        return str(dist.get_rank())


class TestAutoTPRollout(DistributedTest):
    """Tensor-parallel ranks must roll out the same records, or they would run different forwards."""
    world_size = 2

    def _engine(self, tie_word_embeddings=False):
        from transformers import AutoModelForCausalLM, Qwen3Config

        config = Qwen3Config(vocab_size=64,
                             hidden_size=64,
                             intermediate_size=128,
                             num_hidden_layers=2,
                             num_attention_heads=4,
                             num_key_value_heads=2,
                             head_dim=16,
                             max_position_embeddings=128,
                             tie_word_embeddings=tie_word_embeddings)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config).eval()
        reference = copy.deepcopy(model).to(get_accelerator().current_device_name()).eval()
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 0
            },
            "hybrid_engine": {
                "max_out_tokens": 128
            },
        }
        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        engine.module.eval()
        return engine, reference

    @staticmethod
    def _gather(records):
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, [(record.tokens, record.rollout_log_probs) for record in records])
        return gathered

    @pytest.mark.parametrize("use_graph_capture", [False, True])
    @pytest.mark.parametrize("temperature", [0.0, 1.0])
    def test_ranks_sample_the_same_tokens(self, use_graph_capture, temperature):
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            pytest.skip("requires graph capture")
        engine, reference = self._engine()
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None, decode=lambda ids, **_kwargs: "")
        rollout = HybridEngineRollout(
            engine, tokenizer, HybridEngineRolloutConfig(use_graph_capture=use_graph_capture, return_log_probs=True))
        prompts = [[0, 0, 5, 9, 12], [3, 7, 21, 30, 44]]
        request = RolloutRequest(samples=[
            RolloutRecord(tokens=prompt, attention_mask=[int(token != 0) for token in prompt]) for prompt in prompts
        ])
        # Each rank gets a different seed, so only the rollout's own synchronization can keep sampling identical.
        torch.manual_seed(dist.get_rank())
        sampling = SamplingConfig(max_new_tokens=8, temperature=temperature, n_samples_per_prompt=2)

        records = rollout.generate(request, sampling).samples

        gathered = self._gather(records)
        assert gathered[0] == gathered[1]
        if temperature == 0.0:
            device = get_accelerator().current_device_name()
            prompt_ids = torch.tensor(prompts, device=device)
            with torch.no_grad():
                expected = reference.generate(prompt_ids,
                                              attention_mask=(prompt_ids != 0).long(),
                                              max_new_tokens=8,
                                              do_sample=False,
                                              pad_token_id=0)
            assert [record.tokens for record in records[::2]] == expected.tolist()

    def test_ranks_sample_the_same_tokens_after_a_tool_turn(self):
        engine, _ = self._engine()
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None, decode=lambda ids, **_kwargs: "")
        rollout = HybridEngineRollout(engine,
                                      tokenizer,
                                      HybridEngineRolloutConfig(multi_turn=True, max_turns=2, return_log_probs=True),
                                      tools=ToolSet([_RankTool()]))
        # Sampled responses rarely end in EOS, so the first turn is made to call the tool regardless.
        _force_tool_calls(rollout, turns=1)
        prompts = [[0, 0, 5, 9, 12], [3, 7, 21, 30, 44]]
        request = RolloutRequest(samples=[
            RolloutRecord(tokens=prompt, attention_mask=[int(token != 0) for token in prompt]) for prompt in prompts
        ])
        # Each rank gets a different seed, so only the rollout's own synchronization can keep sampling identical.
        torch.manual_seed(dist.get_rank())
        sampling = SamplingConfig(max_new_tokens=8, max_new_tokens_per_turn=4, temperature=1.0)

        with patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses",
                   side_effect=_rank_tool_tokens):
            records = rollout.generate(request, sampling).samples

        assert all(len(record.tokens) == 5 + 4 + 1 + 4 for record in records)
        gathered = self._gather(records)
        assert gathered[0] == gathered[1]

    @pytest.mark.parametrize("use_graph_capture", [False, True])
    @pytest.mark.parametrize("continuous_batch_size", [None, 2])
    def test_tied_vocabulary_parallel_head_decodes_over_the_whole_vocabulary(self, use_graph_capture,
                                                                             continuous_batch_size):
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            pytest.skip("requires graph capture")
        if (continuous_batch_size is not None and not use_graph_capture and hasattr(Cache, "get_query_offset")
                and not hasattr(DeepSpeedStaticCache, "get_query_offset")):
            pytest.skip("DeepSpeedStaticCache lacks get_query_offset for this Transformers version")
        engine, reference = self._engine(tie_word_embeddings=True)
        head = engine.module.get_output_embeddings()
        # AutoTP shards a tied head by vocabulary only when the Transformers tp_plan asks for it.
        if not getattr(head, "is_vocab_parallel_lm_head", False):
            pytest.skip("AutoTP keeps this tied output head whole with this Transformers version")
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None, decode=lambda ids, **_kwargs: "")
        rollout = HybridEngineRollout(
            engine, tokenizer,
            HybridEngineRolloutConfig(use_graph_capture=use_graph_capture,
                                      continuous_batch_size=continuous_batch_size,
                                      return_log_probs=True))
        prompts = [[0, 0, 5, 9, 12], [3, 7, 21, 30, 44]]

        def request():
            return RolloutRequest(samples=[
                RolloutRecord(tokens=prompt, attention_mask=[int(token != 0) for token in prompt])
                for prompt in prompts
            ])

        try:
            # The repetition penalty also needs the whole vocabulary to track each row's history.
            greedy = rollout.generate(request(),
                                      SamplingConfig(max_new_tokens=8, temperature=0.0,
                                                     repetition_penalty=1.3)).samples
            # Each rank gets a different seed, so only the rollout's own synchronization keeps sampling identical.
            torch.manual_seed(dist.get_rank())
            sampled = rollout.generate(request(), SamplingConfig(max_new_tokens=8, temperature=1.0)).samples
        finally:
            # A live graph holding NCCL work hangs process-group teardown.
            del rollout
            gc.collect()
            get_accelerator().synchronize()

        device = get_accelerator().current_device_name()
        prompt_ids = torch.tensor(prompts, device=device)
        with torch.no_grad():
            expected = reference.generate(prompt_ids,
                                          attention_mask=(prompt_ids != 0).long(),
                                          max_new_tokens=8,
                                          do_sample=False,
                                          repetition_penalty=1.3,
                                          pad_token_id=0)
        assert [record.tokens for record in greedy] == expected.tolist()
        for records in (greedy, sampled):
            gathered = self._gather(records)
            assert gathered[0] == gathered[1]
        # Training goes back to rank-local logits for the vocabulary-parallel loss.
        assert head.gather_output is False

    @pytest.mark.parametrize("use_graph_capture", [False, True])
    def test_continuous_batching_admits_tool_turns_at_the_same_step_on_every_rank(self, use_graph_capture):
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            pytest.skip("requires graph capture")
        if (not use_graph_capture and hasattr(Cache, "get_query_offset")
                and not hasattr(DeepSpeedStaticCache, "get_query_offset")):
            pytest.skip("DeepSpeedStaticCache lacks get_query_offset for this Transformers version")
        engine, _ = self._engine()
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None, decode=lambda ids, **_kwargs: "")
        prompts = [[3, 7, 21, 30, 44], [5, 9, 12], [8, 17, 2, 40], [11, 6, 33, 25, 19, 4]]
        sampling = SamplingConfig(max_new_tokens=12, max_new_tokens_per_turn=4, temperature=0.0)

        def roll_out(continuous_batch_size):
            tool = _SlowRankTool()
            rollout = HybridEngineRollout(engine,
                                          tokenizer,
                                          HybridEngineRolloutConfig(multi_turn=True,
                                                                    max_turns=3,
                                                                    use_graph_capture=use_graph_capture,
                                                                    continuous_batch_size=continuous_batch_size,
                                                                    return_log_probs=True),
                                          tools=ToolSet([tool]))
            # Greedy responses here never end in EOS, so the first two turns are made to call the tool.
            _force_tool_calls(rollout, turns=2)
            request = RolloutRequest(samples=[_prompt(prompt) for prompt in prompts])
            try:
                with patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses",
                           side_effect=_rank_tool_tokens):
                    records = rollout.generate(request, sampling).samples
            finally:
                # The override refers back to the rollout, and a live graph holding NCCL work hangs process-group
                # teardown, so break the cycle and free the graph now.
                del rollout._finish_turn
                del rollout
                gc.collect()
                get_accelerator().synchronize()
            return records, tool.calls

        # Two slots for four trajectories, so turns return from tools while other rows decode.
        records, calls = roll_out(continuous_batch_size=2)
        static_records, _ = roll_out(continuous_batch_size=None)

        gathered = self._gather(records)
        assert gathered[0] == gathered[1]
        # Only the first rank runs the tools, and every rank appends its output.
        assert calls == (2 * len(prompts) if dist.get_rank() == 0 else 0)
        for prompt, record in zip(prompts, records):
            assert len(record.tokens) == len(prompt) + 4 + 1 + 4 + 1 + 4
            assert record.tokens[len(prompt) + 4] == record.tokens[len(prompt) + 9] == 50
        assert [record.tokens for record in records] == [record.tokens for record in static_records]


# ZeRO-3 gathers parameters in every forward, so its ranks must run the same forwards: single-turn static batching
# on HF generate only, the same number of prompts per rank, and no early stop.
_DATA_PARALLEL_SETUPS = {
    "zero2": {
        "zero_optimization": {
            "stage": 2
        }
    },
    "zero2_offload": {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    },
    "zero2_fp16": {
        "zero_optimization": {
            "stage": 2
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 8
        }
    },
    "zero3": {
        "zero_optimization": {
            "stage": 3,
            # The test model's parameters are below the default threshold, so ZeRO-3 would keep them all whole.
            "stage3_param_persistence_threshold": 0
        }
    },
}


class TestDataParallelRollout(DistributedTest):
    """Data-parallel ranks roll out their own prompts independently, before and after a training step."""
    world_size = 2

    @pytest.mark.parametrize(
        "setup, use_graph_capture, continuous_batch_size",
        [
            ("zero2", False, None),
            ("zero2", True, None),
            ("zero2", False, 2),
            ("zero2", True, 2),
            # Optimizer offload changes only the step; the graph must still read the weights it writes.
            ("zero2_offload", True, 2),
            ("zero2_fp16", True, None),
            ("zero2_fp16", True, 2),
            # ZeRO-3 runs on static batching with HF generate only.
            ("zero3", False, None),
        ])
    def test_each_rank_matches_its_own_greedy_generation(self, setup, use_graph_capture, continuous_batch_size):
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            pytest.skip("requires graph capture")
        if setup == "zero2_fp16" and not get_accelerator().is_fp16_supported():
            pytest.skip("requires fp16")
        from transformers import AutoModelForCausalLM, Qwen3Config

        config = Qwen3Config(vocab_size=64,
                             hidden_size=64,
                             intermediate_size=128,
                             num_hidden_layers=2,
                             num_attention_heads=4,
                             num_key_value_heads=2,
                             head_dim=16,
                             max_position_embeddings=128)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        optimizer_params = {"lr": 1e-2}
        if setup != "zero2_offload":
            optimizer_params["torch_adam"] = True
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "AdamW",
                "params": optimizer_params
            },
            "hybrid_engine": {
                "max_out_tokens": 128
            },
            **_DATA_PARALLEL_SETUPS[setup],
        }
        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        module = engine.module
        device = get_accelerator().current_device_name()
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None, decode=lambda ids, **_kwargs: "")
        rollout = HybridEngineRollout(
            engine, tokenizer,
            HybridEngineRolloutConfig(use_graph_capture=use_graph_capture,
                                      continuous_batch_size=continuous_batch_size))
        # Each rank has its own prompts, of different lengths and counts, so nothing keeps the ranks in step.
        rank = dist.get_rank()
        prompts = [[3 + rank, 7, 21, 30], [5, 9 + rank, 12], [8, 2, 40 + rank, 11, 6]]
        prompts = prompts[:2] if setup == "zero3" else prompts[:2 + rank]
        sampling = SamplingConfig(max_new_tokens=6, temperature=0.0)

        def check_against_hf_generate():
            module.eval()
            records = rollout.generate(
                RolloutRequest(samples=[RolloutRecord(tokens=p, attention_mask=[1] * len(p)) for p in prompts]),
                sampling).samples
            for prompt, record in zip(prompts, records):
                with torch.no_grad():
                    expected = module.generate(torch.tensor([prompt], device=device),
                                               max_new_tokens=6,
                                               do_sample=False,
                                               pad_token_id=0)
                # Records keep the request's left padding, so compare the real tokens.
                real = [token for token, keep in zip(record.tokens, record.attention_mask) if keep]
                assert real == expected[0].tolist()
            return [record.tokens for record in records]

        before = check_against_hf_generate()
        # A training step changes the weights in place, and the next rollout must read the new ones.
        module.train()
        input_ids = torch.tensor([prompts[0] * 2], device=device)
        loss = engine(input_ids, labels=input_ids).loss
        engine.backward(loss)
        engine.step()
        after = check_against_hf_generate()
        assert after != before

    @pytest.mark.parametrize("over_cap", [False, True])
    def test_zero3_ranks_stay_in_step_under_max_out_tokens(self, over_cap):
        """``max_out_tokens`` gives each rank a different budget, or none, and ZeRO-3 ranks must still agree."""
        from transformers import AutoModelForCausalLM, Qwen3Config

        config = Qwen3Config(vocab_size=64,
                             hidden_size=64,
                             intermediate_size=128,
                             num_hidden_layers=2,
                             num_attention_heads=4,
                             num_key_value_heads=2,
                             head_dim=16,
                             max_position_embeddings=128)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        max_out_tokens, max_new_tokens = 12, 6
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-2,
                    "torch_adam": True
                }
            },
            "hybrid_engine": {
                "max_out_tokens": max_out_tokens
            },
            **_DATA_PARALLEL_SETUPS["zero3"],
        }
        engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        module = engine.module.eval()
        device = get_accelerator().current_device_name()
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=None, decode=lambda ids, **_kwargs: "")
        rollout = HybridEngineRollout(engine, tokenizer, HybridEngineRolloutConfig())
        # Rank 0 has the budget for every token; rank 1 for two, or with over_cap for none.
        if dist.get_rank() == 0:
            prompts = [[3, 7, 21, 30], [5, 9, 12]]
        else:
            length = 13 if over_cap else 10
            prompts = [list(range(1, length + 1)), list(range(2, length + 2))]

        records = rollout.generate(
            RolloutRequest(samples=[RolloutRecord(tokens=p, attention_mask=[1] * len(p)) for p in prompts]),
            SamplingConfig(max_new_tokens=max_new_tokens, temperature=0.0)).samples

        for prompt, record in zip(prompts, records):
            budget = max(min(max_new_tokens, max_out_tokens - len(prompt)), 0)
            # Every rank generates the same number of tokens here, so the reference stays in step too.
            with torch.no_grad():
                expected = module.generate(torch.tensor([prompt], device=device),
                                           max_new_tokens=max_new_tokens,
                                           do_sample=False,
                                           pad_token_id=0)
            real = [token for token, keep in zip(record.tokens, record.attention_mask) if keep]
            assert real == expected[0, :len(prompt) + budget].tolist()
            assert record.status == RolloutRecord.Status.TRUNCATED


def _prompt(tokens):
    return RolloutRecord(tokens=list(tokens), attention_mask=[1] * len(tokens))


def _tiny_llama(device):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=64)
    return LlamaForCausalLM(config).to(device).eval()


def _small_qwen3(device, dtype):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=128,
                         hidden_size=128,
                         intermediate_size=256,
                         num_hidden_layers=4,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         head_dim=64,
                         max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).to(device, dtype).eval()


def _left_pad(prompts, device):
    width = max(len(prompt) for prompt in prompts)
    prompt_ids = torch.zeros(len(prompts), width, dtype=torch.long, device=device)
    prompt_attn = torch.zeros_like(prompt_ids)
    for row, prompt in enumerate(prompts):
        prompt_ids[row, width - len(prompt):] = torch.tensor(prompt, device=device)
        prompt_attn[row, width - len(prompt):] = 1
    return prompt_ids, prompt_attn


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_graph_capture_decodes_left_padded_rows_as_if_alone(dtype):
    """The mask hides pad slots, so a left-padded row must decode as it would alone and unpadded."""
    if not get_accelerator().is_available() or not get_accelerator().is_graph_capture_supported():
        pytest.skip("requires graph capture")
    device = torch.device(get_accelerator().device_name())
    model = _small_qwen3(device, dtype)
    # An EOS id that is never generated keeps every row decoding to max_new_tokens.
    rollout = HybridEngineRollout(_engine_for(model), SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  HybridEngineRolloutConfig(return_log_probs=True, use_graph_capture=True))
    torch.manual_seed(1)
    prompts = [torch.randint(1, 128, (length, )).tolist() for length in (9, 5, 7, 9, 3)]
    prompt_ids, prompt_attn = _left_pad(prompts, device)
    width, max_new_tokens, temperature = prompt_ids.shape[1], 6, 0.8

    with torch.no_grad():
        output_ids, log_probs = rollout._generate_graph(
            prompt_ids, prompt_attn, max_new_tokens,
            SamplingConfig(max_new_tokens=max_new_tokens, temperature=temperature), model, device)
        responses = output_ids[:, width:]
        for row, prompt in enumerate(prompts):
            # The reference is a plain forward over the row alone, unpadded, scored the way training scores it.
            sequence = torch.cat((torch.tensor(prompt, device=device), responses[row])).unsqueeze(0)
            logits = model(sequence).logits[0, len(prompt) - 1:-1].float()
            expected = torch.log_softmax(logits / temperature, dim=-1).gather(1, responses[row].unsqueeze(1))
            atol = 1e-3 if dtype == torch.float32 else 5e-2
            torch.testing.assert_close(log_probs[row], expected.squeeze(1), atol=atol, rtol=0)

        if dtype == torch.float32:
            greedy, _ = rollout._generate_graph(prompt_ids, prompt_attn, max_new_tokens,
                                                SamplingConfig(max_new_tokens=max_new_tokens, temperature=0.0), model,
                                                device)
            for row, prompt in enumerate(prompts):
                alone = torch.tensor([prompt], device=device)
                reference = model.generate(alone, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=0)
                assert greedy[row, width:].tolist() == reference[0, len(prompt):].tolist(), f"row {row}"


_OK_CALL = '<tool_call>{"name": "ok", "arguments": {}}</tool_call>'

_TOOL_TOKENS = [9, 10]


def _continuous_case(model_factory, dtype):
    """A tiny model, prompts, an EOS id its greedy decoding produces, and the response length that calls a tool."""
    device = torch.device(get_accelerator().device_name())
    model = model_factory(device, dtype) if dtype is not None else model_factory(device)
    torch.manual_seed(1)
    prompts = [torch.randint(11, 60, (length, )).tolist() for length in (3, 7, 5, 9, 4)]
    with torch.no_grad():
        probe = [
            model.generate(torch.tensor([prompt], device=device), max_new_tokens=8, do_sample=False,
                           pad_token_id=0)[0, len(prompt):].tolist() for prompt in prompts
        ]
    # The most frequent greedy token ends turns often, so trajectories run several turns.
    flat = [token for response in probe for token in response]
    eos = max(set(flat), key=flat.count)
    tool_length = min(response.index(eos) + 1 for response in probe if eos in response)
    return model, prompts, eos, tool_length


def _continuous_rollout(model, eos, tool_length, **cfg_kwargs):
    # A response of tool_length tokens calls the ok tool, so turns end in tool calls and in answers.
    tokenizer = SimpleNamespace(pad_token_id=0,
                                eos_token_id=eos,
                                decode=lambda ids, **_kwargs: _OK_CALL if len(ids) == tool_length else "")
    cfg = HybridEngineRolloutConfig(multi_turn=True, max_turns=4, **cfg_kwargs)
    engine = _engine_for(model, max_out_tokens=48)
    return HybridEngineRollout(engine, tokenizer, cfg, tools=ToolSet([_OkTool()]))


def _check_as_if_alone(model, prompts, samples, n_samples_per_prompt, eos, tool_length, temperature, atol):
    """Every generated token has the log-prob a plain forward over its trajectory alone gives it, greedy picks
    the top token, and tool output follows exactly the turns that ended in a tool call."""
    device = next(model.parameters()).device
    for index, sample in enumerate(samples):
        prompt = prompts[index // n_samples_per_prompt]
        real = [token for token, keep in zip(sample.tokens, sample.attention_mask) if keep]
        assert real[:len(prompt)] == prompt
        turns = real[len(prompt):]
        assert len(turns) == sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)
        with torch.no_grad():
            logits = model(torch.tensor([real], device=device)).logits[0].float()
        position = 0
        while position < len(turns):
            end = position
            while end < len(turns) and sample.loss_mask[end]:
                end += 1
            if end < len(turns):
                # Environment text comes only after a turn that ended in EOS and called a tool.
                response = turns[position:end]
                assert response and response[-1] == eos and len(response) == tool_length
                assert turns[end:end + len(_TOOL_TOKENS)] == _TOOL_TOKENS
                assert sample.loss_mask[end:end + len(_TOOL_TOKENS)] == [0] * len(_TOOL_TOKENS)
                end += len(_TOOL_TOKENS)
            position = end
        for position, (token, keep) in enumerate(zip(turns, sample.loss_mask)):
            if not keep:
                continue
            step = logits[len(prompt) + position - 1]
            scale = temperature if temperature > 0 else 1.0
            expected = torch.log_softmax(step / scale, dim=-1)[token].item()
            assert abs(sample.rollout_log_probs[position] - expected) < atol, f"sample {index} position {position}"
            if temperature <= 0:
                # Greedy, up to ties within numerical noise.
                assert step[token] >= step.max() - atol


@pytest.mark.parametrize("model_name, dtype, use_graph_capture, temperature", [
    ("qwen3", torch.float32, True, 0.0),
    ("qwen3", torch.float32, False, 0.0),
    ("qwen3", torch.bfloat16, True, 0.8),
    ("qwen3", torch.float32, True, 0.8),
    ("llama", None, True, 0.0),
])
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=_TOOL_TOKENS)
def test_multi_turn_continuous_batching_generates_each_turn_as_if_alone(_tokenize_tool_responses, model_name, dtype,
                                                                        use_graph_capture, temperature):
    """With fewer decode rows than trajectories, every turn waits for a row, is prefilled with its whole
    conversation, and must decode exactly as the trajectory would alone."""
    if not get_accelerator().is_available():
        pytest.skip("multi-turn continuous batching requires an accelerator")
    model_factory = _small_qwen3 if model_name == "qwen3" else _tiny_llama
    model, prompts, eos, tool_length = _continuous_case(model_factory, dtype)
    rollout = _continuous_rollout(model,
                                  eos,
                                  tool_length,
                                  use_graph_capture=use_graph_capture,
                                  continuous_batch_size=3,
                                  return_log_probs=True)
    sampling = SamplingConfig(max_new_tokens=24,
                              max_new_tokens_per_turn=8,
                              temperature=temperature,
                              n_samples_per_prompt=2)

    torch.manual_seed(2)
    samples = rollout.generate(RolloutRequest(samples=[_prompt(prompt) for prompt in prompts]), sampling).samples

    assert len(samples) == 10
    assert all(sample.status != RolloutRecord.Status.PENDING for sample in samples)
    if temperature <= 0:
        # Greedy first turns repeat the probe, so some trajectories call a tool and run more turns.
        assert sum(0 in sample.loss_mask for sample in samples) >= 2
    atol = 1e-3 if dtype != torch.bfloat16 else 5e-2
    _check_as_if_alone(model, prompts, samples, 2, eos, tool_length, temperature, atol)


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=_TOOL_TOKENS)
def test_multi_turn_continuous_repetition_penalty_matches_generate(_tokenize_tool_responses):
    """With fewer rows than trajectories, a reused row must penalize its new trajectory's tokens, not its last one's."""
    if not get_accelerator().is_available():
        pytest.skip("multi-turn continuous batching requires an accelerator")
    model, prompts, _, _ = _continuous_case(_small_qwen3, torch.float32)
    device = next(model.parameters()).device
    # eos -1 never matches, so each trajectory is one turn of the full length.
    model.generation_config.repetition_penalty = 1.3
    rollout = _continuous_rollout(model, -1, 0, use_graph_capture=True, continuous_batch_size=2)
    sampling = SamplingConfig(max_new_tokens=8, temperature=0)

    samples = rollout.generate(RolloutRequest(samples=[_prompt(prompt) for prompt in prompts]), sampling).samples

    for prompt, sample in zip(prompts, samples):
        with torch.no_grad():
            alone = model.generate(torch.tensor([prompt], device=device),
                                   max_new_tokens=8,
                                   do_sample=False,
                                   eos_token_id=None,
                                   pad_token_id=0)
        real = [token for token, keep in zip(sample.tokens, sample.attention_mask) if keep]
        assert real == alone[0].tolist()


@pytest.mark.parametrize("use_graph_capture", [True, False])
def test_multi_turn_continuous_batching_admits_into_a_freed_row_on_the_next_step(use_graph_capture):
    """A finished turn frees its row for the next queued turn at once, so no row idles while a turn waits."""
    if not get_accelerator().is_available():
        pytest.skip("multi-turn continuous batching requires an accelerator")
    model, prompts, _, _ = _continuous_case(_small_qwen3, torch.float32)
    # eos -1 never matches, so each trajectory is one turn of exactly max_new_tokens tokens.
    rollout = _continuous_rollout(model,
                                  -1,
                                  0,
                                  use_graph_capture=use_graph_capture,
                                  enable_profiling=True,
                                  continuous_batch_size=2)
    max_new_tokens = 8
    sampling = SamplingConfig(max_new_tokens=max_new_tokens, temperature=0)

    samples = rollout.generate(RolloutRequest(samples=[_prompt(prompt) for prompt in prompts]), sampling).samples

    assert [sample.response_length for sample in samples] == [max_new_tokens] * len(prompts)
    # Turns run in waves of two rows, each taking max_new_tokens - 1 decode steps after its prefill picks the first
    # token. A row that waited for a later check before taking the next turn would add steps.
    waves = -(-len(prompts) // 2)
    assert rollout.get_last_profile()["num_decode_forwards"] == waves * (max_new_tokens - 1)


@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=_TOOL_TOKENS)
def test_multi_turn_continuous_batching_matches_static_batching(_tokenize_tool_responses):
    """At temperature 0 both schedules must produce the same trajectories, turn by turn."""
    if not get_accelerator().is_available():
        pytest.skip("multi-turn continuous batching requires an accelerator")
    model, prompts, eos, tool_length = _continuous_case(_small_qwen3, torch.float32)
    request = RolloutRequest(samples=[_prompt(prompt) for prompt in prompts])
    sampling = SamplingConfig(max_new_tokens=24, max_new_tokens_per_turn=8, temperature=0, n_samples_per_prompt=2)

    static = _continuous_rollout(model, eos, tool_length, use_graph_capture=True,
                                 return_log_probs=True).generate(request, sampling)
    continuous_rollout = _continuous_rollout(model,
                                             eos,
                                             tool_length,
                                             use_graph_capture=True,
                                             enable_profiling=True,
                                             continuous_batch_size=3,
                                             return_log_probs=True)
    continuous = continuous_rollout.generate(request, sampling)

    assert sum(0 in sample.loss_mask for sample in continuous.samples) >= 2
    # The step trace accounts for every decoded token (each turn's first comes from its prefill), and never counts
    # more decoding rows than rows holding a turn.
    profile = continuous_rollout.get_last_profile()
    trace = profile["continuous_step_trace"]
    assert trace[-1][4] == profile["num_generated_tokens"] - profile["num_turns"]
    assert all(decoding <= held <= 3 for _, _, held, decoding, *_ in trace)
    for expected, actual in zip(static.samples, continuous.samples):
        assert actual.tokens == expected.tokens
        assert actual.loss_mask == expected.loss_mask
        assert actual.status == expected.status
        torch.testing.assert_close(actual.rollout_log_probs, expected.rollout_log_probs, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "layout_kwargs",
    [
        {},
        {
            "align_decode_fronts": True
        },
        {
            "enable_cache_trimming": True
        },
        # With one row, every returning turn finds the column reset to the prompt width and moves it forward.
        {
            "continuous_batch_size": 1
        },
    ])
@patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses", return_value=_TOOL_TOKENS)
def test_multi_turn_shared_column_matches_static_batching(_tokenize_tool_responses, layout_kwargs):
    """Without graph capture, rows share one write column; at temperature 0 the trajectories match static batching."""
    if not get_accelerator().is_available():
        pytest.skip("multi-turn continuous batching requires an accelerator")
    model, prompts, eos, tool_length = _continuous_case(_small_qwen3, torch.float32)
    request = RolloutRequest(samples=[_prompt(prompt) for prompt in prompts])
    sampling = SamplingConfig(max_new_tokens=24, max_new_tokens_per_turn=8, temperature=0, n_samples_per_prompt=2)
    continuous_kwargs = {"continuous_batch_size": 3, **layout_kwargs}

    static = _continuous_rollout(model, eos, tool_length, return_log_probs=True).generate(request, sampling)
    continuous = _continuous_rollout(model, eos, tool_length, return_log_probs=True,
                                     **continuous_kwargs).generate(request, sampling)

    assert sum(0 in sample.loss_mask for sample in continuous.samples) >= 2
    for expected, actual in zip(static.samples, continuous.samples):
        assert actual.tokens == expected.tokens
        assert actual.loss_mask == expected.loss_mask
        assert actual.status == expected.status
        torch.testing.assert_close(actual.rollout_log_probs, expected.rollout_log_probs, atol=1e-4, rtol=0)


def test_continuous_graph_recaptures_only_when_tensor_layout_changes():
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("graph capture is required for continuous graph rollout coverage")

    from transformers import LlamaConfig, LlamaForCausalLM

    device = torch.device(accelerator.device_name())
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=64)
    config._attn_implementation = "sdpa"
    eager_model = LlamaForCausalLM(config).to(device).eval()
    graph_model = LlamaForCausalLM(config).to(device).eval()
    graph_model.load_state_dict(eager_model.state_dict())
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=-1, decode=lambda ids, **_kwargs: "")
    eager_rollout = HybridEngineRollout(_engine_for(eager_model), tokenizer)
    # A pinned capacity keeps the decode shape fixed across prompt widths.
    graph_rollout = HybridEngineRollout(
        _engine_for(graph_model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(use_graph_capture=True, continuous_batch_size=2, continuous_cache_capacity=24),
    )
    sampling = SamplingConfig(max_new_tokens=8, temperature=0)
    generator = torch.Generator().manual_seed(1)
    short_prompts = torch.randint(1, 64, (3, 8), generator=generator).to(device)
    long_prompts = torch.randint(1, 64, (3, 12), generator=generator).to(device)

    # The continuous path also calls create_graph() as a support probe, so count captures instead.
    original_capture_to_graph = accelerator.capture_to_graph
    capture_count = [0]

    def counted_capture_to_graph(*args, **kwargs):
        capture_count[0] += 1
        return original_capture_to_graph(*args, **kwargs)

    accelerator.capture_to_graph = counted_capture_to_graph
    try:
        outputs = []
        for step, prompt_ids in enumerate((short_prompts, short_prompts, long_prompts, short_prompts, short_prompts)):
            if step == 1:
                # Apply the same in-place update an optimizer step would make.
                with torch.no_grad():
                    for eager_param, graph_param in zip(eager_model.parameters(), graph_model.parameters()):
                        delta = torch.randn_like(eager_param) * 0.05
                        eager_param.add_(delta)
                        graph_param.add_(delta)
            if step == 3:
                # Same address, new strides: replay would read the old layout.
                with torch.no_grad():
                    eager_model.lm_head.weight.transpose_(0, 1)
                    graph_model.lm_head.weight.transpose_(0, 1)
            if step == 4:
                # A re-registered buffer lives at an address the captured kernels never read.
                for model in (eager_model, graph_model):
                    rotary = model.model.rotary_emb
                    rotary.register_buffer("inv_freq", rotary.inv_freq * 0.5, persistent=False)
            attention_mask = torch.ones_like(prompt_ids)
            expected = eager_rollout.generate(_records(prompt_ids, attention_mask), sampling).samples
            output = graph_rollout.generate(_records(prompt_ids, attention_mask), sampling).samples
            assert [sample.tokens for sample in output] == [sample.tokens for sample in expected]
            outputs.append([sample.tokens for sample in output])
    finally:
        accelerator.capture_to_graph = original_capture_to_graph

    # The update must change the tokens, otherwise stale weights would also pass.
    assert outputs[0] != outputs[1]
    assert outputs[3] != outputs[4]
    assert capture_count[0] == 3

    cache_bytes = graph_rollout.get_last_continuous_stats()["cache_memory_bytes"]
    allocated = accelerator.memory_allocated()
    graph_rollout.shutdown()
    assert accelerator.memory_allocated() <= allocated - cache_bytes
