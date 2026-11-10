# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Accelerator-backed v1 HybridEngineRollout tests."""

import copy
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.config import HybridEngineConfig
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout import Tool, ToolSet
from deepspeed.runtime.rollout.base import RolloutRecord, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
from unit.common import DistributedTest


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
    # Three requests and two slots force refill. Each graph call captures once, however many turns it admits.
    for rollout in (eager_rollout, graph_rollout, graph_rollout):
        output = rollout.generate(request, sampling).samples
        assert [sample.tokens for sample in output] == reference.tolist()
        assert [sample.attention_mask for sample in output] == expected_attention
        assert all(sample.response_length == sampling.max_new_tokens for sample in output)
    assert capture_count == 2


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
@pytest.mark.parametrize("continuous_batch_size", [None, 2])
def test_tool_output_that_would_pass_max_out_tokens_ends_the_trajectory(use_graph_capture, continuous_batch_size):
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
                                                            continuous_batch_size=continuous_batch_size),
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


class TestAutoTPRollout(DistributedTest):
    """Tensor-parallel ranks must roll out the same records, or they would run different forwards."""
    world_size = 2

    def _engine(self):
        from transformers import AutoModelForCausalLM, Qwen3Config

        config = Qwen3Config(vocab_size=64,
                             hidden_size=64,
                             intermediate_size=128,
                             num_hidden_layers=2,
                             num_attention_heads=4,
                             num_key_value_heads=2,
                             head_dim=16,
                             max_position_embeddings=128,
                             tie_word_embeddings=False)
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

    def test_ranks_append_the_first_ranks_tool_output(self):
        engine, reference = self._engine()
        prompt = [3, 7, 21, 30, 44]
        device = get_accelerator().current_device_name()
        with torch.no_grad():
            first = reference.generate(torch.tensor([prompt], device=device), max_new_tokens=1,
                                       do_sample=False)[0, -1].item()
        # The first greedy token ends the turn and calls the tool, so the second turn reads the tool output.
        tokenizer = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=first,
            decode=lambda ids, **_kwargs: '<tool_call>{"name": "rank", "arguments": {}}</tool_call>')
        rollout = HybridEngineRollout(engine,
                                      tokenizer,
                                      HybridEngineRolloutConfig(multi_turn=True, max_turns=2, return_log_probs=True),
                                      tools=ToolSet([_RankTool()]))
        sampling = SamplingConfig(max_new_tokens=6, max_new_tokens_per_turn=3, temperature=0.0)

        with patch("deepspeed.runtime.rollout.hybrid_engine_rollout.tokenize_tool_responses",
                   side_effect=lambda messages, *_args: [50 + int(message["content"]) for message in messages]):
            record = rollout.generate(RolloutRequest(samples=[RolloutRecord(tokens=prompt, attention_mask=[1] * 5)]),
                                      sampling).samples[0]

        assert record.tokens[5:7] == [first, 50]
        gathered = self._gather([record])
        assert gathered[0] == gathered[1]


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

    @pytest.mark.parametrize("setup", list(_DATA_PARALLEL_SETUPS))
    @pytest.mark.parametrize("use_graph_capture", [False, True])
    @pytest.mark.parametrize("continuous_batch_size", [None, 2])
    def test_each_rank_matches_its_own_greedy_generation(self, use_graph_capture, continuous_batch_size, setup):
        if use_graph_capture and not get_accelerator().is_graph_capture_supported():
            pytest.skip("requires graph capture")
        if setup == "zero3" and (use_graph_capture or continuous_batch_size is not None):
            pytest.skip("ZeRO-3 runs on static batching with HF generate only")
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
