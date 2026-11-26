# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Accelerator-backed v1 HybridEngineRollout tests."""

import copy
import gc
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
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
            mask = attention_mask[:, 0, 0, :]
            # Boolean masks mark valid columns True, additive ones with 0.
            valid = mask if mask.dtype == torch.bool else mask == 0
        else:
            valid = torch.zeros((values.shape[0], values.shape[2]), dtype=torch.bool, device=input_ids.device)
            valid[:, :attention_mask.shape[1]] = attention_mask.bool()
        cache_sums = (values * valid[:, None, :, None]).sum(dim=(1, 2, 3)).long()
        next_tokens = cache_sums.remainder(29) + 3
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 32),
                             dtype=self.weight.dtype,
                             device=input_ids.device)
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


def test_static_graph_attention_mask_uses_additive_values():
    attention = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
    causal = torch.empty(2, 1, 1, 3)

    HybridEngineRollout._update_static_causal_attention(attention, causal)

    minimum = torch.finfo(causal.dtype).min
    assert torch.equal(causal[0, 0, 0], torch.tensor([0, 0, minimum], dtype=causal.dtype))
    assert torch.equal(causal[1, 0, 0], torch.tensor([minimum, 0, 0], dtype=causal.dtype))


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


@pytest.mark.parametrize("batch_size, repetition_penalty", [(1, 1.3), (2, 1.0)])
def test_static_graph_full_step_kernel_preconditions(batch_size, repetition_penalty, monkeypatch):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("requires graph capture")

    from deepspeed.ops import module_inject

    calls = []

    class GraphOp:

        def decode_step_graph(self, *args):
            calls.append(args)

    monkeypatch.setattr(module_inject, "get_fused_glu_op", GraphOp)
    device = torch.device(accelerator.device_name())
    model = _LocalKVModel(2).to(device).to(torch.bfloat16).eval()
    model.generation_config = SimpleNamespace(repetition_penalty=repetition_penalty)
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    prompt_ids = torch.tensor([[1, 3, 4]] * batch_size, device=device)

    output = rollout.generate(RolloutRequest(prompt_ids, torch.ones_like(prompt_ids)),
                              SamplingConfig(max_new_tokens=3, temperature=0))

    assert output.input_ids.shape == (batch_size, 6)
    # The captured step kernel cannot apply penalties and is reserved for
    # the single-row fast path, so neither case may mount it.
    assert calls == []


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
                             max_position_embeddings=64,
                             eos_token_id=None)
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
                                      SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                      cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph))
        try:
            # Three requests on two slots force a refill into an already allocated cache row.
            output = rollout.generate(
                RolloutRequest(prompt_ids, prompt_attn),
                SamplingConfig(max_new_tokens=new_tokens, temperature=0, continuous_batch_size=2))
            assert torch.equal(output.input_ids, expected)
            assert model.config.num_key_value_heads == 4
        finally:
            # A live graph holding NCCL work hangs process-group teardown.
            del rollout
            gc.collect()
            accelerator.synchronize()


def _tied_autotp_engine():
    """A tied-embedding tiny Llama under AutoTP 2, whose output head AutoTP makes vocabulary-parallel, and an
    unsharded copy of the same weights."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=128,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=4,
                         max_position_embeddings=64,
                         tie_word_embeddings=True,
                         eos_token_id=None)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    device = torch.device(get_accelerator().current_device_name())
    reference = copy.deepcopy(model).to(device).eval()
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "tensor_parallel": {
            "autotp_size": 2
        },
        "zero_optimization": {
            "stage": 0
        }
    }
    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    model.eval()
    assert any(getattr(module, "is_vocab_parallel_lm_head", False) for module in model.modules())
    prompt_ids = torch.tensor([[5, 17, 42, 9], [33, 8, 71, 2], [64, 11, 19, 90]], device=device)
    return engine, reference, RolloutRequest(prompt_ids, torch.ones_like(prompt_ids))


def _same_on_every_rank(tensor):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, tensor.tolist())
    return all(rank_tokens == gathered[0] for rank_tokens in gathered)


class TestAutoTPVocabularyParallelHeadRollout(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("use_graph, continuous_batch_size", [(False, None), (True, None), (False, 2), (True, 2)])
    def test_greedy_records_match_one_gpu_on_every_rank(self, use_graph, continuous_batch_size):
        accelerator = get_accelerator()
        if use_graph and accelerator.create_graph() is None:
            pytest.skip("requires graph capture")
        if continuous_batch_size and not use_graph and hasattr(
                Cache, "get_query_offset") and not hasattr(DeepSpeedStaticCache, "get_query_offset"):
            pytest.skip("DeepSpeedStaticCache lacks get_query_offset for this Transformers version")
        engine, reference, request = _tied_autotp_engine()
        new_tokens = 6
        with torch.no_grad():
            expected = reference.generate(request.prompt_ids,
                                          attention_mask=request.prompt_attention_mask,
                                          max_new_tokens=new_tokens,
                                          do_sample=False,
                                          pad_token_id=0)

        # The rollout is the first forward after initialize, so with graph capture it also meets AutoTP's input
        # check with a DeepSpeedStaticCache argument.
        rollout = HybridEngineRollout(engine,
                                      SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                      cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph))
        try:
            output = rollout.generate(
                request,
                SamplingConfig(max_new_tokens=new_tokens, temperature=0, continuous_batch_size=continuous_batch_size))
            assert torch.equal(output.input_ids, expected)
            assert _same_on_every_rank(output.input_ids)
            # Training gets rank-local logits back.
            assert not any(
                getattr(module, "gather_output", False)
                for module in engine.module.modules() if getattr(module, "is_vocab_parallel_lm_head", False))
        finally:
            del rollout
            gc.collect()
            accelerator.synchronize()

    def test_ranks_sample_the_same_tokens_when_seeded_differently(self):
        engine, _, request = _tied_autotp_engine()
        get_accelerator().manual_seed(1234 + dist.get_rank())
        rollout = HybridEngineRollout(engine, SimpleNamespace(pad_token_id=0, eos_token_id=-1))

        output = rollout.generate(request, SamplingConfig(max_new_tokens=8, temperature=1.0))

        assert _same_on_every_rank(output.input_ids)


class TestAutoTPGraphRecapture(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("continuous_batch_size", [None, 2])
    def test_ranks_recapture_together(self, continuous_batch_size, monkeypatch):
        accelerator = get_accelerator()
        if accelerator.create_graph() is None:
            pytest.skip("requires graph capture")

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
        }
        engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
        model.eval()

        device = torch.device(accelerator.current_device_name())
        prompt_ids = torch.tensor([[5, 17, 42, 9], [33, 8, 71, 2]], device=device)
        prompt_attn = torch.ones_like(prompt_ids)
        new_tokens = 6
        with torch.no_grad():
            expected = model.generate(prompt_ids,
                                      attention_mask=prompt_attn,
                                      max_new_tokens=new_tokens,
                                      do_sample=False,
                                      eos_token_id=None,
                                      pad_token_id=0)

        original_capture_to_graph = accelerator.capture_to_graph
        capture_count = 0

        def counted_capture_to_graph(*args, **kwargs):
            nonlocal capture_count
            capture_count += 1
            return original_capture_to_graph(*args, **kwargs)

        monkeypatch.setattr(accelerator, "capture_to_graph", counted_capture_to_graph)
        rollout = HybridEngineRollout(engine,
                                      SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                      cfg=HybridEngineRolloutConfig(use_graph_capture=True))
        sampling = SamplingConfig(max_new_tokens=new_tokens,
                                  temperature=0,
                                  continuous_batch_size=continuous_batch_size)
        try:
            for step in range(2):
                if step == 1 and dist.get_rank() == 1:
                    # Only this rank's signature changes; its peer must capture with it,
                    # since capture runs forwards that hold TP collectives.
                    weight = model.model.layers[0].mlp.down_proj.weight
                    weight.data = weight.data.clone()
                output = rollout.generate(RolloutRequest(prompt_ids, prompt_attn), sampling)
                assert torch.equal(output.input_ids, expected)
            assert capture_count == 2
        finally:
            # A live graph holding NCCL work hangs process-group teardown.
            del rollout
            gc.collect()
            accelerator.synchronize()


class TestSegKIDataParallelRollout(DistributedTest):
    world_size = 2

    def test_each_rank_matches_its_own_generation(self):
        accelerator = get_accelerator()
        if accelerator.create_graph() is None:
            pytest.skip("requires graph capture")
        try:
            from deepspeed.ops.module_inject import get_fused_glu_op
            get_fused_glu_op()
        except Exception as e:
            pytest.skip(f"requires the segKI native op: {e}")

        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(vocab_size=128,
                             hidden_size=128,
                             intermediate_size=256,
                             num_hidden_layers=2,
                             num_attention_heads=4,
                             num_key_value_heads=2,
                             max_position_embeddings=64)
        config._attn_implementation = "sdpa"
        torch.manual_seed(0)
        device = torch.device(accelerator.current_device_name())
        model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
        reference = copy.deepcopy(model)
        # Pure data parallel: every rank holds the whole model and decodes its own prompt.
        generator = torch.Generator().manual_seed(100 + dist.get_rank())
        prompt_ids = torch.randint(1, config.vocab_size, (1, 8), generator=generator).to(device)
        attention_mask = torch.ones_like(prompt_ids)
        new_tokens = 8
        with torch.no_grad():
            expected = reference.generate(prompt_ids,
                                          attention_mask=attention_mask,
                                          max_new_tokens=new_tokens,
                                          do_sample=False,
                                          eos_token_id=None,
                                          pad_token_id=0)

        rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                      SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                      cfg=HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
        try:
            output = rollout.generate(RolloutRequest(prompt_ids, attention_mask),
                                      SamplingConfig(max_new_tokens=new_tokens, temperature=0))
            assert hasattr(model.lm_head, "_ki_gemm_op"), "segKI did not install its lm_head"
            # A head split over the world would decode each rank's prompt with half of its peer's logits.
            assert output.input_ids.tolist() == expected.tolist()
        finally:
            del rollout
            gc.collect()
            accelerator.synchronize()

    def test_vocab_parallel_head_is_not_split_over_the_tp_group(self):
        """segKI must not patch a vocab-parallel lm_head.

        AutoTP's VocabParallelLinear head holds a shard of the vocabulary; patching it would make
        segKI halve that shard and all-gather, silently dropping most of the rows from the logits.
        It is not caught by the collective guard because VocabParallelLinear subclasses the
        collective-bearing LinearLayer, not LinearAllreduce, so install_lm_head needs its own
        ``is_vocab_parallel_lm_head`` check (a pre-existing master bug fixed by this PR).
        """
        accelerator = get_accelerator()
        if accelerator.create_graph() is None:
            pytest.skip("requires graph capture")
        # Without the native op install_lm_head is skipped entirely, so only
        # here is the is_vocab_parallel_lm_head guard under test in the first place.
        try:
            from deepspeed.ops.module_inject import get_fused_glu_op
            get_fused_glu_op()
        except Exception as e:
            pytest.skip(f"requires the segKI native op: {e}")

        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(vocab_size=128,
                             hidden_size=128,
                             intermediate_size=256,
                             num_hidden_layers=2,
                             num_attention_heads=4,
                             num_key_value_heads=2,
                             max_position_embeddings=64)
        torch.manual_seed(0)
        device = torch.device(accelerator.current_device_name())
        model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
        reference = copy.deepcopy(model)
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            # segKI's lm_head install requires bf16 weights, so cast on initialize instead of after.
            "bf16": {
                "enabled": True
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "vocab_parallel_lm_head": True
            },
            "zero_optimization": {
                "stage": 0
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
        model.eval()
        # The config must actually shard the head, else this is not the case under test.
        heads = [m for m in model.modules() if getattr(m, "is_vocab_parallel_lm_head", False)]
        assert heads and heads[0].weight.shape[0] < config.vocab_size

        prompt_ids = torch.tensor([[5, 17, 42, 9, 30, 61, 100, 3]], device=device)
        attention_mask = torch.ones_like(prompt_ids)
        new_tokens = 8
        with torch.no_grad():
            expected = reference.generate(prompt_ids,
                                          attention_mask=attention_mask,
                                          max_new_tokens=new_tokens,
                                          do_sample=False,
                                          eos_token_id=None,
                                          pad_token_id=0)

        rollout = HybridEngineRollout(engine,
                                      SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                      cfg=HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))
        try:
            output = rollout.generate(RolloutRequest(prompt_ids, attention_mask),
                                      SamplingConfig(max_new_tokens=new_tokens, temperature=0))
            assert not any(hasattr(m, "_ki_gemm_op") for m in heads), \
                "segKI patched a vocab-parallel lm_head"
            # A patched head halves its shard and gathers, leaving the greedy tokens to wander over
            # the missing rows.
            assert output.input_ids.tolist() == expected.tolist()
        finally:
            del rollout
            gc.collect()
            accelerator.synchronize()


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
                         max_position_embeddings=128,
                         eos_token_id=None)
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


def test_segki_graphs_recapture_after_refresh_or_a_shared_model_reinstall(monkeypatch):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("requires graph capture")

    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(4321)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=64,
                         intermediate_size=128,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=64)
    config._attn_implementation = "sdpa"
    device = torch.device(accelerator.device_name())
    model = LlamaForCausalLM(config).to(device).eval()
    prompt_ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(2)).to(device)
    attention_mask = torch.ones_like(prompt_ids)
    sampling = SamplingConfig(max_new_tokens=8, temperature=0)
    with torch.no_grad():
        expected = model.generate(prompt_ids,
                                  attention_mask=attention_mask,
                                  max_new_tokens=8,
                                  do_sample=False,
                                  eos_token_id=None,
                                  pad_token_id=0)

    original_capture_to_graph = accelerator.capture_to_graph
    capture_count = 0

    def counted_capture_to_graph(*args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        return original_capture_to_graph(*args, **kwargs)

    monkeypatch.setattr(accelerator, "capture_to_graph", counted_capture_to_graph)

    def make_rollout():
        return HybridEngineRollout(SimpleNamespace(module=model),
                                   SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                   cfg=HybridEngineRolloutConfig(use_graph_capture=True, use_segki=True))

    first, second = make_rollout(), make_rollout()
    # first, first (replay), second (reinstalls the model's decode kernels), first (must recapture),
    # then a refresh, after which first must recapture once more and replay again.
    for rollout, captures in ((first, 1), (first, 1), (second, 2), (first, 3), (first, 3), ("refresh", 4), (first, 4)):
        if rollout == "refresh":
            first.refresh_weights()
            rollout = first
        output = rollout.generate(RolloutRequest(prompt_ids, attention_mask), sampling)
        assert output.input_ids.tolist() == expected.tolist()
        assert capture_count == captures


def test_static_graph_is_reused_across_rollouts_with_the_same_decode_shape(monkeypatch):
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
    config._attn_implementation = "sdpa"
    device = torch.device(accelerator.device_name())
    model = LlamaForCausalLM(config).to(device).eval()
    generator = torch.Generator().manual_seed(1)
    short_prompts = torch.randint(1, 64, (4, 8), generator=generator).to(device)
    long_prompts = torch.randint(1, 64, (4, 12), generator=generator).to(device)
    # eos -1 never matches, so every row generates the full length.
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))

    original_capture_to_graph = accelerator.capture_to_graph
    capture_count = 0

    def counted_capture_to_graph(*args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        return original_capture_to_graph(*args, **kwargs)

    monkeypatch.setattr(accelerator, "capture_to_graph", counted_capture_to_graph)
    # The decode graph depends on the batch size and prompt width plus max_new_tokens, not on the
    # prompt width alone: 8 + 16 and 12 + 12 share one graph, while 8 + 8 needs another.
    outputs = []
    for step, (prompt_ids, new_tokens) in enumerate(
        ((short_prompts, 16), (short_prompts, 16), (long_prompts, 12), (short_prompts, 8))):
        if step == 1:
            # Apply the same in-place update an optimizer step would make.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.05)
            rollout.sync_weights(step=1)
        attention_mask = torch.ones_like(prompt_ids)
        with torch.no_grad():
            expected = model.generate(prompt_ids,
                                      attention_mask=attention_mask,
                                      max_new_tokens=new_tokens,
                                      do_sample=False,
                                      eos_token_id=None,
                                      pad_token_id=0)
        output = rollout.generate(RolloutRequest(prompt_ids, attention_mask),
                                  SamplingConfig(max_new_tokens=new_tokens, temperature=0))
        assert output.input_ids.tolist() == expected.tolist()
        outputs.append(output.input_ids)

    # The update must change the tokens, otherwise a graph replaying stale weights would also pass.
    assert not torch.equal(outputs[0], outputs[1])
    assert capture_count == 2

    # A graph captured in eval mode must not be replayed in training mode.
    model.train()
    rollout.generate(RolloutRequest(short_prompts, torch.ones_like(short_prompts)),
                     SamplingConfig(max_new_tokens=8, temperature=0))
    model.eval()
    assert capture_count == 3

    # Child module mode changes must also invalidate a retained graph.
    child = model.model.layers[0]
    child.train()
    rollout.generate(RolloutRequest(short_prompts, torch.ones_like(short_prompts)),
                     SamplingConfig(max_new_tokens=8, temperature=0))
    child.eval()
    assert capture_count == 4

    # Nor may a graph captured without autocast be replayed under it. Autocast
    # still changes kernels at the model's own dtype, e.g. softmax runs in fp32.
    bf16_rollout = HybridEngineRollout(SimpleNamespace(module=copy.deepcopy(model).to(torch.bfloat16)),
                                       SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                       cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    bf16_request = RolloutRequest(short_prompts, torch.ones_like(short_prompts))
    bf16_rollout.generate(bf16_request, SamplingConfig(max_new_tokens=8, temperature=0))
    with torch.autocast(device_type=accelerator.device_name(), dtype=torch.bfloat16):
        bf16_rollout.generate(bf16_request, SamplingConfig(max_new_tokens=8, temperature=0))
    bf16_rollout.shutdown()
    assert capture_count == 6

    head_dim = config.hidden_size // config.num_attention_heads
    # Keys and values for every layer at batch 4 and cache length 8 + 8.
    cache_bytes = 2 * config.num_hidden_layers * 4 * config.num_key_value_heads * 16 * head_dim * 4
    allocated = accelerator.memory_allocated()
    rollout.shutdown()
    assert accelerator.memory_allocated() <= allocated - cache_bytes


@pytest.mark.parametrize("continuous_batch_size", [None, 2])
def test_graph_revalidates_tensors_replaced_during_prefill(monkeypatch, continuous_batch_size):
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
    config._attn_implementation = "eager"
    device = torch.device(accelerator.device_name())
    model = LlamaForCausalLM(config).to(device).eval()
    prompt_ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(1)).to(device)
    attention_mask = torch.ones_like(prompt_ids)
    sampling = SamplingConfig(max_new_tokens=6, temperature=0, continuous_batch_size=continuous_batch_size)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=-1)
    rollout = HybridEngineRollout(SimpleNamespace(module=model),
                                  tokenizer,
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    capture_count = 0
    original_capture_to_graph = accelerator.capture_to_graph

    def counted_capture_to_graph(*args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        return original_capture_to_graph(*args, **kwargs)

    monkeypatch.setattr(accelerator, "capture_to_graph", counted_capture_to_graph)
    first = rollout.generate(RolloutRequest(prompt_ids, attention_mask), sampling)
    assert capture_count == 1

    replaced = [False]
    handle = None

    def replace_buffer_once(*_args, **_kwargs):
        if not replaced[0]:
            rotary = model.model.rotary_emb
            rotary.inv_freq = rotary.inv_freq.clone()
            replaced[0] = True
            handle.remove()

    handle = model.register_forward_pre_hook(replace_buffer_once)
    second = rollout.generate(RolloutRequest(prompt_ids, attention_mask), sampling)
    assert replaced[0]
    assert torch.equal(second.input_ids, first.input_ids)
    assert capture_count == 2


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
    for rollout_index, rollout in enumerate((eager_rollout, graph_rollout, graph_rollout)):
        if rollout_index == 2:
            rollout.sync_weights(step=1)
        output = rollout.generate(request, sampling)
        assert torch.equal(output.input_ids, reference)
        assert torch.equal(output.attention_mask, expected_attention)
        assert torch.equal(output.response_start_idx, expected_response_start)
    assert capture_count == 1


@pytest.mark.parametrize("align_decode_fronts", [False, True])
def test_bucketed_prefill_refills_fixed_graph_slot_on_accelerator(monkeypatch, align_decode_fronts):
    from transformers import LlamaConfig, LlamaForCausalLM

    accelerator = get_accelerator()
    if not accelerator.is_graph_capture_supported():
        pytest.skip("requires graph capture")
    torch.manual_seed(1234)
    config = LlamaConfig(vocab_size=64,
                         hidden_size=32,
                         intermediate_size=64,
                         num_hidden_layers=2,
                         num_attention_heads=4,
                         num_key_value_heads=2,
                         max_position_embeddings=64,
                         pad_token_id=0,
                         eos_token_id=None,
                         attn_implementation="sdpa")
    device = accelerator.device_name()
    model = LlamaForCausalLM(config).to(device).eval()
    # The first request retires while the second continues. The shorter refill
    # must overwrite physical slot zero and leave slot one's history intact.
    ids = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4], [1, 6, 7, 8, 9, 10, 11, 12], [0, 0, 0, 0, 0, 0, 13, 14]],
                       device=device)
    mask = ids.ne(0).long()
    eos = model.generate(ids[:1], attention_mask=mask[:1], max_new_tokens=1, do_sample=False)[0, -1].item()
    expected = [
        model.generate(row[None], attention_mask=attn[None], max_new_tokens=4, eos_token_id=eos,
                       do_sample=False)[0, ids.shape[1]:] for row, attn in zip(ids, mask)
    ]
    assert len(expected[0]) == 1 and len(expected[1]) >= 3 and len(expected[2]) >= 2
    rollout = HybridEngineRollout(
        SimpleNamespace(module=model), SimpleNamespace(pad_token_id=0, eos_token_id=eos),
        HybridEngineRolloutConfig(use_graph_capture=True,
                                  adaptive_prefill=True,
                                  align_decode_fronts=align_decode_fronts,
                                  enable_profiling=True,
                                  prefill_max_tokens=8))
    capture = accelerator.capture_to_graph
    captures = []

    def counted_capture(*args, **kwargs):
        captures.append(1)
        return capture(*args, **kwargs)

    monkeypatch.setattr(accelerator, "capture_to_graph", counted_capture)
    for _ in range(2):
        result = rollout.generate(RolloutRequest(ids, mask),
                                  SamplingConfig(max_new_tokens=4, temperature=0, continuous_batch_size=2))
        for i, tokens in enumerate(expected):
            response = result.input_ids[i, ids.shape[1]:]
            valid = result.attention_mask[i, ids.shape[1]:].bool()
            assert torch.equal(response[valid], tokens)
        profile = rollout.get_last_profile()
        assert profile["generation_strategy"] == "bucketed_graph"
        assert profile["num_prefill_forwards"] == 3
    assert len(captures) == 1


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
                         max_position_embeddings=128,
                         eos_token_id=None)
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


@pytest.mark.parametrize("use_graph_capture, continuous_batch_size", [(True, None), (True, 2), (False, 2)])
@pytest.mark.parametrize("supports_cache_class", [None, True])
def test_rollout_under_autocast_matches_hf_generate(use_graph_capture, continuous_batch_size, supports_cache_class):
    # Under autocast an fp32 model emits fp32 keys (rotary embeddings promote them) but bf16 values,
    # which the pre-allocated KV caches used to reject with a dtype mismatch (#9102).
    # Continuous prefill goes through generate() unless the model sets _supports_cache_class.
    accelerator = get_accelerator()
    if not accelerator.is_bf16_supported():
        pytest.skip("bf16 autocast is not supported")
    if use_graph_capture and not accelerator.is_graph_capture_supported():
        pytest.skip("graph capture is not supported")

    device = torch.device(accelerator.device_name())
    model = _tiny_llama(device)
    if supports_cache_class is not None:
        model._supports_cache_class = supports_cache_class
    new_tokens = 8
    request = _random_request(batch_size=4, prompt_len=8, device=device)
    sampling = SamplingConfig(max_new_tokens=new_tokens, temperature=0, continuous_batch_size=continuous_batch_size)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = _rollout(model, use_graph_capture).generate(request, sampling)
        with torch.no_grad():
            expected = model.generate(request.prompt_ids,
                                      attention_mask=request.prompt_attention_mask,
                                      max_new_tokens=new_tokens,
                                      do_sample=False,
                                      eos_token_id=None,
                                      pad_token_id=0)

    assert output.input_ids.tolist() == expected.tolist()


def test_static_graph_is_recaptured_when_autocast_changes(monkeypatch):
    # A retained fp32 graph replays fp32 kernels, so entering bf16 autocast must recapture rather than reuse it.
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph decode coverage")
    if not accelerator.is_bf16_supported():
        pytest.skip("bf16 autocast is not supported")

    device = torch.device(accelerator.device_name())
    model = _tiny_llama(device)
    rollout = _rollout(model, use_graph_capture=True)
    captures = []
    original_capture = rollout._capture_static_graph

    def capture(*args, **kwargs):
        captures.append(torch.is_autocast_enabled(device.type))
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(rollout, "_capture_static_graph", capture)
    new_tokens = 8
    request = _random_request(batch_size=4, prompt_len=8, device=device)
    sampling = SamplingConfig(max_new_tokens=new_tokens, temperature=0)

    def generate_and_compare():
        output = rollout.generate(request, sampling)
        with torch.no_grad():
            expected = model.generate(request.prompt_ids,
                                      attention_mask=request.prompt_attention_mask,
                                      max_new_tokens=new_tokens,
                                      do_sample=False,
                                      eos_token_id=None,
                                      pad_token_id=0)
        assert output.input_ids.tolist() == expected.tolist()

    for _ in range(2):
        generate_and_compare()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for _ in range(2):
            generate_and_compare()

    assert captures == [False, True]


def test_static_graph_is_recaptured_after_a_peer_capture_failure(monkeypatch):
    # This rank's capture succeeded but a peer's failed: replaying the kept graph
    # would wait on a peer that has none, so the next call must capture again.
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph decode coverage")

    device = torch.device(accelerator.device_name())
    model = _tiny_llama(device)
    rollout = _rollout(model, use_graph_capture=True)
    captures = 0

    def capture_on_all_ranks(_module, capture):
        nonlocal captures
        captures += 1
        result = capture()
        if captures == 1:
            raise RuntimeError("decode graph capture failed on a peer rank")
        return result

    monkeypatch.setattr(rollout, "_capture_on_all_ranks", capture_on_all_ranks)
    new_tokens = 8
    request = _random_request(batch_size=4, prompt_len=8, device=device)
    sampling = SamplingConfig(max_new_tokens=new_tokens, temperature=0)

    with pytest.raises(RuntimeError, match="peer rank"):
        rollout.generate(request, sampling)
    output = rollout.generate(request, sampling)
    with torch.no_grad():
        expected = model.generate(request.prompt_ids,
                                  attention_mask=request.prompt_attention_mask,
                                  max_new_tokens=new_tokens,
                                  do_sample=False,
                                  eos_token_id=None,
                                  pad_token_id=0)

    assert captures == 2
    assert output.input_ids.tolist() == expected.tolist()


def test_static_graph_stops_once_every_row_has_produced_eos(monkeypatch):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph decode coverage")

    device = torch.device(accelerator.device_name())
    model = _tiny_llama(device)
    new_tokens = 48
    request = _random_request(batch_size=8, prompt_len=8, device=device)
    sampling = SamplingConfig(max_new_tokens=new_tokens, temperature=0)
    unstopped = _rollout(model, use_graph_capture=False).generate(request, sampling).input_ids
    # Every row's third generated token is an EOS id, so all rows finish early.
    eos_ids = sorted(set(unstopped[:, 8 + 2].tolist()))

    def rollout(use_graph_capture):
        return HybridEngineRollout(SimpleNamespace(module=model),
                                   SimpleNamespace(pad_token_id=0, eos_token_id=eos_ids),
                                   cfg=HybridEngineRolloutConfig(use_graph_capture=use_graph_capture))

    expected = rollout(use_graph_capture=False).generate(request, sampling)
    replay_count = 0
    original_replay_graph = accelerator.replay_graph

    def counted_replay_graph(graph):
        nonlocal replay_count
        replay_count += 1
        return original_replay_graph(graph)

    monkeypatch.setattr(accelerator, "replay_graph", counted_replay_graph)
    output = rollout(use_graph_capture=True).generate(request, sampling)

    assert output.input_ids.tolist() == expected.input_ids.tolist()
    assert output.attention_mask.tolist() == expected.attention_mask.tolist()
    assert replay_count < new_tokens - 1


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
    request.prompt_ids[:4, :3] = 0
    request.prompt_attention_mask[:4, :3] = 0
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


@pytest.mark.parametrize("use_graph_capture, continuous_batch_size", [(False, None), (True, None), (True, 8),
                                                                      (False, 8)])
def test_every_path_stops_on_a_generation_config_eos_id(use_graph_capture, continuous_batch_size):
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.device_name() != "cuda":
        pytest.skip("CUDA is required for CUDA graph decode coverage")

    device = torch.device(accelerator.device_name())
    model = _tiny_llama(device)
    new_tokens = 16
    request = _random_request(batch_size=8, prompt_len=8, device=device)
    prompt_len = request.prompt_ids.shape[1]

    def hf_generate():
        with torch.no_grad():
            return model.generate(request.prompt_ids,
                                  attention_mask=request.prompt_attention_mask,
                                  max_new_tokens=new_tokens,
                                  do_sample=False,
                                  pad_token_id=0)[:, prompt_len:].tolist()

    # The model's generation config lists an EOS id the tokenizer does not have: a token the model generates midway.
    full_length = hf_generate()
    eos = full_length[0][new_tokens // 2]
    model.generation_config.eos_token_id = [eos]
    expected = [row[:row.index(eos) + 1] if eos in row else row for row in hf_generate()]
    assert any(len(row) < new_tokens for row in expected)

    output = _rollout(model, use_graph_capture).generate(
        request, SamplingConfig(max_new_tokens=new_tokens,
                                temperature=0.0,
                                continuous_batch_size=continuous_batch_size))

    # Each row matches generate() up to its EOS, and nothing after it is attended.
    for row, mask, want in zip(output.input_ids[:, prompt_len:].tolist(),
                               output.attention_mask[:, prompt_len:].tolist(), expected):
        assert row[:len(want)] == want
        assert not any(mask[len(want):])


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
                         max_position_embeddings=64,
                         eos_token_id=None)
    config._attn_implementation = "sdpa"
    eager_model = LlamaForCausalLM(config).to(device).eval()
    graph_model = LlamaForCausalLM(config).to(device).eval()
    graph_model.load_state_dict(eager_model.state_dict())
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=-1)
    eager_rollout = HybridEngineRollout(SimpleNamespace(module=eager_model), tokenizer)
    # A pinned capacity keeps the decode shape fixed across prompt widths.
    graph_rollout = HybridEngineRollout(
        SimpleNamespace(module=graph_model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(use_graph_capture=True, continuous_cache_capacity=24),
    )
    eager_sampling = SamplingConfig(max_new_tokens=8, temperature=0)
    graph_sampling = SamplingConfig(max_new_tokens=8, temperature=0, continuous_batch_size=2)
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
            request = RolloutRequest(prompt_ids, torch.ones_like(prompt_ids))
            expected = eager_rollout.generate(request, eager_sampling)
            output = graph_rollout.generate(request, graph_sampling)
            assert torch.equal(output.input_ids, expected.input_ids)
            outputs.append(output.input_ids)
    finally:
        accelerator.capture_to_graph = original_capture_to_graph

    # The update must change the tokens, otherwise stale weights would also pass.
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[3], outputs[4])
    assert capture_count[0] == 3

    cache_bytes = graph_rollout.get_last_continuous_stats()["cache_memory_bytes"]
    allocated = accelerator.memory_allocated()
    graph_rollout.shutdown()
    assert accelerator.memory_allocated() <= allocated - cache_bytes


def test_graph_capture_falls_back_for_zero3_partitioned_weights(monkeypatch):
    """ZeRO-3 gathers fresh parameter buffers each forward; a replayed graph would read
    the stale buffers captured earlier. With use_graph_capture on, a partitioned-weights
    engine must therefore take the eager path (which syncs ZeRO-3) and capture nothing."""
    accelerator = get_accelerator()
    if not accelerator.is_available() or accelerator.create_graph() is None:
        pytest.skip("requires graph capture")
    device = torch.device(accelerator.device_name())

    prompt_ids = torch.tensor([[1, 3, 4, 2], [3, 4, 5, 1]], device=device)
    prompt_attn = torch.ones_like(prompt_ids)
    new_tokens = 4

    # Eager reference: prompt plus one deterministic token per step.
    expected = torch.cat([prompt_ids, (prompt_ids.sum(dim=1, keepdim=True).remainder(29) + 3).repeat(1, new_tokens)],
                         dim=1)

    # If the graph path is (wrongly) taken, capture must fail so the test does.
    def no_capture(*args, **kwargs):
        raise RuntimeError("graph capture must not run on a ZeRO-3-partitioned engine")

    monkeypatch.setattr(accelerator, "create_graph", no_capture)

    # A ZeRO-3 engine reports partitioned weights; the eager path reads module.generate.
    eager_calls = []

    def fake_generate(prompt_ids, **kwargs):
        eager_calls.append(kwargs)
        return expected

    engine = SimpleNamespace(module=SimpleNamespace(generate=fake_generate, config=object()),
                             zero_optimization_partition_weights=lambda: True)
    rollout = HybridEngineRollout(engine,
                                  SimpleNamespace(pad_token_id=0, eos_token_id=-1),
                                  cfg=HybridEngineRolloutConfig(use_graph_capture=True))
    output = rollout.generate(RolloutRequest(prompt_ids, prompt_attn),
                              SamplingConfig(max_new_tokens=new_tokens, temperature=0))
    assert eager_calls, "ZeRO-3 engine must fall back to the eager generation path"
    assert torch.equal(output.input_ids, expected)
