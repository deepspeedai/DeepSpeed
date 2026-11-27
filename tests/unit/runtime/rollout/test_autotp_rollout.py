# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""CPU unit tests for AutoTPRollout.

Two simulated TP ranks are built in one process: AutoTP layers read their rank from
``deepspeed.comm``, which is patched, so the layers partition and record their layout metadata
exactly as they do under ``init_inference``. The ZeRO gather is patched to return the full weight.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from deepspeed.module_inject import layers as autotp_layers
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer
from deepspeed.runtime.rollout import build_rollout
from deepspeed.runtime.rollout import autotp_rollout
from deepspeed.runtime.rollout.autotp_rollout import AutoTPRollout, sync_autotp_weights
from deepspeed.runtime.rollout.base import RolloutBatch, RolloutConfig, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig

TP_SIZE = 2
PAD = 0
RESPONSE_TOKENS = 1


class _Block(torch.nn.Module):
    """Embedding, a column-parallel and a row-parallel projection, and a norm: one of each layout."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 8)
        self.up = torch.nn.Linear(8, 12, bias=True)
        self.down = torch.nn.Linear(12, 8, bias=True)
        self.norm = torch.nn.LayerNorm(8)


def _train_module():
    """The training engine's module, with weights unrelated to the AutoTP copy's initial ones."""
    torch.manual_seed(0)
    return _Block()


def _tp_copy(tp_rank, dtype):
    """Rank ``tp_rank``'s AutoTP copy, initialized with weights unrelated to the training module."""
    torch.manual_seed(1)
    block = _Block().to(dtype)
    with patch.object(autotp_layers.dist, "get_world_size", return_value=TP_SIZE), \
            patch.object(autotp_layers.dist, "get_rank", return_value=tp_rank):
        block.up = LinearLayer(block.up, mp_group=object(), name="up")
        block.down = LinearAllreduce(block.down, mp_group=object(), name="down")
    return block


def _sync(train_module, tp_module):
    # ZeRO-2 keeps a full low-precision copy on every rank, so the gather's result is its fp32 value.
    with patch.object(autotp_rollout, "safe_get_full_fp32_param", side_effect=lambda p: p.detach().float()):
        return sync_autotp_weights(train_module, tp_module)


@pytest.mark.parametrize("tp_rank", range(TP_SIZE))
def test_sync_copies_each_layout_into_its_shard(tp_rank):
    train = _train_module()
    tp_module = _tp_copy(tp_rank, torch.float32)
    full = dict(train.named_parameters())
    assert tp_module.up.weight.shape == (12 // TP_SIZE, 8)
    assert not torch.equal(tp_module.up.weight, full["up.weight"].chunk(TP_SIZE, dim=0)[tp_rank])

    assert _sync(train, tp_module) == len(full)

    tp = dict(tp_module.named_parameters())
    # Column-parallel: weight and bias split on the output dimension.
    assert torch.equal(tp["up.weight"], full["up.weight"].chunk(TP_SIZE, dim=0)[tp_rank])
    assert torch.equal(tp["up.bias"], full["up.bias"].chunk(TP_SIZE, dim=0)[tp_rank])
    # Row-parallel: weight split on the input dimension, bias replicated.
    assert torch.equal(tp["down.weight"], full["down.weight"].chunk(TP_SIZE, dim=1)[tp_rank])
    assert torch.equal(tp["down.bias"], full["down.bias"])
    # Modules AutoTP did not replace are copied whole.
    for name in ("embed.weight", "norm.weight", "norm.bias"):
        assert torch.equal(tp[name], full[name])


def test_sync_rounds_fp32_masters_to_the_inference_dtype():
    train = _train_module()
    tp_module = _tp_copy(0, torch.bfloat16)
    _sync(train, tp_module)
    full = dict(train.named_parameters())
    assert tp_module.down.weight.dtype == torch.bfloat16
    assert torch.equal(tp_module.down.weight, full["down.weight"].chunk(TP_SIZE, dim=1)[0].to(torch.bfloat16))


def test_sync_rejects_an_inference_copy_missing_a_parameter():
    train = _train_module()
    tp_module = _tp_copy(0, torch.float32)
    del tp_module.norm
    with pytest.raises(AssertionError, match="no parameter norm.weight"):
        _sync(train, tp_module)


def test_sync_rejects_a_training_model_missing_a_parameter():
    train = _train_module()
    del train.norm
    with pytest.raises(AssertionError, match="synced"):
        _sync(train, _tp_copy(0, torch.float32))


@pytest.mark.parametrize("option", ["use_graph_capture", "use_shared_prefill"])
def test_constructor_rejects_unsupported_generation_paths(option):
    # Both paths drive engine and cache internals that the AutoTP copy does not provide.
    with pytest.raises(NotImplementedError, match=option):
        AutoTPRollout(MagicMock(), MagicMock(), cfg=HybridEngineRolloutConfig(**{option: True}))


def test_engine_factory_autotp_requires_student_engine():
    with pytest.raises(ValueError, match="needs both"):
        build_rollout(RolloutConfig(engine="autotp"))


# -- generate: one TP group spans all ranks, so each rank generates the union of all prompts ------


def _rollout():
    rollout = AutoTPRollout.__new__(AutoTPRollout)
    rollout.tp_size = TP_SIZE
    rollout.tokenizer = SimpleNamespace(pad_token_id=PAD, eos_token_id=2)
    rollout._seed = 1234
    rollout._generate_calls = 0
    return rollout


def _prompts(*rows):
    """A left-padded request; a row's zeros are padding."""
    prompt_ids = torch.tensor(rows, dtype=torch.long)
    return RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=(prompt_ids != PAD).long())


def _fake_union_generate(seen):
    """Stand-in for HybridEngineRollout.generate under the RolloutBatch contract: n adjacent copies per
    prompt in prompt order, with the row's position in the union as its response token."""

    def generate(self, request, sampling):
        seen.append((torch.initial_seed(), request))
        n = sampling.n_samples_per_prompt
        prompt_ids = request.prompt_ids.repeat_interleave(n, dim=0)
        prompt_mask = request.prompt_attention_mask.repeat_interleave(n, dim=0)
        total = prompt_ids.shape[0]
        responses = 100 + torch.arange(total).unsqueeze(1)
        return RolloutBatch(input_ids=torch.cat([prompt_ids, responses], dim=1),
                            attention_mask=torch.cat([prompt_mask, torch.ones_like(responses)], dim=1),
                            response_start_idx=torch.full((total, ), prompt_ids.shape[1], dtype=torch.long))

    return generate


def _generate(rank, per_rank_requests, n_samples_per_prompt=1, gathered_tokens=None, rollout=None):
    rollout = rollout or _rollout()
    seen = []

    def all_gather_object(output, obj):
        if isinstance(obj, tuple):
            output[:] = [(request.prompt_ids, request.prompt_attention_mask) for request in per_rank_requests]
        else:  # the token check: every rank reports what this rank generated, unless overridden
            output[:] = gathered_tokens(obj) if gathered_tokens else [obj] * TP_SIZE

    with patch.object(autotp_rollout.dist, "get_rank", return_value=rank), \
            patch.object(autotp_rollout.dist, "all_gather_object", side_effect=all_gather_object), \
            patch.object(HybridEngineRollout, "generate", _fake_union_generate(seen)):
        sampling = SamplingConfig(max_new_tokens=RESPONSE_TOKENS, n_samples_per_prompt=n_samples_per_prompt)
        batch = rollout.generate(per_rank_requests[rank], sampling)
    return rollout, batch, seen


@pytest.mark.parametrize("rank, expected_prompts, expected_responses", [(0, [[7], [8]], [100, 101, 102, 103]),
                                                                        (1, [[9]], [104, 105])])
def test_generate_returns_only_this_ranks_rows(rank, expected_prompts, expected_responses):
    per_rank = [_prompts([7], [8]), _prompts([9])]
    _, batch, _ = _generate(rank, per_rank, n_samples_per_prompt=2)
    expected_rows = [
        prompt + [response]
        for prompt, response in zip([p for p in expected_prompts for _ in range(2)], expected_responses)
    ]
    assert batch.input_ids.tolist() == expected_rows


@pytest.mark.parametrize("rank, expected_ids, expected_mask, expected_start", [
    (0, [[0, 7, 100]], [[0, 1, 1]], [2]),
    (1, [[8, 9, 10, 101]], [[1, 1, 1, 1]], [3]),
])
def test_generate_keeps_each_ranks_own_layout_when_prompt_widths_differ(rank, expected_ids, expected_mask,
                                                                        expected_start):
    # A rank with short prompts must get back [B', T_p + T_r] for its own T_p, or its response
    # boundary would not match the request it made.
    per_rank = [_prompts([0, 7]), _prompts([8, 9, 10])]
    _, batch, _ = _generate(rank, per_rank)
    assert batch.input_ids.tolist() == expected_ids
    assert batch.attention_mask.tolist() == expected_mask
    assert batch.response_start_idx.tolist() == expected_start


def test_generate_left_pads_the_union_without_attending_to_the_padding():
    per_rank = [_prompts([7]), _prompts([8, 9])]
    _, _, seen = _generate(0, per_rank)
    _, union = seen[0]
    assert union.prompt_ids.tolist() == [[PAD, 7], [8, 9]]
    assert union.prompt_attention_mask.tolist() == [[0, 1], [1, 1]]


def test_generate_seeds_every_rank_alike_and_each_call_differently():
    per_rank = [_prompts([7]), _prompts([9])]
    seeds_per_rank = []
    for rank in range(TP_SIZE):
        rollout = _rollout()
        seeds = []
        for _ in range(2):
            _, _, seen = _generate(rank, per_rank, rollout=rollout)
            seeds += [seed for seed, _ in seen]
        seeds_per_rank.append(seeds)
    # Ranks that sample from different seeds pick different tokens and deadlock in AutoTP's all-reduce.
    assert seeds_per_rank[0] == seeds_per_rank[1]
    # Reusing one seed would make every rollout step draw the same samples for the same prompts.
    assert seeds_per_rank[0][0] != seeds_per_rank[0][1]


def test_generate_leaves_the_callers_random_stream_unchanged():
    # Otherwise, turning rollout on would change the training run's dropout and data shuffling.
    per_rank = [_prompts([7]), _prompts([9])]
    torch.manual_seed(42)
    expected = torch.rand(1)
    torch.manual_seed(42)
    _generate(0, per_rank)
    assert torch.equal(torch.rand(1), expected)


def test_generate_fails_loudly_when_ranks_disagree():
    per_rank = [_prompts([7]), _prompts([9])]

    def diverged(input_ids):
        other = input_ids.clone()
        other[0, -1] = 99
        return [input_ids, other]

    with pytest.raises(RuntimeError, match="different tokens"):
        _generate(0, per_rank, gathered_tokens=diverged)


def test_generate_rejects_continuous_batching():
    with pytest.raises(NotImplementedError, match="continuous batching"):
        _rollout().generate(_prompts([7]), SamplingConfig(max_new_tokens=1, continuous_batch_size=4))
