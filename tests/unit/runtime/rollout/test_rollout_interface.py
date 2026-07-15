# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Conformance tests for the RolloutEngine interface.

Validates the dataclass invariants and exercises the interface against a
``FakeRollout`` so the contract is testable without GPUs or a model. The real
backends are tested manually with a launched training script (see README).
"""

import pytest
import torch

from deepspeed.runtime.rollout import (
    RolloutBatch,
    RolloutEngine,
    RolloutRequest,
    SamplingConfig,
    build_rollout,
)

# --- dataclass invariants ---------------------------------------------------


def test_rollout_request_validates_shapes():
    with pytest.raises(ValueError, match="must be 2-D"):
        RolloutRequest(prompt_ids=torch.zeros(8), prompt_attention_mask=torch.ones(8))
    with pytest.raises(ValueError, match="does not match"):
        RolloutRequest(prompt_ids=torch.zeros(2, 4, dtype=torch.long), prompt_attention_mask=torch.ones(2, 5))


def test_rollout_batch_validates_shapes():
    with pytest.raises(ValueError, match="must be 2-D"):
        RolloutBatch(input_ids=torch.zeros(8, dtype=torch.long),
                     attention_mask=torch.ones(8),
                     response_start_idx=torch.tensor([4]))
    with pytest.raises(ValueError, match="does not match"):
        RolloutBatch(input_ids=torch.zeros(2, 4, dtype=torch.long),
                     attention_mask=torch.ones(2, 5),
                     response_start_idx=torch.tensor([4, 4]))
    with pytest.raises(ValueError, match="1-D of length"):
        RolloutBatch(input_ids=torch.zeros(2, 4, dtype=torch.long),
                     attention_mask=torch.ones(2, 4),
                     response_start_idx=torch.tensor([4]))


def test_rollout_batch_accessors():
    batch = RolloutBatch(
        input_ids=torch.zeros(3, 12, dtype=torch.long),
        attention_mask=torch.ones(3, 12),
        response_start_idx=torch.tensor([4, 5, 6]),
    )
    assert batch.batch_size == 3
    assert batch.seq_len == 12


def test_sampling_config_defaults():
    cfg = SamplingConfig(max_new_tokens=32)
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.top_k == -1
    assert cfg.n_samples_per_prompt == 1


# --- interface conformance via FakeRollout ---------------------------------


class FakeRollout(RolloutEngine):
    """Deterministic stub: appends ``[42] * max_new_tokens`` to each prompt."""

    name = "fake"

    def __init__(self, response_token: int = 42):
        self.response_token = response_token
        self.sync_calls: list = []

    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        B, T_p = request.prompt_ids.shape
        n = sampling.n_samples_per_prompt
        T_r = sampling.max_new_tokens

        prompts_expanded = request.prompt_ids.repeat_interleave(n, dim=0)
        attn_p_expanded = request.prompt_attention_mask.repeat_interleave(n, dim=0)
        response = torch.full((B * n, T_r), self.response_token, dtype=request.prompt_ids.dtype)
        response_attn = torch.ones((B * n, T_r), dtype=attn_p_expanded.dtype)

        input_ids = torch.cat([prompts_expanded, response], dim=1)
        attention_mask = torch.cat([attn_p_expanded, response_attn], dim=1)
        response_start_idx = torch.full((B * n, ), T_p, dtype=torch.long)
        return RolloutBatch(input_ids=input_ids, attention_mask=attention_mask, response_start_idx=response_start_idx)

    def sync_weights(self, step: int) -> None:
        self.sync_calls.append(step)


def test_fake_rollout_shape_basic():
    fake = FakeRollout()
    req = RolloutRequest(prompt_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                         prompt_attention_mask=torch.ones(2, 3, dtype=torch.long))
    out = fake.generate(req, SamplingConfig(max_new_tokens=4))
    assert out.input_ids.shape == (2, 7)
    assert out.attention_mask.shape == (2, 7)
    # With left-padded (fully real here) prompts of width 3, response begins
    # at column 3 for every sample.
    assert out.response_start_idx.tolist() == [3, 3]


def test_fake_rollout_with_n_samples():
    fake = FakeRollout()
    req = RolloutRequest(prompt_ids=torch.tensor([[1, 2], [3, 4]]),
                         prompt_attention_mask=torch.ones(2, 2, dtype=torch.long))
    out = fake.generate(req, SamplingConfig(max_new_tokens=3, n_samples_per_prompt=4))
    assert out.input_ids.shape == (8, 5)
    assert out.response_start_idx.tolist() == [2] * 8


def test_fake_rollout_left_padded_prompts():
    fake = FakeRollout()
    # left-padded prompts: prompt B has only the last 2 positions real, but
    # response_start_idx still equals the prompt column width T_p.
    prompt_ids = torch.tensor([[1, 2, 3, 4], [0, 0, 5, 6]])
    attn = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
    req = RolloutRequest(prompt_ids=prompt_ids, prompt_attention_mask=attn)
    out = fake.generate(req, SamplingConfig(max_new_tokens=2))
    assert out.response_start_idx.tolist() == [4, 4]


def test_sync_records_steps():
    fake = FakeRollout()
    fake.sync_weights(0)
    fake.sync_weights(5)
    assert fake.sync_calls == [0, 5]


def test_engine_factory_unknown_raises():
    from deepspeed.runtime.rollout.base import RolloutConfig

    with pytest.raises(ValueError, match="Unknown rollout engine"):
        build_rollout(RolloutConfig(engine="totally_made_up"))


def test_engine_factory_hybrid_requires_student_engine():
    from deepspeed.runtime.rollout.base import RolloutConfig

    with pytest.raises(ValueError, match="needs both"):
        build_rollout(RolloutConfig(engine="hybrid_engine"))
