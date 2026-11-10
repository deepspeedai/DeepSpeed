# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Conformance tests for the RolloutEngine interface.

Validates the dataclass invariants and exercises the interface against a
``FakeRollout`` so the contract is testable without GPUs or a model. The real
backends are tested manually with a launched training script (see README).
"""

from dataclasses import replace

import pytest
import torch

import deepspeed.runtime.rollout as rollout_module
from deepspeed.runtime.rollout import (
    GRPOLossOutput,
    JSDLossOutput,
    RolloutBatch,
    RolloutEngine,
    RolloutRequest,
    ResponseTokenBatch,
    SamplingConfig,
    build_rollout,
    generalized_jsd_loss,
    group_advantages,
    grpo_policy_loss,
    old_log_probs_from_rollout,
)
from deepspeed.runtime.rollout.base import RolloutRecord

# --- dataclass invariants ---------------------------------------------------


def test_rollout_request_validates_shapes():
    RolloutRequest(samples=[RolloutRecord(prompt="not tokenized yet")])
    with pytest.raises(ValueError, match="does not match"):
        RolloutRequest(samples=[RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1])])


def test_rollout_batch_validates_shapes():
    with pytest.raises(ValueError, match="does not match"):
        RolloutBatch(samples=[
            RolloutRecord(tokens=[1, 2], attention_mask=[1, 1]),
            RolloutRecord(tokens=[1, 2], attention_mask=[1])
        ])


def test_rollout_batch_accessors():
    batch = RolloutBatch(
        samples=[RolloutRecord(tokens=[1] * length, attention_mask=[1] * length) for length in (4, 12, 7)])
    assert batch.batch_size == 3
    assert batch.seq_len == 12


def test_sampling_config_defaults():
    cfg = SamplingConfig(max_new_tokens=32)
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.top_k == -1
    assert cfg.n_samples_per_prompt == 1
    assert not hasattr(cfg, "continuous_batch_size")
    assert not hasattr(cfg, "return_log_probs")


def test_continuous_batching_details_are_not_public_exports():
    assert "ContinuousBatchRequest" not in rollout_module.__all__
    assert "ContinuousBatchScheduler" not in rollout_module.__all__
    assert "ContinuousBatchUpdate" not in rollout_module.__all__


def test_response_token_batch_masks_prompts_padding_and_tool_output():
    rollout = RolloutBatch(samples=[
        RolloutRecord(tokens=[0, 11, 12, 21, 2], attention_mask=[0, 1, 1, 1, 1], response_length=2),
        RolloutRecord(tokens=[13, 14, 15, 22, 23, 2], attention_mask=[1] * 6, response_length=3),
        # A multi-turn record with generated 31, tool output 41 (loss mask 0), generated 32 and EOS.
        RolloutRecord(
            tokens=[16, 17, 31, 41, 32, 2], attention_mask=[1] * 6, response_length=4, loss_mask=[1, 0, 1, 1]),
    ])

    response_batch = ResponseTokenBatch.from_rollout(rollout, pad_token_id=0)

    assert response_batch.input_ids.tolist() == [[0, 0, 11, 12, 21, 2], [13, 14, 15, 22, 23, 2],
                                                 [16, 17, 31, 41, 32, 2]]
    assert response_batch.attention_mask.tolist() == [[0, 0, 1, 1, 1, 1], [1] * 6, [1] * 6]
    assert response_batch.target_ids.tolist() == [[0, 11, 12, 21, 2], [14, 15, 22, 23, 2], [17, 31, 41, 32, 2]]
    assert response_batch.response_mask.tolist() == [[False, False, False, True,
                                                      True], [False, False, True, True, True],
                                                     [False, True, False, True, True]]


@pytest.mark.parametrize(
    "sample",
    [
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=3),
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=4),
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=-1),
        RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=2, loss_mask=[1]),
    ],
)
def test_response_token_batch_rejects_invalid_response_boundaries(sample):
    with pytest.raises(ValueError):
        ResponseTokenBatch.from_rollout(RolloutBatch(samples=[sample]))


def test_response_token_batch_aligns_causal_logits():
    rollout = RolloutBatch(samples=[RolloutRecord(tokens=[1, 2, 3], attention_mask=[1, 1, 1], response_length=1)])
    logits = torch.randn(1, 3, 5)

    response_batch = ResponseTokenBatch.from_rollout(rollout)

    assert torch.equal(response_batch.select_causal_logits(logits), logits[:, :-1])
    with pytest.raises(ValueError, match="sequence"):
        response_batch.select_causal_logits(torch.randn(1, 2, 5))


@pytest.mark.parametrize(
    "beta, expected",
    [
        (
            0.0,
            0.5 * torch.log(torch.tensor(0.5 / 0.8)) + 0.5 * torch.log(torch.tensor(0.5 / 0.2)),
        ),
        (
            1.0,
            0.8 * torch.log(torch.tensor(0.8 / 0.5)) + 0.2 * torch.log(torch.tensor(0.2 / 0.5)),
        ),
        (
            0.5,
            0.5 * (0.5 * torch.log(torch.tensor(0.5 / 0.65)) + 0.5 * torch.log(torch.tensor(0.5 / 0.35))) + 0.5 *
            (0.8 * torch.log(torch.tensor(0.8 / 0.65)) + 0.2 * torch.log(torch.tensor(0.2 / 0.35))),
        ),
    ],
)
def test_generalized_jsd_loss_covers_beta_variants(beta, expected):
    student_logits = torch.log(torch.tensor([[[0.8, 0.2], [0.5, 0.5]]]))
    teacher_logits = torch.log(torch.tensor([[[0.5, 0.5], [0.9, 0.1]]]))
    response_mask = torch.tensor([[True, False]])

    output = generalized_jsd_loss(student_logits, teacher_logits, response_mask, beta=beta)

    assert isinstance(output, JSDLossOutput)
    assert output.valid_token_count.item() == 1
    assert torch.allclose(output.loss_sum, expected)
    assert torch.allclose(output.loss, expected)


def test_generalized_jsd_loss_supports_top_k_and_empty_response_masks():
    student_logits = torch.tensor([[[20.0, 0.0, 0.0]]])
    teacher_logits = torch.tensor([[[0.0, 10.0, 9.0]]])
    response_mask = torch.tensor([[True]])

    top_one = generalized_jsd_loss(student_logits, teacher_logits, response_mask, teacher_top_k=1)
    empty = generalized_jsd_loss(student_logits, teacher_logits, torch.tensor([[False]]))

    assert top_one.loss.item() == pytest.approx(0.0)
    assert empty.valid_token_count.item() == 0
    assert empty.loss.item() == pytest.approx(0.0)


# --- interface conformance via FakeRollout ---------------------------------


class FakeRollout(RolloutEngine):
    """A deterministic stub that appends ``[42] * max_new_tokens`` to ``n_samples_per_prompt`` copies of each prompt."""

    name = "fake"

    def __init__(self, response_token: int = 42):
        self.response_token = response_token
        self.sync_calls: list = []

    def generate(self, request: RolloutRequest, sampling: SamplingConfig) -> RolloutBatch:
        response = [self.response_token] * sampling.max_new_tokens
        samples = []
        for group_index, prompt in enumerate(request.samples):
            for _ in range(sampling.n_samples_per_prompt):
                samples.append(
                    replace(prompt,
                            group_index=group_index,
                            tokens=prompt.tokens + response,
                            attention_mask=prompt.attention_mask + [1] * len(response),
                            loss_mask=[1] * len(response),
                            response_length=len(response),
                            status=RolloutRecord.Status.TRUNCATED))
        return RolloutBatch(samples=samples)

    def sync_weights(self, step: int) -> None:
        self.sync_calls.append(step)


def _prompt(tokens, attention_mask=None):
    return RolloutRecord(tokens=tokens, attention_mask=attention_mask or [1] * len(tokens))


def test_fake_rollout_shape_basic():
    fake = FakeRollout()
    req = RolloutRequest(samples=[_prompt([1, 2, 3]), _prompt([4, 5, 6])])
    out = fake.generate(req, SamplingConfig(max_new_tokens=4))
    assert out.batch_size == 2
    assert out.seq_len == 7
    assert [sample.tokens[3:] for sample in out.samples] == [[42] * 4] * 2
    assert [sample.response_length for sample in out.samples] == [4, 4]


def test_fake_rollout_with_n_samples():
    fake = FakeRollout()
    req = RolloutRequest(samples=[_prompt([1, 2]), _prompt([3, 4])])
    out = fake.generate(req, SamplingConfig(max_new_tokens=3, n_samples_per_prompt=4))
    assert out.batch_size == 8
    assert out.seq_len == 5
    assert [sample.group_index for sample in out.samples] == [0] * 4 + [1] * 4


def test_fake_rollout_left_padded_prompts():
    fake = FakeRollout()
    # Prompt B's first two positions are padding, so the response still follows its
    # last real token and the padding stays masked out.
    req = RolloutRequest(samples=[_prompt([1, 2, 3, 4]), _prompt([0, 0, 5, 6], [0, 0, 1, 1])])
    out = fake.generate(req, SamplingConfig(max_new_tokens=2))
    assert out.samples[1].tokens == [0, 0, 5, 6, 42, 42]
    assert out.samples[1].attention_mask == [0, 0, 1, 1, 1, 1]


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


# --- GRPO ---------------------------------------------------------------------


def _grouped(group_indices):
    return [RolloutRecord(group_index=group_index) for group_index in group_indices]


def test_group_advantages_normalize_within_each_prompt_group():
    records = _grouped([0, 0, 1, 1, 2])
    rewards = [1.0, -1.0, 1.0, 1.0, 1.0]

    # Group 0 has mean 0 and sample std sqrt(2). Group 1 has equal rewards and group 2 one record, so both get 0.
    expected = torch.tensor([2**-0.5, -2**-0.5, 0.0, 0.0, 0.0])
    assert torch.allclose(group_advantages(records, rewards, eps=0.0), expected)
    assert group_advantages(records, rewards, scale_by_std=False).tolist() == [1.0, -1.0, 0.0, 0.0, 0.0]


def test_group_advantages_reject_ungrouped_records_and_mismatched_rewards():
    with pytest.raises(ValueError, match="group_index"):
        group_advantages([RolloutRecord()], [1.0])
    with pytest.raises(ValueError, match="rewards"):
        group_advantages(_grouped([0, 0]), [1.0])


def test_old_log_probs_line_up_with_the_response_targets():
    # A multi-turn record with a model token, a tool token (log-prob 0.0, loss mask 0) and a model token.
    multi_turn = RolloutRecord(tokens=[5, 6, 7, 8, 9],
                               attention_mask=[1] * 5,
                               response_length=3,
                               loss_mask=[1, 0, 1],
                               rollout_log_probs=[-0.1, 0.0, -0.3])
    single_turn = RolloutRecord(tokens=[5, 6, 7], attention_mask=[1] * 3, response_length=1, rollout_log_probs=[-0.2])
    rollout = RolloutBatch(samples=[multi_turn, single_turn])

    response_batch = ResponseTokenBatch.from_rollout(rollout)
    old_log_probs = old_log_probs_from_rollout(rollout)

    assert old_log_probs.shape == response_batch.response_mask.shape
    # Under the response mask, each target gets the log-prob the rollout recorded for that token.
    assert old_log_probs[response_batch.response_mask].tolist() == pytest.approx([-0.1, -0.3, -0.2])
    assert response_batch.target_ids[response_batch.response_mask].tolist() == [7, 9, 7]


def test_old_log_probs_need_the_rollout_to_record_them():
    record = RolloutRecord(tokens=[5, 6], attention_mask=[1, 1], response_length=1)

    with pytest.raises(ValueError, match="return_log_probs"):
        old_log_probs_from_rollout(RolloutBatch(samples=[record]))


def _ppo_case():
    """Ratios 1.5 / 0.5 under advantages +1 and -1; the last token is masked out."""
    old_log_probs = torch.zeros(2, 2)
    log_probs = torch.log(torch.tensor([[1.5, 0.5], [0.5, 1.5]]))
    advantages = torch.tensor([1.0, -1.0])
    response_mask = torch.tensor([[True, True], [True, False]])
    return log_probs, old_log_probs, advantages, response_mask


@pytest.mark.parametrize(
    "aggregation, expected_loss",
    [
        # Row 0's token losses are [-1.28 (ratio clipped at 1 + 0.28), -0.5] and row 1's [0.8 (clipped at 1 - 0.2)].
        ("token_mean", (-1.28 - 0.5 + 0.8) / 3),
        ("sample_mean", ((-1.28 - 0.5) / 2 + 0.8) / 2),
    ],
)
def test_grpo_policy_loss_clips_both_sides_and_aggregates(aggregation, expected_loss):
    log_probs, old_log_probs, advantages, response_mask = _ppo_case()

    output = grpo_policy_loss(log_probs,
                              old_log_probs,
                              advantages,
                              response_mask,
                              clip_low=0.2,
                              clip_high=0.28,
                              aggregation=aggregation)

    assert isinstance(output, GRPOLossOutput)
    assert output.loss.item() == pytest.approx(expected_loss)
    assert output.loss_sum.item() == pytest.approx(expected_loss * output.loss_normalizer.item())
    assert output.loss_normalizer.item() == (3 if aggregation == "token_mean" else 2)
    assert output.valid_token_count.item() == 3
    # Two of the three response tokens were clipped.
    assert output.clip_fraction.item() == pytest.approx(2 / 3)
    assert output.kl is None


def test_grpo_policy_loss_gradient_is_the_policy_gradient_inside_the_clip_range():
    log_probs = torch.zeros(1, 2, requires_grad=True)
    advantages = torch.tensor([2.0])

    output = grpo_policy_loss(log_probs, torch.zeros(1, 2), advantages, torch.tensor([[True, False]]))
    output.loss.backward()

    # At ratio 1, d(-A * ratio) / d(log_prob) = -A on the one response token, and nothing on the masked one.
    assert log_probs.grad.tolist() == [[-2.0, 0.0]]


def test_grpo_policy_loss_adds_the_k3_kl_penalty():
    log_probs = torch.zeros(1, 2)
    ref_log_probs = torch.log(torch.tensor([[2.0, 1.0]]))
    response_mask = torch.ones(1, 2, dtype=torch.bool)

    output = grpo_policy_loss(log_probs,
                              log_probs,
                              torch.zeros(1),
                              response_mask,
                              ref_log_probs=ref_log_probs,
                              kl_coef=0.1)

    # k3 = exp(d) - d - 1 with d = log(2) on the first token and 0 on the second.
    token_kl = 1.0 - torch.log(torch.tensor(2.0)).item()
    assert output.kl.item() == pytest.approx(token_kl / 2)
    assert output.loss.item() == pytest.approx(0.1 * token_kl / 2)


def test_grpo_policy_loss_of_an_empty_response_is_zero():
    log_probs, old_log_probs, advantages, _ = _ppo_case()

    for aggregation in ("token_mean", "sample_mean"):
        output = grpo_policy_loss(log_probs,
                                  old_log_probs,
                                  advantages,
                                  torch.zeros(2, 2, dtype=torch.bool),
                                  aggregation=aggregation)
        assert output.loss.item() == 0.0
        assert output.loss_normalizer.item() == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({
            "kl_coef": 0.1
        }, "ref_log_probs"),
        ({
            "aggregation": "seq_sum"
        }, "aggregation"),
        ({
            "clip_low": -0.1
        }, "non-negative"),
    ],
)
def test_grpo_policy_loss_rejects_invalid_options(kwargs, message):
    log_probs, old_log_probs, advantages, response_mask = _ppo_case()

    with pytest.raises(ValueError, match=message):
        grpo_policy_loss(log_probs, old_log_probs, advantages, response_mask, **kwargs)
