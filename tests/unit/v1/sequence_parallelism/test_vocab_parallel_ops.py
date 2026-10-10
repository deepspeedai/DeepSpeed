# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F

import deepspeed.comm as dist
import deepspeed.sequence.vocab_parallel_ops as vocab_parallel_ops
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size_list
from deepspeed.sequence.vocab_parallel_ops import vocab_parallel_entropy, vocab_parallel_kl_div, vocab_parallel_logprobs
from unit.common import DistributedTest


def reference_logprobs(logits, index, ignore_index=-100):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    valid = index != ignore_index
    gathered = log_probs.gather(-1, index.clamp(min=0))
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def reference_entropy(logits):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def reference_kl(student_logits, teacher_logits, reverse):
    log_q = F.log_softmax(student_logits.float(), dim=-1)
    log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    if reverse:
        return F.kl_div(log_p, log_q, reduction="none", log_target=True).sum(dim=-1)
    return F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(dim=-1)


def leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


@pytest.mark.parametrize("top_k", [None, 3])
def test_logprobs_matches_log_softmax(top_k):
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 19)
    index_shape = (2, 5) if top_k is None else (2, 5, top_k)
    index = torch.randint(0, 19, index_shape)
    index[0, 1] = -100
    actual_logits, expected_logits = leaf(logits), leaf(logits)
    weight = torch.randn(index_shape)

    actual = vocab_parallel_logprobs(actual_logits, index)
    expected_index = index if top_k is not None else index.unsqueeze(-1)
    expected = reference_logprobs(expected_logits, expected_index)
    if top_k is None:
        expected = expected.squeeze(-1)

    torch.testing.assert_close(actual, expected)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


def test_entropy_matches_reference():
    torch.manual_seed(1)
    logits = torch.randn(3, 4, 23) * 3
    actual_logits, expected_logits = leaf(logits), leaf(logits)
    weight = torch.randn(3, 4)

    actual = vocab_parallel_entropy(actual_logits)
    expected = reference_entropy(expected_logits)

    torch.testing.assert_close(actual, expected)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


@pytest.mark.parametrize("reverse", [False, True])
def test_kl_div_matches_reference_for_student_and_teacher(reverse):
    torch.manual_seed(2)
    student = torch.randn(2, 6, 29) * 2
    teacher = torch.randn(2, 6, 29) * 2
    actual_student, actual_teacher = leaf(student), leaf(teacher)
    expected_student, expected_teacher = leaf(student), leaf(teacher)
    weight = torch.randn(2, 6)

    actual = vocab_parallel_kl_div(actual_student, actual_teacher, reverse=reverse)
    expected = reference_kl(expected_student, expected_teacher, reverse)

    torch.testing.assert_close(actual, expected)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_student.grad, expected_student.grad)
    torch.testing.assert_close(actual_teacher.grad, expected_teacher.grad)


def test_top_k_teacher_forward_kl():
    # Distillation from a teacher that only exposes top-k (id, logprob) pairs: the student
    # gradient must match the same loss computed on full-vocabulary student log-probabilities.
    torch.manual_seed(3)
    student = torch.randn(2, 4, 31)
    teacher_topk_logprobs, teacher_topk_ids = F.log_softmax(torch.randn(2, 4, 31), dim=-1).topk(5, dim=-1)
    actual_student, expected_student = leaf(student), leaf(student)

    def top_k_forward_kl(student_topk_logprobs):
        return (teacher_topk_logprobs.exp() * (teacher_topk_logprobs - student_topk_logprobs)).sum(dim=-1)

    actual = top_k_forward_kl(vocab_parallel_logprobs(actual_student, teacher_topk_ids))
    expected = top_k_forward_kl(reference_logprobs(expected_student, teacher_topk_ids))

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_student.grad, expected_student.grad)


def test_row_chunking_matches_reference(monkeypatch):
    # One row per chunk, so every per-token statistic is assembled from several chunks.
    monkeypatch.setattr(vocab_parallel_ops, "_CHUNK_NUMEL", 1)
    torch.manual_seed(6)
    student = torch.randn(3, 4, 13)
    teacher = torch.randn(3, 4, 13)
    index = torch.randint(0, 13, (3, 4, 2))
    actual_student, actual_teacher = leaf(student), leaf(teacher)
    expected_student, expected_teacher = leaf(student), leaf(teacher)

    actual = (vocab_parallel_logprobs(actual_student, index).sum(dim=-1) + vocab_parallel_entropy(actual_student) +
              vocab_parallel_kl_div(actual_student, actual_teacher, reverse=True))
    expected = (reference_logprobs(expected_student, index).sum(dim=-1) + reference_entropy(expected_student) +
                reference_kl(expected_student, expected_teacher, True))

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_student.grad, expected_student.grad)
    torch.testing.assert_close(actual_teacher.grad, expected_teacher.grad)


def test_empty_token_batch():
    logits = torch.randn(0, 11, requires_grad=True)

    outputs = [
        vocab_parallel_logprobs(logits, torch.zeros(0, dtype=torch.long)),
        vocab_parallel_logprobs(logits, torch.zeros(0, 3, dtype=torch.long)),
        vocab_parallel_entropy(logits),
        vocab_parallel_kl_div(logits, torch.randn(0, 11)),
    ]
    sum(output.sum() for output in outputs).backward()

    assert [tuple(output.shape) for output in outputs] == [(0, ), (0, 3), (0, ), (0, )]
    assert logits.grad.shape == logits.shape


def test_outputs_support_in_place_masking():
    torch.manual_seed(7)
    student = torch.randn(2, 3, 11, requires_grad=True)
    teacher = torch.randn(2, 3, 11)
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    entropy = vocab_parallel_entropy(student)
    kl = vocab_parallel_kl_div(student, teacher, reverse=True)
    (entropy.mul_(mask).sum() + kl.mul_(mask).sum()).backward()

    expected_student = leaf(student)
    ((reference_entropy(expected_student) + reference_kl(expected_student, teacher, True)) * mask).sum().backward()
    torch.testing.assert_close(student.grad, expected_student.grad)


def test_double_backward_raises_instead_of_returning_wrong_values():
    logits = torch.randn(2, 7, requires_grad=True)

    (grad, ) = torch.autograd.grad(vocab_parallel_entropy(logits).sum(), logits, create_graph=True)

    with pytest.raises(RuntimeError):
        grad.sum().backward()


def test_detached_teacher_gets_no_gradient():
    student = torch.randn(2, 3, 11, requires_grad=True)
    teacher = torch.randn(2, 3, 11)

    vocab_parallel_kl_div(student, teacher, reverse=True).sum().backward()

    assert student.grad is not None
    assert teacher.grad is None


def test_bf16_inputs_match_fp32_reference():
    torch.manual_seed(4)
    student = torch.randn(2, 3, 13, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(2, 3, 13, dtype=torch.bfloat16)
    reference_student = leaf(student.float())
    weight = torch.randn(2, 3)

    loss = vocab_parallel_kl_div(student, teacher)
    expected = reference_kl(reference_student, teacher.float(), reverse=False)
    (loss * weight).sum().backward()
    (expected * weight).sum().backward()

    assert loss.dtype == torch.float32
    assert student.grad.dtype == torch.bfloat16
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(student.grad.float(), reference_student.grad, atol=1e-2, rtol=1e-2)


def _all_ops(logits, other_logits, index):
    return {
        "logprobs": lambda z: vocab_parallel_logprobs(z, index),
        "entropy": vocab_parallel_entropy,
        "forward_kl": lambda z: vocab_parallel_kl_div(z, other_logits),
        "reverse_kl": lambda z: vocab_parallel_kl_div(z, other_logits, reverse=True),
    }


@pytest.mark.parametrize("op", ["logprobs", "entropy", "forward_kl", "reverse_kl"])
def test_large_common_offset_is_exact(op):
    # Softmax is shift-invariant, so an offset large enough to swallow the log-normalizer in
    # fp32 must still reproduce the unshifted result. Logits are multiples of 8 so that adding
    # 1e8 is exact in fp32.
    torch.manual_seed(5)
    base = torch.randint(-2, 3, (2, 3, 7)).float() * 8
    other = torch.randn(2, 3, 7)
    index = torch.randint(0, 7, (2, 3))
    shifted_logits, base_logits = leaf(base + 1e8), leaf(base)
    weight = torch.randn(2, 3)

    actual = _all_ops(shifted_logits, other, index)[op](shifted_logits)
    expected = _all_ops(base_logits, other, index)[op](base_logits)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(shifted_logits.grad, base_logits.grad)


@pytest.mark.parametrize("op", ["logprobs", "entropy", "forward_kl", "reverse_kl"])
def test_partially_masked_vocabulary_matches_unmasked_subset(op):
    # Masking tokens to -inf is the same distribution as dropping them from the vocabulary, so
    # the reduced vocabulary is an independent oracle for both values and gradients.
    torch.manual_seed(6)
    keep = torch.tensor([True, False, True, True, False, True, False])
    logits = torch.randn(2, 3, 7)
    other = torch.randn(2, 3, 7)
    logits[..., ~keep] = float("-inf")
    other[..., ~keep] = float("-inf")
    kept_ids = keep.nonzero().squeeze(-1)
    subset_index = torch.randint(0, len(kept_ids), (2, 3))
    masked_logits, subset_logits = leaf(logits), leaf(logits[..., keep])
    weight = torch.randn(2, 3)

    actual = _all_ops(masked_logits, other, kept_ids[subset_index])[op](masked_logits)
    expected = _all_ops(subset_logits, other[..., keep], subset_index)[op](subset_logits)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(masked_logits.grad[..., keep], subset_logits.grad)
    torch.testing.assert_close(masked_logits.grad[..., ~keep], torch.zeros_like(masked_logits.grad[..., ~keep]))


def test_deterministic_distribution_has_zero_entropy_and_kl():
    logits = leaf(torch.tensor([[0.0, float("-inf")]]))

    entropy = vocab_parallel_entropy(logits)
    kl = vocab_parallel_kl_div(logits, logits.detach())
    (entropy + kl).sum().backward()

    torch.testing.assert_close(entropy, torch.zeros(1))
    torch.testing.assert_close(kl, torch.zeros(1))
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_kl_is_infinite_where_target_mass_is_masked():
    # Forward KL(teacher || student) is genuinely infinite when the teacher puts mass on a token
    # the student masked out; the zero-probability handling must not hide that.
    student = torch.tensor([[0.0, float("-inf")]])
    teacher = torch.tensor([[0.0, 0.0]])

    assert torch.isinf(vocab_parallel_kl_div(student, teacher)).all()
    torch.testing.assert_close(vocab_parallel_kl_div(student, teacher, reverse=True), torch.tensor([0.6931472]))


def test_kl_stays_infinite_when_target_probability_underflows():
    # exp(-200) underflows to 0 in fp32, but the teacher still has support on a token the
    # student masked out, so KL(teacher || student) must remain infinite rather than 0 or NaN.
    student = torch.tensor([[0.0, float("-inf")]])
    teacher = torch.tensor([[0.0, -200.0]])

    assert torch.isposinf(vocab_parallel_kl_div(student, teacher)).all()
    assert torch.isposinf(vocab_parallel_kl_div(teacher, student, reverse=True)).all()


@pytest.mark.parametrize("top_k", [None, 2])
def test_ignored_fully_masked_row_gets_zero_gradient(top_k):
    # Padding rows are often fully masked and ignored; their constant 0 output must not
    # inject NaN gradients into the shared logits tensor.
    logits = leaf(torch.tensor([[1.0, 2.0, 0.5], [float("-inf")] * 3]))
    index = torch.tensor([0, -100]) if top_k is None else torch.tensor([[0, 2], [-100, -100]])

    vocab_parallel_logprobs(logits, index).sum().backward()

    torch.testing.assert_close(logits.grad[1], torch.zeros(3))
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("op", ["logprobs", "entropy", "forward_kl", "reverse_kl"])
def test_masked_undefined_row_gets_zero_gradient(op):
    # A fully -inf padding row has no distribution. Once the caller masks its output, it must not
    # inject NaN gradients, and the valid row must still match its own unpadded computation.
    torch.manual_seed(7)
    valid_logits, valid_other = torch.randn(1, 5), torch.randn(1, 5)
    padding = torch.full((1, 5), float("-inf"))
    index = torch.tensor([3, 1])
    padded_logits, unpadded_logits = leaf(torch.cat([valid_logits, padding])), leaf(valid_logits)
    padded_other = leaf(torch.cat([valid_other, padding]))

    actual = _all_ops(padded_logits, padded_other, index)[op](padded_logits)
    expected = _all_ops(unpadded_logits, valid_other, index[:1])[op](unpadded_logits)
    assert torch.isnan(actual[1])
    actual.masked_fill(torch.tensor([False, True]), 0.0).sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual[:1], expected)
    torch.testing.assert_close(padded_logits.grad[:1], unpadded_logits.grad)
    torch.testing.assert_close(padded_logits.grad[1], torch.zeros(5))
    if op.endswith("kl"):
        torch.testing.assert_close(padded_other.grad[1], torch.zeros(5))
        assert torch.isfinite(padded_other.grad).all()


def test_logprobs_trusted_vocab_size_matches_validated_path():
    torch.manual_seed(8)
    logits = torch.randn(2, 3, 11)
    index = torch.randint(0, 11, (2, 3))

    trusted = vocab_parallel_logprobs(logits, index, vocab_start_index=0, vocab_end_index=11, global_vocab_size=11)

    torch.testing.assert_close(trusted, vocab_parallel_logprobs(logits, index))
    with pytest.raises(ValueError, match="out of range"):
        vocab_parallel_logprobs(logits,
                                torch.full((2, 3), 11),
                                vocab_start_index=0,
                                vocab_end_index=11,
                                global_vocab_size=11)


def test_validates_inputs():
    logits = torch.randn(2, 3, 11)
    with pytest.raises(ValueError, match="non-vocabulary dimensions"):
        vocab_parallel_logprobs(logits, torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="out of range"):
        vocab_parallel_logprobs(logits, torch.full((2, 3), 11))
    with pytest.raises(ValueError, match="provided together"):
        vocab_parallel_logprobs(logits, torch.zeros(2, 3, dtype=torch.long), vocab_start_index=0)
    with pytest.raises(ValueError, match="same shape"):
        vocab_parallel_kl_div(logits, torch.randn(2, 3, 12))
    with pytest.raises(ValueError, match="requires explicit"):
        vocab_parallel_logprobs(logits, torch.zeros(2, 3, dtype=torch.long), global_vocab_size=11)


class TestVocabParallelOpsTP(DistributedTest):
    world_size = 2

    def test_uneven_shards_match_full_vocabulary(self):
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        tp_group = dist.get_world_group()
        vocab_size = 17
        partition_sizes = get_shard_size_list(vocab_size, self.world_size, AutoTPMeta(), "lm_head")
        start = sum(partition_sizes[:rank])
        end = start + partition_sizes[rank]

        torch.manual_seed(5)
        student = torch.randn(2, 4, vocab_size, device=device) * 2
        teacher = torch.randn(2, 4, vocab_size, device=device) * 2
        index = torch.randint(0, vocab_size, (2, 4, 3), device=device)
        index[1, 2, 0] = -100
        weight = torch.randn(2, 4, device=device)

        full_student, full_teacher = leaf(student), leaf(teacher)
        local_student = leaf(student[..., start:end])
        local_teacher = leaf(teacher[..., start:end])

        actual = (vocab_parallel_logprobs(
            local_student, index, tp_group=tp_group, vocab_start_index=start, vocab_end_index=end).sum(dim=-1) +
                  vocab_parallel_entropy(local_student, tp_group=tp_group) +
                  vocab_parallel_kl_div(local_student, local_teacher, tp_group=tp_group) +
                  vocab_parallel_kl_div(local_student, local_teacher, reverse=True, tp_group=tp_group))
        expected = (reference_logprobs(full_student, index).sum(dim=-1) + reference_entropy(full_student) +
                    reference_kl(full_student, full_teacher, False) + reference_kl(full_student, full_teacher, True))

        torch.testing.assert_close(actual, expected)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
        torch.testing.assert_close(local_student.grad, full_student.grad[..., start:end])
        torch.testing.assert_close(local_teacher.grad, full_teacher.grad[..., start:end])

    def test_empty_shard_does_not_desynchronize_ranks(self):
        # Rank 0 owns no vocabulary. Entropy and KL need no shard bounds, so they must still match
        # the full vocabulary instead of failing on one rank while the other waits in a collective;
        # logprobs must reject the layout on both ranks together.
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        tp_group = dist.get_world_group()
        vocab_size = 3
        start, end = (0, 0) if rank == 0 else (0, vocab_size)

        torch.manual_seed(9)
        student = torch.randn(2, 4, vocab_size, device=device)
        teacher = torch.randn(2, 4, vocab_size, device=device)
        weight = torch.randn(2, 4, device=device)
        full_student, full_teacher = leaf(student), leaf(teacher)
        local_student = leaf(student[..., start:end])
        local_teacher = leaf(teacher[..., start:end])

        actual = (vocab_parallel_entropy(local_student, tp_group=tp_group) +
                  vocab_parallel_kl_div(local_student, local_teacher, tp_group=tp_group) +
                  vocab_parallel_kl_div(local_student, local_teacher, reverse=True, tp_group=tp_group))
        expected = (reference_entropy(full_student) + reference_kl(full_student, full_teacher, False) +
                    reference_kl(full_student, full_teacher, True))

        torch.testing.assert_close(actual, expected)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
        torch.testing.assert_close(local_student.grad, full_student.grad[..., start:end])
        torch.testing.assert_close(local_teacher.grad, full_teacher.grad[..., start:end])

        index = torch.zeros(2, 4, dtype=torch.long, device=device)
        with pytest.raises(ValueError, match="empty vocabulary shard"):
            vocab_parallel_logprobs(local_student,
                                    index,
                                    tp_group=tp_group,
                                    vocab_start_index=start,
                                    vocab_end_index=end)
