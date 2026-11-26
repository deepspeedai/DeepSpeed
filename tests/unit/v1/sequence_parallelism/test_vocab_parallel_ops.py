# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F

import deepspeed.comm as dist
import deepspeed.sequence.vocab_parallel_ops as vocab_parallel_ops
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size_list
from deepspeed.sequence.vocab_parallel_ops import (vocab_parallel_entropy, vocab_parallel_jsd, vocab_parallel_kl_div,
                                                   vocab_parallel_logprobs)
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


def reference_jsd(student_logits, teacher_logits, beta):
    log_q = F.log_softmax(student_logits.double(), dim=-1)
    log_p = F.log_softmax(teacher_logits.double(), dim=-1)
    if beta == 0:
        return F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(-1)
    if beta == 1:
        return F.kl_div(log_p, log_q, reduction="none", log_target=True).sum(-1)
    mixture = (1 - beta) * log_q.exp() + beta * log_p.exp()
    return ((1 - beta) * F.kl_div(mixture.log(), log_q, reduction="none", log_target=True) +
            beta * F.kl_div(mixture.log(), log_p, reduction="none", log_target=True)).sum(-1)


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


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("frozen", ["neither", "teacher", "student"])
def test_jsd_matches_dense_oracle(beta, dtype, frozen):
    torch.manual_seed(12)
    # A sequence slice exercises the non-contiguous causal-shift layout.
    student = torch.randn(2, 4, 19).to(dtype)[:, :-1]
    teacher = torch.randn(2, 4, 19).to(dtype)[:, :-1]
    actual_student = student.detach().requires_grad_(frozen != "student")
    actual_teacher = teacher.detach().requires_grad_(frozen != "teacher")
    expected_student, expected_teacher = leaf(student.double()), leaf(teacher.double())
    weight = torch.randn(2, 3)
    actual = vocab_parallel_jsd(actual_student, actual_teacher, beta=beta)
    expected = reference_jsd(expected_student, expected_teacher, beta)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-2)
    for actual_input, expected_input in [(actual_student, expected_student), (actual_teacher, expected_teacher)]:
        if actual_input.requires_grad:
            torch.testing.assert_close(actual_input.grad.double(), expected_input.grad, atol=atol, rtol=rtol)
        else:
            assert actual_input.grad is None


def test_jsd_disjoint_support_and_undefined_rows():
    student = leaf(torch.tensor([[0.0, -float("inf")], [-float("inf"), -float("inf")]]))
    teacher = leaf(torch.tensor([[-float("inf"), 0.0], [1.0, 0.0]]))
    actual = vocab_parallel_jsd(student, teacher)
    torch.testing.assert_close(actual[:1], torch.tensor([2.0]).log())
    assert torch.isnan(actual[1])
    actual.masked_fill_(torch.tensor([False, True]), 0).sum().backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))
    torch.testing.assert_close(teacher.grad, torch.zeros_like(teacher))


def test_jsd_independent_logit_shifts():
    base = torch.tensor([[0.0, 8.0, -8.0, 16.0]])
    other = torch.tensor([[16.0, -8.0, 8.0, 0.0]])
    shifted_student, shifted_teacher = leaf(base + 1e8), leaf(other - 1e8)
    student, teacher = leaf(base), leaf(other)
    actual = vocab_parallel_jsd(shifted_student, shifted_teacher, beta=0.3)
    expected = vocab_parallel_jsd(student, teacher, beta=0.3)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(shifted_student.grad, student.grad)
    torch.testing.assert_close(shifted_teacher.grad, teacher.grad)


@pytest.mark.parametrize("beta", [-0.1, 1.1, float("nan")])
def test_jsd_rejects_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        vocab_parallel_jsd(torch.randn(2, 7), torch.randn(2, 7), beta=beta)


def test_jsd_rejects_invalid_logits():
    with pytest.raises(ValueError, match="same shape"):
        vocab_parallel_jsd(torch.randn(2, 7), torch.randn(2, 8))
    with pytest.raises(ValueError, match="vocabulary dimension"):
        vocab_parallel_jsd(torch.tensor(1.0), torch.tensor(2.0))
    with pytest.raises(ValueError, match="floating point"):
        vocab_parallel_jsd(torch.ones(2, 7, dtype=torch.long), torch.randn(2, 7))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout,beta", [("contiguous", 0.1), ("sequence_slice", 0.5), ("strided_vocab", 0.9)])
def test_jsd_triton_matches_eager(dtype, layout, beta):
    device = torch.device(get_accelerator().current_device_name())
    probe = torch.empty(0, device=device)
    if not vocab_parallel_ops._use_jsd_triton(probe, probe):
        pytest.skip("JSD Triton kernels are not supported on this device")
    torch.manual_seed(15)
    student = (torch.randn(6, 5000) * 3).to(dtype)
    teacher = (torch.randn(6, 5000) * 3).to(dtype)
    student[1, :2500] = -float("inf")
    teacher[1, 2500:] = -float("inf")
    student[2] = -float("inf")
    student[3, :2500] = -float("inf")
    teacher[3, :2500] = -float("inf")
    student[4] = teacher[4]  # coincident distributions
    expected_student, expected_teacher = leaf(student), leaf(teacher)
    expected = vocab_parallel_jsd(expected_student, expected_teacher, beta=beta)
    weight = torch.linspace(-1, 1, 6)
    expected.masked_fill(torch.isnan(expected), 0).mul(weight).sum().backward()

    def embed(logits):
        if layout == "sequence_slice":
            padded = torch.full((2, 4, 5000), float("nan"), dtype=dtype)
            padded[:, :-1] = logits.view(2, 3, 5000)
            padded = leaf(padded.to(device))
            return padded, padded[:, :-1]
        if layout == "strided_vocab":
            padded = torch.full((6, 10000), float("nan"), dtype=dtype)
            padded[:, ::2] = logits
            padded = leaf(padded.to(device))
            return padded, padded[:, ::2]
        padded = leaf(logits.to(device))
        return padded, padded

    padded_student, local_student = embed(student)
    padded_teacher, local_teacher = embed(teacher)
    actual = vocab_parallel_jsd(local_student, local_teacher, beta=beta)
    actual.masked_fill(torch.isnan(actual), 0).mul(weight.to(device).view_as(actual)).sum().backward()
    torch.testing.assert_close(actual.detach().cpu().reshape(6), expected, atol=1e-5, rtol=1e-4, equal_nan=True)
    atol, rtol = (1e-6, 1e-4) if dtype == torch.float32 else (1e-3, 2e-2)
    for padded, reference in [(padded_student, expected_student), (padded_teacher, expected_teacher)]:
        grad = padded.grad
        if layout == "sequence_slice":
            torch.testing.assert_close(grad[:, -1], torch.zeros_like(grad[:, -1]))
            grad = grad[:, :-1]
        elif layout == "strided_vocab":
            torch.testing.assert_close(grad[:, 1::2], torch.zeros_like(grad[:, 1::2]))
            grad = grad[:, ::2]
        torch.testing.assert_close(grad.cpu().reshape(6, 5000), reference.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("frozen", ["student", "teacher"])
def test_jsd_triton_frozen_inputs_and_extreme_logits(frozen):
    device = torch.device(get_accelerator().current_device_name())
    probe = torch.empty(0, device=device)
    if not vocab_parallel_ops._use_jsd_triton(probe, probe):
        pytest.skip("JSD Triton kernels are not supported on this device")
    student = torch.full((2, 5000), -3e38, device=device)
    student[0, 0] = 3e38
    student[1, -1] = 3e38
    teacher = student.clone()
    student.requires_grad_(frozen != "student")
    teacher.requires_grad_(frozen != "teacher")
    out = vocab_parallel_jsd(student, teacher, beta=0.3)
    out.sum().backward()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-7, rtol=0)
    for tensor in (student, teacher):
        if tensor.requires_grad:
            torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor), atol=1e-7, rtol=0)
        else:
            assert tensor.grad is None


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
              vocab_parallel_kl_div(actual_student, actual_teacher, reverse=True) +
              vocab_parallel_jsd(actual_student, actual_teacher))
    expected = (reference_logprobs(expected_student, index).sum(dim=-1) + reference_entropy(expected_student) +
                reference_kl(expected_student, expected_teacher, True) +
                reference_jsd(expected_student, expected_teacher, 0.5).float())

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
        vocab_parallel_jsd(logits, torch.randn(0, 11)),
    ]
    sum(output.sum() for output in outputs).backward()

    assert [tuple(output.shape) for output in outputs] == [(0, ), (0, 3), (0, ), (0, ), (0, )]
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


def _triton_device():
    from deepspeed.sequence.vocab_parallel_ops import _use_triton
    device = torch.device(get_accelerator().current_device_name())
    if not _use_triton(torch.empty(0, device=device)):
        pytest.skip("the Triton kernels are not used on this device")
    return device


def _triton_edge_case_logits(dtype):
    torch.manual_seed(9)
    # 5000 columns span two kernel blocks, so the running max and sum cross a block boundary.
    student = (torch.randn(6, 5000) * 3).to(dtype)
    teacher = (torch.randn(6, 5000) * 3).to(dtype)
    student[1, :2500] = float("-inf")
    teacher[1, :2500] = float("-inf")
    student[2] = float("-inf")
    teacher[3, 4321] = float("-inf")
    student[3, 4321] = -200.0
    student[4] += 1000.0
    teacher[4] += 1000.0
    return student, teacher


def _run_on(device, op, student, teacher, index):
    student = leaf(student.to(device))
    teacher = leaf(teacher.to(device))
    if op == "logprobs":
        out = vocab_parallel_logprobs(student, index.to(device))
    elif op == "entropy":
        out = vocab_parallel_entropy(student)
    else:
        out = vocab_parallel_kl_div(student, teacher, reverse=(op == "reverse_kl"))
    weight = torch.linspace(-1.0, 1.0, out.numel()).view_as(out)
    finite = torch.isfinite(out.detach())
    (out.where(finite, 0.0) * weight.to(device)).sum().backward()
    teacher_grad = teacher.grad.cpu() if teacher.grad is not None else None
    return out.detach().cpu(), student.grad.cpu(), teacher_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op", ["logprobs", "entropy", "forward_kl", "reverse_kl"])
def test_triton_kernels_match_eager_path(op, dtype):
    device = _triton_device()
    student, teacher = _triton_edge_case_logits(dtype)
    index = torch.randint(0, 5000, (6, 4))
    index[0, 1] = index[0, 0]  # duplicate ids accumulate
    index[1, 2] = -100
    index[5] = -100  # a fully ignored row gets zero gradient

    expected = _run_on("cpu", op, student, teacher, index)
    actual = _run_on(device, op, student, teacher, index)

    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1.6e-2, 1e-3)
    for actual_tensor, expected_tensor in zip(actual, expected):
        if expected_tensor is None:
            assert actual_tensor is None
            continue
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op", ["entropy", "forward_kl", "reverse_kl"])
def test_triton_kernels_handle_finite_extreme_logits(op, dtype):
    # The gap between -3e38 and 3e38 overflows fp32, so rescaling between the two maxima pairs a zero
    # scale with an infinite shift. Each row is still one-hot: entropy and KL(p || p) are 0 with zero
    # gradient. The peak sits in the first block in one row and in the last block in the other, so the
    # maximum is reached before and after the other lanes' running maxima.
    device = _triton_device()
    logits = torch.full((2, 5000), -3e38)
    logits[0, 0] = 3e38
    logits[1, -1] = 3e38
    logits = logits.to(dtype)
    index = torch.zeros(2, 1, dtype=torch.long)

    out, student_grad, teacher_grad = _run_on(device, op, logits, logits.clone(), index)

    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(student_grad, torch.zeros_like(student_grad))
    if teacher_grad is not None:
        assert torch.equal(teacher_grad, torch.zeros_like(teacher_grad))


@pytest.mark.parametrize("layout", ["sequence_slice", "strided_vocab"])
@pytest.mark.parametrize("op", ["logprobs", "entropy", "forward_kl", "reverse_kl"])
def test_triton_kernels_match_eager_path_on_non_contiguous_logits(op, layout):
    # A next-token shift such as logits[:, :-1] cannot be viewed as [rows, vocab] without a copy, and a
    # strided vocabulary cannot be read in place at all. Positions outside the view hold NaN, so reading
    # any of them would show up in the outputs or gradients.
    device = _triton_device()
    student, teacher = _triton_edge_case_logits(torch.bfloat16)
    index = torch.randint(0, 5000, (6, 4))
    expected = _run_on("cpu", op, student, teacher, index)

    def embed(logits):
        if layout == "sequence_slice":
            padded = torch.full((2, 4, 5000), float("nan"), dtype=logits.dtype)
            padded[:, :-1] = logits.view(2, 3, 5000)
            padded = leaf(padded.to(device))
            return padded, padded[:, :-1]
        padded = torch.full((6, 10000), float("nan"), dtype=logits.dtype)
        padded[:, ::2] = logits
        padded = leaf(padded.to(device))
        return padded, padded[:, ::2]

    padded_student, student_rows = embed(student)
    padded_teacher, teacher_rows = embed(teacher)
    if op == "logprobs":
        out = vocab_parallel_logprobs(student_rows, index.view(*student_rows.shape[:-1], 4).to(device))
    elif op == "entropy":
        out = vocab_parallel_entropy(student_rows)
    else:
        out = vocab_parallel_kl_div(student_rows, teacher_rows, reverse=(op == "reverse_kl"))
    weight = torch.linspace(-1.0, 1.0, out.numel()).view_as(out)
    finite = torch.isfinite(out.detach())
    (out.where(finite, 0.0) * weight.to(device)).sum().backward()

    def rows_of(padded):
        grad = padded.grad[:, :-1] if layout == "sequence_slice" else padded.grad[:, ::2]
        return grad.cpu().reshape(6, 5000)

    expected_out, expected_student_grad, expected_teacher_grad = expected
    rtol, atol = 1.6e-2, 1e-3
    torch.testing.assert_close(out.detach().cpu().view_as(expected_out),
                               expected_out,
                               rtol=rtol,
                               atol=atol,
                               equal_nan=True)
    # As in the test above, a row whose KL is infinite has NaN gradients under a zero output gradient.
    torch.testing.assert_close(rows_of(padded_student), expected_student_grad, rtol=rtol, atol=atol, equal_nan=True)
    if expected_teacher_grad is not None:
        torch.testing.assert_close(rows_of(padded_teacher),
                                   expected_teacher_grad,
                                   rtol=rtol,
                                   atol=atol,
                                   equal_nan=True)


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

    def test_finite_extreme_logits_across_shards(self):
        # Only rank 1 holds the peak, so rank 0's maximum is rescaled by exp(-6e38) = 0 with an
        # infinite shift when the shards are combined. The distribution is one-hot: entropy and KL(p || p)
        # are 0 with zero gradient.
        device = torch.device(get_accelerator().current_device_name())
        tp_group = dist.get_world_group()
        local = torch.full((2, 3), -3e38, device=device)
        if dist.get_rank() == 1:
            local[:, 1] = 3e38
        student, teacher = leaf(local), leaf(local.clone())

        actual = (vocab_parallel_entropy(student, tp_group=tp_group) +
                  vocab_parallel_kl_div(student, teacher, tp_group=tp_group) +
                  vocab_parallel_kl_div(student, teacher, reverse=True, tp_group=tp_group))
        actual.sum().backward()

        assert torch.equal(actual, torch.zeros_like(actual))
        assert torch.equal(student.grad, torch.zeros_like(student.grad))
        assert torch.equal(teacher.grad, torch.zeros_like(teacher.grad))

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


class TestVocabParallelJSDTP(DistributedTest):
    world_size = [1, 2, 4]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("beta", [0.0, 0.2, 0.5, 1.0])
    def test_loss_and_gradients_match_dense(self, dtype, beta):
        device = torch.device(get_accelerator().current_device_name())
        group = dist.get_world_group()
        tp_size = dist.get_world_size(group)
        torch.manual_seed(13)
        student = torch.randn(2, 3, 17, device=device).to(dtype)
        teacher = torch.randn_like(student)
        weight = torch.randn(2, 3, device=device)
        rank = dist.get_rank()
        start = sum(t.shape[-1] for t in torch.tensor_split(student, tp_size, dim=-1)[:rank])
        local_student = leaf(torch.tensor_split(student, tp_size, dim=-1)[rank])
        local_teacher = leaf(torch.tensor_split(teacher, tp_size, dim=-1)[rank])
        full_student, full_teacher = leaf(student.double()), leaf(teacher.double())
        actual = vocab_parallel_jsd(local_student, local_teacher, beta=beta, tp_group=group)
        expected = reference_jsd(full_student, full_teacher, beta)
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
        end = start + local_student.shape[-1]
        atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-2)
        torch.testing.assert_close(local_student.grad.double(),
                                   full_student.grad[..., start:end],
                                   atol=atol,
                                   rtol=rtol)
        torch.testing.assert_close(local_teacher.grad.double(),
                                   full_teacher.grad[..., start:end],
                                   atol=atol,
                                   rtol=rtol)

    def test_empty_shard_and_fully_masked_local_support(self):
        device = torch.device(get_accelerator().current_device_name())
        group = dist.get_world_group()
        tp_size = dist.get_world_size(group)
        # With TP4 two ranks have empty shards, and the two distributions live on different ranks.
        full_student = torch.tensor([[0.0, -float("inf")]], device=device)
        full_teacher = full_student.flip(-1)
        rank = dist.get_rank()
        student = leaf(torch.tensor_split(full_student, tp_size, dim=-1)[rank])
        teacher = leaf(torch.tensor_split(full_teacher, tp_size, dim=-1)[rank])
        actual = vocab_parallel_jsd(student, teacher, tp_group=group)
        torch.testing.assert_close(actual, torch.tensor([2.0], device=device).log())
        actual.sum().backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))
        torch.testing.assert_close(teacher.grad, torch.zeros_like(teacher))

    def test_head_optimizer_updates_match_dense(self):
        device = torch.device(get_accelerator().current_device_name())
        group = dist.get_world_group()
        torch.manual_seed(14)
        weight = torch.randn(17, 8, device=device) * 0.1
        teacher = torch.randn(2, 3, 17, device=device)
        rank, tp_size = dist.get_rank(), dist.get_world_size(group)
        local_weight = leaf(torch.tensor_split(weight, tp_size, dim=0)[rank])
        full_weight = leaf(weight)
        local_optimizer = torch.optim.SGD([local_weight], lr=0.1)
        full_optimizer = torch.optim.SGD([full_weight], lr=0.1)
        local_teacher = torch.tensor_split(teacher, tp_size, dim=-1)[rank].contiguous()
        mask = torch.tensor([[True, False, True], [False, True, True]], device=device)
        for _ in range(3):
            hidden = torch.randn(2, 3, 8, device=device)
            actual = vocab_parallel_jsd(F.linear(hidden, local_weight), local_teacher, beta=0.3, tp_group=group)
            expected = reference_jsd(F.linear(hidden, full_weight), teacher, 0.3)
            actual_loss = actual.masked_fill(~mask, 0).sum() / mask.sum()
            expected_loss = expected.masked_fill(~mask, 0).sum() / mask.sum()
            torch.testing.assert_close(actual_loss.double(), expected_loss, atol=1e-6, rtol=1e-5)
            local_optimizer.zero_grad()
            full_optimizer.zero_grad()
            actual_loss.backward()
            expected_loss.backward()
            local_optimizer.step()
            full_optimizer.step()
            torch.testing.assert_close(local_weight, torch.tensor_split(full_weight, tp_size, dim=0)[rank])
        assert not torch.equal(full_weight, weight)
