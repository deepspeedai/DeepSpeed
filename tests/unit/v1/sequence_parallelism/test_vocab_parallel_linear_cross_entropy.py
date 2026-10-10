# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size_list
from deepspeed.sequence.linear_cross_entropy import vocab_parallel_linear_cross_entropy
from unit.common import DistributedTest


def _inputs(vocab_size=17, hidden_size=8, device="cpu", dtype=torch.float32, bias=True):
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    bias_tensor = torch.randn(vocab_size, device=device, dtype=dtype) if bias else None
    target = torch.randint(0, vocab_size, (2, 5), device=device)
    target[0, 1] = -100
    target[1, 4] = -100
    return hidden, weight, bias_tensor, target


def _leaf(tensor):
    return None if tensor is None else tensor.detach().clone().requires_grad_(True)


def _reference(hidden, weight, bias, target, reduction):
    logits = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
    return F.cross_entropy(logits.view(-1, weight.shape[0]), target.view(-1), reduction=reduction)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("chunk_size", [None, 1, 3])
def test_matches_linear_then_cross_entropy(reduction, use_bias, chunk_size):
    hidden, weight, bias, target = _inputs(bias=use_bias)
    ref_args = [_leaf(hidden), _leaf(weight), _leaf(bias)]
    args = [_leaf(hidden), _leaf(weight), _leaf(bias)]

    expected = _reference(*ref_args, target, reduction)
    if reduction == "none":
        expected = expected.view_as(target)
    actual = vocab_parallel_linear_cross_entropy(args[0],
                                                 args[1],
                                                 target,
                                                 bias=args[2],
                                                 reduction=reduction,
                                                 chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected)

    upstream = torch.rand_like(expected) + 0.5
    actual.backward(upstream)
    expected.backward(upstream)
    for got, want in zip(args, ref_args):
        if want is not None:
            torch.testing.assert_close(got.grad, want.grad)


def test_all_ignored_is_zero():
    hidden, weight, _, target = _inputs(bias=False)
    hidden, weight = _leaf(hidden), _leaf(weight)
    loss = vocab_parallel_linear_cross_entropy(hidden, weight, torch.full_like(target, -100))

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


def test_frozen_weight_only_produces_hidden_grad():
    hidden, weight, _, target = _inputs(bias=False)
    hidden = _leaf(hidden)
    vocab_parallel_linear_cross_entropy(hidden, weight, target).backward()

    reference_hidden = _leaf(hidden)
    _reference(reference_hidden, weight, None, target, "mean").backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    assert weight.grad is None


def test_fp16_loss_scale_does_not_overflow():
    # A loss scale of 65536 is not representable in fp16, but the scaled gradients are.
    hidden = _leaf(torch.ones(4, 1, dtype=torch.float16))
    weight = _leaf(torch.zeros(2, 1, dtype=torch.float16))
    bias = _leaf(torch.zeros(2, dtype=torch.float16))
    target = torch.zeros(4, dtype=torch.long)
    reference = [_leaf(hidden), _leaf(weight), _leaf(bias)]

    (vocab_parallel_linear_cross_entropy(hidden, weight, target, bias=bias) * 65536).backward()
    (_reference(*reference, target, "mean") * 65536).backward()

    for got, want in zip((hidden, weight, bias), reference):
        torch.testing.assert_close(got.grad, want.grad)


def test_fp16_product_cancelled_by_bias_stays_finite():
    # The product 256 * 256 overflows fp16 but the bias brings the logits back to 32, as in F.linear.
    hidden = _leaf(torch.tensor([[256.0]], dtype=torch.float16))
    weight = _leaf(torch.tensor([[256.0], [256.0]], dtype=torch.float16))
    bias = _leaf(torch.tensor([-65504.0, -65504.0], dtype=torch.float16))
    target = torch.tensor([0])
    args = [_leaf(hidden), _leaf(weight), _leaf(bias)]

    actual = vocab_parallel_linear_cross_entropy(*args[:2], target, bias=args[2])
    expected = F.cross_entropy(F.linear(hidden, weight, bias).float(), target)
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    for got, want in zip(args, (hidden, weight, bias)):
        torch.testing.assert_close(got.grad, want.grad)


def test_many_low_precision_chunks_accumulate_parameter_gradients_exactly():
    # The weight gradient sums 1024 per-token contributions; it must not depend on chunk_size.
    hidden = torch.ones(1024, 1, dtype=torch.bfloat16)
    weight = _leaf(torch.zeros(2, 1, dtype=torch.bfloat16))
    bias = _leaf(torch.zeros(2, dtype=torch.bfloat16))
    target = torch.zeros(1024, dtype=torch.long)
    reference = [_leaf(weight), _leaf(bias)]

    vocab_parallel_linear_cross_entropy(hidden, weight, target, bias=bias, chunk_size=1).backward()
    _reference(hidden, *reference, target, "mean").backward()

    torch.testing.assert_close(weight.grad, reference[0].grad)
    torch.testing.assert_close(bias.grad, reference[1].grad)


def test_hidden_gradient_sums_many_vocab_entries_exactly():
    # Each token's hidden gradient sums 1023 contributions of 1/1024. A bf16 running sum stalls
    # near 0.5, so the result must not depend on how the vocabulary is split into blocks.
    hidden = torch.zeros(1024, 1, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(1024, 1, dtype=torch.bfloat16)
    weight[0] = 0
    target = torch.zeros(1024, dtype=torch.long)
    reference = _leaf(hidden)

    vocab_parallel_linear_cross_entropy(hidden, weight, target, reduction="sum", chunk_size=1).backward()
    _reference(reference, weight, None, target, "sum").backward()

    torch.testing.assert_close(hidden.grad, reference.grad)


def test_autocast_low_precision_activations_with_fp32_parameters():
    hidden, weight, bias, target = _inputs()
    hidden = _leaf(hidden.bfloat16())
    weight, bias = _leaf(weight), _leaf(bias)
    reference = [_leaf(hidden), _leaf(weight), _leaf(bias)]

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = vocab_parallel_linear_cross_entropy(hidden, weight, target, bias=bias)
        logits = F.linear(*reference)
    expected = F.cross_entropy(logits.float().view(-1, weight.shape[0]), target.view(-1))
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for got, want in zip((hidden, weight, bias), reference):
        assert got.grad.dtype == want.dtype
        torch.testing.assert_close(got.grad, want.grad, atol=2e-2, rtol=2e-2)


def test_validates_inputs():
    hidden, weight, _, target = _inputs(bias=False)
    with pytest.raises(ValueError, match="Unsupported reduction"):
        vocab_parallel_linear_cross_entropy(hidden, weight, target, reduction="avg")
    with pytest.raises(ValueError, match="matching non-hidden dimensions"):
        vocab_parallel_linear_cross_entropy(hidden, weight, target[:, :-1])
    with pytest.raises(ValueError, match="local_vocab, hidden"):
        vocab_parallel_linear_cross_entropy(hidden, weight[:, :-1], target)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        vocab_parallel_linear_cross_entropy(hidden, weight, target, chunk_size=0)
    with pytest.raises(ValueError, match="out of range"):
        vocab_parallel_linear_cross_entropy(hidden, weight, torch.full_like(target, 17))


class TestVocabParallelLinearCrossEntropyTP(DistributedTest):
    world_size = 2

    # chunk_size=None checks that ranks with different shard sizes still agree on the chunking,
    # since every chunk issues collectives.
    @pytest.mark.parametrize("chunk_size", [None, 3])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_uneven_vocab_matches_full_vocab(self, dtype, chunk_size):
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        vocab_size = 17
        partition_sizes = get_shard_size_list(vocab_size, self.world_size, AutoTPMeta(), "lm_head")
        start = sum(partition_sizes[:rank])
        end = start + partition_sizes[rank]

        hidden, weight, bias, target = _inputs(vocab_size, device=device, dtype=dtype)
        ref_args = [_leaf(hidden), _leaf(weight), _leaf(bias)]
        args = [_leaf(hidden), _leaf(weight[start:end]), _leaf(bias[start:end])]

        expected = _reference(*ref_args, target, "mean")
        actual = vocab_parallel_linear_cross_entropy(args[0],
                                                     args[1],
                                                     target,
                                                     bias=args[2],
                                                     tp_group=dist.get_world_group(),
                                                     vocab_start_index=start,
                                                     vocab_end_index=end,
                                                     chunk_size=chunk_size)
        actual.backward()
        expected.backward()

        tolerance = {} if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(args[0].grad, ref_args[0].grad, **tolerance)
        torch.testing.assert_close(args[1].grad, ref_args[1].grad[start:end], **tolerance)
        torch.testing.assert_close(args[2].grad, ref_args[2].grad[start:end], **tolerance)
