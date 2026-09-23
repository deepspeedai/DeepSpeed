# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""The chunked causal-LM loss must equal Hugging Face's ForCausalLMLoss, in value and in gradient."""

import pytest
import torch

from deepspeed.runtime.chunked_cross_entropy import (ChunkedCausalLMLoss, chunked_cross_entropy,
                                                     install_chunked_causal_lm_loss)

loss_utils = pytest.importorskip("transformers.loss.loss_utils")


def _inputs(batch, seq, vocab, dtype, seed=0, ignore_every=5):
    generator = torch.Generator().manual_seed(seed)
    # A wide logit range exercises the log-sum-exp stabilization.
    logits = (torch.randn(batch, seq, vocab, generator=generator) * 6).to(dtype)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    labels[:, ::ignore_every] = -100
    return logits, labels


def _loss_and_grad(loss_function, logits, labels, vocab, **kwargs):
    leaf = logits.clone().requires_grad_(True)
    loss = loss_function(leaf, labels, vocab, **kwargs)
    loss.backward()
    return loss.detach(), leaf.grad


def _ordered_bits(tensor):
    """Integers whose difference is the distance in representable values, for 16-bit floats."""
    bits = tensor.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -32768 - bits, bits)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("block_rows", [None, 7])
def test_matches_hugging_face_loss_and_gradient(dtype, block_rows):
    vocab = 257
    logits, labels = _inputs(2, 33, vocab, dtype)

    expected_loss, expected_grad = _loss_and_grad(loss_utils.ForCausalLMLoss, logits, labels, vocab)
    loss, grad = _loss_and_grad(ChunkedCausalLMLoss(block_rows=block_rows), logits, labels, vocab)

    assert loss.dtype == expected_loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)
    assert grad.dtype == dtype
    if dtype == torch.float32:
        torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-8)
    else:
        # Both round the same FP32 gradient formula to the logits' dtype once; FP32 evaluation order
        # may move a value that sits on a rounding boundary by one representable step.
        distance = (_ordered_bits(grad) - _ordered_bits(expected_grad)).abs()
        assert distance.max().item() <= 1
        assert (distance == 0).float().mean().item() >= 0.99


def test_normalizes_by_num_items_in_batch_like_hugging_face():
    vocab = 64
    logits, labels = _inputs(2, 17, vocab, torch.bfloat16, seed=3)
    num_items = torch.tensor(50)

    expected_loss, expected_grad = _loss_and_grad(loss_utils.ForCausalLMLoss,
                                                  logits,
                                                  labels,
                                                  vocab,
                                                  num_items_in_batch=num_items)
    loss, grad = _loss_and_grad(ChunkedCausalLMLoss(), logits, labels, vocab, num_items_in_batch=num_items)

    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)
    distance = (_ordered_bits(grad) - _ordered_bits(expected_grad)).abs()
    assert distance.max().item() <= 1


def test_uses_given_shift_labels_like_hugging_face():
    vocab = 64
    logits, labels = _inputs(1, 19, vocab, torch.float32, seed=4)
    shift_labels = torch.roll(labels, shifts=-3, dims=-1)

    expected_loss, expected_grad = _loss_and_grad(loss_utils.ForCausalLMLoss,
                                                  logits,
                                                  labels,
                                                  vocab,
                                                  shift_labels=shift_labels)
    loss, grad = _loss_and_grad(ChunkedCausalLMLoss(), logits, labels, vocab, shift_labels=shift_labels)

    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-8)


def test_ignored_rows_get_no_gradient_and_all_ignored_is_nan():
    logits = torch.randn(6, 11, requires_grad=True)
    target = torch.tensor([3, -100, 5, -100, 0, 10])

    chunked_cross_entropy(logits, target, block_rows=4).backward()
    assert torch.all(logits.grad[1] == 0) and torch.all(logits.grad[3] == 0)
    assert torch.all(logits.grad[0] != 0)

    all_ignored = torch.full((6, ), -100)
    expected = torch.nn.functional.cross_entropy(logits.detach(), all_ignored)
    assert torch.isnan(expected) and torch.isnan(chunked_cross_entropy(logits.detach(), all_ignored))


def test_saves_no_full_vocabulary_float32_tensor():
    """Backward may keep only the given logits and per-row values, not an FP32 copy the size of the logits."""
    rows, vocab = 64, 1000
    logits = torch.randn(rows, vocab, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randint(0, vocab, (rows, ))
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        chunked_cross_entropy(logits, target, block_rows=16)
    float_sizes = [tensor.numel() for tensor in saved if tensor.dtype == torch.float32]
    assert float_sizes and max(float_sizes) <= rows


def _tiny_causal_lm(seed):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(vocab_size=97,
                                      hidden_size=32,
                                      intermediate_size=64,
                                      num_hidden_layers=2,
                                      num_attention_heads=4,
                                      num_key_value_heads=2,
                                      max_position_embeddings=64)
    return transformers.LlamaForCausalLM(config)


def test_installed_loss_trains_a_hugging_face_model_like_the_stock_loss():
    stock = _tiny_causal_lm(seed=11)
    chunked = _tiny_causal_lm(seed=11)
    installed = install_chunked_causal_lm_loss(chunked, block_rows=5)
    assert chunked.loss_function is installed
    input_ids = torch.randint(0, 97, (2, 23), generator=torch.Generator().manual_seed(1))
    labels = input_ids.clone()
    labels[:, :4] = -100

    stock_loss = stock(input_ids=input_ids, labels=labels).loss
    chunked_loss = chunked(input_ids=input_ids, labels=labels).loss
    stock_loss.backward()
    chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, stock_loss, rtol=1e-6, atol=1e-6)
    for (name, stock_parameter), chunked_parameter in zip(stock.named_parameters(), chunked.parameters()):
        torch.testing.assert_close(chunked_parameter.grad, stock_parameter.grad, rtol=1e-4, atol=1e-7, msg=name)


def test_install_refuses_a_model_that_does_not_use_the_stock_loss():
    model = _tiny_causal_lm(seed=0)
    install_chunked_causal_lm_loss(model)
    with pytest.raises(ValueError, match="stock Hugging Face ForCausalLMLoss"):
        install_chunked_causal_lm_loss(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures CUDA allocator peaks")  #ignore-cuda
def test_real_vocabulary_backward_peak_drops_by_the_float32_tensors():
    """At Qwen3's vocabulary, the stock loss holds three FP32 [tokens, vocab] tensors when backward starts."""
    cuda = torch.cuda  #ignore-cuda
    tokens, vocab = 8192, 151936
    generator = torch.Generator(device="cuda").manual_seed(5)
    logits = (torch.randn(1, tokens, vocab, device="cuda", generator=generator) * 3).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (1, tokens), device="cuda", generator=generator)
    float32_logits_bytes = tokens * vocab * 4

    def backward_peak(loss_function):
        leaf = logits.clone().requires_grad_(True)
        cuda.synchronize()
        cuda.reset_peak_memory_stats()
        before = cuda.memory_allocated()
        loss = loss_function(leaf, labels, vocab)
        loss.backward()
        cuda.synchronize()
        return loss.detach(), leaf.grad, cuda.max_memory_allocated() - before

    expected_loss, expected_grad, stock_peak = backward_peak(loss_utils.ForCausalLMLoss)
    loss, grad, chunked_peak = backward_peak(ChunkedCausalLMLoss())

    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-5)
    distance = (_ordered_bits(grad) - _ordered_bits(expected_grad)).abs()
    assert distance.max().item() <= 1
    assert (distance == 0).float().mean().item() >= 0.999
    assert stock_peak - chunked_peak >= 2 * float32_logits_bytes, (stock_peak, chunked_peak)
