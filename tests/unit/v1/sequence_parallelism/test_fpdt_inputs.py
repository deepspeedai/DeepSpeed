# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace

import pytest
import torch

from deepspeed.sequence.fpdt_layer import FPDT_InputConstruct


def construct(tokens, labels=None, weights=None, mask=None, positions=None, chunk=8, size=1, rank=0):
    args = SimpleNamespace(ds_sequence_parallel_fpdt_chunk_size=chunk)
    return FPDT_InputConstruct(tokens, labels, weights, mask, positions, args, size, rank)


def rank_indices(length, chunk, size):
    # Oracle: split ordinary global chunks evenly among ranks, in source order.
    chunks = torch.arange(length).split(chunk)
    return [torch.cat([part.tensor_split(size)[rank] for part in chunks]) for rank in range(size)]


@pytest.mark.parametrize('length', [8, 3, 9, 19])
@pytest.mark.parametrize('size', [1, 2, 4])
@pytest.mark.parametrize('trailing,dtype', [((), torch.int64), ((2, 3), torch.float64)])
def test_padding_values_and_rank_order(length, size, trailing, dtype):
    """Catch dropped/duplicated real tokens, rank-order errors and nonzero padding weights."""
    chunk, batch = 8, 2
    count = batch * length
    for dim in trailing:
        count *= dim
    tokens = torch.arange(1, count + 1, dtype=dtype).reshape(batch, length, *trailing)
    labels = torch.arange(batch * length, dtype=torch.int64).reshape(batch, length) % 7
    weights = torch.arange(1, batch * length + 1, dtype=torch.float32).reshape(batch, length) / 4
    positions = torch.stack((torch.arange(length) * 3 + 7, torch.arange(length) * 2 - 4)).to(torch.int32)
    originals = [value.clone() for value in (tokens, labels, weights, positions)]
    padded_length = ((length + chunk - 1) // chunk) * chunk
    padding = padded_length - length
    expected = []
    for value in originals[:3]:
        zeros = value.new_zeros((batch, padding, *value.shape[2:]))
        expected.append(torch.cat((value, zeros), dim=1))
    extra_positions = positions[:, -1:] + torch.arange(1, padding + 1, dtype=positions.dtype)
    expected.append(torch.cat((positions, extra_positions), dim=1))
    indices = rank_indices(padded_length, chunk, size)
    mask_order = torch.cat(indices)
    outputs = []
    for rank in range(size):
        builder = construct(tokens, labels, weights, positions=positions, size=size, rank=rank)
        out_tokens, out_labels, out_weights, out_mask, out_positions = builder.generate()
        assert out_mask is None
        for actual, full in zip((out_tokens, out_labels, out_positions), (expected[0], expected[1], expected[3])):
            torch.testing.assert_close(actual, full[:, indices[rank]], rtol=0, atol=0)
            assert actual.dtype == full.dtype and actual.device == full.device
        torch.testing.assert_close(out_weights, expected[2][:, mask_order], rtol=0, atol=0)
        assert out_weights.dtype == weights.dtype and out_weights.device == weights.device
        assert out_weights.sum() == weights.sum()
        outputs.append(out_tokens)
    # All-to-all source-rank concatenation must restore each ordinary global chunk.
    for i, expected_chunk in enumerate(expected[0].split(chunk, dim=1)):
        local_chunks = [value.split(chunk // size, dim=1)[i] for value in outputs]
        torch.testing.assert_close(torch.cat(local_chunks, dim=1), expected_chunk, rtol=0, atol=0)
    for value, before in zip((tokens, labels, weights, positions), originals):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
    outputs[0].zero_()
    torch.testing.assert_close(tokens, originals[0], rtol=0, atol=0)


@pytest.mark.parametrize('size', [1, 2, 4])
def test_aligned_optional_fields_and_mask(size):
    """Aligned callers retain optional None fields and the original attention mask."""
    tokens = torch.arange(16).reshape(2, 8)
    mask = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    builder = construct(tokens, mask=mask, size=size)
    out_tokens, labels, weights, out_mask, positions = builder.generate()
    assert builder.tokens is tokens
    assert out_mask is mask
    assert labels is None and weights is None and positions is None
    torch.testing.assert_close(out_tokens, tokens[:, rank_indices(8, 8, size)[0]], rtol=0, atol=0)


def test_padded_optional_fields_and_noncontiguous_input():
    """Padding accepts absent labels/positions and preserves trailing feature axes."""
    tokens = torch.arange(36, dtype=torch.float32).reshape(2, 2, 9).transpose(1, 2)
    before = tokens.clone()
    out, labels, weights, mask, positions = construct(tokens, weights=torch.ones(2, 9), size=2).generate()
    assert labels is None and mask is None and positions is None
    assert out.shape == (2, 8, 2)
    assert weights.shape == (2, 16) and weights.sum() == 18
    torch.testing.assert_close(tokens, before, rtol=0, atol=0)


@pytest.mark.parametrize('length,chunk,size,rank', [(0, 8, 1, 0), (8, 0, 1, 0), (8, -8, 1, 0), (8, 8, 0, 0),
                                                    (8, 8, -1, 0), (8, 7, 2, 0), (8, 2, 4, 0), (8, 8, 2, -1),
                                                    (8, 8, 2, 2)])
def test_invalid_geometry(length, chunk, size, rank):
    """Invalid geometry must fail before division, reshaping or rank slicing."""
    with pytest.raises(ValueError) as error:
        construct(torch.ones(2, length), weights=torch.ones(2, length), chunk=chunk, size=size, rank=rank)
    for name, value in zip(("S", "C", "P", "rank"), (length, chunk, size, rank)):
        assert f"{name}={value}" in str(error.value)


def test_padding_requires_loss_weights():
    with pytest.raises(ValueError, match='loss_mask') as error:
        construct(torch.ones(2, 9))
    assert all(value in str(error.value) for value in ('S=9', 'C=8', 'P=1', 'rank=0'))


def test_padding_rejects_attention_mask():
    with pytest.raises(ValueError, match='attention_mask') as error:
        construct(torch.ones(2, 9), weights=torch.ones(2, 9), mask=torch.ones(9, 9))
    assert all(value in str(error.value) for value in ('S=9', 'C=8', 'P=1', 'rank=0'))
