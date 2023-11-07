# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List, Optional, Tuple

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.kernels.ragged_ops import RaggedEmbeddingKernel
from ....v2.inference_test_utils import allclose, get_dtypes
from .ragged_testing_utils import build_batch_and_manager


def baseline_implementation(token_ids: torch.Tensor,
                            embedding_table: torch.Tensor,
                            unpadded_size: int,
                            positional_embedding_table: Optional[torch.Tensor] = None,
                            positional_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Baseline implementation for our ragged embedding kernel.
    """
    if unpadded_size == token_ids.shape[0]:
        token_embed = torch.nn.functional.embedding(token_ids, embedding_table)

        if positional_embedding_table is not None:
            pos_embed = torch.nn.functional.embedding(positional_ids, positional_embedding_table)
            token_embed += pos_embed
        return token_embed
    else:
        real_token_ids = token_ids[:unpadded_size]
        output = torch.empty((token_ids.shape[0], embedding_table.shape[1]),
                             dtype=embedding_table.dtype,
                             device=get_accelerator().current_device())
        unpadded_output = torch.nn.functional.embedding(real_token_ids, embedding_table)

        # Positional embeddings aren't padded because it's simulated
        if positional_embedding_table is not None:
            pos_embed = torch.nn.functional.embedding(positional_ids, positional_embedding_table)
            unpadded_output += pos_embed

        output[:unpadded_size] = unpadded_output
        return output


def _ragged_embed_test_helper(sequence_config: List[Tuple[int, int]],
                              embed_dtype: torch.dtype,
                              token_dtype: torch.dtype,
                              embed_dim: int,
                              vocab_size: int,
                              do_padding: bool = False,
                              pos_embed_size: int = -1,
                              pos_embed_offset: int = 0) -> None:
    """
    Helper for embedding test to limit the number of tests to run.

    Params:
        embed_dim (int): Model dimension
        vocab_size (int): Leading dimension on embedding weight
        pos_embed_size (int): Size of positional embedding. If negative, no positional embedding
            is used.
        pos_embed_offset (int): Offset for positional embedding. Effectively, the raw offsets
            of a token into a sequence are offset by this amount into the embedding matrix. (
            i.e. the shape of the positional embeddings is (pos_embed_size + pos_embed_offset
            embed_dim)
    """
    device = get_accelerator().current_device()

    # Heads/Block size are irrelevant here but need something.
    batch, _, _, = build_batch_and_manager(sequence_config, 64, 16, 64, vocab_range=vocab_size, padding=do_padding)

    embedding_table = torch.randn((vocab_size, embed_dim), dtype=embed_dtype, device=device)

    if pos_embed_size > 0:
        pos_embedding_table = torch.randn((pos_embed_size + pos_embed_offset, embed_dim),
                                          dtype=embed_dtype,
                                          device=device)
        positional_ids = torch.cat([
            torch.arange(start_idx, start_idx + seq_len, dtype=token_dtype, device=device)
            for seq_len, start_idx in sequence_config
        ]) + pos_embed_offset
    else:
        pos_embedding_table = None
        positional_ids = None

    baseline_output = baseline_implementation(batch.input_ids().to(token_dtype), embedding_table, batch.current_tokens,
                                              pos_embedding_table, positional_ids)

    kernel = RaggedEmbeddingKernel(embed_dtype, token_dtype, embed_dim)
    output = torch.empty_like(baseline_output)

    kernel(output,
           batch,
           embedding_table,
           position_embed_weight=pos_embedding_table,
           position_embed_offset=pos_embed_offset)

    if do_padding:
        assert output.shape[0] != batch.current_tokens

    assert allclose(output[:batch.current_tokens], baseline_output[:batch.current_tokens])


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize('token_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('embed_dtype', get_dtypes())
def test_dtype_permutations(token_dtype: torch.dtype, embed_dtype: torch.dtype) -> None:
    """
    Validate (on a single problem size) that the kernel support for different data types
    is correct.
    """
    embed_dim = 4096
    vocab_size = 50304

    _ragged_embed_test_helper([(256, 0)], embed_dtype, token_dtype, embed_dim, vocab_size)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize('vocab_size, embed_dim', [(1024, 1024), (32000, 5120), (50304, 6144)])
def test_problem_size_permutations(vocab_size: int, embed_dim: int) -> None:
    """
    Validate on wider range of problem sizes.
    """

    _ragged_embed_test_helper([(256, 0)], torch.float16, torch.int32, embed_dim, vocab_size)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize('seq_lens', [[128, 64, 192, 32], [57, 112, 63, 89, 1, 1, 1, 1]])
@pytest.mark.parametrize('do_padding', [True, False])
def test_complex_sequences(seq_lens: List[int], do_padding: bool) -> None:
    """
    Validate on different ragged batch construction scenarios.
    """
    embed_dim = 4096
    vocab_size = 50304

    _ragged_embed_test_helper([(seq_len, 0) for seq_len in seq_lens],
                              torch.float16,
                              torch.int32,
                              embed_dim,
                              vocab_size,
                              do_padding=do_padding)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("seq_lens", [[(256, 0)], [(256, 0),
                                                   (128, 0)], [(256, 0), (128, 0),
                                                               (64, 0)], [(1, 877), (619, 0), (213, 372), (1, 45)]])
def test_positional_embedding(seq_lens: List[Tuple[int, int]]) -> None:
    """
    Validate that positional embedding works correctly.
    """
    embed_dim = 4096
    vocab_size = 50304

    _ragged_embed_test_helper(seq_lens, torch.float16, torch.int32, embed_dim, vocab_size, pos_embed_size=2048)


@pytest.mark.inference_v2_ops
def test_positional_embedding_offset() -> None:
    """
    Validate that positional embedding works correctly with an offset.
    """
    embed_dim = 4096
    vocab_size = 50304
    seq_config = [(1, 877), (619, 0), (213, 372), (1, 45)]

    _ragged_embed_test_helper(seq_config,
                              torch.float16,
                              torch.int32,
                              embed_dim,
                              vocab_size,
                              pos_embed_size=2048,
                              pos_embed_offset=2)
