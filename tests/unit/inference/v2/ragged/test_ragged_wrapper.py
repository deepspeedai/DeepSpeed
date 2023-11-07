# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.ragged import (
    PlaceholderSequenceDescriptor,
    RaggedBatchWrapper,
    DSStateManagerConfig,
)


@pytest.mark.inference_v2
@pytest.mark.parametrize('max_ragged_sequence_count, max_ragged_batch_size', [(128, 512), (128, 1024)])
def test_wrapper_initialization(max_ragged_sequence_count: int, max_ragged_batch_size: int) -> None:
    config = DSStateManagerConfig(max_tracked_sequences=max_ragged_sequence_count,
                                  max_ragged_batch_size=max_ragged_batch_size,
                                  max_ragged_sequence_count=max_ragged_sequence_count)

    batch = RaggedBatchWrapper(config)

    assert batch.current_tokens == 0
    assert batch.current_sequences == 0


@pytest.mark.inference_v2
@pytest.mark.parametrize('seq_len', [1, 37, 128, 512])
def test_single_sequence_batch(seq_len: int) -> None:
    """
    Test we successfully construct single sequence batches and the on device metadata is accurate.
    """

    config = DSStateManagerConfig()
    batch = RaggedBatchWrapper(config)

    batch.clear()

    assert batch.current_tokens == 0
    assert batch.current_sequences == 0

    seq_desc = PlaceholderSequenceDescriptor()
    tokens = torch.randint(0, 100, (seq_len, ))
    batch.insert_sequence(seq_desc, tokens)

    batch.finalize()

    assert batch.current_tokens == seq_len
    assert batch.current_sequences == 1
    assert torch.equal(batch.input_ids(), tokens.to(get_accelerator().current_device()))
    assert torch.equal(batch.tokens_to_seq(), torch.zeros_like(tokens, device=get_accelerator().current_device()))
    assert torch.equal(batch.batch_metadata_buffer(),
                       torch.tensor([seq_len, 1], device=get_accelerator().current_device()))

    batch.clear()

    assert batch.current_tokens == 0
    assert batch.current_sequences == 0


@pytest.mark.inference_v2
@pytest.mark.parametrize('seq_lens', [[128, 128], [1, 32, 243], [64, 1, 1, 1, 1, 393, 27, 2]])
def test_multi_sequence_batch(seq_lens: List[int]) -> None:
    """
    Test sequentially adding new tokens to a batch and validate device data structures hold
    the appropriate data.
    """
    config = DSStateManagerConfig()
    batch = RaggedBatchWrapper(config)

    batch.clear()

    assert batch.current_tokens == 0
    assert batch.current_sequences == 0

    all_toks = [torch.randint(0, 100, (seq_len, )) for seq_len in seq_lens]

    for i, toks in enumerate(all_toks):
        seq_desc = PlaceholderSequenceDescriptor()
        batch.insert_sequence(seq_desc, toks)

        assert batch.current_tokens == sum(seq_lens[:i + 1])
        assert batch.current_sequences == i + 1

    batch.finalize()

    assert batch.current_tokens == sum(seq_lens)
    assert batch.current_sequences == len(seq_lens)

    assert torch.equal(batch.input_ids(), torch.cat(all_toks, dim=0).to(get_accelerator().current_device()))
    assert torch.equal(
        batch.tokens_to_seq(),
        torch.cat([torch.full((seq_len, ), i, dtype=torch.int32) for i, seq_len in enumerate(seq_lens)],
                  dim=0).to(get_accelerator().current_device()))

    for i, seq_len in enumerate(seq_lens):
        assert batch.inflight_seq_descriptors()[i][0] == sum(seq_lens[:i])
        assert batch.inflight_seq_descriptors()[i][1] == seq_len
        assert batch.inflight_seq_descriptors()[i][2] == 0

    assert torch.equal(batch.batch_metadata_buffer(),
                       torch.tensor([sum(seq_lens), len(seq_lens)], device=get_accelerator().current_device()))

    batch.clear()

    assert batch.current_tokens == 0
    assert batch.current_sequences == 0
