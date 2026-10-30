# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.runtime.data_pipeline.data_sampling.variable_batch_size_and_lr import batch_by_seqlens


def batched_sample_ids(microbatch_ids):
    return sorted(sample_id for _, sample_ids in microbatch_ids for sample_id in sample_ids)


@pytest.mark.parametrize("required_microbatches_of_same_size", [False, True])
@pytest.mark.parametrize("n_samples,max_tokens", [(5, 100), (6, 10), (8, 30), (9, 20)])
def test_every_sample_is_batched(n_samples, max_tokens, required_microbatches_of_same_size):
    """With one microbatch per batch nothing is aligned away and the trim below is a no-op,
    so the epoch has to contain every sample. `batch_end` is the exclusive end of
    `metrics[batch_init:batch_end]`, so a range stopping at `len(metrics)` never offered the
    slice that reaches the last sample and it was silently dropped."""
    seqlens = [5] * n_samples

    microbatch_ids, _, _ = batch_by_seqlens(
        seqlens=seqlens,
        max_tokens=max_tokens,
        effective_batch_size=1,
        required_microbatches_of_same_size=required_microbatches_of_same_size,
    )

    assert batched_sample_ids(microbatch_ids) == list(range(n_samples))


@pytest.mark.parametrize("effective_batch_size,required_microbatches_of_same_size,n_samples,max_tokens", [
    (2, True, 6, 10),
    (2, True, 8, 30),
    (2, True, 10, 10),
    (2, False, 8, 30),
    (4, True, 8, 30),
])
def test_larger_effective_batches_also_reach_the_last_sample(effective_batch_size, required_microbatches_of_same_size,
                                                             n_samples, max_tokens):
    """The loop steps by `equal_size_multiple`, which is the effective batch size when
    microbatches must match, so the exclusive end matters at every step size and not only at
    one. These are the shapes where the last stride lands exactly on `len(metrics)`: each of
    them lost its tail before the fix, half the dataset in the last case."""
    seqlens = [5] * n_samples

    microbatch_ids, _, _ = batch_by_seqlens(
        seqlens=seqlens,
        max_tokens=max_tokens,
        effective_batch_size=effective_batch_size,
        required_microbatches_of_same_size=required_microbatches_of_same_size,
    )

    assert batched_sample_ids(microbatch_ids) == list(range(n_samples))


@pytest.mark.parametrize("effective_batch_size", [1, 2, 4])
@pytest.mark.parametrize("required_microbatches_of_same_size", [False, True])
@pytest.mark.parametrize("n_samples,max_tokens", [(9, 10), (10, 10), (12, 10)])
def test_only_an_unalignable_tail_is_dropped(effective_batch_size, required_microbatches_of_same_size, n_samples,
                                             max_tokens):
    """Above one microbatch per batch the epoch is deliberately allowed to fall short: the
    microbatch list is trimmed to a multiple of the effective batch size so every dataloader
    gets the same count. What has to hold in every mode is that the samples which survive are
    the leading ones in order, with nothing repeated and nothing over the token cap."""
    seqlens = [5] * n_samples

    microbatch_ids, _, _ = batch_by_seqlens(
        seqlens=seqlens,
        max_tokens=max_tokens,
        effective_batch_size=effective_batch_size,
        required_microbatches_of_same_size=required_microbatches_of_same_size,
    )

    sample_ids = batched_sample_ids(microbatch_ids)
    assert sample_ids == list(range(len(sample_ids))), "a sample was dropped from the middle, not the tail"
    assert len(microbatch_ids) % effective_batch_size == 0, "every dataloader must get the same microbatch count"
    for _, ids_in_microbatch in microbatch_ids:
        assert sum(seqlens[sample_id] for sample_id in ids_in_microbatch) <= max_tokens


def test_microbatches_stay_within_max_tokens():
    """Packing every sample must not come at the cost of the token cap."""
    seqlens = [3, 7, 2, 8, 5, 5, 1, 9]
    max_tokens = 12

    microbatch_ids, _, _ = batch_by_seqlens(seqlens=seqlens, max_tokens=max_tokens, effective_batch_size=1)

    assert batched_sample_ids(microbatch_ids) == list(range(len(seqlens)))
    for _, sample_ids in microbatch_ids:
        assert sum(seqlens[sample_id] for sample_id in sample_ids) <= max_tokens
