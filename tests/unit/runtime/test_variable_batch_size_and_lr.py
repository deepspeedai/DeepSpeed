# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.runtime.data_pipeline.data_sampling.variable_batch_size_and_lr import batch_by_seqlens


def batched_sample_ids(microbatch_ids):
    return sorted(sample_id for _, sample_ids in microbatch_ids for sample_id in sample_ids)


@pytest.mark.parametrize("n_samples,max_tokens", [(5, 100), (6, 10), (8, 30), (9, 20)])
def test_every_sample_is_batched(n_samples, max_tokens):
    """No sample may be left out of the epoch. `batch_end` is the exclusive end of
    `metrics[batch_init:batch_end]`, so a range stopping at `len(metrics)` never
    offered the slice that reaches the last sample and it was silently dropped."""
    seqlens = [5] * n_samples

    microbatch_ids, _, _ = batch_by_seqlens(seqlens=seqlens, max_tokens=max_tokens, effective_batch_size=1)

    assert batched_sample_ids(microbatch_ids) == list(range(n_samples))


def test_microbatches_stay_within_max_tokens():
    """Packing every sample must not come at the cost of the token cap."""
    seqlens = [3, 7, 2, 8, 5, 5, 1, 9]
    max_tokens = 12

    microbatch_ids, _, _ = batch_by_seqlens(seqlens=seqlens, max_tokens=max_tokens, effective_batch_size=1)

    assert batched_sample_ids(microbatch_ids) == list(range(len(seqlens)))
    for _, sample_ids in microbatch_ids:
        assert sum(seqlens[sample_id] for sample_id in sample_ids) <= max_tokens
