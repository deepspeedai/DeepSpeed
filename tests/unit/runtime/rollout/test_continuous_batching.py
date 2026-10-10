# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from itertools import product

import pytest

from deepspeed.runtime.rollout.continuous_batching import (ContinuousBatchRequest, ContinuousBatchScheduler,
                                                           plan_prefill_buckets)


def _request(request_id):
    return ContinuousBatchRequest(request_id)


def test_scheduler_admits_fifo_and_respects_capacity():
    scheduler = ContinuousBatchScheduler(max_batch_size=2, max_new_tokens=3)
    scheduler.submit(_request("a"))
    scheduler.submit(_request("b"))
    scheduler.submit(_request("c"))

    update = scheduler.schedule()
    assert update.active_ids == ("a", "b")
    assert update.keep_slots == ()
    assert update.admitted == (_request("a"), _request("b"))
    assert update.admitted_slots == (0, 1)
    assert scheduler.pending == (_request("c"), )


def test_scheduler_compacts_survivors_and_admits_pending_request():
    scheduler = ContinuousBatchScheduler(max_batch_size=2, max_new_tokens=3)
    scheduler.submit(_request("a"))
    scheduler.submit(_request("b"))
    scheduler.submit(_request("c"))
    scheduler.schedule()

    update = scheduler.schedule(finished_ids=("a", ))
    assert update.keep_slots == (1, )
    assert update.retired == ("a", )
    assert update.admitted == (_request("c"), )
    assert update.admitted_slots == (1, )
    assert update.active_ids == ("b", "c")


def test_scheduler_admits_pending_request_after_retirement():
    scheduler = ContinuousBatchScheduler(max_batch_size=1, max_new_tokens=3)
    scheduler.submit(_request("long"))
    scheduler.submit(_request("short"))

    update = scheduler.schedule()
    assert update.active_ids == ("long", )
    assert scheduler.pending == (_request("short"), )

    update = scheduler.advance(finished_ids=("long", ))
    assert update.retired == ("long", )
    assert update.admitted == (_request("short"), )


def test_scheduler_advance_retires_by_budget():
    scheduler = ContinuousBatchScheduler(max_batch_size=2, max_new_tokens=1)
    scheduler.submit(_request("a"))
    scheduler.submit(_request("b"))
    scheduler.schedule()

    update = scheduler.advance()
    assert update.retired == ("a", "b")
    assert update.keep_slots == ()
    assert update.active_ids == ()


def test_scheduler_rejects_invalid_transitions():
    with pytest.raises(ValueError, match="max_batch_size"):
        ContinuousBatchScheduler(max_batch_size=0, max_new_tokens=3)
    with pytest.raises(ValueError, match="max_new_tokens"):
        ContinuousBatchScheduler(max_batch_size=1, max_new_tokens=0)

    scheduler = ContinuousBatchScheduler(max_batch_size=1, max_new_tokens=3)
    scheduler.submit(_request("a"))
    with pytest.raises(ValueError, match="duplicate"):
        scheduler.submit(_request("a"))
    with pytest.raises(ValueError, match="not active"):
        scheduler.schedule(finished_ids=("missing", ))


@pytest.mark.parametrize("lengths,max_tokens", [([4, 1, 16, 4], None), ([2, 8, 3, 7, 1], 16)])
def test_prefill_buckets_match_exhaustive_partition_cost(lengths, max_tokens):

    def cost(count, width):
        return 5.0 + count * width + 0.01 * count * width * width

    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    reference = float("inf")
    for cuts in product([False, True], repeat=len(lengths) - 1):
        starts = [0] + [i + 1 for i, split in enumerate(cuts) if split]
        ends = starts[1:] + [len(lengths)]
        groups = [order[a:b] for a, b in zip(starts, ends)]
        if max_tokens is not None and any(len(g) * max(lengths[i] for i in g) > max_tokens for g in groups):
            continue
        reference = min(reference, sum(cost(len(g), max(lengths[i] for i in g)) for g in groups))

    buckets, predicted = plan_prefill_buckets(lengths, cost, max_tokens)
    assert sorted(i for bucket in buckets for i in bucket) == list(range(len(lengths)))
    assert predicted == pytest.approx(reference)
    assert predicted == pytest.approx(sum(cost(len(g), max(lengths[i] for i in g)) for g in buckets))


@pytest.mark.parametrize("fixed_cost,expected_count", [(25.0, 2), (5000.0, 1)])
def test_prefill_bucket_count_responds_to_forward_cost(fixed_cost, expected_count):
    lengths = [16] * 31 + [512]
    buckets, _ = plan_prefill_buckets(lengths, lambda count, width: fixed_cost + 0.1 * count * width)
    assert len(buckets) == expected_count


def test_prefill_bucket_capacity_supports_long_lengths_without_token_expansion():
    lengths = [65536, 16, 16]
    buckets, _ = plan_prefill_buckets(lengths, lambda count, width: 1 + count * width, 65536)
    assert sorted(i for bucket in buckets for i in bucket) == [0, 1, 2]
    assert all(len(g) * max(lengths[i] for i in g) <= 65536 for g in buckets)
    with pytest.raises(ValueError, match="prefill.*limit"):
        plan_prefill_buckets([65537], lambda count, width: count * width, 65536)
    assert plan_prefill_buckets([], lambda count, width: count * width) == ((), 0.0)
    with pytest.raises(ValueError, match="positive"):
        plan_prefill_buckets([0], lambda count, width: count * width)


def test_prefill_calibration_fit_recovers_nonnegative_costs():
    from benchmarks.rollout_prefill import nonnegative_fit

    matrix = [[1, 1, 1], [1, 2, 4], [1, 4, 16], [1, 8, 64]]
    assert nonnegative_fit(matrix, [5.5, 10, 22, 58]) == pytest.approx([2, 3, 0.5])
    # Decreasing noisy measurements must not produce negative execution costs.
    assert nonnegative_fit(matrix, [4, 3, 2, 1]) == pytest.approx([2.5, 0, 0])
