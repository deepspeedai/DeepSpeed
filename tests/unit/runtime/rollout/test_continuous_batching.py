# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.runtime.rollout.continuous_batching import (ContinuousBatchRequest, ContinuousBatchScheduler)


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


def test_scheduler_can_defer_pending_admission_until_capacity_is_available():
    scheduler = ContinuousBatchScheduler(max_batch_size=1, max_new_tokens=3)
    scheduler.submit(_request("long"))
    scheduler.submit(_request("short"))

    update = scheduler.schedule(admit_if=lambda request: request.request_id == "long")
    assert update.active_ids == ("long",)
    assert scheduler.pending == (_request("short"), )

    update = scheduler.advance(finished_ids=("long", ), admit_if=lambda request: True)
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
