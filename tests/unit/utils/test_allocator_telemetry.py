# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json

from deepspeed.utils import allocator_telemetry as telemetry


def test_disabled_empty_cache_is_a_transparent_call(monkeypatch):
    monkeypatch.delenv("DEEPSPEED_ALLOCATOR_TELEMETRY", raising=False)
    telemetry._reset_for_tests()
    calls = []

    assert telemetry.record_empty_cache("test.site", lambda value: calls.append(value) or "ok", 7) == "ok"
    assert calls == [7]
    assert telemetry.summary()["empty_cache_counts"] == {}


def test_enabled_empty_cache_records_site_and_memory(monkeypatch):
    monkeypatch.setenv("DEEPSPEED_ALLOCATOR_TELEMETRY", "1")
    telemetry._reset_for_tests()
    samples = iter([{"num_alloc_retries": 2}, {"num_alloc_retries": 3}, {"num_alloc_retries": 3}])
    monkeypatch.setattr(telemetry, "memory_stats", lambda: next(samples))
    telemetry.set_step(11, "measured")

    telemetry.record_empty_cache("backend.post-profile", lambda: None)
    result = telemetry.summary()

    assert result["empty_cache_counts"] == {"backend.post-profile": 1}
    assert result["empty_cache_observations"] == [{
        "site": "backend.post-profile",
        "ordinal": 1,
        "step": 11,
        "phase": "measured",
        "before": {
            "num_alloc_retries": 2
        },
        "after": {
            "num_alloc_retries": 3
        },
        "error": None,
    }]


def test_scheduler_and_allocator_retry_terminal_summary(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSPEED_ALLOCATOR_TELEMETRY", "1")
    telemetry._reset_for_tests()
    monkeypatch.setattr(telemetry, "memory_stats", lambda: {"num_alloc_retries": 4})
    telemetry.set_step(12, "measured")
    telemetry.record_scheduler_decision(graph_id=3,
                                        backward=False,
                                        budget_source="incomplete_profile_minimum_gather_residency",
                                        disabled_reason=None,
                                        max_gathered_bytes=1024,
                                        max_live_gathered_bytes=1024,
                                        budget_rejections=2,
                                        over_budget_fallbacks=0,
                                        minimum_rejected_candidate_peak_bytes=1536)
    telemetry.record_allocator_retry_sample(3, 4)

    telemetry.close_recorder()
    output = capsys.readouterr().out.strip()
    prefix = "DEEPSPEED_ALLOCATOR_TELEMETRY_SUMMARY "
    assert output.startswith(prefix)
    payload = json.loads(output[len(prefix):])

    assert payload["allocator_retry_total"] == 1
    assert payload["scheduler_decisions"][0]["budget_source"] == ("incomplete_profile_minimum_gather_residency")
    assert payload["scheduler_decisions"][0]["minimum_rejected_candidate_peak_bytes"] == 1536
    assert payload["final_memory"]["num_alloc_retries"] == 4
