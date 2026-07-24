# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Small opt-in allocator diagnostics for the DeepCompile memory experiment."""

from collections import Counter
import json
import os
from typing import Any, Callable, Dict, Optional

MAX_OBSERVATIONS = 64

_step: Optional[int] = None
_phase: Optional[str] = None
_empty_cache_counts = Counter()
_empty_cache_observations = []
_scheduler_decisions = []
_allocator_retry_samples = []
_allocator_retry_total = 0
_dropped_observations = Counter()
_closed = False


def enabled() -> bool:
    return os.environ.get("DEEPSPEED_ALLOCATOR_TELEMETRY", "0") in {"1", "true", "True"}


def memory_stats() -> Dict[str, Any]:
    result = {
        "num_alloc_retries": None,
        "allocated_bytes": None,
        "allocated_peak_bytes": None,
        "reserved_bytes": None,
        "reserved_peak_bytes": None,
        "inactive_split_bytes": None,
    }
    if not enabled():
        return result
    try:
        from deepspeed.accelerator import get_accelerator

        stats = get_accelerator().memory_stats() or {}
        mapping = {
            "num_alloc_retries": "num_alloc_retries",
            "allocated_bytes": "allocated_bytes.all.current",
            "allocated_peak_bytes": "allocated_bytes.all.peak",
            "reserved_bytes": "reserved_bytes.all.current",
            "reserved_peak_bytes": "reserved_bytes.all.peak",
            "inactive_split_bytes": "inactive_split_bytes.all.current",
        }
        for output_name, stats_name in mapping.items():
            value = stats.get(stats_name)
            result[output_name] = int(value) if value is not None else None
    except Exception as exc:
        result["error"] = type(exc).__name__
    return result


def set_step(step: Optional[int], phase: Optional[str] = None) -> None:
    global _step, _phase
    if enabled():
        _step = step
        _phase = phase


def _append(kind: str, collection: list, observation: Dict[str, Any]) -> None:
    if len(collection) < MAX_OBSERVATIONS:
        collection.append(observation)
    else:
        _dropped_observations[kind] += 1


def record_empty_cache(site: str, callable_: Callable[..., Any], *args, **kwargs):
    if not enabled():
        return callable_(*args, **kwargs)
    before = memory_stats()
    _empty_cache_counts[site] += 1
    error = None
    try:
        return callable_(*args, **kwargs)
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        _append(
            "empty_cache", _empty_cache_observations, {
                "site": site,
                "ordinal": _empty_cache_counts[site],
                "step": _step,
                "phase": _phase,
                "before": before,
                "after": memory_stats(),
                "error": error,
            })


def record_scheduler_decision(*,
                              graph_id: int,
                              backward: bool,
                              budget_source: Optional[str],
                              disabled_reason: Optional[str],
                              max_gathered_bytes: Optional[int],
                              max_live_gathered_bytes: int,
                              budget_rejections: int,
                              over_budget_fallbacks: int,
                              minimum_rejected_candidate_peak_bytes: Optional[int] = None) -> None:
    if not enabled():
        return
    _append(
        "scheduler", _scheduler_decisions, {
            "graph_id":
            int(graph_id),
            "backward":
            bool(backward),
            "budget_source":
            budget_source,
            "disabled_reason":
            disabled_reason,
            "max_gathered_bytes":
            int(max_gathered_bytes) if max_gathered_bytes is not None else None,
            "max_live_gathered_bytes":
            int(max_live_gathered_bytes),
            "budget_rejections":
            int(budget_rejections),
            "over_budget_fallbacks":
            int(over_budget_fallbacks),
            "minimum_rejected_candidate_peak_bytes": (int(minimum_rejected_candidate_peak_bytes)
                                                      if minimum_rejected_candidate_peak_bytes is not None else None),
        })


def record_allocator_retry_sample(tracker_before: int, tracker_after: int) -> None:
    global _allocator_retry_total
    if not enabled():
        return
    delta = max(0, int(tracker_after) - int(tracker_before))
    _allocator_retry_total += delta
    if delta:
        _append(
            "allocator_retry", _allocator_retry_samples, {
                "step": _step,
                "phase": _phase,
                "tracker_before": int(tracker_before),
                "tracker_after": int(tracker_after),
                "retry_delta": delta,
            })


def summary() -> Dict[str, Any]:
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "last_step": _step,
        "last_phase": _phase,
        "final_memory": memory_stats(),
        "empty_cache_counts": dict(sorted(_empty_cache_counts.items())),
        "empty_cache_observations": list(_empty_cache_observations),
        "scheduler_decisions": list(_scheduler_decisions),
        "allocator_retry_total": _allocator_retry_total,
        "allocator_retry_samples": list(_allocator_retry_samples),
        "dropped_observations": dict(sorted(_dropped_observations.items())),
    }


def close_recorder() -> None:
    global _closed
    if enabled() and not _closed:
        print("DEEPSPEED_ALLOCATOR_TELEMETRY_SUMMARY " + json.dumps(summary(), sort_keys=True), flush=True)
        _closed = True


def _reset_for_tests() -> None:
    global _step, _phase, _allocator_retry_total, _closed
    _step = None
    _phase = None
    _empty_cache_counts.clear()
    _empty_cache_observations.clear()
    _scheduler_decisions.clear()
    _allocator_retry_samples.clear()
    _allocator_retry_total = 0
    _dropped_observations.clear()
    _closed = False
