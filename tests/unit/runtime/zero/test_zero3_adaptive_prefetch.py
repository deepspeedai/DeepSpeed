# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Unit tests for ZeRO-3 adaptive prefetch bucket size.

The adaptation logic lives entirely in PartitionedParameterCoordinator and
uses only CPU wall-clock time, so these tests run without a GPU.
"""

import time
import pytest
from unittest.mock import patch

from deepspeed.runtime.zero.partitioned_param_coordinator import (
    InflightParamRegistry,
    PartitionedParameterCoordinator,
)


def _make_coordinator(prefetch_bucket_sz=50_000_000,
                      adaptive_prefetch=True,
                      adaptive_prefetch_min_sz=10_000_000,
                      adaptive_prefetch_max_sz=500_000_000):
    """Instantiate a coordinator with no GPU resources for unit testing."""
    return PartitionedParameterCoordinator(
        prefetch_bucket_sz=prefetch_bucket_sz,
        max_reuse_distance_in_numel=int(1e9),
        max_available_parameters_in_numel=int(1e9),
        allgather_stream=None,
        inflight_param_registry=InflightParamRegistry(),
        prefetch_nvme=False,
        timers=None,
        adaptive_prefetch=adaptive_prefetch,
        adaptive_prefetch_min_sz=adaptive_prefetch_min_sz,
        adaptive_prefetch_max_sz=adaptive_prefetch_max_sz,
    )


def _drive_adaptation(coordinator, wait_ratio, n_steps=10):
    """Simulate n_steps calls to __update_adaptive_prefetch with a fixed wait/step ratio.

    wait_ratio is the fraction of each step that looks like fetch-wait time.
    """
    step_duration = 0.01  # 10 ms per step
    wait_duration = step_duration * wait_ratio

    update_fn = coordinator._PartitionedParameterCoordinator__update_adaptive_prefetch

    t = time.perf_counter()
    for _ in range(n_steps):
        # Simulate: wait started `wait_duration` seconds before "now"
        wait_t0 = t - wait_duration
        with patch("deepspeed.runtime.zero.partitioned_param_coordinator.time") as mock_time:
            mock_time.perf_counter.return_value = t
            update_fn(wait_t0)
        t += step_duration


class TestAdaptivePrefetchConfig:

    def test_disabled_by_default(self):
        coord = _make_coordinator(adaptive_prefetch=False)
        assert coord._PartitionedParameterCoordinator__adaptive_prefetch == False

    def test_enabled(self):
        coord = _make_coordinator(adaptive_prefetch=True)
        assert coord._PartitionedParameterCoordinator__adaptive_prefetch == True

    def test_bounds_stored(self):
        coord = _make_coordinator(adaptive_prefetch_min_sz=5_000_000,
                                  adaptive_prefetch_max_sz=200_000_000)
        assert coord._PartitionedParameterCoordinator__adaptive_prefetch_min_sz == 5_000_000
        assert coord._PartitionedParameterCoordinator__adaptive_prefetch_max_sz == 200_000_000


class TestAdaptivePrefetchLogic:

    def test_high_wait_ratio_grows_bucket(self):
        """When wait time is >15% of step time, bucket should grow."""
        initial = 50_000_000
        coord = _make_coordinator(prefetch_bucket_sz=initial)
        # Drive 20 steps at 30% wait ratio so EMAs converge enough to trigger
        _drive_adaptation(coord, wait_ratio=0.30, n_steps=20)
        sz = coord._PartitionedParameterCoordinator__prefetch_bucket_sz
        assert sz > initial, f"expected bucket to grow from {initial}, got {sz}"

    def test_low_wait_ratio_shrinks_bucket(self):
        """When wait time is <5% of step time, bucket should shrink."""
        initial = 50_000_000
        coord = _make_coordinator(prefetch_bucket_sz=initial)
        _drive_adaptation(coord, wait_ratio=0.01, n_steps=20)
        sz = coord._PartitionedParameterCoordinator__prefetch_bucket_sz
        assert sz < initial, f"expected bucket to shrink from {initial}, got {sz}"

    def test_mid_ratio_leaves_bucket_unchanged(self):
        """Wait ratio in [5%, 15%] should not trigger a resize."""
        initial = 50_000_000
        coord = _make_coordinator(prefetch_bucket_sz=initial)
        _drive_adaptation(coord, wait_ratio=0.09, n_steps=20)
        sz = coord._PartitionedParameterCoordinator__prefetch_bucket_sz
        assert sz == initial, f"expected bucket unchanged at {initial}, got {sz}"

    def test_bucket_clamped_to_min(self):
        """Bucket should never drop below adaptive_prefetch_min_sz."""
        min_sz = 40_000_000
        coord = _make_coordinator(prefetch_bucket_sz=min_sz + 1_000_000,
                                  adaptive_prefetch_min_sz=min_sz)
        # Very low wait ratio drives the bucket down
        _drive_adaptation(coord, wait_ratio=0.001, n_steps=100)
        sz = coord._PartitionedParameterCoordinator__prefetch_bucket_sz
        assert sz >= min_sz, f"bucket {sz} dropped below min {min_sz}"

    def test_bucket_clamped_to_max(self):
        """Bucket should never exceed adaptive_prefetch_max_sz."""
        max_sz = 60_000_000
        coord = _make_coordinator(prefetch_bucket_sz=max_sz - 1_000_000,
                                  adaptive_prefetch_max_sz=max_sz)
        # Very high wait ratio drives the bucket up
        _drive_adaptation(coord, wait_ratio=0.99, n_steps=100)
        sz = coord._PartitionedParameterCoordinator__prefetch_bucket_sz
        assert sz <= max_sz, f"bucket {sz} exceeded max {max_sz}"

    def test_no_update_before_interval(self):
        """Bucket should not change before the update interval (10 steps) is reached."""
        initial = 50_000_000
        coord = _make_coordinator(prefetch_bucket_sz=initial)
        # Only 5 steps — not enough to trigger a resize even at high wait ratio
        _drive_adaptation(coord, wait_ratio=0.99, n_steps=5)
        sz = coord._PartitionedParameterCoordinator__prefetch_bucket_sz
        assert sz == initial, f"bucket changed before update interval, got {sz}"
