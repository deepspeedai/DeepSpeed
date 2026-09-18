# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Regression tests for zero/division safety gaps reported in #7838."""

import math

import pytest

from deepspeed.utils.groups import _ensure_divisibility
from deepspeed.utils.timer import ThroughputTimer
from deepspeed.inference.v2.inference_utils import ceil_div


class _DummyTimerConfig:
    enabled = False
    synchronized = False


def test_ensure_divisibility_rejects_zero_denominator():
    with pytest.raises(AssertionError, match="non-zero"):
        _ensure_divisibility(8, 0)


def test_ensure_divisibility_accepts_valid_inputs():
    _ensure_divisibility(8, 2)
    _ensure_divisibility(0, 4)


def test_ceil_div_rejects_zero_divisor():
    with pytest.raises(ValueError, match="non-zero"):
        ceil_div(10, 0)


def test_ceil_div_matches_math_ceil():
    assert ceil_div(10, 3) == math.ceil(10 / 3)
    assert ceil_div(9, 3) == 3
    assert ceil_div(1, 1) == 1


def test_throughput_timer_rejects_zero_steps_per_output():
    with pytest.raises(ValueError, match="positive"):
        ThroughputTimer(_DummyTimerConfig(), batch_size=1, steps_per_output=0)


def test_throughput_timer_rejects_negative_steps_per_output():
    with pytest.raises(ValueError, match="positive"):
        ThroughputTimer(_DummyTimerConfig(), batch_size=1, steps_per_output=-1)


def test_throughput_timer_report_boundary_guards_mutated_zero():
    timer = ThroughputTimer(_DummyTimerConfig(), batch_size=1, steps_per_output=2)
    timer.steps_per_output = 0
    with pytest.raises(ValueError, match="positive"):
        timer._is_report_boundary()


def test_throughput_timer_report_boundary_none_is_safe():
    timer = ThroughputTimer(_DummyTimerConfig(), batch_size=1, steps_per_output=None)
    assert timer._is_report_boundary() is False
