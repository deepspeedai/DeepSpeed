# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import SimpleNamespace
import pytest

from deepspeed.utils.timer import ThroughputTimer


def _timer_config():
    return SimpleNamespace(enabled=True, synchronized=False)


def test_steps_per_output_rejects_zero():
    with pytest.raises(ValueError, match="steps_per_output must be greater than 0"):
        ThroughputTimer(config=_timer_config(), batch_size=1, steps_per_output=0)


def test_steps_per_output_rejects_non_integral():
    with pytest.raises(ValueError, match="steps_per_output must be a positive integer or None"):
        ThroughputTimer(config=_timer_config(), batch_size=1, steps_per_output=1.5)


def test_report_boundary_for_valid_steps_per_output():
    timer = ThroughputTimer(config=_timer_config(), batch_size=1, steps_per_output=3)
    timer.global_step_count = 6
    assert timer._is_report_boundary()
