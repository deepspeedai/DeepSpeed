# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.utils.timer import SynchronizedWallClockTimer

# Allowed difference in ms between the timer and the reference events recorded right inside it.
TOLERANCE_MS = 0.05


@pytest.fixture
def run_matmuls():
    accelerator = get_accelerator()
    if accelerator.use_host_timers():
        pytest.skip("this accelerator times with host timers, not device events")
    x = torch.randn(2048, 2048, device=accelerator.device_name())
    y = torch.empty_like(x)

    def run(num_matmuls):
        for _ in range(num_matmuls):
            torch.mm(x, x, out=y)

    return run


def time_intervals(timer, run_matmuls, matmuls_per_interval):
    # Each interval is also measured by its own reference events, recorded just inside the timer's
    # events. The reference durations are the expected values.
    reference_events = []
    for num_matmuls in matmuls_per_interval:
        # Queue work first so the timer's and the reference's start events complete together on the
        # device, however long the host takes between recording them.
        run_matmuls(10)
        reference_start = get_accelerator().Event(enable_timing=True)
        reference_stop = get_accelerator().Event(enable_timing=True)
        timer.start()
        reference_start.record()
        run_matmuls(num_matmuls)
        reference_stop.record()
        timer.stop()
        reference_events.append((reference_start, reference_stop))
    get_accelerator().synchronize()
    return [start.elapsed_time(stop) for start, stop in reference_events]


def test_elapsed_is_latest_interval_and_elapsed_total_is_sum(run_matmuls):
    timer = SynchronizedWallClockTimer()('op')
    expected = time_intervals(timer, run_matmuls, [5, 20, 10])

    assert timer.elapsed(reset=False) == pytest.approx(expected[-1], abs=TOLERANCE_MS)
    assert timer.elapsed_total(reset=False) == pytest.approx(sum(expected), abs=TOLERANCE_MS)
    assert timer.mean() == pytest.approx(sum(expected) / len(expected), abs=TOLERANCE_MS)
    # A read without reset must not consume anything: the engine reads its timers twice per step.
    assert timer.elapsed_total(reset=False) == pytest.approx(sum(expected), abs=TOLERANCE_MS)


def test_reset_starts_a_new_total(run_matmuls):
    timer = SynchronizedWallClockTimer()('op')
    expected = time_intervals(timer, run_matmuls, [5, 20])
    assert timer.elapsed_total(reset=True) == pytest.approx(sum(expected), abs=TOLERANCE_MS)

    expected = time_intervals(timer, run_matmuls, [10])
    assert timer.elapsed(reset=False) == pytest.approx(expected[0], abs=TOLERANCE_MS)
    assert timer.elapsed_total(reset=False) == pytest.approx(expected[0], abs=TOLERANCE_MS)
    assert timer.mean() == pytest.approx(expected[0], abs=TOLERANCE_MS)


def test_log_reports_memory_only_with_memory_breakdown(capsys):
    # The engine logs its micro timers with memory_breakdown taken from the config
    # (DeepSpeedEngine._stop_timers), so a True flag that logs no memory leaves that config key
    # with no effect, while a False flag that logs extra fields breaks wall_clock_breakdown parsers.
    timers = SynchronizedWallClockTimer()
    timers('op').start()
    timers('op').stop()

    timers.log(['op'], reset=False)
    timings_only = capsys.readouterr().out.strip()
    timers.log(['op'], reset=False, memory_breakdown=False)
    assert capsys.readouterr().out.strip() == timings_only

    # Sampled just before the log so the accelerator itself, not this test, is the oracle for the
    # reported figure. GB is the unit every memory report in DeepSpeed uses.
    expected_gb = get_accelerator().memory_allocated() / (1024**3)
    timers.log(['op'], reset=False, memory_breakdown=True)
    with_memory = capsys.readouterr().out.strip()

    assert timings_only in with_memory
    reported_gb = [float(value) for value in re.findall(r"(\d+\.\d+) GB", with_memory)]
    assert any(gb == pytest.approx(expected_gb, abs=0.05) for gb in reported_gb), with_memory
