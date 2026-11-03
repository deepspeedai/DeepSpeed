# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.runtime.fp16.loss_scaler import DynamicLossScaler


@pytest.fixture(autouse=True)
def _patch_dist(monkeypatch):
    """Stub out dist.get_rank so update_scale can run without a process group."""
    import deepspeed.comm as dist
    monkeypatch.setattr(dist, "get_rank", lambda: 0)


INIT_SCALE = 2**16
SCALE_WINDOW = 1000
MIN_SCALE = 1


def _make_scaler(delayed_shift, consecutive_hysteresis=False):
    return DynamicLossScaler(
        init_scale=INIT_SCALE,
        scale_window=SCALE_WINDOW,
        min_scale=MIN_SCALE,
        delayed_shift=delayed_shift,
        consecutive_hysteresis=consecutive_hysteresis,
    )


class TestDynamicLossScalerHysteresis:
    """Ensure DynamicLossScaler lowers its scale on overflow for all hysteresis values."""

    @pytest.mark.parametrize("delayed_shift", [0, -1, -5])
    def test_zero_or_negative_hysteresis_lowers_scale(self, delayed_shift):
        scaler = _make_scaler(delayed_shift)
        scaler.update_scale(overflow=True)
        assert scaler.cur_scale < INIT_SCALE

    def test_hysteresis_one_lowers_scale_immediately(self):
        scaler = _make_scaler(delayed_shift=1)
        scaler.update_scale(overflow=True)
        assert scaler.cur_scale == INIT_SCALE / 2

    def test_hysteresis_two_delays_one_step(self):
        scaler = _make_scaler(delayed_shift=2)
        scaler.update_scale(overflow=True)
        assert scaler.cur_scale == INIT_SCALE
        scaler.update_scale(overflow=True)
        assert scaler.cur_scale == INIT_SCALE / 2

    @pytest.mark.parametrize("delayed_shift", [0, 1, 2])
    def test_repeated_overflow_reaches_min_scale(self, delayed_shift):
        scaler = _make_scaler(delayed_shift)
        for _ in range(200):
            try:
                scaler.update_scale(overflow=True)
            except Exception:
                break
        assert scaler.cur_scale == MIN_SCALE
