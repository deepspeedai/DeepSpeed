# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from pydantic import ValidationError

from deepspeed.runtime.precision_config import DeepSpeedFP16Config


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [0, -1])
def test_fp16_dynamic_scale_rejects_nonpositive_when_dynamic(field, value):
    # Dynamic loss scaling is active when fp16 is enabled and loss_scale == 0.
    with pytest.raises(ValidationError):
        DeepSpeedFP16Config(enabled=True, loss_scale=0, **{field: value})


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [1, 1000])
def test_fp16_dynamic_scale_accepts_positive_when_dynamic(field, value):
    cfg = DeepSpeedFP16Config(enabled=True, loss_scale=0, **{field: value})
    assert getattr(cfg, field) > 0


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [0, -1])
def test_fp16_dynamic_scale_ignored_with_static_loss_scale(field, value):
    # With a static loss scale (loss_scale > 0) these fields are unused, so a
    # non-positive value must not fail config construction (compatibility).
    cfg = DeepSpeedFP16Config(enabled=True, loss_scale=128, **{field: value})
    assert getattr(cfg, field) == value


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [0, -1])
def test_fp16_dynamic_scale_ignored_when_fp16_disabled(field, value):
    # When fp16 is disabled the dynamic scaling fields are unused.
    cfg = DeepSpeedFP16Config(enabled=False, loss_scale=0, **{field: value})
    assert getattr(cfg, field) == value


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [True, False])
def test_fp16_dynamic_scale_rejects_bool(field, value):
    # Pydantic coerces bool to int (True -> 1), which would otherwise slip past
    # the positivity check. Bools must be rejected before coercion.
    with pytest.raises(ValidationError):
        DeepSpeedFP16Config(enabled=True, loss_scale=0, **{field: value})


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [float("inf"), float("nan"), "abc", None])
def test_fp16_dynamic_scale_rejects_non_integer(field, value):
    # Non-finite and non-numeric values must be rejected rather than coerced.
    with pytest.raises(ValidationError):
        DeepSpeedFP16Config(enabled=True, loss_scale=0, **{field: value})
