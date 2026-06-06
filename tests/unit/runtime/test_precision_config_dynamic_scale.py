# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from pydantic import ValidationError

from deepspeed.runtime.precision_config import DeepSpeedFP16Config


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_fp16_dynamic_scale_rejects_invalid_values(field, value):
    with pytest.raises(ValidationError):
        DeepSpeedFP16Config(**{field: value})


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [1, 1000, "2"])
def test_fp16_dynamic_scale_accepts_valid_values(field, value):
    cfg = DeepSpeedFP16Config(**{field: value})
    assert getattr(cfg, field) > 0


@pytest.mark.parametrize("field", ["loss_scale_window", "min_loss_scale"])
@pytest.mark.parametrize("value", [[], {}])
def test_fp16_dynamic_scale_invalid_type_has_clear_error(field, value):
    with pytest.raises(ValidationError) as excinfo:
        DeepSpeedFP16Config(**{field: value})
    assert "must be a number" in str(excinfo.value)
