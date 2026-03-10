import math

import pytest
from pydantic import ValidationError

from deepspeed.runtime.precision_config import DeepSpeedFP16Config


@pytest.mark.parametrize("loss_scale", [-1, float("inf"), float("nan"), True])
def test_fp16_loss_scale_rejects_invalid_values(loss_scale):
    with pytest.raises(ValidationError):
        DeepSpeedFP16Config(loss_scale=loss_scale)


@pytest.mark.parametrize("loss_scale", [0, 1, 2.0, "3"])
def test_fp16_loss_scale_accepts_valid_values(loss_scale):
    cfg = DeepSpeedFP16Config(loss_scale=loss_scale)
    assert math.isfinite(cfg.loss_scale)
    assert cfg.loss_scale >= 0


@pytest.mark.parametrize("loss_scale", [[], {}])
def test_fp16_loss_scale_invalid_type_has_clear_error(loss_scale):
    with pytest.raises(ValidationError) as excinfo:
        DeepSpeedFP16Config(loss_scale=loss_scale)
    assert "must be a number" in str(excinfo.value)
