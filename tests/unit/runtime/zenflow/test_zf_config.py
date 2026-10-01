# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from pydantic import ValidationError

from deepspeed.runtime.zero.config import DeepSpeedZeroConfig, ZeroStageEnum
from deepspeed.runtime.zenflow.zenflow_config import ZenFlowConfig
from deepspeed.runtime.zero.offload_config import DeepSpeedZeroOffloadOptimizerConfig


def test_stage_enum_accepts_int_and_enum():
    """`stage` can be passed as either an int or the ZeroStageEnum."""
    c1 = DeepSpeedZeroConfig(stage=2)
    assert c1.stage == ZeroStageEnum.gradients
    c2 = DeepSpeedZeroConfig(stage=ZeroStageEnum.weights)
    assert c2.stage == ZeroStageEnum.weights


def test_offload_optimizer_config_from_dict():
    """A dict for offload_optimizer should be coerced into DeepSpeedZeroOffloadOptimizerConfig."""
    cfg = DeepSpeedZeroConfig(offload_optimizer={"device": "cpu", "pin_memory": True})
    assert isinstance(cfg.offload_optimizer, DeepSpeedZeroOffloadOptimizerConfig)
    assert cfg.offload_optimizer.device == "cpu"
    assert cfg.offload_optimizer.pin_memory is True


def test_invalid_offload_optimizer_type_raises():
    """Passing a non-dict to offload_optimizer must error out."""
    with pytest.raises(ValidationError):
        DeepSpeedZeroConfig(offload_optimizer="not a dict")


def test_zenflow_config_from_dict():
    """A dict for zenflow should be coerced into ZenFlowConfig."""
    zenflow_payload = {
        "topk_ratio": 0.25,
        "select_strategy": "auto",
        "select_interval": 4,
        "update_interval": 8,
        "full_warm_up_rounds": 1,
        "overlap_step": True
    }
    cfg = DeepSpeedZeroConfig(zenflow=zenflow_payload)
    assert isinstance(cfg.zenflow, ZenFlowConfig)
    assert cfg.zenflow.topk_ratio == 0.25
    assert cfg.zenflow.select_strategy == "auto"
    assert cfg.zenflow.select_interval == 4
    assert cfg.zenflow.update_interval == 8
    assert cfg.zenflow.full_warm_up_rounds == 1
    assert cfg.zenflow.overlap_step is True


def test_invalid_zenflow_type_raises():
    """Passing a non-dict to zenflow must error out."""
    with pytest.raises(ValidationError):
        DeepSpeedZeroConfig(zenflow=123)


@pytest.mark.parametrize("topk_ratio", [0.0, 1.0])
def test_topk_ratio_excludes_endpoints(topk_ratio):
    with pytest.raises(ValidationError):
        ZenFlowConfig(topk_ratio=topk_ratio)


@pytest.mark.parametrize("update_interval", [0, -5])
def test_update_interval_rejects_non_positive_values(update_interval):
    with pytest.raises(ValidationError):
        ZenFlowConfig(update_interval=update_interval)


@pytest.mark.parametrize("update_interval", [1, "auto"])
def test_update_interval_accepts_valid_boundary_values(update_interval):
    config = ZenFlowConfig(update_interval=update_interval)

    assert config.update_interval == update_interval


def test_offload_and_zenflow_combined():
    """
    offload_optimizer and zenflow can be used together under stage 2
    without validation errors.
    """
    payload = {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "zenflow": {
            "topk_ratio": 0.3,
            "select_strategy": "epoch",
            "select_interval": 3,
            "update_interval": 6,
            "full_warm_up_rounds": 0,
            "overlap_step": False
        }
    }
    cfg = DeepSpeedZeroConfig(**payload)
    assert isinstance(cfg.offload_optimizer, DeepSpeedZeroOffloadOptimizerConfig)
    assert cfg.offload_optimizer.device == "cpu"
    assert isinstance(cfg.zenflow, ZenFlowConfig)
    assert cfg.zenflow.select_strategy == "epoch"
