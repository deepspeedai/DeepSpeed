# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging
import types

import pytest

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.config import get_log_level
from deepspeed.runtime.constants import LOG_LEVEL_DEFAULT
from deepspeed.utils import logging as ds_logging
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


class TestLogLevelConfig:

    def test_default_is_none(self):
        assert get_log_level({}) is None
        assert LOG_LEVEL_DEFAULT is None

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_explicit_value(self, level):
        assert get_log_level({"log_level": level}) == level

    def test_engine_getter_returns_config_value(self):
        engine = types.SimpleNamespace(_config=types.SimpleNamespace(log_level="ERROR"))
        assert DeepSpeedEngine.log_level(engine) == "ERROR"


class TestLogLevelEndToEnd(DistributedTest):
    world_size = 1

    def _config(self, log_level=None):
        config = {
            "train_batch_size": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3,
                    "torch_adam": True
                }
            },
        }
        if log_level is not None:
            config["log_level"] = log_level
        return config

    def _init(self, log_level=None):
        model = SimpleModel(hidden_dim=8)
        engine, _, _, _ = deepspeed.initialize(config=self._config(log_level),
                                               model=model,
                                               model_parameters=model.parameters())
        return engine

    def test_init_with_error_log_level(self):
        saved = ds_logging.logger.level
        ds_logging.logger.setLevel(logging.DEBUG)
        try:
            engine = self._init("ERROR")
            assert engine.log_level() == "ERROR"
            assert ds_logging.logger.getEffectiveLevel() == logging.ERROR
        finally:
            ds_logging.logger.setLevel(saved)

    def test_omit_log_level_defaults_to_warning(self):
        saved = ds_logging.logger.level
        try:
            ds_logging.logger.setLevel(logging.WARNING)
            for handler in ds_logging.logger.handlers:
                handler.setLevel(logging.WARNING)
            engine = self._init()
            assert engine.log_level() is None
            assert ds_logging.logger.getEffectiveLevel() == logging.WARNING
        finally:
            ds_logging.logger.setLevel(saved)
