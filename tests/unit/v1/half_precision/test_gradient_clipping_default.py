# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.config import get_gradient_clipping
from deepspeed.runtime.constants import GRADIENT_CLIPPING_DEFAULT
from unit.common import DistributedTest
from unit.simple_model import SimpleModel
import pytest


class TestGradientClippingConfig:

    def test_default_is_one(self):
        assert get_gradient_clipping({}) == GRADIENT_CLIPPING_DEFAULT == 1.0

    @pytest.mark.parametrize("gradient_clipping", [0.5, 0.0])
    def test_explicit_value_is_used(self, gradient_clipping):
        assert get_gradient_clipping({"gradient_clipping": gradient_clipping}) == gradient_clipping

    @pytest.mark.parametrize("gradient_clipping", [0.5, 0.0])
    def test_engine_getter_returns_config_value(self, gradient_clipping):
        engine = types.SimpleNamespace(_config=types.SimpleNamespace(gradient_clipping=gradient_clipping))
        assert DeepSpeedEngine.gradient_clipping(engine) == gradient_clipping


class TestGradientClippingEndToEnd(DistributedTest):
    world_size = 1

    def _config(self, gradient_clipping=None):
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
        if gradient_clipping is not None:
            config["gradient_clipping"] = gradient_clipping
        return config

    def _init(self, gradient_clipping=None):
        model = SimpleModel(hidden_dim=8)
        engine, _, _, _ = deepspeed.initialize(config=self._config(gradient_clipping),
                                               model=model,
                                               model_parameters=model.parameters())
        return engine

    def test_init_without_gradient_clipping_defaults_to_one(self):
        engine = self._init()
        assert engine.gradient_clipping() == 1.0

    def test_explicit_zero_disables_clipping(self):
        engine = self._init(gradient_clipping=0.0)
        assert engine.gradient_clipping() == 0.0
