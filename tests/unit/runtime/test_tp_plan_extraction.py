# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


def _get_hf_tp_plan(model_or_config):
    """Extract tp_plan from HF model or config"""
    if hasattr(model_or_config, '_tp_plan'):
        return model_or_config._tp_plan

    if hasattr(model_or_config, 'config') and hasattr(model_or_config.config, 'base_model_tp_plan'):
        return model_or_config.config.base_model_tp_plan

    if hasattr(model_or_config, 'base_model_tp_plan'):
        return model_or_config.base_model_tp_plan

    return None


class TestTPPlanExtraction:

    def test_extract_tp_plan_from_mock_model(self):

        class MockHFModel:

            def __init__(self):
                self._tp_plan = {"layers.*.self_attn.q_proj": "colwise", "layers.*.self_attn.o_proj": "rowwise"}

        model = MockHFModel()
        tp_plan = _get_hf_tp_plan(model)

        assert tp_plan is not None
        assert "layers.*.self_attn.q_proj" in tp_plan
        assert tp_plan["layers.*.self_attn.q_proj"] == "colwise"

    def test_extract_tp_plan_from_config(self):

        class MockHFConfig:
            base_model_tp_plan = {
                "layers.*.mlp.gate_proj": "colwise",
                "layers.*.mlp.up_proj": "colwise",
                "layers.*.mlp.down_proj": "rowwise"
            }

        config = MockHFConfig()
        tp_plan = _get_hf_tp_plan(config)

        assert tp_plan is not None
        assert tp_plan == config.base_model_tp_plan

    def test_extract_tp_plan_from_model_with_config(self):

        class MockHFConfig:
            base_model_tp_plan = {"layers.*.self_attn.q_proj": "colwise"}

        class MockHFModel:

            def __init__(self, config):
                self.config = config

        config = MockHFConfig()
        model = MockHFModel(config)
        tp_plan = _get_hf_tp_plan(model)

        assert tp_plan is not None
        assert "layers.*.self_attn.q_proj" in tp_plan

    def test_no_tp_plan_model(self):
        model = torch.nn.Linear(10, 10)
        tp_plan = _get_hf_tp_plan(model)

        assert tp_plan is None

    def test_empty_tp_plan(self):

        class MockHFModel:

            def __init__(self):
                self._tp_plan = {}

        model = MockHFModel()
        tp_plan = _get_hf_tp_plan(model)

        assert tp_plan is not None
        assert len(tp_plan) == 0

    def test_none_tp_plan(self):

        class MockHFModel:

            def __init__(self):
                pass

        model = MockHFModel()
        tp_plan = _get_hf_tp_plan(model)

        assert tp_plan is None

    def test_priority_model_over_config(self):

        class MockHFConfig:
            base_model_tp_plan = {"config_plan": "colwise"}

        class MockHFModel:

            def __init__(self, config):
                self.config = config
                self._tp_plan = {"model_plan": "colwise"}

        config = MockHFConfig()
        model = MockHFModel(config)
        tp_plan = _get_hf_tp_plan(model)

        assert tp_plan is not None
        assert "model_plan" in tp_plan
        assert "config_plan" not in tp_plan
