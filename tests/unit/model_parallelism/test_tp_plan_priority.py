# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from deepspeed.runtime.tensor_parallel.config import resolve_tp_config


def _get_hf_tp_plan(model_or_config):
    if hasattr(model_or_config, '_tp_plan'):
        return model_or_config._tp_plan
    if hasattr(model_or_config, 'config') and hasattr(model_or_config.config, 'base_model_tp_plan'):
        return model_or_config.config.base_model_tp_plan
    if hasattr(model_or_config, 'base_model_tp_plan'):
        return model_or_config.base_model_tp_plan
    return None


class TestTPPlanPriority:

    def test_custom_config_highest_priority(self):

        class MockModel:
            _tp_plan = {"layers.*.q_proj": "colwise"}

        model = MockModel()
        ds_config = {
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": {
                    "layer_specs": [{
                        "patterns": [".*\\.custom_proj\\.weight$"],
                        "partition_type": "column"
                    }]
                }
            }
        }

        tp_config = resolve_tp_config(model, ds_config)

        assert tp_config is not None
        assert len(tp_config.layer_specs) == 1
        assert "custom_proj" in tp_config.layer_specs[0].patterns[0]

    def test_tp_plan_medium_priority(self):

        class MockModel:
            _tp_plan = {"layers.*.self_attn.q_proj": "colwise", "layers.*.self_attn.o_proj": "rowwise"}

        model = MockModel()
        ds_config = {"tensor_parallel": {"autotp_size": 2}}

        tp_config = resolve_tp_config(model, ds_config)

        assert tp_config is not None
        assert len(tp_config.layer_specs) >= 2
        assert any("q_proj" in spec.patterns[0] for spec in tp_config.layer_specs)
        assert any("o_proj" in spec.patterns[0] for spec in tp_config.layer_specs)

    def test_preset_lowest_priority(self):

        class MockModel:
            pass

        model = MockModel()
        ds_config = {"tensor_parallel": {"autotp_size": 2, "preset_model": "llama"}}

        tp_config = resolve_tp_config(model, ds_config)

        assert tp_config is not None
        assert len(tp_config.layer_specs) > 0

    def test_no_config_fails(self):

        class MockModel:
            pass

        model = MockModel()
        ds_config = {"tensor_parallel": {"autotp_size": 2}}

        with pytest.raises(ValueError, match="No TP configuration"):
            resolve_tp_config(model, ds_config)

    def test_priority_custom_over_tp_plan(self):

        class MockModel:
            _tp_plan = {"layers.*.q_proj": "colwise"}

        model = MockModel()
        ds_config = {
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": {
                    "layer_specs": [{
                        "patterns": [".*\\.custom\\.weight$"],
                        "partition_type": "row"
                    }]
                }
            }
        }

        tp_config = resolve_tp_config(model, ds_config)

        assert "custom" in tp_config.layer_specs[0].patterns[0]
        assert "q_proj" not in tp_config.layer_specs[0].patterns[0]

    def test_priority_tp_plan_over_preset(self):

        class MockModel:
            _tp_plan = {"layers.*.custom_proj": "colwise"}

        model = MockModel()
        ds_config = {"tensor_parallel": {"autotp_size": 2, "preset_model": "llama"}}

        tp_config = resolve_tp_config(model, ds_config)

        assert any("custom_proj" in spec.patterns[0] for spec in tp_config.layer_specs)
