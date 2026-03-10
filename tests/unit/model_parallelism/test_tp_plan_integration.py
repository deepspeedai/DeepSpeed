# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from deepspeed.module_inject.tp_plan_converter import TPPlanConverter
from deepspeed.runtime.tensor_parallel.config import resolve_tp_config, _get_hf_tp_plan


class TestTPPlanIntegration:
    """Integration tests - no GPU required"""

    def test_full_pipeline_conversion(self):
        """Test complete conversion pipeline"""
        # 1. Mock HF tp_plan
        hf_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.mlp.gate_proj": "colwise",
            "layers.*.mlp.up_proj": "colwise",
            "layers.*.mlp.down_proj": "rowwise",
        }

        # 2. Convert to DeepSpeed format
        layer_specs = TPPlanConverter.convert(hf_plan)

        # 3. Verify
        assert len(layer_specs) == 7

        # Verify Q/K/V are column parallel
        qkv_specs = [s for s in layer_specs if any(proj in s.patterns[0] for proj in ['q_proj', 'k_proj', 'v_proj'])]
        assert len(qkv_specs) == 3
        assert all(s.partition_type.value == "column" for s in qkv_specs)

        # Verify o_proj and down_proj are row parallel
        row_specs = [s for s in layer_specs if any(proj in s.patterns[0] for proj in ['o_proj', 'down_proj'])]
        assert len(row_specs) == 2
        assert all(s.partition_type.value == "row" for s in row_specs)

    def test_resolve_config_with_mock_model(self):
        """Test complete config resolution flow"""

        # Create a mock HF model
        class MockHFModel:

            def __init__(self):
                self._tp_plan = {"layers.*.attention.q_proj": "colwise", "layers.*.attention.o_proj": "rowwise"}

        model = MockHFModel()
        ds_config = {"tensor_parallel": {"autotp_size": 2}}

        # Resolve config
        tp_config = resolve_tp_config(model, ds_config)

        # Verify
        assert tp_config is not None
        assert tp_config.tp_size == 2
        assert len(tp_config.layer_specs) == 2

        # Verify correct patterns are included (note the escaping)
        assert any('attention' in s.patterns[0] and 'q_proj' in s.patterns[0] for s in tp_config.layer_specs)

    def test_pattern_matches_real_param_names(self):
        """Test generated regex matches real parameter names"""
        hf_plan = {"layers.*.self_attn.q_proj": "colwise", "layers.*.mlp.down_proj": "rowwise"}

        layer_specs = TPPlanConverter.convert(hf_plan)

        # Simulate real parameter names
        real_param_names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.5.self_attn.q_proj.weight",
            "model.layers.10.self_attn.q_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
        ]

        import re
        for spec in layer_specs:
            pattern = spec.patterns[0]
            matching_params = [name for name in real_param_names if re.match(pattern, name)]

            if "q_proj" in pattern:
                assert len(matching_params) == 3
            elif "down_proj" in pattern:
                assert len(matching_params) == 2

    def test_empty_and_none_cases(self):
        """Test edge cases"""
        # Empty tp_plan
        specs = TPPlanConverter.convert({})
        assert len(specs) == 0

        # Model without tp_plan
        class NoPlanModel:
            pass

        plan = _get_hf_tp_plan(NoPlanModel())
        assert plan is None

        # Should raise error
        with pytest.raises(ValueError, match="No TP configuration"):
            resolve_tp_config(NoPlanModel(), {"tensor_parallel": {"autotp_size": 2}})

    def test_priority_order_complete(self):
        """Complete test of priority order"""

        class ModelWithPlan:
            _tp_plan = {"layers.*.q_proj": "colwise"}

        # Case 1: Custom config has highest priority
        custom_config = {
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": {
                    "layer_specs": [{
                        "patterns": [".*\\.custom\\.weight$"],
                        "partition_type": "column"
                    }]
                }
            }
        }
        result = resolve_tp_config(ModelWithPlan(), custom_config)
        assert "custom" in result.layer_specs[0].patterns[0]

        # Case 2: tp_plan has medium priority
        tp_only_config = {"tensor_parallel": {"autotp_size": 2}}
        result = resolve_tp_config(ModelWithPlan(), tp_only_config)
        assert any("q_proj" in s.patterns[0] for s in result.layer_specs)

        # Case 3: preset has lowest priority
        class NoPlanModel:
            pass

        preset_config = {"tensor_parallel": {"autotp_size": 2, "preset_model": "llama"}}
        result = resolve_tp_config(NoPlanModel(), preset_config)
        assert result is not None
        assert len(result.layer_specs) > 0

    def test_mixed_layer_types(self):
        """Test mixed layer types (attention + mlp)"""
        hf_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.mlp.fc1": "colwise",
            "layers.*.mlp.fc2": "rowwise",
        }

        layer_specs = TPPlanConverter.convert(hf_plan)

        # Count colwise and rowwise
        colwise_count = sum(1 for s in layer_specs if s.partition_type.value == "column")
        rowwise_count = sum(1 for s in layer_specs if s.partition_type.value == "row")

        assert colwise_count == 4  # q, k, v, fc1
        assert rowwise_count == 2  # o_proj, fc2
