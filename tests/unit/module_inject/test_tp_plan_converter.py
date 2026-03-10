# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
from deepspeed.module_inject.tp_plan_converter import TPPlanConverter
from deepspeed.module_inject.autotp_config import PartitionType


class TestTPPlanConverter:

    def test_wildcard_to_regex_basic(self):
        assert TPPlanConverter._wildcard_to_regex("layers.*.q_proj") == r".*layers\..*\.q_proj"
        assert TPPlanConverter._wildcard_to_regex("self_attn.*.weight") == r".*self_attn\..*\.weight"
        assert TPPlanConverter._wildcard_to_regex("layers.*.self_attn.q_proj") == r".*layers\..*\.self_attn\.q_proj"

    def test_wildcard_to_regex_special_chars(self):
        assert TPPlanConverter._wildcard_to_regex("layers.0.q_proj") == r".*layers\.0\.q_proj"
        assert TPPlanConverter._wildcard_to_regex("mlp.gate_proj") == r".*mlp\.gate_proj"

    def test_colwise_rowwise_conversion(self):
        hf_plan = {"layers.*.q_proj": "colwise", "layers.*.o_proj": "rowwise"}
        specs = TPPlanConverter.convert(hf_plan)

        assert len(specs) == 2

        q_spec = [s for s in specs if "q_proj" in s.patterns[0]][0]
        o_spec = [s for s in specs if "o_proj" in s.patterns[0]][0]

        assert q_spec.partition_type == PartitionType.COLUMN
        assert o_spec.partition_type == PartitionType.ROW

    def test_pattern_weight_suffix(self):
        hf_plan = {"layers.*.q_proj": "colwise"}
        specs = TPPlanConverter.convert(hf_plan)

        assert len(specs) == 1
        assert specs[0].patterns[0].endswith(r"\.weight$")

    def test_pattern_weight_suffix_already_present(self):
        hf_plan = {"layers.*.q_proj.weight": "colwise"}
        specs = TPPlanConverter.convert(hf_plan)

        assert len(specs) == 1
        assert specs[0].patterns[0].endswith(r"\.weight$")

    def test_empty_plan(self):
        hf_plan = {}
        specs = TPPlanConverter.convert(hf_plan)

        assert len(specs) == 0

    def test_multiple_patterns(self):
        hf_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.mlp.gate_proj": "colwise",
            "layers.*.mlp.up_proj": "colwise",
            "layers.*.mlp.down_proj": "rowwise",
        }
        specs = TPPlanConverter.convert(hf_plan)

        assert len(specs) == 7

        colwise_count = sum(1 for s in specs if s.partition_type == PartitionType.COLUMN)
        rowwise_count = sum(1 for s in specs if s.partition_type == PartitionType.ROW)

        assert colwise_count == 5
        assert rowwise_count == 2

    def test_pattern_matches_param_name(self):
        import re

        hf_plan = {"layers.*.self_attn.q_proj": "colwise", "layers.*.mlp.down_proj": "rowwise"}
        specs = TPPlanConverter.convert(hf_plan)

        q_pattern = [s for s in specs if "q_proj" in s.patterns[0]][0]
        down_pattern = [s for s in specs if "down_proj" in s.patterns[0]][0]

        assert re.match(q_pattern.patterns[0], "model.layers.0.self_attn.q_proj.weight")
        assert re.match(q_pattern.patterns[0], "model.layers.10.self_attn.q_proj.weight")
        assert not re.match(q_pattern.patterns[0], "model.layers.0.self_attn.k_proj.weight")

        assert re.match(down_pattern.patterns[0], "model.layers.5.mlp.down_proj.weight")

    def test_invalid_partition_type(self):
        hf_plan = {"layers.*.q_proj": "invalid_type"}

        with pytest.raises(ValueError):
            TPPlanConverter.convert(hf_plan)
