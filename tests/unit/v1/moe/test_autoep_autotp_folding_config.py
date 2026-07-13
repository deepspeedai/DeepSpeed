# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP + AutoTP folding config and topology tests."""

from types import SimpleNamespace

import pytest

from deepspeed.module_inject.auto_ep_config import AutoEPConfig, parse_autoep_config, validate_autoep_config
from deepspeed.module_inject.auto_ep_folding import (
    FoldingGroupHandles,
    FoldingGroupTables,
    ParallelFoldingSpec,
    build_folding_spec,
    expected_folding_group_tables,
)


def test_folding_symbols_import_without_distributed_init():
    spec = build_folding_spec(world_size=8, pp_size=1, tp_size=2, ep_size=4, etp_size=1)
    assert isinstance(spec, ParallelFoldingSpec)
    assert isinstance(expected_folding_group_tables(spec), FoldingGroupTables)
    assert FoldingGroupHandles.__name__ == "FoldingGroupHandles"


@pytest.mark.parametrize(
    "world_size,tp_size,ep_size,expected_dp,expected_edp",
    [(8, 2, 4, 4, 2), (16, 2, 4, 8, 4), (4, 2, 2, 2, 2)],
)
def test_valid_folding_spec_derives_dense_and_expert_dp(world_size, tp_size, ep_size, expected_dp, expected_edp):
    config = AutoEPConfig(enabled=True, autoep_size=ep_size, expert_tensor_parallel_size=1)
    validate_autoep_config(config, world_size=world_size, pp_size=1, tp_size=tp_size, sp_size=1)
    spec = build_folding_spec(world_size=world_size, pp_size=1, tp_size=tp_size, ep_size=ep_size, etp_size=1)
    assert spec.dp_size == expected_dp
    assert spec.edp_size == expected_edp


def test_backward_config_compatibility_defaults_etp_to_one():
    config = parse_autoep_config({"enabled": True, "autoep_size": 2, "preset_model": "mixtral"})
    assert config.expert_tensor_parallel_size == 1
    validate_autoep_config(config, world_size=4, pp_size=1, tp_size=1, sp_size=1)


def test_expected_folding_tables_match_design_examples():
    spec8 = build_folding_spec(world_size=8, pp_size=1, tp_size=2, ep_size=4, etp_size=1)
    tables8 = expected_folding_group_tables(spec8)
    assert tables8.tp_groups == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert tables8.dense_dp_groups == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert tables8.ep_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert tables8.edp_groups == ((0, 4), (1, 5), (2, 6), (3, 7))

    spec16 = build_folding_spec(world_size=16, pp_size=1, tp_size=2, ep_size=4, etp_size=1)
    tables16 = expected_folding_group_tables(spec16)
    assert tables16.ep_groups == ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    assert tables16.edp_groups == ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15))


def test_expected_folding_tables_keep_ep8_node_local_with_node_contiguous_ranks():
    spec = build_folding_spec(world_size=16, pp_size=1, tp_size=2, ep_size=8, etp_size=1)
    tables = expected_folding_group_tables(spec)

    assert tables.tp_groups == ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15))
    assert tables.dense_dp_groups == ((0, 2, 4, 6, 8, 10, 12, 14), (1, 3, 5, 7, 9, 11, 13, 15))
    assert tables.ep_groups == ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13, 14, 15))
    assert tables.edp_groups == (
        (0, 8),
        (1, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (5, 13),
        (6, 14),
        (7, 15),
    )


def _assert_rejects(match, **kwargs):
    config_kwargs = {
        "enabled": True,
        "autoep_size": kwargs.pop("ep_size", 2),
        "expert_tensor_parallel_size": kwargs.pop("etp_size", 1),
        "preset_model": kwargs.pop("ep_preset", None),
    }
    validate_kwargs = {
        "world_size": kwargs.pop("world_size", 4),
        "pp_size": kwargs.pop("pp_size", 1),
        "tp_size": kwargs.pop("tp_size", 2),
        "sp_size": kwargs.pop("sp_size", 1),
    }
    validate_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=match):
        validate_autoep_config(AutoEPConfig(**config_kwargs), **validate_kwargs)


def test_validation_rule_g1_pp_divisibility_and_pp_rejection():
    _assert_rejects("pp_size=2 must divide world_size=7", world_size=7, pp_size=2, tp_size=1, ep_size=1)
    _assert_rejects("pp_size=1 only", world_size=8, pp_size=2, tp_size=2, ep_size=2)


def test_nonfolded_autoep_preserves_pipeline_parallel_compatibility():
    config = AutoEPConfig(enabled=True, autoep_size=2, expert_tensor_parallel_size=1)
    validate_autoep_config(config, world_size=8, pp_size=2, tp_size=1, sp_size=1)


def test_validation_rule_g2_tp_divisibility_names_valid_divisors():
    _assert_rejects("autotp_size=3.*Valid autotp_size values", world_size=8, tp_size=3, ep_size=1)


def test_validation_rule_g3_expert_width_divisibility_names_valid_divisors():
    _assert_rejects("autoep_size \\* expert_parallel\\.expert_tensor_parallel_size.*Valid expert-width values",
                    world_size=8,
                    tp_size=2,
                    ep_size=3)


def test_validation_rule_g4_etp_reserved_message():
    _assert_rejects("expert_tensor_parallel_size=2 is reserved", world_size=8, tp_size=2, ep_size=2, etp_size=2)


def test_validation_rule_g5_tp_sp_exclusive():
    _assert_rejects("mutually exclusive", world_size=4, tp_size=2, ep_size=2, sp_size=2)


def test_validation_rule_g6_preset_consistency():
    _assert_rejects("must match", world_size=4, tp_size=2, ep_size=2, ep_preset="mixtral", tp_preset_model="qwen3_moe")


def test_validation_rule_g7_zero3_lane_pointer():
    _assert_rejects("ZeRO stage 3.*separate ZeRO-3 composition lane", zero_stage=3)


def test_validation_rule_g8_mpu_conflict():
    mpu = SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 4,
                          get_pipeline_model_parallel_world_size=lambda: 1)
    _assert_rejects("mpu tensor/model parallel world size", mpu=mpu)


def test_validation_rule_g9_ep_one_rejected_with_autotp():
    _assert_rejects("autoep_size > 1", world_size=4, tp_size=2, ep_size=1)


def test_validation_rule_g10_data_before_expert_parallel_rejected():
    _assert_rejects("use_data_before_expert_parallel_", use_data_before_expert_parallel=True)


@pytest.mark.parametrize("offload_key", ["zero_offload_optimizer", "zero_offload_param"])
def test_validation_rule_g11_zero_offload_rejected(offload_key):
    _assert_rejects("offload", **{offload_key: True})


def test_deepcompile_folded_rejected():
    config = AutoEPConfig(enabled=True, autoep_size=2, expert_tensor_parallel_size=1)
    with pytest.raises(ValueError, match="DeepCompile.*AutoEP\\+AutoTP folding"):
        validate_autoep_config(config, world_size=4, pp_size=1, tp_size=2, sp_size=1, deepcompile_enabled=True)


def test_deepcompile_nonfolded_accepted():
    config = AutoEPConfig(enabled=True, autoep_size=2, expert_tensor_parallel_size=1)
    validate_autoep_config(config, world_size=4, pp_size=1, tp_size=1, sp_size=1, deepcompile_enabled=True)


@pytest.mark.parametrize(
    "world_size,tp_size,ep_size,expected_dp,expected_edp",
    [
        (4, 4, 4, 1, 1),  # EP group == TP group == {0,1,2,3}
        (4, 2, 4, 2, 1),  # ep>dp AND dp % ep != 0; EP spans both TP lanes and both DP ranks
        (8, 4, 4, 2, 2),  # cross-lane with expert replication (edp>1)
    ],
)
def test_cross_lane_ep_groups_accepted(world_size, tp_size, ep_size, expected_dp, expected_edp):
    # Cross-lane EP (expert_width = ep*etp may exceed dp, and need not divide dp) is now
    # supported: EP groups may span TP lanes. The earlier "temporary limitation" and
    # "dp % (ep*etp) == 0" fail-fasts are removed; only non-tiling shapes are rejected.
    config = AutoEPConfig(enabled=True, autoep_size=ep_size, expert_tensor_parallel_size=1)
    validate_autoep_config(config, world_size=world_size, pp_size=1, tp_size=tp_size, sp_size=1)
    spec = build_folding_spec(world_size=world_size, pp_size=1, tp_size=tp_size, ep_size=ep_size, etp_size=1)
    assert spec.dp_size == expected_dp
    assert spec.edp_size == expected_edp


def test_cross_lane_expected_folding_tables():
    # world=4 tp4 ep4 dp1: the EP group is the whole TP group; one expert per rank (edp=1).
    spec_tp4 = build_folding_spec(world_size=4, pp_size=1, tp_size=4, ep_size=4, etp_size=1)
    tables_tp4 = expected_folding_group_tables(spec_tp4)
    assert tables_tp4.tp_groups == ((0, 1, 2, 3), )
    assert tables_tp4.ep_groups == ((0, 1, 2, 3), )
    assert tables_tp4.edp_groups == ((0, ), (1, ), (2, ), (3, ))

    # world=4 tp2 ep4: EP group spans both TP lanes.
    spec_tp2 = build_folding_spec(world_size=4, pp_size=1, tp_size=2, ep_size=4, etp_size=1)
    tables_tp2 = expected_folding_group_tables(spec_tp2)
    assert tables_tp2.tp_groups == ((0, 1), (2, 3))
    assert tables_tp2.ep_groups == ((0, 1, 2, 3), )
    assert tables_tp2.edp_groups == ((0, ), (1, ), (2, ), (3, ))

    # world=8 tp4 ep4 (edp=2): two EP groups, each spanning TP lanes.
    spec_w8 = build_folding_spec(world_size=8, pp_size=1, tp_size=4, ep_size=4, etp_size=1)
    tables_w8 = expected_folding_group_tables(spec_w8)
    assert tables_w8.tp_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert tables_w8.ep_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert tables_w8.edp_groups == ((0, 4), (1, 5), (2, 6), (3, 7))
