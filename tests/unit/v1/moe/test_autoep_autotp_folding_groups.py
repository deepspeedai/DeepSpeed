# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP + AutoTP folding group-table and handle contract tests."""

import pytest

from deepspeed.module_inject.auto_ep_folding import (
    FoldingGroupHandles,
    assert_group_matches_spec,
    build_folding_spec,
    expected_folding_group_tables,
    local_folding_ranks,
)


def test_8gpu_tp2_ep4_tables_match_design():
    spec = build_folding_spec(world_size=8, pp_size=1, tp_size=2, ep_size=4, etp_size=1)
    tables = expected_folding_group_tables(spec)

    assert tables.tp_groups == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert tables.dense_dp_groups == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert tables.ep_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert tables.edp_groups == ((0, 4), (1, 5), (2, 6), (3, 7))


def test_16gpu_tp2_ep4_tables_match_design():
    spec = build_folding_spec(world_size=16, pp_size=1, tp_size=2, ep_size=4, etp_size=1)
    tables = expected_folding_group_tables(spec)

    assert tables.ep_groups == ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    assert tables.edp_groups == ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15))


def test_local_folding_ranks_match_helper_tables():
    spec = build_folding_spec(world_size=8, pp_size=1, tp_size=2, ep_size=4, etp_size=1)
    assert local_folding_ranks(5, spec) == {
        "tp": (4, 5),
        "dense_dp": (1, 3, 5, 7),
        "ep": (4, 5, 6, 7),
        "edp": (1, 5),
    }


def test_stale_registry_rank_lists_are_rejected():
    spec = build_folding_spec(world_size=8, pp_size=1, tp_size=2, ep_size=4, etp_size=1)
    stale_legacy_ep = ((0, 2, 4, 6), )
    stale_legacy_edp = ((0, 1), )

    with pytest.raises(RuntimeError, match="does not match folding spec"):
        assert_group_matches_spec((stale_legacy_ep, stale_legacy_edp), spec)


def test_group_handle_container_carries_explicit_groups_and_rank_tables():
    spec = build_folding_spec(world_size=4, pp_size=1, tp_size=2, ep_size=2, etp_size=1)
    local = local_folding_ranks(2, spec)
    handles = FoldingGroupHandles(
        spec=spec,
        tp_group=object(),
        dense_dp_group=object(),
        ep_group=object(),
        edp_group=object(),
        ep_group_name="ep_size_2",
        tp_ranks=local["tp"],
        dense_dp_ranks=local["dense_dp"],
        ep_ranks=local["ep"],
        edp_ranks=local["edp"],
    )

    assert handles.ep_group_name == "ep_size_2"
    assert handles.tp_ranks == (2, 3)
    assert handles.ep_ranks == (2, 3)
    assert handles.edp_ranks == (0, 2)
