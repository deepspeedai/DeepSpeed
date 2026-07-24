# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Regression coverage for #8084: zero.Init silently falls back to a single-rank (unsharded) group when a size-1
# process group exists at zero.Init time under a multi-process launcher (e.g. `from_pretrained` before
# `deepspeed.init_distributed()`), so every rank allocates the full model and OOMs. ZeRO-3 cannot work correctly with
# a process group that contradicts the launcher world, so the detection helper must produce an error message only
# when the launcher reports a multi-process world but the resolved group collapsed to one rank.

import pytest

from deepspeed.runtime.zero.partition_parameters import _contradicting_single_rank_pg_error


def test_errors_when_launcher_multiprocess_but_group_is_single_rank():
    msg = _contradicting_single_rank_pg_error(dp_world_size=1, data_parallel_group=None, env={"WORLD_SIZE": "8"})
    assert msg is not None
    assert "WORLD_SIZE=8" in msg
    assert "init_distributed" in msg


def test_no_error_for_genuine_single_process():
    assert _contradicting_single_rank_pg_error(dp_world_size=1, data_parallel_group=None,
                                               env={"WORLD_SIZE": "1"}) is None
    assert _contradicting_single_rank_pg_error(dp_world_size=1, data_parallel_group=None, env={}) is None


def test_no_error_when_group_actually_shards():
    assert _contradicting_single_rank_pg_error(dp_world_size=8, data_parallel_group=None,
                                               env={"WORLD_SIZE": "8"}) is None


def test_no_error_when_explicit_dp_group_supplied():
    # An explicitly provided size-1 data_parallel_group is treated as intentional.
    sentinel_group = object()
    assert _contradicting_single_rank_pg_error(dp_world_size=1, data_parallel_group=sentinel_group,
                                               env={"WORLD_SIZE": "8"}) is None


@pytest.mark.parametrize("bad", ["", "not-an-int", None])
def test_malformed_world_size_does_not_raise(bad):
    assert _contradicting_single_rank_pg_error(dp_world_size=1, data_parallel_group=None,
                                               env={"WORLD_SIZE": bad}) is None
