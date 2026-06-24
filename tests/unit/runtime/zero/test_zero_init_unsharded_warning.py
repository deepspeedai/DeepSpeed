# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Regression coverage for #8084: zero.Init silently falls back to a single-rank (unsharded) group when the
# distributed process group is not initialized before it runs (e.g. `from_pretrained` before
# `deepspeed.init_distributed()`), so every rank allocates the full model and OOMs. The detection helper must warn
# only when the launcher reports a multi-process world but the resolved group collapsed to one rank.

import pytest

from deepspeed.runtime.zero.partition_parameters import _unsharded_single_rank_warning


def test_warns_when_launcher_multiprocess_but_group_is_single_rank():
    msg = _unsharded_single_rank_warning(dp_world_size=1, data_parallel_group=None, env={"WORLD_SIZE": "8"})
    assert msg is not None
    assert "WORLD_SIZE=8" in msg
    assert "init_distributed" in msg


def test_no_warning_for_genuine_single_process():
    assert _unsharded_single_rank_warning(dp_world_size=1, data_parallel_group=None, env={"WORLD_SIZE": "1"}) is None
    assert _unsharded_single_rank_warning(dp_world_size=1, data_parallel_group=None, env={}) is None


def test_no_warning_when_group_actually_shards():
    assert _unsharded_single_rank_warning(dp_world_size=8, data_parallel_group=None, env={"WORLD_SIZE": "8"}) is None


def test_no_warning_when_explicit_dp_group_supplied():
    # An explicitly provided size-1 data_parallel_group is treated as intentional.
    sentinel_group = object()
    assert _unsharded_single_rank_warning(dp_world_size=1, data_parallel_group=sentinel_group, env={"WORLD_SIZE":
                                                                                                    "8"}) is None


@pytest.mark.parametrize("bad", ["", "not-an-int", None])
def test_malformed_world_size_does_not_raise(bad):
    assert _unsharded_single_rank_warning(dp_world_size=1, data_parallel_group=None, env={"WORLD_SIZE": bad}) is None
