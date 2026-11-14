# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the universal checkpoint state-key discovery helpers.

The checkpoint converter used to hard-code Adam-specific state names (`exp_avg`,
`exp_avg_sq`). This broke any non-Adam optimizer — Muon, Lion, SGD, or custom ones.
The helpers introduced for fix #8608 discover state names from what the optimizer
actually saved, per param group, rather than by hard-coded name.
"""

import logging
import os
import tempfile

import pytest
import torch

from deepspeed.checkpoint.ds_to_universal import _flat_state, _scale_power, _states_on_disk
from deepspeed.checkpoint.affine import SCALE_POWER_BY_STATE


def test_scale_power_table_includes_momentum_buffer():
    """Muon keeps `momentum_buffer` for the matrices it orthogonalizes.

    Before this fix, `momentum_buffer` was not in the scale-power table at all,
    so any tensor-parallel Muon parameter would crash the merge. It takes scale
    power -1 (gradient-like), same as Adam's `exp_avg`.
    """
    assert 'momentum_buffer' in SCALE_POWER_BY_STATE
    assert SCALE_POWER_BY_STATE['momentum_buffer'] == -1


def test_scale_power_fp32_is_1():
    """The parameter itself takes scale power 1.

    This is unchanged by the fix but worth pinning so the table does not get
    reordered or corrupted accidentally.
    """
    assert _scale_power('fp32', 'some.param') == 1


def test_scale_power_exp_avg_is_minus_1():
    """Adam's first moment is inverse-gradient-like."""
    assert _scale_power('exp_avg', 'some.param') == -1


def test_scale_power_exp_avg_sq_is_minus_2():
    """Adam's second moment is inverse-squared-gradient-like."""
    assert _scale_power('exp_avg_sq', 'some.param') == -2


def test_scale_power_raises_for_unknown_state():
    """An optimizer state that is not in the table must raise, not silently guess.

    Silent guessing would corrupt the optimizer state after a tensor-parallel
    affine map is applied, changing the resume trajectory.
    """
    with pytest.raises(ValueError, match='no known scaling'):
        _scale_power('some_future_optimizer_state', 'param.weight')


def test_flat_state_collects_matching_tensor_states():
    """Every tensor state shaped like the partition should be collected.

    This is the contract the extraction path needs: whatever states the optimizer
    actually saved, per param group, get forwarded for dumping.
    """
    fp32 = torch.randn(64)
    state = {
        'exp_avg': torch.randn(64),
        'exp_avg_sq': torch.randn(64),
        'momentum_buffer': torch.randn(64),  # Muon in partition layout
        'step': 7,
    }
    result = _flat_state(param_group_id=0, param_group_state=state, fp32_flat_group=fp32)
    # fp32 is always present; every tensor state shaped like the partition is kept.
    assert set(result.keys()) >= {'fp32', 'exp_avg', 'exp_avg_sq', 'momentum_buffer', 'step'}
    # The values must be the same objects — no copy.
    assert result['exp_avg'] is state['exp_avg']
    assert result['momentum_buffer'] is state['momentum_buffer']
    assert result['step'] == 7


def test_flat_state_skips_non_tensor_entries():
    """Non-tensor entries like param-group settings are not optimizer states."""
    fp32 = torch.randn(32)
    state = {
        'exp_avg': torch.randn(32),
        'some_float': 3.14,
        'some_list': [1, 2, 3],
    }
    result = _flat_state(param_group_id=0, param_group_state=state, fp32_flat_group=fp32)
    # Only fp32 and exp_avg should be present; the non-tensor entries are skipped.
    assert set(result.keys()) == {'fp32', 'exp_avg'}


def test_flat_state_warns_but_does_not_raise_on_shape_mismatch(caplog):
    """A state whose shape does not match the partition is warned and skipped.

    ZeRO-1/2 replicate Muon's momentum buffer whole per partition for Newton-Schulz.
    The fix must not crash the entire conversion when that buffer is encountered —
    instead log a warning and continue for the states that do match. A proper
    whole-parameter state layout (PR #8609) handles this correctly, but until then
    warn-and-skip is the right behaviour.
    """
    fp32 = torch.randn(64)
    state = {
        'exp_avg': torch.randn(64),
        'momentum_buffer': torch.randn(128),  # wrong shape
    }
    with caplog.at_level(logging.WARNING):
        result = _flat_state(param_group_id=3, param_group_state=state, fp32_flat_group=fp32)

    # The well-shaped state is collected; the mismatched one is not.
    assert set(result.keys()) == {'fp32', 'exp_avg'}
    # A warning was emitted mentioning the state name and the param group.
    assert any('momentum_buffer' in record.message and 'param group 3' in record.message for record in caplog.records)


def test_states_on_disk_reads_state_names_from_fragment_files():
    """The merge path must discover state names from what extraction wrote, not by name.

    This is the fix for the hard-coded `("fp32", "exp_avg", "exp_avg_sq")` tuples:
    after extraction writes fragment files under each state name, the merge reads
    them back off disk so it works with any optimizer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate what extraction wrote for one parameter under tp=1.
        tp_path = os.path.join(tmpdir, "0")
        os.makedirs(tp_path)
        torch.save(torch.tensor([1.0]), os.path.join(tp_path, "fp32.00"))
        torch.save(torch.tensor([2.0]), os.path.join(tp_path, "exp_avg.00"))
        torch.save(torch.tensor([3.0]), os.path.join(tp_path, "momentum_buffer.00"))
        torch.save(5, os.path.join(tp_path, "step.00"))  # step must be excluded

        states = _states_on_disk(tmpdir, tp_degree=1)
        # fp32 first, step excluded, alphabetical after that.
        assert states == ['fp32', 'exp_avg', 'momentum_buffer']


def test_states_on_disk_handles_multiple_tp_ranks():
    """Merge path with tensor parallelism reads state names across all TP ranks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for tp in (0, 1):
            tp_path = os.path.join(tmpdir, str(tp))
            os.makedirs(tp_path)
            torch.save(torch.tensor([1.0]), os.path.join(tp_path, "fp32.00"))
            if tp == 0:
                # Only tp=0 happens to have exp_avg (should never actually happen, but robust).
                torch.save(torch.tensor([2.0]), os.path.join(tp_path, "exp_avg.00"))

        states = _states_on_disk(tmpdir, tp_degree=2)
        # States that appear on any TP rank should be in the result.
        assert 'fp32' in states
        assert 'exp_avg' in states


def test_states_on_disk_empty_folder_returns_empty():
    """Parameter that had no optimizer state produces no states to merge."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_tp0 = os.path.join(tmpdir, "0")
        os.makedirs(empty_tp0)
        states = _states_on_disk(tmpdir, tp_degree=1)
        assert states == []


def test_flat_state_step_is_preserved():
    """`step` is the only non-tensor optimizer state that should survive extraction."""
    fp32 = torch.randn(16)
    state = {'exp_avg': torch.randn(16), 'step': 42}
    result = _flat_state(param_group_id=0, param_group_state=state, fp32_flat_group=fp32)
    assert result['step'] == 42


def test_flat_state_no_tensor_states_beyond_fp32():
    """A param group with only fp32 (edge case) still produces a valid dict."""
    fp32 = torch.randn(8)
    state = {}
    result = _flat_state(param_group_id=0, param_group_state=state, fp32_flat_group=fp32)
    assert result == {'fp32': fp32}
