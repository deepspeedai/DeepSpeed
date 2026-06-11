# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from deepspeed.runtime.engine import DeepSpeedEngine


def test_eigenvalue_summary_events_use_block_eigenvalue_values():
    engine = object.__new__(DeepSpeedEngine)
    engine.block_eigenvalue = {
        "block_a": (0.25, 0),
        "block_b": (0.5, 1),
    }
    engine.gas_boundary_ctr = 4
    engine.global_samples = 128
    engine.eigenvalue_enabled = lambda: True
    engine.eigenvalue_gas_boundary_resolution = lambda: 2

    assert engine._eigenvalue_summary_events() == [
        ("Train/Eigenvalues/ModelBlockParam_0", 0.25, 128),
        ("Train/Eigenvalues/ModelBlockParam_1", 0.5, 128),
    ]


def test_eigenvalue_summary_events_skip_non_boundary_steps():
    engine = object.__new__(DeepSpeedEngine)
    engine.block_eigenvalue = {"block_a": (0.25, 0)}
    engine.gas_boundary_ctr = 3
    engine.eigenvalue_enabled = lambda: True
    engine.eigenvalue_gas_boundary_resolution = lambda: 2

    assert engine._eigenvalue_summary_events() == []
