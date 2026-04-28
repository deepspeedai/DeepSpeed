# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.engine import DeepSpeedEngine


def test_eigenvalue_monitor_events_use_block_eigenvalue_values(capsys):
    engine = DeepSpeedEngine.__new__(DeepSpeedEngine)
    engine.block_eigenvalue = {
        "first_param": (0.25, 0),
        "second_param": (0.5, 1),
    }
    engine.global_samples = 32
    expected_events = [
        ("Train/Eigenvalues/ModelBlockParam_0", 0.25, 32),
        ("Train/Eigenvalues/ModelBlockParam_1", 0.5, 32),
    ]
    actual_events = engine._get_eigenvalue_monitor_events()

    with capsys.disabled():
        print(f"\nblock_eigenvalue: {engine.block_eigenvalue}")
        print(f"generated eigenvalue monitor events: {actual_events}")

    assert actual_events == expected_events
