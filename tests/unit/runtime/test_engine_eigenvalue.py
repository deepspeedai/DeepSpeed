# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.engine import _eigenvalue_summary_events


def test_eigenvalue_summary_events_use_block_values():
    block_eigenvalue = {
        "layer0.weight": (1.25, 0),
        "layer1.weight": (0.5, 1),
    }

    assert _eigenvalue_summary_events(block_eigenvalue, global_samples=128) == [
        ("Train/Eigenvalues/ModelBlockParam_0", 1.25, 128),
        ("Train/Eigenvalues/ModelBlockParam_1", 0.5, 128),
    ]
