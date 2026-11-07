# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn as nn

from deepspeed.module_inject.auto_tp import Loading


def test_load_buffer_skips_unset_buffers():
    # InstanceNorm1d with track_running_stats=False (the default) registers
    # running_mean/running_var/num_batches_tracked as None, which used to crash
    # load_buffer with AttributeError: 'NoneType' object has no attribute 'data'.
    norm = nn.InstanceNorm1d(4)
    assert all(buf is None for buf in norm._buffers.values())

    Loading.load_buffer(norm, state_dict={}, prefix="norm.")

    assert all(buf is None for buf in norm._buffers.values())


def test_load_buffer_still_loads_present_buffers():
    bn = nn.BatchNorm1d(4)
    state_dict = {"bn.running_mean": torch.ones(4)}

    Loading.load_buffer(bn, state_dict, prefix="bn.")

    assert torch.equal(bn._buffers["running_mean"].data, torch.ones(4))
