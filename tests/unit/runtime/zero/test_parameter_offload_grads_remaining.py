# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.runtime.zero import parameter_offload as po


def test_ensure_ds_grads_remaining_sets_default():
    module = torch.nn.Linear(2, 2)
    assert not hasattr(module, "ds_grads_remaining")
    assert po._ensure_ds_grads_remaining(module) == 0
    assert module.ds_grads_remaining == 0


def test_ensure_ds_grads_remaining_preserves_existing():
    module = torch.nn.Linear(2, 2)
    module.ds_grads_remaining = 3
    assert po._ensure_ds_grads_remaining(module) == 3
    assert module.ds_grads_remaining == 3
