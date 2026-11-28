# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from torch import nn

from deepspeed.module_inject.replace_module import replace_module


@pytest.mark.parametrize("norm_cls", [nn.InstanceNorm1d, nn.BatchNorm1d])
@pytest.mark.parametrize("track_running_stats", [False, True])
def test_checkpoint_loading_preserves_optional_buffers(tmp_path, norm_cls, track_running_stats):

    def make_model():
        return nn.Sequential(norm_cls(4, affine=False, track_running_stats=track_running_stats), nn.Linear(3, 2))

    reference = make_model().eval()
    if track_running_stats:
        reference[0].running_mean.fill_(2)
        reference[0].running_var.fill_(3)
        reference[0].num_batches_tracked.fill_(7)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(reference.state_dict(), checkpoint)

    with torch.device("meta"):
        model = make_model().eval()

    def replace_relu(child, policy, layer_id, **kwargs):
        return child

    loaded = replace_module(model, nn.ReLU, replace_relu, None, checkpoint=str(checkpoint))

    torch.testing.assert_close(loaded.state_dict(), reference.state_dict())
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    torch.testing.assert_close(loaded(inputs), reference(inputs))
    if not track_running_stats:
        assert loaded[0].running_mean is None
        assert loaded[0].running_var is None
        assert loaded[0].num_batches_tracked is None
