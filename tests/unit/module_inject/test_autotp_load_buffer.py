# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.module_inject.auto_tp import Loading


class ModuleWithNoneBuffer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", None)
        self.register_buffer("scale", torch.zeros(3))


def test_load_buffer_skips_none_buffers():
    module = ModuleWithNoneBuffer()
    state_dict = {"layer.scale": torch.ones(3)}

    Loading.load_buffer(module, state_dict, "layer.")

    assert module.running_mean is None
    assert torch.equal(module.scale, torch.ones(3))


def test_load_buffer_on_instance_norm_without_running_stats():
    module = torch.nn.InstanceNorm1d(4)
    assert module.running_mean is None

    Loading.load_buffer(module, {}, "norm.")

    assert module.running_mean is None
    assert module.running_var is None
