# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools

import pytest
import torch
import deepspeed.runtime.zero.linear as zero_linear
from deepspeed.runtime.zero.linear import LinearModuleForZeroStage3
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.torch import required_torch_version
from unit.common import DistributedTest


@pytest.mark.parametrize('half_op', [False, True])
class TestAutoCastDisable(DistributedTest):

    def test_missing_amp_autocast(self, half_op):
        hidden_dim = 4
        if half_op:
            input = torch.randn(hidden_dim).to(get_accelerator().device_name()).half()
            ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name()).half()
        else:
            input = torch.randn(hidden_dim).to(get_accelerator().device_name())
            ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name())

        output = ds_linear(input)
        assert output.dtype == ds_linear.weight.dtype

    def test_disable_autocast_linear(self, half_op):
        hidden_dim = 4
        if half_op:
            input = torch.randn(hidden_dim).to(get_accelerator().device_name()).half()
            ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name()).half()
        else:
            input = torch.randn(hidden_dim).to(get_accelerator().device_name())
            ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name())

        with torch.amp.autocast(device_type=get_accelerator().device_name(), enabled=False):
            output = ds_linear(input)
            assert output.dtype == ds_linear.weight.dtype


@pytest.mark.parametrize('half_input, half_weight', [(False, False), (False, True), (True, False), (True, True)])
class TestAutoCastEnable(DistributedTest):

    def test_autocast_linear(self, tmpdir, half_input, half_weight):
        hidden_dim = 4
        input = torch.randn(hidden_dim).to(get_accelerator().device_name())
        ds_linear = LinearModuleForZeroStage3(hidden_dim, hidden_dim).to(get_accelerator().device_name())

        if half_input:
            input = input.half()

        if half_weight:
            ds_linear = ds_linear.half()

        with torch.amp.autocast(device_type=get_accelerator().device_name()):
            output = ds_linear(input)
            assert output.dtype == torch.half or output.dtype == torch.bfloat16


def test_get_autocast_decorators_use_torch_amp_on_torch_2_4_or_newer():
    if not required_torch_version(min_version=2.4):
        pytest.skip('torch.amp.custom_fwd/custom_bwd are only available on torch >= 2.4')

    device_type = get_accelerator().device_name()

    assert isinstance(zero_linear.autocast_custom_fwd, functools.partial)
    assert isinstance(zero_linear.autocast_custom_bwd, functools.partial)
    assert zero_linear.autocast_custom_fwd.func is torch.amp.custom_fwd
    assert zero_linear.autocast_custom_bwd.func is torch.amp.custom_bwd
    assert zero_linear.autocast_custom_fwd.keywords == {'device_type': device_type}
    assert zero_linear.autocast_custom_bwd.keywords == {'device_type': device_type}


def test_get_autocast_decorators_use_legacy_amp_or_noop_before_torch_2_4():
    if required_torch_version(min_version=2.4):
        pytest.skip('legacy AMP fallback only applies on torch < 2.4')

    device_type = get_accelerator().device_name()
    legacy_amp = getattr(getattr(torch, device_type, None), 'amp', None)
    expected_custom_fwd = getattr(legacy_amp, 'custom_fwd', zero_linear.noop_decorator)
    expected_custom_bwd = getattr(legacy_amp, 'custom_bwd', zero_linear.noop_decorator)

    assert zero_linear.autocast_custom_fwd is expected_custom_fwd
    assert zero_linear.autocast_custom_bwd is expected_custom_bwd
