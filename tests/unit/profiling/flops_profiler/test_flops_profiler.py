# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
import torch
import pytest
import deepspeed
from types import SimpleNamespace
from deepspeed.profiling.flops_profiler import get_model_profile, FlopsProfiler
from unit.simple_model import SimpleModel, random_dataloader
from unit.common import DistributedTest
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator

pytestmark = pytest.mark.skipif(not required_torch_version(min_version=1.3),
                                reason='requires Pytorch version 1.3 or above')


def within_range(val, target, tolerance):
    return abs(val - target) / target < tolerance


TOLERANCE = 0.05


@pytest.mark.sequential
@pytest.mark.skipif(not required_torch_version(min_version=2.0), reason="requires Pytorch version 2.0 or above")
def test_repeated_profile_restores_operations():

    class AttentionModel(torch.nn.Module):

        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    model = AttentionModel()
    inputs = [torch.randn(2, 4, 16, 8) for _ in range(3)]
    prof = FlopsProfiler(model)
    original_operations = (torch.nn.functional.scaled_dot_product_attention, torch.Tensor.__matmul__, torch.bmm)

    profiles = []
    for _ in range(3):
        prof.start_profile()
        model(*inputs)
        prof.stop_profile()
        profiles.append((prof.get_total_flops(), prof.get_total_macs()))
        prof.end_profile()

        restored_operations = (torch.nn.functional.scaled_dot_product_attention, torch.Tensor.__matmul__, torch.bmm)
        assert restored_operations == original_operations

    assert profiles == [(65536, 32768)] * 3


@pytest.mark.sequential
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("stride, padding, output_padding", [(1, 0, 0), (2, 1, 1)])
def test_conv_transpose_flops(groups, stride, padding, output_padding):
    """A transposed convolution scatters every input element across the kernel and adds the
    bias once per output element, so the two counts follow different shapes."""
    in_channels, out_channels, kernel_size = 4, 8, 3
    model = torch.nn.ConvTranspose2d(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     output_padding=output_padding,
                                     groups=groups)
    inputs = torch.randn(2, in_channels, 8, 8)

    prof = FlopsProfiler(model)
    prof.start_profile()
    outputs = model(inputs)
    prof.stop_profile()
    flops, macs = prof.get_total_flops(), prof.get_total_macs()
    prof.end_profile()

    input_elements = inputs.shape[0] * inputs[0, 0].numel()
    output_elements = outputs.shape[0] * outputs[0, 0].numel()
    expected_macs = input_elements * in_channels * (out_channels // groups) * kernel_size * kernel_size
    expected_bias_flops = out_channels * output_elements

    assert macs == expected_macs
    assert flops == 2 * expected_macs + expected_bias_flops


@pytest.mark.sequential
def test_conv_transpose_flops_with_output_size():
    """`output_size=` makes torch compute the per-dimension output_padding itself, and it
    hands it over as a list rather than a tuple."""
    in_channels, out_channels, kernel_size = 4, 8, 3
    model = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1)
    inputs = torch.randn(2, in_channels, 8, 8)

    prof = FlopsProfiler(model)
    prof.start_profile()
    outputs = model(inputs, output_size=(2, out_channels, 16, 16))
    prof.stop_profile()
    flops, macs = prof.get_total_flops(), prof.get_total_macs()
    prof.end_profile()

    input_elements = inputs.shape[0] * inputs[0, 0].numel()
    output_elements = outputs.shape[0] * outputs[0, 0].numel()
    expected_macs = input_elements * in_channels * out_channels * kernel_size * kernel_size
    expected_bias_flops = out_channels * output_elements

    assert macs == expected_macs
    assert flops == 2 * expected_macs + expected_bias_flops


class LeNet5(torch.nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            torch.nn.Tanh(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return logits, probs


class TestFlopsProfiler(DistributedTest):
    world_size = 1

    def test(self):
        if torch.half not in get_accelerator().supported_dtypes():
            pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}")

        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001,
                }
            },
            "zero_optimization": {
                "stage": 0
            },
            "fp16": {
                "enabled": True,
            },
            "flops_profiler": {
                "enabled": True,
                "step": 1,
                "module_depth": -1,
                "top_modules": 3,
            },
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim, empty_grad=False)

        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.half)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            if n == 3: break
        assert within_range(model.flops_profiler.flops, 200, tolerance=TOLERANCE)
        assert model.flops_profiler.params == 110

    def test_flops_profiler_in_inference(self):
        mod = LeNet5(10)
        batch_size = 1024
        input = torch.randn(batch_size, 1, 32, 32)
        flops, macs, params = get_model_profile(
            mod,
            tuple(input.shape),
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=3,
            warm_up=1,
            as_string=False,
            ignore_modules=None,
        )
        print(flops, macs, params)
        assert within_range(flops, 866076672, TOLERANCE)
        assert within_range(macs, 426516480, TOLERANCE)
        assert params == 61706


@pytest.mark.sequential
def test_print_model_profile_with_none_dp_world_size(capsys):
    # Regression test for https://github.com/deepspeedai/DeepSpeed/issues/7483
    # Under sequence parallelism (Ulysses) the engine reports dp_world_size as None, which used to
    # crash print_model_profile with "unsupported format string passed to NoneType.__format__".
    model = torch.nn.Sequential(torch.nn.Linear(128, 128))
    prof = FlopsProfiler(model)
    # Mimic a DeepSpeed engine configured with Ulysses sequence parallelism, where dp_world_size is
    # None and the effective data-parallel replication is the sequence-data-parallel group.
    prof.ds_engine = SimpleNamespace(world_size=8,
                                     dp_world_size=None,
                                     seq_dp_world_size=4,
                                     mp_world_size=1,
                                     has_moe_layers=False,
                                     train_micro_batch_size_per_gpu=lambda: 1,
                                     wall_clock_breakdown=lambda: False)

    prof.start_profile()
    # A few forward passes give the profiled modules a non-zero measured duration.
    for _ in range(3):
        model(torch.randn(64, 128))
    prof.print_model_profile(profile_step=1, detailed=False)
    prof.end_profile()

    out = capsys.readouterr().out
    match = re.search(r"data parallel size:\s+(\S+)", out)
    assert match is not None
    # The sequence-data-parallel world size is reported in place of the None dp_world_size.
    assert match.group(1) == "4"


@pytest.mark.sequential
@pytest.mark.parametrize("lhs_shape, rhs_shape", [
    ((2, 3, 4), (4, )),
    ((4, ), (2, 3, 4)),
    ((8, ), (2, 8)),
    ((3, 4), (4, )),
    ((2, 3, 4), (2, 3, 4)),
    ((4, 1, 7), (1, 5, 7)),
])
def test_elementwise_broadcast_flops(lhs_shape, rhs_shape):
    """An elementwise op costs one flop per element of its broadcast result, and
    broadcasting lines the operand shapes up from the trailing dimension."""

    class Elementwise(torch.nn.Module):

        def forward(self, lhs, rhs):
            return torch.mul(lhs, rhs)

    model = Elementwise()
    lhs, rhs = torch.randn(*lhs_shape), torch.randn(*rhs_shape)

    prof = FlopsProfiler(model)
    prof.start_profile()
    result = model(lhs, rhs)
    prof.stop_profile()
    flops = prof.get_total_flops()
    prof.end_profile()

    assert flops == result.numel()


class Block(torch.nn.Module):

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)


class SplitLinears(torch.nn.Module):
    # Three blocks over three 100-wide slices of the input; with shared=True one Linear object
    # is held by all three. The blocks are load bearing: named_children() already skips a
    # repeat among immediate siblings, so a shared module must sit under distinct parents.
    def __init__(self, shared):
        super().__init__()
        shared_linear = torch.nn.Linear(100, 100, bias=False)
        self.blocks = torch.nn.ModuleList(
            [Block(shared_linear if shared else torch.nn.Linear(100, 100, bias=False)) for _ in range(3)])

    def forward(self, x):
        return tuple(block(part) for block, part in zip(self.blocks, torch.split(x, 100, dim=1)))


@pytest.mark.sequential
@pytest.mark.parametrize("shared", [True, False])
def test_flops_profiler_counts_shared_module_once(shared):
    # Regression test for https://github.com/deepspeedai/DeepSpeed/issues/7256
    # Sharing a submodule changes how many tree positions reach it, not how much work runs, so
    # both models report the same totals: three Linear(100, 100) calls, 100 * 100 MACs each.
    flops, macs, params = get_model_profile(model=SplitLinears(shared).eval(),
                                            input_shape=(1, 300),
                                            print_profile=False,
                                            detailed=False,
                                            warm_up=1,
                                            as_string=False)
    assert flops == 60000
    assert macs == 30000
    assert params == (10000 if shared else 30000)


@pytest.mark.sequential
@pytest.mark.parametrize("bias", [True, False])
def test_linear_bias_flops(bias):
    """A bias is added once per output element, exactly as the convolution counter charges for
    its own bias."""
    model = torch.nn.Linear(8, 16, bias=bias)
    inputs = torch.randn(4, 8)

    prof = FlopsProfiler(model)
    prof.start_profile()
    result = model(inputs)
    prof.stop_profile()
    flops, macs = prof.get_total_flops(), prof.get_total_macs()
    prof.end_profile()

    assert macs == 4 * 8 * 16
    assert flops == 2 * macs + (result.numel() if bias else 0)


@pytest.mark.sequential
@pytest.mark.parametrize("lhs_shape, rhs_shape", [
    ((8, ), (8, )),
    ((4, 8), (8, )),
    ((8, ), (8, 5)),
    ((4, 8), (8, 5)),
    ((3, 4, 8), (3, 8, 5)),
    ((2, 1, 4, 8), (3, 8, 5)),
])
def test_matmul_flops(lhs_shape, rhs_shape):
    """matmul costs one multiply-accumulate per output element per contracted element. The
    singleton dimension matmul adds for a 1-D operand is dropped from the result, so it must not
    be counted."""

    class Matmul(torch.nn.Module):

        def forward(self, lhs, rhs):
            return torch.matmul(lhs, rhs)

    model = Matmul()
    lhs, rhs = torch.randn(*lhs_shape), torch.randn(*rhs_shape)

    prof = FlopsProfiler(model)
    prof.start_profile()
    result = model(lhs, rhs)
    prof.stop_profile()
    flops, macs = prof.get_total_flops(), prof.get_total_macs()
    prof.end_profile()

    expected_macs = max(result.numel(), 1) * lhs_shape[-1]
    assert macs == expected_macs
    assert flops == 2 * expected_macs


@pytest.mark.sequential
@pytest.mark.parametrize("bias_shape", [(4, 5), (5, ), (1, 5), (4, 1), (1, )])
def test_addmm_broadcast_bias_flops(bias_shape):
    """The added tensor reaches every output element, so a broadcast bias costs one flop per
    output element rather than one per element of its own shape."""

    class Addmm(torch.nn.Module):

        def forward(self, bias, mat1, mat2):
            return torch.addmm(bias, mat1, mat2)

    model = Addmm()
    bias, mat1, mat2 = torch.randn(*bias_shape), torch.randn(4, 8), torch.randn(8, 5)

    prof = FlopsProfiler(model)
    prof.start_profile()
    result = model(bias, mat1, mat2)
    prof.stop_profile()
    flops, macs = prof.get_total_flops(), prof.get_total_macs()
    prof.end_profile()

    assert macs == 4 * 8 * 5
    assert flops == 2 * macs + result.numel()
