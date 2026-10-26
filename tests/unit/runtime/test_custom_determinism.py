# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Policy checks that do not require compiling a CUDA extension."""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def deterministic_policy():
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])


def test_stochastic_layer_rejected_before_extension_load(deterministic_policy):
    from deepspeed.ops.transformer.transformer import DeepSpeedTransformerConfig, DeepSpeedTransformerLayer
    with pytest.raises(RuntimeError, match="stochastic_mode=False"):
        DeepSpeedTransformerLayer(DeepSpeedTransformerConfig(stochastic_mode=True))


def test_stochastic_forward_rejected_after_construction(deterministic_policy):
    from deepspeed.ops.transformer.transformer import DeepSpeedTransformerFunction
    # No tensor/extension access should precede the strict-policy rejection.
    config = SimpleNamespace(stochastic_mode=True)
    with pytest.raises(RuntimeError, match="stochastic_mode=False"):
        DeepSpeedTransformerFunction.forward(*([None] * 18), config)


@pytest.mark.parametrize("enabled", [False, True])
def test_grouped_kernel_launch_policy(enabled):
    from deepspeed.ops.triton_ops.group_gemm_triton import _launch_group_gemm
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    calls = []

    class Kernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    kernel = Kernel()
    kernel.fn = Kernel()
    config = SimpleNamespace(kwargs={"BLOCK_M": 32}, num_warps=8, num_stages=4)
    try:
        torch.use_deterministic_algorithms(enabled)
        _launch_group_gemm(kernel, [config], (1, 1), "input", NUM_GROUPS=2)
        assert calls == [((1, 1), ("input", ), {
            "NUM_GROUPS": 2,
            **({
                "BLOCK_M": 32,
                "num_warps": 8,
                "num_stages": 4
            } if enabled else {})
        })]
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
