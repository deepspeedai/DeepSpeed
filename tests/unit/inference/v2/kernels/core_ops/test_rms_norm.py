# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import DtypeEnum
from deepspeed.inference.v2.kernels.core_ops import CUDARMSNorm, CUDARMSPreNorm
from ....v2.inference_test_utils import get_dtypes, allclose


def reference_rms_norm(vals: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals


def reference_rms_pre_norm(vals: torch.Tensor,
                           residual: torch.Tensor,
                           gamma: torch.Tensor,
                           epsilon: float = 1e-5) -> torch.Tensor:
    residual = residual + vals
    return residual, reference_rms_norm(residual, gamma, epsilon)


def _rms_norm_testing_helper(rows: int, channels: int, do_residual: bool, dtype: DtypeEnum) -> None:
    device = get_accelerator().current_device_name()
    t_dtype = dtype.value

    vals = torch.randn((rows, channels), dtype=t_dtype, device=device)
    gamma = torch.randn((channels), dtype=t_dtype, device=device)
    epsilon = 1e-5

    if do_residual:
        residual_in = torch.randn((rows, channels), dtype=t_dtype, device=device)
        ds_residual = residual_in.clone()

        ref_residual, ref_output = reference_rms_pre_norm(vals, residual_in, gamma, epsilon)

        kernel = CUDARMSPreNorm(channels, t_dtype, epsilon=epsilon)
        ds_out = torch.empty_like(ds_residual)

        kernel(ds_residual, ds_out, residual_in, vals, gamma)

        assert allclose(ds_out, ref_output)
        assert allclose(ds_residual, ref_residual)
    else:

        ref_output = reference_rms_norm(vals, gamma, epsilon)

        kernel = CUDARMSNorm(channels, t_dtype, epsilon=epsilon)
        ds_out = torch.empty_like(vals)

        kernel(ds_out, vals, gamma)

        assert allclose(ds_out, ref_output)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("dtype", get_dtypes())
@pytest.mark.parametrize("do_residual", [True, False])
def test_rms_dtypes(dtype: DtypeEnum, do_residual: bool) -> None:
    _rms_norm_testing_helper(883, 1024, do_residual, DtypeEnum(dtype))


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("rows, cols", [(1, 4096), (37, 2048), (112, 14432), (1024, 6144)])
@pytest.mark.parametrize("do_residual", [True, False])
def test_rms_shapes(rows: int, cols: int, do_residual: bool) -> None:
    _rms_norm_testing_helper(rows, cols, do_residual, DtypeEnum.fp16)
