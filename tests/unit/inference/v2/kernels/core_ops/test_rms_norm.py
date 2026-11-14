# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import DtypeEnum
from deepspeed.inference.v2.kernels.core_ops import CUDARMSNorm, CUDARMSPreNorm
from ....v2.inference_test_utils import (get_dtypes, allclose, allocate_output_with_canary, assert_rows_independent,
                                         norm_tolerances, CANARY_VALUE)


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
        ds_residual, residual_canary = allocate_output_with_canary(rows, channels, t_dtype)

        ref_residual, ref_output = reference_rms_pre_norm(vals, residual_in, gamma, epsilon)

        kernel = CUDARMSPreNorm(channels, t_dtype, epsilon=epsilon)
        ds_out, out_canary = allocate_output_with_canary(rows, channels, t_dtype)

        kernel(ds_residual, ds_out, residual_in, vals, gamma)

        def run_rows(start: int, end: int):
            residual_single = torch.empty_like(vals[start:end])
            out_single = torch.empty_like(vals[start:end])
            kernel(residual_single, out_single, residual_in[start:end], vals[start:end], gamma)
            return out_single, residual_single

        assert allclose(ds_out, ref_output, norm_tolerances(t_dtype))
        assert allclose(ds_residual, ref_residual, norm_tolerances(t_dtype))
        assert (out_canary == CANARY_VALUE).all()
        assert (residual_canary == CANARY_VALUE).all()
        assert_rows_independent(run_rows, (ds_out, ds_residual))
    else:

        ref_output = reference_rms_norm(vals, gamma, epsilon)

        kernel = CUDARMSNorm(channels, t_dtype, epsilon=epsilon)
        ds_out, out_canary = allocate_output_with_canary(rows, channels, t_dtype)

        kernel(ds_out, vals, gamma)

        def run_rows(start: int, end: int):
            out_single = torch.empty_like(vals[start:end])
            kernel(out_single, vals[start:end], gamma)
            return (out_single, )

        assert allclose(ds_out, ref_output, norm_tolerances(t_dtype))
        assert (out_canary == CANARY_VALUE).all()
        assert_rows_independent(run_rows, (ds_out, ))


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


# For hidden sizes <= 128 several rows share one thread block, from 16 rows per block (16 threads per row) down
# to 256 (one thread per row for fp32/4 and fp16/8). Each case uses one row more than fits in a block, so the
# last block is partially filled and must not read or write rows past the end. 128 / 64 channels is the
# per-head Q/K norm shape of EXAONE4 (rows = tokens * heads).
@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("dtype, channels, rows", [(DtypeEnum.fp32, 4, 257), (DtypeEnum.fp32, 32, 33),
                                                   (DtypeEnum.fp32, 64, 17), (DtypeEnum.fp16, 8, 257),
                                                   (DtypeEnum.fp16, 16, 129), (DtypeEnum.fp16, 32, 65),
                                                   (DtypeEnum.fp16, 64, 33), (DtypeEnum.fp16, 128, 17),
                                                   (DtypeEnum.bf16, 128, 17)])
@pytest.mark.parametrize("do_residual", [True, False])
def test_rms_small_channels(dtype: DtypeEnum, channels: int, rows: int, do_residual: bool) -> None:
    if dtype == DtypeEnum.bf16 and not get_accelerator().is_bf16_supported():
        pytest.skip("bf16 is not supported on this system")
    _rms_norm_testing_helper(rows, channels, do_residual, dtype)
