# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder  # type: ignore
from deepspeed.ops.transformer.inference.op_binding.pre_rms_norm import PreRMSNormOp
from deepspeed.ops.transformer.inference.op_binding.rms_norm import RMSNormOp
from .inference_test_utils import allclose, get_dtypes, run_with_output_canaries, CANARY_VALUE

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)


def ref_implementation(vals, gamma, epsilon):
    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals


def ds_implementation(vals, gamma, epsilon):
    return RMSNormOp()(vals, gamma, epsilon)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 17, 33, 128])
@pytest.mark.parametrize("channels", [32, 64, 384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_rms_norm(batch, seq_len, channels, dtype):
    device = get_accelerator().current_device_name()
    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    gamma = torch.randn((channels), dtype=dtype, device=device)
    epsilon = 1e-5

    ref_output = ref_implementation(vals, gamma, epsilon)
    new_output = ds_implementation(vals, gamma, epsilon)

    assert allclose(new_output, ref_output)


def pre_ds_implementation(vals, residual, gamma, epsilon):
    return PreRMSNormOp()(vals, residual, gamma, epsilon)


def pre_ref_implementation(vals, residual, gamma, epsilon):
    residual = vals.to(torch.float32) + residual.to(torch.float32)
    vals = residual

    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals, residual.to(gamma.dtype)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("seq_len", [1, 17, 33, 128])
@pytest.mark.parametrize("channels", [32, 64, 384, 512, 768, 1024, 2048, 8192, 14432])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pre_norm(batch, seq_len, channels, dtype):
    device = get_accelerator().current_device_name()
    vals = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    residual = torch.randn((batch, seq_len, channels), dtype=dtype, device=device)
    gamma = torch.randn((channels), dtype=dtype, device=device)
    epsilon = 1e-5

    ref_output = pre_ref_implementation(vals, residual, gamma, epsilon)
    new_output = pre_ds_implementation(vals, residual, gamma, epsilon)

    assert allclose(new_output[0], ref_output[0])
    #assert allclose(new_output[1], ref_output[1])


# For hidden sizes <= 128 several rows share one thread block; with 17 / 33 rows the last block is only
# partially filled. Nothing may be written past the last row of an output (checked with canaries placed
# right after each output, so this fails under plain pytest, not only under compute-sanitizer).
@pytest.mark.inference_ops
@pytest.mark.parametrize("rows", [17, 33])
@pytest.mark.parametrize("dtype, channels", [(torch.float32, 32), (torch.float32, 64), (torch.float16, 64),
                                             (torch.float16, 128), (torch.bfloat16, 128)])
@pytest.mark.parametrize("op", ["rms_norm", "pre_rms_norm"])
def test_rms_norm_no_write_past_last_row(rows, dtype, channels, op):
    if dtype not in get_dtypes():
        pytest.skip(f"{dtype} is not supported on this system")
    inference_module = InferenceBuilder().load()
    device = get_accelerator().current_device_name()
    vals = torch.randn((1, rows, channels), dtype=dtype, device=device)
    residual = torch.randn((1, rows, channels), dtype=dtype, device=device)
    gamma = torch.randn((channels), dtype=dtype, device=device)
    epsilon = 1e-5

    if op == "rms_norm":
        run, num_outputs = lambda: inference_module.rms_norm(vals, gamma, epsilon), 1
    else:
        run, num_outputs = lambda: inference_module.pre_rms_norm(vals, residual, gamma, epsilon), 2
    _, canaries = run_with_output_canaries(run,
                                           rows * channels,
                                           dtype,
                                           num_outputs=num_outputs,
                                           canary_numel=32 * channels)
    for canary in canaries:
        assert (canary == CANARY_VALUE).all()
