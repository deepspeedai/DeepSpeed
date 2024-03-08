# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import ActivationType, DtypeEnum, is_gated
from deepspeed.inference.v2.modules import ConfigBundle
from deepspeed.inference.v2.modules.configs import DSLinearConfig
from deepspeed.inference.v2.modules.interfaces import DSLinearRegistry
from ...v2.inference_test_utils import allclose


def reference_implementation(hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                             act_type: ActivationType) -> torch.Tensor:
    dtype = hidden_states.dtype
    out_states = torch.nn.functional.linear(hidden_states, weight, bias)
    out_states.float()

    if is_gated(act_type):
        act_func_map = {
            ActivationType.ReGLU: torch.nn.functional.relu,
            ActivationType.GEGLU: lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
            ActivationType.SiGLU: torch.nn.functional.silu,
        }

        act_act = out_states[..., ::2]
        act_linear = out_states[..., 1::2]

        act_act = act_func_map[act_type](act_act)
        out_states = act_act * act_linear
    else:
        act_func_map = {
            ActivationType.RELU: torch.nn.functional.relu,
            ActivationType.GELU: torch.nn.functional.gelu,
            ActivationType.SILU: torch.nn.functional.silu,
            ActivationType.IDENTITY: lambda x: x,
        }

        out_states = act_func_map[act_type](out_states)
    return out_states.to(dtype)


def _fp6_quant_dequant_weights(weight: torch.Tensor) -> torch.Tensor:
    from deepspeed.inference.v2.modules.implementations.linear.quantized_linear import fp_quantize
    weight_quantized_fake_fp6, scales = fp_quantize(weight, num_bits=6, exp_bits=3)
    return weight_quantized_fake_fp6 * scales


def quant_dequant_implementation(hidden_states: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                                 act_type: ActivationType) -> torch.Tensor:
    dtype = hidden_states.dtype
    weight_dequantized = _fp6_quant_dequant_weights(weight)
    out_states = torch.nn.functional.linear(hidden_states, weight_dequantized, bias)
    out_states.float()

    if is_gated(act_type):
        act_func_map = {
            ActivationType.ReGLU: torch.nn.functional.relu,
            ActivationType.GEGLU: lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
            ActivationType.SiGLU: torch.nn.functional.silu,
        }

        act_act = out_states[..., ::2]
        act_linear = out_states[..., 1::2]

        act_act = act_func_map[act_type](act_act)
        out_states = act_act * act_linear
    else:
        act_func_map = {
            ActivationType.RELU: torch.nn.functional.relu,
            ActivationType.GELU: torch.nn.functional.gelu,
            ActivationType.SILU: torch.nn.functional.silu,
            ActivationType.IDENTITY: lambda x: x,
        }

        out_states = act_func_map[act_type](out_states)
    return out_states.to(dtype)


def _fp6_quantized_linear_helper(tokens: int,
                                 in_channels: int,
                                 out_channels: int,
                                 dtype: DtypeEnum,
                                 act_fn: ActivationType,
                                 use_bias: bool = True,
                                 expect_failure: bool = False) -> None:
    # The current FP6 kernel only supports NVIDIA Ampere GPUs.
    if not 'cuda' in get_accelerator().current_device_name():
        return
    major, _ = torch.cuda.get_device_capability()  #ignore-cuda
    if major != 8:
        return

    # Input vals
    hidden_states = torch.randn(
        (tokens, in_channels), dtype=dtype.value, device=get_accelerator().current_device_name()) * .01

    weight_out_channels = 2 * \
        out_channels if is_gated(act_fn) else out_channels
    weight = torch.randn(
        (weight_out_channels, in_channels), dtype=dtype.value, device=get_accelerator().current_device_name()) * .01
    if use_bias:
        bias = torch.randn(
            (weight_out_channels), dtype=dtype.value, device=get_accelerator().current_device_name()) * .01
    else:
        bias = None

    # quantize and dequantize output
    ref_quant_dequant_output = quant_dequant_implementation(hidden_states, weight, bias, act_fn)

    linear_config = DSLinearConfig(max_tokens=2048,
                                   in_channels=in_channels,
                                   out_channels=out_channels,
                                   activation=act_fn,
                                   input_dtype=dtype,
                                   output_dtype=dtype)
    bundle = ConfigBundle(name='quantized_wf6af16_linear', config=linear_config)
    fp6_linear_module = DSLinearRegistry.instantiate_config(bundle)
    weight_fp6 = fp6_linear_module.transform_param(weight.clone().cpu()).to(get_accelerator().current_device_name())

    if expect_failure:
        with pytest.raises(ValueError) as excinfo:
            ds_output = fp6_linear_module(hidden_states, weight_fp6, bias)
        assert "The out and in channel should be multiple of 256 and 64 respectively." in str(excinfo.value)
    else:
        ds_output = fp6_linear_module(hidden_states, weight_fp6, bias)
        # The current FP6 kernel uses FP16 Tensor Core.
        tolerances = (3e-2, 2e-3)  # tolerances for fp16

        # Check DeepSpeed implementation
        assert allclose(ds_output, ref_quant_dequant_output, tolerances=tolerances)


all_acts = [
    ActivationType.RELU,
    ActivationType.GELU,
    ActivationType.SILU,
    ActivationType.GEGLU,
    ActivationType.ReGLU,
    ActivationType.SiGLU,
]
all_tokens = [1, 37]
all_in_out_channels = [
    (4096, 4096),
    (8192, 28672),
]


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("tokens", all_tokens)
@pytest.mark.parametrize("in_channels, out_channels", all_in_out_channels)
@pytest.mark.parametrize("act_fn", all_acts)
@pytest.mark.parametrize("use_bias", [True, False])
def test_fp6_quantized_linear_act_fn(tokens: int, in_channels: int, out_channels: int, act_fn: ActivationType,
                                     use_bias: bool) -> None:
    _fp6_quantized_linear_helper(tokens=tokens,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 dtype=DtypeEnum.fp16,
                                 act_fn=act_fn,
                                 use_bias=use_bias)


# Other shapes, not supported by FP6 kernels. Will raise ValueError.
@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("tokens", all_tokens)
@pytest.mark.parametrize("in_channels, out_channels", [(4608, 1728)])
@pytest.mark.parametrize("act_fn", all_acts)
@pytest.mark.parametrize("use_bias", [True, False])
def test_fp6_quantized_linear_act_fn_fail(tokens: int, in_channels: int, out_channels: int, act_fn: ActivationType,
                                          use_bias: bool) -> None:
    _fp6_quantized_linear_helper(tokens=tokens,
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 dtype=DtypeEnum.fp16,
                                 act_fn=act_fn,
                                 use_bias=use_bias,
                                 expect_failure=True)
