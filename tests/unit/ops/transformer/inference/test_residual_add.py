"""
Copyright 2022 The Microsoft DeepSpeed Team
"""

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system",
                allow_module_level=True)


def allclose(x, y):
    assert x.dtype == y.dtype
    rtol, atol = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-2)}[x.dtype]
    return torch.allclose(x, y, rtol=rtol, atol=atol)


@pytest.fixture(scope="module")
def inference_module():
    return InferenceBuilder().load()


def res_add_bias_ref(hidden_state,
                     residual,
                     attn_output,
                     attn_bias,
                     final_bias,
                     mp_size=1,
                     pre_attn_norm=True):
    if pre_attn_norm:
        hidden_state += (residual + final_bias + attn_output + attn_bias) / mp_size
    else:
        hidden_state += residual + final_bias
    return hidden_state


def res_add_bias_ref_gptj(hidden_state,
                          residual,
                          attn_output,
                          attn_bias,
                          final_bias,
                          add_attn_bias,
                          mp_size):
    hidden_state += attn_output + (residual + final_bias) / mp_size
    if add_attn_bias:
        hidden_state += attn_bias / mp_size
    return hidden_state


def run_residual_add_reference(hidden_state,
                               residual,
                               attn_output,
                               attn_bias,
                               final_bias,
                               mlp_after_attn,
                               add_attn_bias,
                               mp_size,
                               pre_attn_norm):
    if mlp_after_attn:
        return res_add_bias_ref(hidden_state,
                                residual,
                                attn_output,
                                attn_bias,
                                final_bias,
                                mp_size,
                                pre_attn_norm)
    else:
        return res_add_bias_ref_gptj(hidden_state,
                                     residual,
                                     attn_output,
                                     attn_bias,
                                     final_bias,
                                     add_attn_bias,
                                     mp_size)


@pytest.mark.inference
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("hidden_dim", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("mlp_after_attn", [True, False])
@pytest.mark.parametrize("add_bias", [True, False])
@pytest.mark.parametrize("mp_size", [1, 2])
@pytest.mark.parametrize("pre_attn_norm", [True, False])
def test_residual_add(inference_module,
                      batch,
                      sequence,
                      hidden_dim,
                      dtype,
                      mlp_after_attn,
                      add_bias,
                      mp_size,
                      pre_attn_norm):
    ds_out = torch.randn((batch, sequence, hidden_dim), dtype=dtype, device='cuda')
    residual = torch.randn((batch, sequence, hidden_dim), dtype=dtype, device='cuda')
    attn_output = torch.randn((batch, sequence, hidden_dim), dtype=dtype, device='cuda')
    final_bias = torch.randn((hidden_dim), dtype=dtype, device='cuda')
    attn_bias = torch.randn((hidden_dim), dtype=dtype, device='cuda')

    ref_out = ds_out.clone()
    ref_out = run_residual_add_reference(ref_out,
                                         residual,
                                         attn_output,
                                         attn_bias,
                                         final_bias,
                                         mlp_after_attn,
                                         add_bias,
                                         mp_size,
                                         pre_attn_norm)

    res_add_args = [
        ds_out,
        residual,
        attn_output,
        attn_bias,
        final_bias,
        mp_size,
        mlp_after_attn,
        add_bias,
        pre_attn_norm
    ]

    if dtype == torch.float16:
        ds_out = inference_module.residual_add_bias_fp16(*res_add_args)
    elif dtype == torch.float32:
        ds_out = inference_module.residual_add_bias_fp32(*res_add_args)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    assert (allclose(ds_out, ref_out))
