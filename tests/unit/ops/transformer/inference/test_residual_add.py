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


@pytest.fixture
def inference_module():
    return InferenceBuilder().load()


def run_residual_add_reference(hidden_state,
                               residual,
                               attention_output,
                               final_bias,
                               attention_output_bias,
                               mlp_after_attn,
                               add_bias,
                               mp_size=1):
    residual_scaled = residual / mp_size
    final_bias_scaled = final_bias / mp_size
    attention_output_scaled = attention_output / mp_size
    attention_output_bias_scaled = attention_output_bias / mp_size

    hidden_state = hidden_state + residual_scaled + final_bias_scaled

    # in case that mlp_after_attn = True, we additionally need to scale attention_output as well
    if mlp_after_attn:
        hidden_state += attention_output_scaled
    else:
        hidden_state += attention_output

    # TODO: The `add_bias` flag is used only for `launch_gptj_residual_add` kernel (aka, mlp_after_attn is False).
    # This is a hack to get the parametarized add_bias to work. We need to fix this after refactoring the kernels.
    add_bias = True if mlp_after_attn else add_bias

    if add_bias:
        hidden_state = hidden_state + attention_output_bias_scaled

    return hidden_state


@pytest.mark.inference
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("hidden_dim", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("mlp_after_attn", [True, False])
@pytest.mark.parametrize("add_bias", [True, False])
@pytest.mark.parametrize("mp_size", [1, 2])
# @pytest.mark.parametrize("preln", [True])  # TODO: add support for preln
def test_residual_add(inference_module,
                      batch,
                      sequence,
                      hidden_dim,
                      dtype,
                      mlp_after_attn,
                      add_bias,
                      mp_size):
    preln = True
    ds_out = torch.randn((batch, sequence, hidden_dim), dtype=dtype, device='cuda')
    residual = torch.randn((batch, sequence, hidden_dim), dtype=dtype, device='cuda')
    attention_output = torch.randn((batch,
                                    sequence,
                                    hidden_dim),
                                   dtype=dtype,
                                   device='cuda')
    final_bias = torch.randn((hidden_dim), dtype=dtype, device='cuda')
    attention_output_bias = torch.randn((hidden_dim), dtype=dtype, device='cuda')

    ref_out = ds_out.clone()
    ref_out = run_residual_add_reference(ref_out,
                                         residual,
                                         attention_output,
                                         final_bias,
                                         attention_output_bias,
                                         mlp_after_attn,
                                         add_bias,
                                         mp_size)

    inference_module.residual_add(
            ds_out,         # in-place update of ds_out. Needs reafactoring to be consistent with other kernels.
            residual,
            attention_output,
            final_bias,
            attention_output_bias,
            mp_size,
            mlp_after_attn,
            add_bias,
            preln)

    assert (allclose(ds_out, ref_out))
