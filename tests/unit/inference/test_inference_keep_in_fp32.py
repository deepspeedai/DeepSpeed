# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""init_inference(dtype=fp16/bf16) keeps the tensors a model names for fp32 in fp32.

transformers loads the tensors a model lists in `_keep_in_fp32_modules` (under fp16) and
`_keep_in_fp32_modules_strict` (under fp16 and bf16) in fp32, because they overflow or lose precision in
half precision. Regression test for https://github.com/deepspeedai/DeepSpeed/issues/3177, where
init_inference cast T5's `wo` projection to fp16 along with the rest of the model.
"""

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator

HIDDEN = 8
# Distinct in fp32, but fp16 and bf16 round all of them to 8.0.
EXACT_VALUES = 8.0 + torch.arange(HIDDEN, dtype=torch.float32) * 1e-4


class ToyModel(torch.nn.Module):
    _keep_in_fp32_modules = ["wo"]  # fp16 only, as T5 lists its `wo` projection
    _keep_in_fp32_modules_strict = ["router_bias"]  # fp16 and bf16, as the MoE models list their routing bias

    def __init__(self):
        super().__init__()
        self.wi = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.wo = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.register_buffer("router_bias", EXACT_VALUES.clone())
        self.register_buffer("scale", torch.ones(HIDDEN))


def _dtypes(model):
    return {name: t.dtype for name, t in list(model.named_parameters()) + list(model.named_buffers())}


@pytest.mark.inference
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_init_inference_keeps_listed_tensors_in_fp32(dtype):
    if dtype not in get_accelerator().supported_dtypes():
        pytest.skip(f"{dtype} not supported by {get_accelerator().device_name()}")
    model = ToyModel().to(get_accelerator().device_name())

    engine = deepspeed.init_inference(model, dtype=dtype)

    dtypes = _dtypes(engine.module)
    assert dtypes["wi.weight"] == dtype
    assert dtypes["scale"] == dtype
    assert dtypes["router_bias"] == torch.float32
    # Kept as loaded, not rounded through half precision and cast back.
    assert torch.equal(engine.module.router_bias.cpu(), EXACT_VALUES)
    # `_keep_in_fp32_modules` applies under fp16 only, the way transformers applies it.
    assert dtypes["wo.weight"] == (torch.float32 if dtype == torch.float16 else dtype)


@pytest.mark.inference
@pytest.mark.skipif(torch.float16 not in get_accelerator().supported_dtypes(), reason="fp16 not supported")
def test_init_inference_fp16_keeps_t5_wo_in_fp32():
    from transformers import T5Config, T5ForConditionalGeneration

    config = T5Config(vocab_size=64,
                      d_model=32,
                      d_kv=8,
                      d_ff=64,
                      num_layers=1,
                      num_decoder_layers=1,
                      num_heads=2,
                      decoder_start_token_id=0)
    model = T5ForConditionalGeneration(config).to(get_accelerator().device_name())

    engine = deepspeed.init_inference(model, dtype=torch.float16)

    for name, param in engine.module.named_parameters():
        expected = torch.float32 if name.endswith("wo.weight") else torch.float16
        assert param.dtype == expected, name
