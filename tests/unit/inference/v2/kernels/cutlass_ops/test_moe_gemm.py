# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import ActivationType, DtypeEnum
from deepspeed.inference.v2.kernels.cutlass_ops import MoEGEMM
from ....v2.inference_test_utils import allclose

SINGLE_EXPERT_CASES = [(13, 2048, 2048), (256, 1024, 4096), (278, 5120, 2048), (893, 5120, 2560)]

PYTORCH_ACT_FN_MAP = {
    ActivationType.GELU: torch.nn.functional.gelu,
    ActivationType.SILU: torch.nn.functional.silu,
    ActivationType.RELU: torch.nn.functional.relu
}


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("n_tokens, in_neurons, out_neurons", SINGLE_EXPERT_CASES)
def test_single_expert(n_tokens: int, in_neurons: int, out_neurons: int) -> None:
    """
    Validate that the GEMM kernel produces identical results for a single GEMM instance.
    """
    device = get_accelerator().current_device()

    activations = torch.rand((n_tokens, in_neurons), device=device, dtype=torch.float16) - 0.5
    weights = torch.rand((1, in_neurons, out_neurons), device=device, dtype=torch.float16) - 0.5
    biases = torch.randn((1, out_neurons), device=device, dtype=torch.float16)

    weights_ref = weights.reshape(in_neurons, out_neurons)
    biases_ref = biases.reshape(out_neurons)
    ref_output = torch.matmul(activations, weights_ref) + biases_ref

    moe_gemm = MoEGEMM(DtypeEnum.fp16, ActivationType.IDENTITY)
    output = torch.empty((n_tokens, out_neurons), device=device, dtype=torch.float16)
    cumsum_rows = torch.tensor([n_tokens], dtype=torch.int64, device=device)

    moe_gemm(output, activations, weights, cumsum_rows, biases)
    assert allclose(output, ref_output, tolerances=(1e-2, 1e-2))
    get_accelerator().synchronize()


def moe_test_helper(in_neurons: int, out_neurons: int, n_experts: int, max_tokens_per_expert: int,
                    act_fn: ActivationType, dtype: DtypeEnum) -> None:
    """
    Helper function for validating the GEMM kernel for a single expert.
    """
    device = get_accelerator().current_device()

    expert_allocations = torch.randint(0, max_tokens_per_expert, (n_experts, ), device=device, dtype=torch.int32)
    cumsum_rows = expert_allocations.cumsum(dim=0)
    print(cumsum_rows.dtype)

    activations = torch.rand((cumsum_rows[-1], in_neurons), device=device, dtype=dtype.value) - 0.5
    weights = torch.rand((n_experts, in_neurons, out_neurons), device=device, dtype=dtype.value) - 0.5
    biases = torch.randn((n_experts, out_neurons), device=device, dtype=dtype.value)

    out_ref = torch.empty((cumsum_rows[-1], out_neurons), device=device, dtype=dtype.value)

    for expert_idx in range(n_experts):
        start = cumsum_rows[expert_idx - 1] if expert_idx > 0 else 0
        end = cumsum_rows[expert_idx]
        activations_slice = activations[start:end]
        weights_slice = weights[expert_idx]
        biases_slice = biases[expert_idx]
        out_ref[start:end] = torch.matmul(activations_slice, weights_slice) + biases_slice

    if act_fn != ActivationType.IDENTITY:
        act_fn_fn = PYTORCH_ACT_FN_MAP[act_fn]
        out_ref = act_fn_fn(out_ref)

    moe_gemm = MoEGEMM(DtypeEnum.fp16, act_fn)
    output = torch.empty((cumsum_rows[-1], out_neurons), device=device, dtype=dtype.value)

    moe_gemm(output, activations, weights, cumsum_rows, biases)

    if dtype == DtypeEnum.bf16:
        assert allclose(output, out_ref, tolerances=(1e-1, 1e-1))
    else:
        assert allclose(output, out_ref, tolerances=(1e-2, 1e-2))
    get_accelerator().synchronize()


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("max_tokens_per_expert", [1, 4, 16, 64, 128])
def test_multi_expert(max_tokens_per_expert: int) -> None:
    """
    Validate for multi-expert GEMM instances that the output is identical to the reference.
    """
    moe_test_helper(5120, 2048, 64, max_tokens_per_expert, ActivationType.IDENTITY, DtypeEnum.fp16)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("act_fn", [ActivationType.GELU, ActivationType.SILU, ActivationType.RELU])
def test_act_fns(act_fn: ActivationType) -> None:
    """
    Validate activation function behavior.
    """
    moe_test_helper(5120, 2048, 64, 32, act_fn, DtypeEnum.fp16)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("dtype", [DtypeEnum.fp16, DtypeEnum.bf16])
def test_dtypes(dtype: DtypeEnum) -> None:
    """
    Validate data type behavior.
    """
    moe_test_helper(5120, 2048, 64, 32, ActivationType.IDENTITY, dtype)
