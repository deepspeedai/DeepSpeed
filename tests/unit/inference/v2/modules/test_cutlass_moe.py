# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.inference_utils import ActivationType, DtypeEnum
from deepspeed.inference.v2.modules import ConfigBundle
from deepspeed.inference.v2.modules.configs import DSMoEConfig
from deepspeed.inference.v2.modules.interfaces import DSMoERegistry

from ..kernels.ragged_ops.ragged_testing_utils import build_simple_batch
from ...v2.inference_test_utils import allclose, get_dtypes


def _gating_reference(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference gating code.
    """
    logits = logits.float()
    probs = torch.nn.functional.softmax(logits, dim=1)

    indices1_s = torch.argmax(probs, dim=-1)
    mask1 = torch.nn.functional.one_hot(indices1_s, num_classes=logits.shape[-1])
    indices_mask = mask1.sum(dim=1) * logits.shape[-1] - 1
    indices1_s = torch.min(indices1_s, indices_mask)

    gates1_s = (probs * mask1).sum(dim=1)

    sorted_indices = indices1_s.sort()[1]
    original_indices = sorted_indices.sort()[1]

    exp_count = torch.bincount(indices1_s, minlength=logits.shape[-1]).long()
    exp_count_cumsum = exp_count.cumsum(dim=0)

    return sorted_indices, original_indices, exp_count_cumsum, gates1_s


def _reference_impl(hidden_states: torch.Tensor, gate_weight: torch.Tensor, mlp_1_w: torch.Tensor,
                    mlp_2_w: torch.Tensor, mlp_1_b: torch.Tensor, mlp_2_b: torch.Tensor,
                    act_fn: ActivationType) -> torch.Tensor:
    """
    Reference implementation of the MoE module.
    """

    act_fn_dict = {
        ActivationType.GELU: torch.nn.functional.gelu,
        ActivationType.RELU: torch.nn.functional.relu,
        ActivationType.SILU: torch.nn.functional.silu,
        ActivationType.IDENTITY: lambda x: x,
    }

    logits = torch.matmul(hidden_states, gate_weight.t())
    sorted_indices, original_indices, exp_count_cumsum, gate_scales = _gating_reference(logits)

    moe_input = hidden_states[sorted_indices]

    output_unordered = torch.empty_like(hidden_states)

    for expert_idx in range(mlp_1_w.shape[0]):
        min_bound = 0 if expert_idx == 0 else exp_count_cumsum[expert_idx - 1]
        max_bound = exp_count_cumsum[expert_idx]

        input_slice = moe_input[min_bound:max_bound]
        intermediate = torch.nn.functional.linear(input_slice, mlp_1_w[expert_idx], mlp_1_b[expert_idx])

        intermediate = act_fn_dict[act_fn](intermediate)
        output_slice = torch.nn.functional.linear(intermediate, mlp_2_w[expert_idx], mlp_2_b[expert_idx])

        output_unordered[min_bound:max_bound] = output_slice

    output = output_unordered[original_indices]

    output.mul_(gate_scales.unsqueeze(-1)).reshape(hidden_states.shape)
    return output


def _cutlass_moe_testing_helper(tokens: int,
                                in_channels: int,
                                intermediate_dim: int,
                                experts: int,
                                dtype: int,
                                activation_type: ActivationType = ActivationType.GELU,
                                use_bias: bool = True,
                                iters: int = 1) -> None:

    config = DSMoEConfig(max_tokens=4096,
                         model_dim=in_channels,
                         intermediate_features=intermediate_dim,
                         n_experts=experts,
                         activation=activation_type,
                         input_dtype=dtype,
                         output_dtype=dtype)

    implementation_config = {"weight_dtype": DtypeEnum(dtype)}

    bundle = ConfigBundle(name='cutlass_multi_gemm_moe', config=config, implementation_config=implementation_config)
    moe_module = DSMoERegistry.instantiate_config(bundle)

    batch = build_simple_batch([tokens])

    # Parameters
    gate_weight = torch.randn(
        (experts, in_channels), dtype=dtype.value, device=get_accelerator().current_device()) * .1

    mlp_1_w = torch.randn(
        (experts, intermediate_dim, in_channels), dtype=dtype.value, device=get_accelerator().current_device()) * .1
    mlp_2_w = torch.randn(
        (experts, in_channels, intermediate_dim), dtype=dtype.value, device=get_accelerator().current_device()) * .1

    if use_bias:
        mlp_1_b = torch.randn(
            (experts, intermediate_dim), dtype=dtype.value, device=get_accelerator().current_device()) * .1
        mlp_2_b = torch.randn(
            (experts, in_channels), dtype=dtype.value, device=get_accelerator().current_device()) * .1
    else:
        mlp_1_b = None
        mlp_2_b = None

    gate_ds = moe_module.transform_gate_param(gate_weight)
    mlp_1_w_ds = moe_module.transform_moe_mlp_1_param(mlp_1_w)
    mlp_1_b_ds = moe_module.transform_moe_mlp_1_param(mlp_1_b)
    mlp_2_w_ds = moe_module.transform_moe_mlp_2_param(mlp_2_w)
    mlp_2_b_ds = moe_module.transform_moe_mlp_2_param(mlp_2_b)

    for _ in range(iters):
        # Input vals
        hidden_states = torch.randn(
            (tokens, in_channels), dtype=dtype.value, device=get_accelerator().current_device()) * .1

        # Reference implementation
        ref_output = _reference_impl(hidden_states, gate_weight, mlp_1_w, mlp_2_w, mlp_1_b, mlp_2_b, activation_type)

        output = moe_module(hidden_states,
                            batch,
                            gate_ds,
                            mlp_1_w_ds,
                            mlp_2_w_ds,
                            mlp_1_b=mlp_1_b_ds,
                            mlp_2_b=mlp_2_b_ds)

        # Increase the tolerance for larger meta ops since the error is additive
        assert allclose(output, ref_output, tolerances=(1e-2, 1e-2))

    get_accelerator().synchronize()


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("experts", [2, 32, 64])
def test_expert_variance(experts: int) -> None:
    _cutlass_moe_testing_helper(tokens=876,
                                in_channels=4096,
                                intermediate_dim=2048,
                                experts=experts,
                                dtype=DtypeEnum.fp16,
                                activation_type=ActivationType.IDENTITY,
                                use_bias=True)


@pytest.mark.inference_v2_ops
def test_successive_inputs():
    """
    The CUTLASS MoE uses persistent state (expert counts) that is assumed to be cleared
    on each forward pass. This ensures that the module is clearing that metadata.
    """
    _cutlass_moe_testing_helper(tokens=876,
                                in_channels=4096,
                                intermediate_dim=2048,
                                experts=64,
                                dtype=DtypeEnum.fp16,
                                activation_type=ActivationType.IDENTITY,
                                use_bias=True,
                                iters=10)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("dtype", get_dtypes(include_float=False))
def test_dtypes(dtype: torch.dtype) -> None:
    _cutlass_moe_testing_helper(tokens=876,
                                in_channels=4096,
                                intermediate_dim=2048,
                                experts=64,
                                dtype=DtypeEnum(dtype),
                                activation_type=ActivationType.IDENTITY,
                                use_bias=True)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("activation_type", [ActivationType.GELU, ActivationType.RELU, ActivationType.SILU])
def test_activation_types(activation_type: ActivationType) -> None:
    _cutlass_moe_testing_helper(tokens=876,
                                in_channels=4096,
                                intermediate_dim=2048,
                                experts=64,
                                dtype=DtypeEnum.fp16,
                                activation_type=activation_type,
                                use_bias=True)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("in_channels, out_channels", [(4096, 2048), (2048, 8192), (6144, 3072)])
def test_in_out_channels(in_channels: int, out_channels: int) -> None:
    _cutlass_moe_testing_helper(tokens=876,
                                in_channels=in_channels,
                                intermediate_dim=out_channels,
                                experts=64,
                                dtype=DtypeEnum.fp16,
                                activation_type=ActivationType.IDENTITY,
                                use_bias=True)
