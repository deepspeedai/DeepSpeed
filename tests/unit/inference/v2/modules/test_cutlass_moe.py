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


def _mixtral_moe_baseline(hidden_states: torch.Tensor,
                          gate_weight: torch.Tensor,
                          mlp_w1: torch.Tensor,
                          mlp_w2: torch.Tensor,
                          mlp_w3: torch.Tensor,
                          force_float: bool = False) -> torch.Tensor:
    """
    Baseline implementation for mixtral MoE module.

    Based on transformers implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
    """
    output_dtype = hidden_states.dtype
    if force_float:
        hidden_states = hidden_states.float()
        gate_weight = gate_weight.float()
        mlp_w1 = mlp_w1.float()
        mlp_w2 = mlp_w2.float()
        mlp_w3 = mlp_w3.float()

    router_logits = torch.nn.functional.linear(hidden_states, gate_weight)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = routing_weights.topk(k=2, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    # NOTE(cmikeh2): This is a difference implementation, ours will preserve the original scale
    # as float32 and perform in-kernel fused FP16->FP32->FP16 conversion.
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros_like(hidden_states)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=gate_weight.shape[0]).permute(2, 1, 0)
    get_accelerator().synchronize()

    for expert_idx in range(gate_weight.shape[0]):
        exp_mlp_w1 = mlp_w1[expert_idx]
        exp_mlp_w2 = mlp_w2[expert_idx]
        exp_mlp_w3 = mlp_w3[expert_idx]

        idx, top_x = torch.where(expert_mask[expert_idx])

        if top_x.shape[0] == 0:
            continue

        top_x_list = top_x.tolist()
        idx_list = idx.tolist()

        current_state = hidden_states[top_x_list]

        linear = torch.nn.functional.linear
        intermediate = torch.nn.functional.silu(linear(current_state, exp_mlp_w1)) * linear(current_state, exp_mlp_w3)
        output = linear(intermediate, exp_mlp_w2) * routing_weights[top_x_list, idx_list].unsqueeze(-1)
        final_hidden_states.index_add_(0, top_x, output.to(final_hidden_states.dtype))

    return final_hidden_states.to(output_dtype)


@pytest.mark.inference_v2_ops
def test_mixtral_moe_config():

    experts = 8
    n_top_k = 2
    in_channels = 4096
    intermediate_dim = 2048
    dtype = DtypeEnum.bf16

    # Parameters
    gate_weight = torch.randn(
        (experts, in_channels), dtype=dtype.value, device=get_accelerator().current_device()) * .1

    mlp_w1 = torch.randn(
        (experts, intermediate_dim, in_channels), dtype=dtype.value, device=get_accelerator().current_device()) * .1
    mlp_w3 = torch.randn(
        (experts, intermediate_dim, in_channels), dtype=dtype.value, device=get_accelerator().current_device()) * .1
    mlp_w2 = torch.randn(
        (experts, in_channels, intermediate_dim), dtype=dtype.value, device=get_accelerator().current_device()) * .1

    n_tokens = 256
    hidden_states = torch.randn(
        (n_tokens, in_channels), dtype=dtype.value, device=get_accelerator().current_device()) * .1

    baseline = _mixtral_moe_baseline(hidden_states, gate_weight, mlp_w1, mlp_w2, mlp_w3)

    mlp_w13_fused = torch.cat([mlp_w1, mlp_w3], dim=-1).reshape(experts, 2 * intermediate_dim, in_channels)

    config = DSMoEConfig(max_tokens=4096,
                         model_dim=in_channels,
                         intermediate_features=intermediate_dim,
                         n_experts=experts,
                         activation=ActivationType.SiGLU,
                         input_dtype=dtype,
                         output_dtype=dtype,
                         top_k=n_top_k,
                         normalize_scores=True)

    implementation_config = {"weight_dtype": DtypeEnum(dtype)}

    bundle = ConfigBundle(name='cutlass_multi_gemm_moe', config=config, implementation_config=implementation_config)
    moe_module = DSMoERegistry.instantiate_config(bundle)

    batch = build_simple_batch([n_tokens])

    gate_ds = moe_module.transform_gate_param(gate_weight)
    mlp_w1_ds = moe_module.transform_moe_mlp_1_param(mlp_w13_fused)
    mlp_w2_ds = moe_module.transform_moe_mlp_2_param(mlp_w2)

    output = moe_module(hidden_states, batch, gate_ds, mlp_w1_ds, mlp_w2_ds)

    # NOTE(cmikeh2): These are higher than the other tests for reasons that aren't quite
    # clear to me. My best guess is that the SiGLU activation is causing larger numerical
    # divergence. The thresholds chosen here is based on the observed error between the
    # float and bfloat16 reference implementations.
    assert allclose(output, baseline.to(dtype.value), tolerances=(5e-2, 5e-2))
