# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from ...model_implementations.parameter_base import ParameterBase
"""
Common QKV Parameter Patterns
"""


class FusedQKVParameter(ParameterBase):
    """
    Traditional fused QKV parameters for QKV projection. This is functionally
    a direct copy.

    src_qkv_w shape: [3 * out_features, in_features]
    qkv_w shape: [3 * out_features, in_features]
    """

    params: torch.Tensor

    def finalize(self) -> torch.Tensor:
        return self.inference_model.transform_qkv_param(self.params)


class UnfusedQKVParameter(ParameterBase):
    """
    QKV parameter container for unfused QKV projection.

    src_param shapes: 3 x [out_features, in_features]
    dst_param shape: [3 x out_features, in_features]
    """

    q_params: torch.Tensor

    k_params: torch.Tensor

    v_params: torch.Tensor

    def finalize(self):
        fused_param = torch.cat([self.q_params, self.k_params, self.v_params], dim=0)
        return self.inference_model.transform_qkv_param(fused_param)


def megatron_qkv_reshape(param: torch.Tensor, head_size: int, n_heads: int) -> torch.Tensor:
    assert param.shape[0] == 3 * n_heads * head_size

    all_heads = torch.chunk(param, chunks=3 * n_heads, dim=0)
    q_heads = all_heads[::3]
    k_heads = all_heads[1::3]
    v_heads = all_heads[2::3]
    return torch.cat([q_heads, k_heads, v_heads], dim=0)


class MegatronQKVParameter(ParameterBase):
    """
    QKV parameter container for Megatron-style QKV projection. Megatron stores the parameter
    as [n_heads, 3, head_size, in_features] whereas our inference system is built around
    [3, n_heads, head_size, in_features]. This container handles the conversion.

    Note: this container expects the model implementation to implement properties for
    `head_size` and `n_heads`.

    src_qkv_w shape: [3 * out_features, in_features]
    qkv_w shape: [3 * out_features, in_features]
    """

    params: torch.Tensor

    def finalize(self) -> torch.Tensor:
        head_size = self.inference_model.head_size
        n_heads = self.inference_model.n_heads

        transposed_param = megatron_qkv_reshape(self.params, head_size, n_heads)
        return self.inference_model.transform_qkv_param(transposed_param)


def transform_gqa_megatron(src_param: torch.Tensor, head_size: int, n_q_heads: int, n_kv_heads: int) -> torch.Tensor:
    assert src_param.shape[0] == (2 * n_kv_heads + n_q_heads) * head_size

    head_ratio = n_q_heads // n_kv_heads

    # Reshape to get the groups as the leading dimension
    groups_leading_view = src_param.reshape(n_kv_heads, 2 + head_ratio, head_size, -1)
    q_heads = groups_leading_view[:, :head_ratio, :, :].reshape(-1, groups_leading_view.shape[-1])
    k_heads = groups_leading_view[:, head_ratio, :, :].reshape(-1, groups_leading_view.shape[-1])
    v_heads = groups_leading_view[:, head_ratio + 1, :, :].reshape(-1, groups_leading_view.shape[-1])
    # Squeeze will remove extra dimension for bias
    return torch.cat([q_heads, k_heads, v_heads], dim=0).squeeze()


class GQAMegatronQKVParameter(ParameterBase):
    """
    QKV parameter for Megatron-style QKV projection with GQA-style QKV projection. In this
    storage format each of the groups is stored consecutively, so there will be multiple q_heads,
    then one k head, and one v head.

    Note: this container expects the model implementation to implement properties for
    `head_size`, `n_q_heads`, and `n_kv_heads`.

    src_qkv_w shape: [(2 * n_kv_heads + n_q_heads) * head_size, in_features]
    qkv_w shape: [(2 * n_kv_heads + n_q_heads) * head_size, in_features]
    """

    params: torch.Tensor

    def finalize(self) -> torch.Tensor:
        head_size = self.inference_model.head_size
        n_q_heads = self.inference_model.n_heads_q
        n_kv_heads = self.inference_model.n_heads_kv
        transposed_param = transform_gqa_megatron(self.params, head_size, n_q_heads, n_kv_heads)
        return self.inference_model.transform_qkv_param(transposed_param)
