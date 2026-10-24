# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""GPTNEOXLayerPolicy loads its fused query_key_value weight from a checkpoint through
`maybe_copy(..., qkv=True, megatron_v2=True, split_qkv=False)`.
See GPTNEOXContainer.load_params in deepspeed/module_inject/containers/gptneox.py.
"""

import types

import torch

from deepspeed.module_inject.auto_tp import ReplaceWithTensorSlicing
from deepspeed.module_inject.policy import maybe_copy


class _FakeWeightQuantizer:
    q_int8 = False

    def quantize(self, weight, **kwargs):
        return weight


def _copy_fused_qkv(hidden_size, heads, mp_size, weight=None):
    if weight is None:
        weight = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float16)
    sd = {"attention.query_key_value.weight": weight}
    dst_module = types.SimpleNamespace(
        attn_qkvw=torch.nn.Parameter(torch.empty(hidden_size, (hidden_size // mp_size) * 3, dtype=torch.float16)))
    mp_replace = ReplaceWithTensorSlicing(mp_group=None, mp_size=mp_size)
    maybe_copy(dst_module,
               sd,
               _FakeWeightQuantizer(),
               mp_replace,
               "attn_qkvw",
               "attention.query_key_value.weight",
               qkv=True,
               megatron_v2=True,
               split_qkv=False,
               heads=heads)
    return dst_module.attn_qkvw


def test_fused_qkv_megatron_v2_checkpoint_load_does_not_crash():
    # Used to raise `RuntimeError: The size of tensor a (...) must match the size of tensor b
    # (...)` for every mp_size, including mp_size=1 (no tensor-parallel split at all), because
    # `tmp` was transposed one extra time on this non-int8 path. See deepspeedai/DeepSpeed#3038.
    for mp_size in (1, 2, 4):
        dst = _copy_fused_qkv(hidden_size=256, heads=8, mp_size=mp_size)
        assert dst.shape == (256, (256 // mp_size) * 3)


def test_fused_qkv_megatron_v2_checkpoint_load_matches_injection_layout():
    # Without a checkpoint, MegatronContainer.transpose() lays the weight out as
    # _align_qkv_transposed(W.t()); loading the same W from a checkpoint must give the same tensor.
    hidden_size, heads = 256, 8
    weight = torch.randn(3 * hidden_size, hidden_size, dtype=torch.float16)
    x = weight.t().contiguous()
    x_1 = x.view(hidden_size, heads, -1)
    q, k, v = torch.split(x_1, x_1.shape[-1] // 3, dim=-1)
    expected = torch.cat((q.reshape(hidden_size, -1), k.reshape(hidden_size, -1), v.reshape(hidden_size, -1)),
                         dim=-1).reshape(x.shape)
    dst = _copy_fused_qkv(hidden_size=hidden_size, heads=heads, mp_size=1, weight=weight)
    assert torch.equal(dst.data, expected)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
