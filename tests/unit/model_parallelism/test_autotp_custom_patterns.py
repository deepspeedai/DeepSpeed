# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed.comm as dist
import deepspeed
from copy import deepcopy
from torch import nn

from unit.common import DistributedTest, preferred_dtype
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
from deepspeed.module_inject.layers import (LinearAllreduce, LinearLayer, SubParamLinearLayer)
from deepspeed.module_inject.autotp_config import AutoTPConfig
from deepspeed.module_inject.auto_tp import AutoTP


def skip_on_device():
    if get_accelerator().device_name() == 'xpu':
        pytest.skip("XPU requires a higher version for test")


class SequentialLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim, nlayers=1):
        super(SequentialLinearModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(nlayers)])

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


def init_tp_engine(tp_size, partition_config=None):
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "tensor_parallel": {
            "autotp_size": tp_size,
        },
        "zero_optimization": {
            "stage": 0,
        }
    }
    if partition_config is not None:
        config_dict["tensor_parallel"]["partition_config"] = partition_config
    if preferred_dtype() is torch.float16:
        config_dict["fp16"] = {"enabled": True}
    elif preferred_dtype() is torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    model = SequentialLinearModel(hidden_dim=8)
    deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)


def apply_autotp_with_partition_config(model, tp_size, partition_config):
    groups._init_tp_mesh_device(tensor_model_parallel_size=tp_size)
    autotp_config = AutoTPConfig.from_dict(partition_config)
    autotp = AutoTP(module=model,
                    all_reduce_linears=[],
                    prefix="",
                    state_dict=None,
                    linear_layer_setting=None,
                    orig_layer_impl=None,
                    keep_module_on_host=False,
                    partition_config=autotp_config)
    autotp.set_tensor_parallel_config(tp_size, groups.get_tensor_model_parallel_group())
    autotp.update_linear_policies()
    autotp._replace_module(model)
    return model


def gather_subparam_output(output, subparam_sizes, mp_group):
    tp_world_size = dist.get_world_size(group=mp_group)
    local_sizes = [size // tp_world_size for size in subparam_sizes]
    output_chunks = torch.split(output, local_sizes, dim=-1)
    gathered_chunks = []
    for chunk in output_chunks:
        chunk = chunk.contiguous()
        gathered = [torch.empty_like(chunk) for _ in range(tp_world_size)]
        dist.all_gather(gathered, chunk, group=mp_group)
        gathered_chunks.append(torch.cat(gathered, dim=-1))
    return torch.cat(gathered_chunks, dim=-1)


class TestAutoTPCustomPatterns(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_custom_pattern_replacement(self):
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*linears\\.1\\.weight$"],
                    "partition_type": "column",
                },
                {
                    "patterns": [".*linears\\.2\\.weight$"],
                    "partition_type": "skip",
                },
            ],
        }
        model = SequentialLinearModel(hidden_dim=16, nlayers=3)
        model = apply_autotp_with_partition_config(model, tp_size=2, partition_config=partition_config)

        assert isinstance(model.linears[0], LinearAllreduce)
        assert isinstance(model.linears[1], LinearLayer)
        assert isinstance(model.linears[2], nn.Linear)

    def test_first_match_precedence(self):
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "skip",
                },
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "column",
                },
            ],
        }
        model = SequentialLinearModel(hidden_dim=16, nlayers=1)
        model = apply_autotp_with_partition_config(model, tp_size=2, partition_config=partition_config)

        assert isinstance(model.linears[0], nn.Linear)


def test_invalid_custom_shape_rejected():
    bad_config = {
        "layer_specs": [{
            "patterns": [".*"],
            "partition_type": "column",
            "shape": [2, [1, 1]],
        }]
    }
    with pytest.raises(ValueError, match="nested tuple only allowed at partition_dim"):
        AutoTPConfig.from_dict(bad_config)


class TestAutoTPFusedWeights(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_gate_up_fused_weight_partition(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        torch.manual_seed(42)
        linear = nn.Linear(hidden_dim,
                           hidden_dim * 2,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=(2, -1),
                                    partition_dim=0,
                                    name="mlp.gate_up_proj")
        assert layer._subparam_sizes == (hidden_dim, hidden_dim)
        assert layer.weight.shape == (hidden_dim, hidden_dim)

        layer.gather_params([layer.weight, layer.bias])
        torch.testing.assert_close(layer.weight.data, full_weight)
        torch.testing.assert_close(layer.bias.data, full_bias)

    def test_gqa_uneven_qkv_fused_weight_partition(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        q_size, k_size, v_size = 8, 4, 4
        torch.manual_seed(123)
        linear = nn.Linear(hidden_dim,
                           q_size + k_size + v_size,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=((q_size, k_size, v_size), -1),
                                    partition_dim=0,
                                    name="self_attn.qkv_proj")
        assert layer._subparam_sizes == (q_size, k_size, v_size)
        assert layer.weight.shape == ((q_size + k_size + v_size) // 2, hidden_dim)

        layer.gather_params([layer.weight, layer.bias])
        torch.testing.assert_close(layer.weight.data, full_weight)
        torch.testing.assert_close(layer.bias.data, full_bias)

    def test_gqa_uneven_qkv_fused_forward(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)

        hidden_dim = 8
        q_size, k_size, v_size = 8, 4, 4
        torch.manual_seed(321)
        linear = nn.Linear(hidden_dim,
                           q_size + k_size + v_size,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device())
        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=((q_size, k_size, v_size), -1),
                                    partition_dim=0,
                                    name="self_attn.qkv_proj")

        torch.manual_seed(42)
        inputs = torch.randn(2, hidden_dim, dtype=preferred_dtype(), device=get_accelerator().current_device())
        full_output = linear(inputs)
        tp_output = layer(inputs)

        gathered_output = gather_subparam_output(tp_output, (q_size, k_size, v_size),
                                                 groups.get_tensor_model_parallel_group())
        atol = 1e-3
        rtol = 2e-2
        if preferred_dtype() is torch.float32:
            atol = 1e-5
            rtol = 1e-5
        torch.testing.assert_close(gathered_output, full_output, atol=atol, rtol=rtol)
