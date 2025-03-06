# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.inference.v2.allocator import on_device
from deepspeed.inference.v2.inference_parameter import InferenceParameter
from deepspeed.inference.v2.model_implementations.parameter_base import ParameterBase, ParamList
from deepspeed.inference.v2.model_implementations.layer_container_base import LayerContainer
from deepspeed.inference.v2.model_implementations.common_parameters import *

from .utils import validate_device


class SimpleMoELayer(LayerContainer):

    moe_mlp_1: UnfusedMoEMLP1Parameter


class DummyInferenceModel:

    def __init__(self, experts_per_rank: int) -> None:
        self._num_experts = experts_per_rank

    @property
    def n_experts(self) -> int:
        return self._num_experts

    @on_device
    def transform_moe_mlp_1_param(self, param: torch.Tensor) -> torch.Tensor:
        return InferenceParameter.initialize(param)


@pytest.mark.inference_v2
def test_simple_moe_layer():

    inference_model = DummyInferenceModel(experts_per_rank=2)

    simple_moe_layer = SimpleMoELayer(inference_model)

    assert simple_moe_layer.moe_mlp_1.experts[0] is None
    assert simple_moe_layer.moe_mlp_1.experts[1] is None

    # Set the first expert
    simple_moe_layer.moe_mlp_1.experts[0] = torch.zeros(16, 16)

    assert simple_moe_layer.moe_mlp_1.experts[0] is not None
    assert simple_moe_layer.moe_mlp_1.experts[1] is None

    assert not simple_moe_layer.is_initialized

    # Set the second expert
    simple_moe_layer.moe_mlp_1.experts[1] = torch.ones(16, 16)

    # We have all the experts, so the layer should be initialized
    assert simple_moe_layer.is_initialized
    assert isinstance(simple_moe_layer.moe_mlp_1, torch.Tensor)

    validate_device(simple_moe_layer.moe_mlp_1)


"""
Check that we can mix the number of elements in lists in the same context and have that
be tracked correctly.
"""


class CustomListParam1(ParameterBase):

    deps: ParamList("attr_1")


class CustomListParam2(ParameterBase):

    deps: ParamList("attr_2")


class MixedLayer(LayerContainer):

    list_1: CustomListParam1
    list_2: CustomListParam2


class MixedInferenceModel:

    @property
    def attr_1(self) -> int:
        return 1

    @property
    def attr_2(self) -> int:
        return 2


@pytest.mark.inference_v2
def test_mixed_param_lists():
    model = MixedInferenceModel()

    layer = MixedLayer(model)

    assert layer.list_1.deps.n_params == 1
    assert layer.list_2.deps.n_params == 2
