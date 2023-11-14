# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.inference.v2.inference_parameter import InferenceParameter
from deepspeed.inference.v2.model_implementations.layer_container_base import LayerContainer

from .utils import SimpleParam, DummyInferenceModel


class ParentLayer(LayerContainer):
    """
    A layer that has a dependency on a simple parameter.
    """

    param_1: SimpleParam


class ChildLayer(ParentLayer):
    """
    A layer that inherits from another layer.
    """

    param_2: SimpleParam


@pytest.mark.inference_v2
def test_layer_inheritance():
    inference_model = DummyInferenceModel()

    multi_param_layer = ChildLayer(inference_model)

    assert multi_param_layer.n_params == 2
    assert multi_param_layer.is_initialized is False

    multi_param_layer.param_1.param = torch.ones(16, 16)

    assert multi_param_layer.is_initialized is False

    multi_param_layer.param_2.param = torch.full((16, 16), 2.0)

    assert multi_param_layer.is_populated is True
    assert isinstance(multi_param_layer.param_1, InferenceParameter)
    assert isinstance(multi_param_layer.param_2, InferenceParameter)
