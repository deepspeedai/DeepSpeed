# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.inference.v2.inference_parameter import InferenceParameter
from deepspeed.inference.v2.model_implementations.layer_container_base import LayerContainer

from .utils import validate_device, SimpleParam, ListParam, DummyInferenceModel


class MultiParameterLayer(LayerContainer):
    """
    Two dependencies, both of which are simple parameters.
    """

    param_1: SimpleParam

    param_2: SimpleParam


class MixedMultiParameterLayer(LayerContainer):
    """
    Two dependencies, one of which is a simple parameter, the other is a list parameter.
    """

    param_1: SimpleParam

    param_2: ListParam


@pytest.mark.inference_v2
def test_multi_parameter_layer():
    inference_model = DummyInferenceModel()

    multi_param_layer = MultiParameterLayer(inference_model)

    assert multi_param_layer.n_params == 2
    assert multi_param_layer.is_populated is False

    multi_param_layer.param_1.param = torch.ones(16, 16)

    assert multi_param_layer.is_populated is False

    multi_param_layer.param_2.param = torch.full((16, 16), 2.0)

    assert multi_param_layer.is_populated is True
    assert isinstance(multi_param_layer.param_1, InferenceParameter)
    assert isinstance(multi_param_layer.param_2, InferenceParameter)


@pytest.mark.inference_v2
def test_mixed_multi_parameter_layer():
    inference_model = DummyInferenceModel()

    mixed_multi_param_layer = MixedMultiParameterLayer(inference_model)

    assert mixed_multi_param_layer.n_params == 2
    assert mixed_multi_param_layer.is_populated is False

    mixed_multi_param_layer.param_2.params[1] = torch.full((16, 16), 2.0)
    assert mixed_multi_param_layer.is_populated is False
    assert not isinstance(mixed_multi_param_layer.param_2, InferenceParameter)

    mixed_multi_param_layer.param_1.param = torch.ones(16, 16)
    assert mixed_multi_param_layer.is_populated is False
    assert isinstance(mixed_multi_param_layer.param_1, InferenceParameter)

    validate_device(mixed_multi_param_layer.param_1)

    mixed_multi_param_layer.param_2.params[0] = torch.full((16, 16), 2.0)

    assert mixed_multi_param_layer.is_populated is True
    assert isinstance(mixed_multi_param_layer.param_2, InferenceParameter)

    validate_device(mixed_multi_param_layer.param_2)
