# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.model_implementations.flat_model_helpers import (
    flatten_inference_model,
    restore_inference_model,
)
from deepspeed.inference.v2.model_implementations.layer_container_base import LayerContainer
from .utils import SimpleParam, DummyInferenceModel


class TransformerLayerContainer(LayerContainer):
    """
    Stub layer container
    """
    PARAM_MAPPING = {
        "param_1": "param_1.param",
        "param_2": "param_2.param",
    }

    param_1: SimpleParam

    param_2: SimpleParam


class NonTransformerContainer(LayerContainer):
    """
    Stub layer container
    """
    PARAM_MAPPING = {
        "param_1": "param_1.param",
        "param_2": "param_2.param",
        "param_3": "param_3.param",
    }

    param_1: SimpleParam

    param_2: SimpleParam

    param_3: SimpleParam


@pytest.mark.inference_v2
def test_contiguify_roundtrip():
    """
    Validate that contiguify round trips and reconstructions are correct.
    """
    model = DummyInferenceModel()

    n_layers = 2
    transformer_params = []
    transformer_containers = []

    # Create parameters and populate them into the containers
    for i in range(n_layers):
        transformer_containers.append(TransformerLayerContainer(model))
        layer_params = []
        for j in range(2):
            layer_params.append(torch.rand(16, 16))
            transformer_containers[i].set_dependency(f"param_{j+1}", layer_params[j])

        layer_params = [p.to(get_accelerator().current_device()) for p in layer_params]

        transformer_params.append(layer_params)
        assert transformer_containers[i].is_populated == True

    non_transformer_params = []
    non_transformer_container = NonTransformerContainer(model)

    for i in range(3):
        non_transformer_params.append(torch.rand(16, 16).permute(1, 0))
        non_transformer_container.set_dependency(f"param_{i+1}", non_transformer_params[i])

    non_transformer_params = [p.to(get_accelerator().current_device()) for p in non_transformer_params]

    def validate_containers(t_containers: List[LayerContainer], n_t_containers: LayerContainer,
                            t_params: List[List[torch.Tensor]], n_t_params: List[torch.Tensor]):
        """
        Validate params match what is on the containers.
        """
        for i in range(n_layers):
            l_c = t_containers[i]

            assert l_c.is_initialized == True

            assert torch.equal(l_c.param_1, t_params[i][0])
            assert torch.equal(l_c.param_2, t_params[i][1])

        assert n_t_containers.is_initialized == True
        assert torch.equal(n_t_containers.param_1, n_t_params[0])
        assert torch.equal(n_t_containers.param_2, n_t_params[1])
        assert torch.equal(n_t_containers.param_3, n_t_params[2])
        assert not n_t_containers.param_1.is_contiguous()
        assert not n_t_containers.param_2.is_contiguous()
        assert not n_t_containers.param_3.is_contiguous()

    buffer, metadata = flatten_inference_model(transformer_containers, non_transformer_container, "NoOpPolicy")

    # Validate containers before contiguify
    validate_containers(transformer_containers, non_transformer_container, transformer_params, non_transformer_params)

    # Validate restore pass
    transformer_containers_r = []
    for i in range(n_layers):
        transformer_containers_r.append(TransformerLayerContainer(model))

    non_transformer_container_r = NonTransformerContainer(model)

    restore_inference_model(buffer, metadata, transformer_containers_r, non_transformer_container_r)

    validate_containers(transformer_containers_r, non_transformer_container_r, transformer_params,
                        non_transformer_params)
