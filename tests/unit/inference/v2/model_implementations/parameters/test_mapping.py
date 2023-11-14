# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.inference.v2.allocator import on_device
from deepspeed.inference.v2.inference_parameter import InferenceParameter
from deepspeed.inference.v2.model_implementations.parameter_base import ParameterBase, ParamList
from deepspeed.inference.v2.model_implementations.layer_container_base import LayerContainer


class MultiDependencyContainer(ParameterBase):

    dependency_1: torch.Tensor

    dependency_2: torch.Tensor

    @on_device
    def finalize(self) -> torch.Tensor:
        param = torch.cat([self.dependency_1, self.dependency_2])
        return InferenceParameter.initialize(param)


class ListDependencyContainer(ParameterBase):

    dependencies: ParamList("list_items")  # noqa: F821

    @on_device
    def finalize(self) -> torch.Tensor:
        param = torch.cat(tuple(self.dependencies))
        return InferenceParameter.initialize(param)


class MappingLayer(LayerContainer):
    PARAM_MAPPING = {
        "model.val.item.d_1": "multi_depend.dependency_1",
        "model.val.item.d_2": "multi_depend.dependency_2",
        "model.list_vals.*.d": "list_depend.dependencies"
    }

    multi_depend: MultiDependencyContainer

    list_depend: ListDependencyContainer


class SubMappingLayer(MappingLayer):
    PARAM_MAPPING = {
        "model.val.item2.d_1": "multi_depend2.dependency_1",
        "model.val.item2.d_2": "multi_depend2.dependency_2",
    }

    multi_depend2: MultiDependencyContainer


class DoubleMappingLayer(LayerContainer):
    PARAM_MAPPING = {
        "model.val.item.d_1": ["multi_depend.dependency_1", "multi_depend.dependency_2"],
    }

    multi_depend: MultiDependencyContainer


class InferenceModel:

    @property
    def list_items(self) -> int:
        return 16


@pytest.mark.inference_v2
def test_mapping_syntax():
    model = InferenceModel()

    mapping_layer = MappingLayer(model)

    mapping_layer.set_dependency("model.val.item.d_1", torch.ones(1))
    mapping_layer.set_dependency("model.val.item.d_2", torch.ones(1) * 2)

    assert isinstance(mapping_layer.multi_depend, torch.Tensor)

    for i in range(16):
        mapping_layer.set_dependency(f"model.list_vals.{i}.d", torch.ones(1) * i)
        if i != 16 - 1:
            assert mapping_layer.is_populated == False

    assert isinstance(mapping_layer.list_depend, InferenceParameter)
    assert mapping_layer.is_populated == True


@pytest.mark.inference_v2
def test_sub_mapping_syntax():
    model = InferenceModel()

    mapping_layer = SubMappingLayer(model)

    mapping_layer.set_dependency("model.val.item.d_1", torch.ones(1))
    mapping_layer.set_dependency("model.val.item.d_2", torch.ones(1) * 2)

    assert isinstance(mapping_layer.multi_depend, InferenceParameter)

    mapping_layer.set_dependency("model.val.item2.d_1", torch.ones(1))
    mapping_layer.set_dependency("model.val.item2.d_2", torch.ones(1) * 2)

    assert isinstance(mapping_layer.multi_depend2, InferenceParameter)

    # We want to check into double digits to make sure that this isn't specific
    # to single difit indexing.
    for i in range(16):
        mapping_layer.set_dependency(f"model.list_vals.{i}.d", torch.ones(1) * i)
        if i != 16 - 1:
            assert mapping_layer.is_populated == False

    assert isinstance(mapping_layer.list_depend, InferenceParameter)
    assert mapping_layer.is_populated == True


@pytest.mark.inference_v2
def test_double_mapping_syntax():
    model = InferenceModel()

    mapping_layer = DoubleMappingLayer(model)
    mapping_layer.set_dependency("model.val.item.d_1", torch.ones(1))

    # The single parameter setting should immediately make the parameter finalized
    # and the whole layer initialized.
    assert isinstance(mapping_layer.multi_depend, InferenceParameter)
    assert mapping_layer.is_populated == True


@pytest.mark.inference_v2
def test_insufficient_mapping_syntax():
    """
    In the above example, we don't have a mapping for `multi_depend2.dependency_2`.
    """

    with pytest.raises(ValueError):

        class InsuffienctMappingLayer(LayerContainer):
            PARAM_MAPPING = {
                "model.val.item.d_1": "multi_depend1.dependency_1",
                "model.val.item.d_2": "multi_depend1.dependency_2",
                "model.val.item2.d_1": "multi_depend2.dependency_1",
            }

            multi_depend1: MultiDependencyContainer

            multi_depend2: MultiDependencyContainer


@pytest.mark.inference_v2
def test_unknown_target_mapping_syntax():
    """
    In the above example, `multi_depend_unknown` does not exist
    """

    with pytest.raises(ValueError):

        class UnknownTargetMappingLayer(LayerContainer):
            PARAM_MAPPING = {
                "model.val.item.d_1": "multi_depend1.dependency_1",
                "model.val.item.d_2": "multi_depend1.dependency_2",
                "model.val.item2.d_1": "multi_depend_unknown.dependency_1",
            }

            multi_depend: MultiDependencyContainer
