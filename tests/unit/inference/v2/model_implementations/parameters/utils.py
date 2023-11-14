# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.allocator import on_device
from deepspeed.inference.v2.inference_parameter import InferenceParameter
from deepspeed.inference.v2.model_implementations.parameter_base import ParameterBase, ParametrizedList


class SimpleParam(ParameterBase):
    """
    Parameter with single dependency.
    """

    param: torch.Tensor

    @on_device
    def finalize(self) -> torch.Tensor:
        return self.inference_model.transform(self.param)


class SimpleParametrizedList(ParametrizedList):
    """
    Parameter list based on `num_dependencies` attribute.
    """

    count_attr: str = "num_dependencies"


class ListParam(ParameterBase):
    """
    Parameter with list dependency.

    NOTE: This uses the tuple workaround for the `ParametrizedList` class
    as described in the docstring of `ParametrizedList`.
    """

    params: SimpleParametrizedList

    @on_device
    def finalize(self) -> torch.Tensor:
        return self.inference_model.transform(torch.cat(tuple(self.params)))


class DummyInferenceModel:

    @property
    def num_dependencies(self) -> int:
        return 2

    def transform(self, param: torch.Tensor) -> torch.Tensor:
        return InferenceParameter.initialize(param)


def validate_device(tensor: torch.Tensor):
    assert tensor.device == torch.device(get_accelerator().current_device())
