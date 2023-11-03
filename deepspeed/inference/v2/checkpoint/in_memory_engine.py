# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Iterable, Tuple
import torch

from .base_engine import CheckpointEngineBase


class InMemoryModelEngine(CheckpointEngineBase):
    """
    This "checkpoint" engine uses the existing interface to enable loading parameters into an
    inference model from a model already instantiated in memory. In general, this is not the
    recommended way to use the inference engine, and should only be used when absolutely necessary.

    The primary limitation of this approach is that the model must be fully instantiated in memory.
    In a tensor parallel scenario, this means that the model is either replicated many times in host
    memory. Currently, it is also recommended to only use this approach for models held in host memory.

    In order to free the memory held by this copy of the model, we delete the model in the first call
    to `parameters`, so it is not safe to make this call twice.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Create virtual checkpoint engine for the provided module.

        Args:
            model (torch.nn.Module): Model to load parameters from.
        """
        super().__init__()
        self.model = model

    def parameters(self) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, parameter in self.model.named_parameters():
            yield name, parameter

        del self.model
