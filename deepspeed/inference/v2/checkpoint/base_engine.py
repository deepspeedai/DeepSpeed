# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import torch

#from .huggingface_engine import HuggingFaceCheckpointEngine

MEGATRON = 'megatron'
HUGGINGFACE = 'huggingface'


class CheckpointEngineBase(ABC):
    """
    Abstract interface for checkpoint engines to implement.

    There is no ``__init__`` method here by design, since the creation of the checkpoint
    engine will happen outside the policy/engine code. The tradeoff being made here is
    that we will write different frontends for different checkpoint engines, but these
    frontends can be tailored to the specific checkpoint engine/model source needs.
    """

    @abstractmethod
    def parameters(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        This method should create a generator of tuples of the form (name, parameter) for
        all parameters in the model. The name should be the fully qualified name of the
        parameter, and the parameter should be a torch.Tensor.

        The expected use of a checkpoint engine is the following:
        ```python
        for name, parameter in checkpoint_engine.parameters():
            container_map.map_param(name, parameter)
        ```
        For a concrete use example, see ``InferenceV2Policy``.
        """
        ...
