# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractstaticmethod
from typing import Any, Dict, Type

import torch

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class DSModuleConfig(DeepSpeedConfigModel):

    max_tokens: int


class DSModuleBase(torch.nn.Module, ABC):
    """
    Base class for all DeepSpeed Inference modules. This class establishes
    the basic attributes of a DSModule. Only abstract functionality modules should inherit
    directly from this class, not specific implementations.
    """

    @abstractstaticmethod
    def name() -> str:
        """
        Return a memorable, human-readable name for this module.

        This will be used as a key in custom inference configurations and should only
        be implemented by the children of functionality modules.
        """
        ...

    @abstractstaticmethod
    def config_class() -> Type[DSModuleConfig]:
        """
        Return the associated config class for this module.

        This should be implemented (along with the config class) by an abstract functionality
        module.
        """
        ...

    @abstractstaticmethod
    def supports_config(config: DSModuleConfig) -> bool:
        """
        Return whether or not this module supports the given config.

        This should be implemented by the children of functionality modules and should report
        whether it would be feasible to instantiate this module with the given config.
        """
        ...

    def __init__(self, config: DSModuleConfig, implementation_config: Dict[str, Any] = {}) -> None:
        """
        Initialize the module with the given config.
        """
        super().__init__()
        self._config = config
        self._implementation_config = implementation_config
