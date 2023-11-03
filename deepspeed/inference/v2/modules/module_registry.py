# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractstaticmethod
from typing import Any, Dict, Type

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from .ds_module import DSModuleBase


class ConfigBundle(DeepSpeedConfigModel):
    """
    A config bundle is a collection of configs that are used to instantiate a model implementation.
    """
    name: str
    config: DeepSpeedConfigModel
    implementation_config: Dict[str, Any] = {}


class DSModuleRegistryBase(ABC):
    """
    Class holding logic for tracking the DSModule implementations of a given interface.
    """

    @classmethod
    def instantiate_config(cls, config_bundle: ConfigBundle) -> DSModuleBase:
        """
        Given a DSModule key, attempt to instantiate
        """
        if config_bundle.name not in cls.registry:
            raise KeyError(f"Unknown DSModule: {config_bundle.name}, cls.registry={cls.registry}")

        target_implementation = cls.registry[config_bundle.name]
        if not target_implementation.supports_config(config_bundle.config):
            raise ValueError(f"Config {config_bundle.config} is not supported by {target_implementation}")

        return cls.registry[config_bundle.name](config_bundle.config, config_bundle.implementation_config)

    @abstractstaticmethod
    def associated_class() -> Type[DSModuleBase]:
        """
        Return the class associated with this registry.
        """
        raise NotImplementedError("Must associated a DSModule class with its registry.")

    @classmethod
    def register_module(cls, child_class: DSModuleBase) -> None:
        """
        Register a module with this registry.
        """
        if not issubclass(child_class, cls.associated_class()):
            raise TypeError(
                f"Can only register subclasses of {cls.associated_class()}, {child_class} does not inherit from {cls.associated_class()}"
            )
        cls.registry[child_class.name()] = child_class
        return child_class
