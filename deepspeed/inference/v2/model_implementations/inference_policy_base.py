# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Iterable, List, Optional, Union

import torch

from ..config_v2 import RaggedInferenceEngineConfig
from ..checkpoint import CheckpointEngineBase
from ..logging import inference_logger
from .layer_container_base import LayerContainer
from .inference_model_base import DSInferenceModelBase
from .flat_model_helpers import (
    flatten_inference_model,
    make_param_filename,
    make_metadata_filename,
    ModelMetadata,
    restore_inference_model,
)

POLICIES = {}


class ContainerMap:

    def __init__(self) -> None:
        self._prefix_map = {}
        self._transformer_params = None
        self._non_transformer_params = None

    @property
    def transformer_params(self) -> Iterable[LayerContainer]:
        return self._transformer_params

    @property
    def non_transformer_params(self) -> LayerContainer:
        return self._non_transformer_params

    def set_transformer_params(self, prefixes: Union[str, Iterable[str]], containers: List[LayerContainer]) -> None:
        if not isinstance(containers, list):
            raise ValueError(
                f"The transformer containers should be a list, of one container per layer, but got {type(containers)} instead."
            )

        self._transformer_prefixes = prefixes if isinstance(prefixes, list) else [prefixes]
        self._transformer_params = containers

    def set_non_transformer_params(self, container: LayerContainer) -> None:
        self._non_transformer_params = container

    def set_unmapped_params(self, prefixes: Union[str, Iterable[str]]) -> None:
        self._unmapped_prefixes = prefixes

    def map_param(self, name, parameter) -> None:
        for unmapped_prefix in self._unmapped_prefixes:
            if name.startswith(unmapped_prefix):
                inference_logger().debug(f"Ignoring: {name} for {unmapped_prefix}")
                return

        for transformer_prefix in self._transformer_prefixes:
            if name.startswith(transformer_prefix):
                popped_name = name[len(transformer_prefix) + 1:]
                layer_idx = popped_name.split(".")[0]
                assert layer_idx.isdigit(
                ), f"expected name to start w. list index but got {layer_idx} instead, name={name}"
                layer_idx = int(layer_idx)
                inference_logger().debug(
                    f"Setting: {'.'.join(popped_name.split('.')[1:])} for layer-idx={layer_idx} to {parameter.shape}")
                self._transformer_params[layer_idx].set_dependency(".".join(popped_name.split(".")[1:]), parameter)
                return

        try:
            inference_logger().debug(f"Setting: {name} to {parameter.shape}")
            self._non_transformer_params.set_dependency(name, parameter)
        except ValueError:
            # Catch the ValueError here from the non_transformer_params because we are knowingly
            # calling it with something that may not match. This should allow us to raise a slightly more
            # informative error message.
            raise ValueError(f"Cannot find container for {name}, please double check the Containers/ContainerMap")

    def validate(self) -> None:
        if not self._non_transformer_params.is_initialized:
            raise RuntimeError("Non-transformer parameters not fully initialized after checkpoint load.")

        for layer_idx, container in enumerate(self._transformer_params):
            if not container.is_initialized:
                raise RuntimeError(
                    f"Transformer container at index {layer_idx} not fully initialized after checkpoint load.")


class PolicyMeta(ABCMeta):

    def __new__(cls, name, bases, dct):
        new_obj = super().__new__(cls, name, bases, dct)
        if name != "InferenceV2Policy":
            POLICIES[name] = new_obj
        return new_obj


class InferenceV2Policy(ABC, metaclass=PolicyMeta):
    """
    The InferenceV2Policy is the base class for all inference policies. An inference policy
    is responsible for instantiating the inference model and mapping the parameters from the
    checkpoint engine to the model itself.
    """

    def __init__(
        self,
        model_config: Any,
        checkpoint_engine: Optional[CheckpointEngineBase] = None,
        inf_checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Create the Policy with sufficient context to build the model. There are two supported
        model creation mechanisms.

        The first is the generalized ``checkpoint_engine`` which
        will iterate over the parameters of the model and provide them to the policy. These in
        turn will be sharded/transformed by the model implementation.

        The second is used to re-create a previously serialized DeepSpeed inference model. These
        checkpoints should not be used across different model backend configurations.

        TODO(cmikeh2): Enforce this in code
        """
        if checkpoint_engine is None and inf_checkpoint_path is None:
            raise ValueError("Either checkpoint_engine or ds_checkpoint_path must be provided.")

        if checkpoint_engine is not None and inf_checkpoint_path is not None:
            raise ValueError("Only one of checkpoint_engine or ds_checkpoint_path can be provided.")

        self._checkpoint_engine = checkpoint_engine
        self._inf_checkpoint_path = inf_checkpoint_path
        self._model_config = model_config

    def build_model(self, engine_config: RaggedInferenceEngineConfig, mp_group: Any) -> DSInferenceModelBase:
        """
        Completely instantiate the inference model. This will both create the ops needed to run the
        model, as well as load the model parameters via the checkpoint engine. For more context
        on each of these components please see ``instantiate_model`` and ``populate_model_parameters``.

        Arguments:
            engine_config: The config that has been used to instantiate the engine. This is used
                to communicate to the model implementation the limits on batches (sequences/tokens)
                and bound the size of intermediate buffers.
            mp_group: Object to enable communication between tensor parallel ranks.

        Returns:
            DSInferenceModelBase: An implementation of the inference model abstraction that will be
                run by the engine.
        """
        self.model = self.instantiate_model(engine_config, mp_group)
        self.populate_model_parameters()
        return self.model

    @abstractmethod
    def instantiate_model(self, engine_config: RaggedInferenceEngineConfig) -> DSInferenceModelBase:
        """
        Instantiate the inference model. Depending on the engine/model config, this could be where
        different model implementations could be selected.

        Arguments:
            engine_config: The config that has been used to instantiate the engine. This is used
                to communicate to the model implementation the limits on batches (sequences/tokens)
                and bound the size of intermediate buffers.

        Returns:
            DSInferenceModelBase: An implementation of the inference model abstraction that will be
                run by the engine.
        """
        ...

    @abstractmethod
    def build_container_map(self) -> ContainerMap:
        """
        Build a dictionary representing the structure of the string prefixes leading
        to the parameters to be mapped to the container.

        Returns:
            ContainerMap: An instantiated mapping describing how checkpoint prefixes map
                to ``LayerContainer`` instances.
        """
        raise NotImplementedError()

    def populate_model_parameters(self) -> None:
        """
        This model will iterate over the parameters (as provided by the checkpoint engine) and
        use the container map built by ``build_container_map`` to populate the model
        """

        container_map = self.build_container_map()

        if self._checkpoint_engine is not None:
            for name, parameter in self._checkpoint_engine.parameters():
                container_map.map_param(name, parameter)

            buffer, metadata = flatten_inference_model(container_map.transformer_params,
                                                       container_map.non_transformer_params, self.__class__.__name__)
        else:

            buffer_path = make_param_filename(self._inf_checkpoint_path, self.model.tp_rank, self.model.tp_size)
            metadata_path = make_metadata_filename(self._inf_checkpoint_path, self.model.tp_rank, self.model.tp_size)

            buffer = torch.load(buffer_path)
            metadata = json.load(open(metadata_path, "r"))
            metadata = ModelMetadata.parse_raw(metadata)

            restore_inference_model(buffer, metadata, container_map.transformer_params,
                                    container_map.non_transformer_params)

        container_map.validate()

        self.model.set_parameters(transformer=container_map.transformer_params,
                                  non_transformer=container_map.non_transformer_params,
                                  flattened_param_buffer=buffer,
                                  flattened_param_metadata=metadata)
