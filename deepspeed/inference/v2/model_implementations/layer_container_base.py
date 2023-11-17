# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
from typing import Type

import torch

from deepspeed.accelerator import get_accelerator
from .parameter_base import ParameterBase, ParametrizedList
from ..inference_parameter import InferenceParameter

# Currently have dependency loops for the type hints.
InferenceModel = Type["InferenceModel"]
LayerContainer = Type["LayerContainer"]

MAPPING_KEY = "PARAM_MAPPING"
PLIST_HELPERS = "_ds_plist_strip_vals"


def make_finalization_callback(all_names: str):
    """
    Helper method for building the finalization callback for a LayerContainer. This
    is not client code and should not be used or called directly.
    """

    def finalization_callback(self, param: ParameterBase, finalized_param: torch.Tensor) -> None:
        """
        Callback for when a parameter is finalized.
        """
        self._finalized_params += 1

        for name in all_names:
            if getattr(self, name) is param:
                setattr(self, name, finalized_param)

    return finalization_callback


class LayerMetaclass(type):
    """
    MetaClass for the LayerContainer base class. This class will parse the annotations
    of the class that correspond to `ParameterBase` and create None initializers for each
    as well as a finalization callback that for when each `ParameterBase` is finalized
    and should be replaced with a Tensor.
    """

    def __new__(cls, clsname, bases, attrs):

        annotations = attrs.get("__annotations__", {})

        for base in bases:
            # We'll pick up all annotations on any base classes. This will allow us to
            # to use inheritance to share common parameter groups in base classes.
            if hasattr(base, "__annotations__"):
                annotations.update(base.__annotations__)

            if hasattr(base, MAPPING_KEY):
                if MAPPING_KEY not in attrs:
                    # This is likely a fail state. If a parent has MAPPING KEY but the child does
                    # not, then we're guaranteed only a subset of the parameters will be mapped.
                    attrs[MAPPING_KEY] = {}
                attrs[MAPPING_KEY].update(getattr(base, MAPPING_KEY))

        all_names = [name for name, annotation in annotations.items() if issubclass(annotation, ParameterBase)]

        if MAPPING_KEY in attrs:
            # If we have a mapping key at all, then we will enter the validation mode for building
            # helpers for mapping and ensuring we have complete mapping.

            # First we'll build a flat list of every dependency for this layer.
            all_deps = set()
            for name in all_names:
                parameter_deps = [
                    name for name, annotation in annotations[name].__annotations__.items()
                    if issubclass(annotation, (torch.Tensor, ParametrizedList))
                ]

                all_deps.update([f"{name}.{dep}" for dep in parameter_deps])

            # Create static helper for doing the string processing only once.
            attrs[PLIST_HELPERS] = []

            # Iterate over all the mappings
            for src_name, target_or_targets in attrs[MAPPING_KEY].items():
                if isinstance(target_or_targets, str):
                    target_or_targets = [target_or_targets]

                actual_targets = []
                for target_name in target_or_targets:
                    base_dependency, dependency_attr = target_name.split(".")

                    # Check for invalid mappings
                    if base_dependency not in all_names:
                        raise ValueError(
                            "Target parameter \"{}\" not found in this layer. Valid targets are {}".format(
                                base_dependency, all_names))
                    if dependency_attr not in annotations[base_dependency].__annotations__:
                        # This check is not universal (see below) if a single dependency is being
                        # mapped to by a single row.
                        raise ValueError(
                            "Target dependency \"{}\" not found on parameter \"{}\". Valid targets are {}".format(
                                dependency_attr, base_dependency, annotations[base_dependency].__annotations__.keys()))
                    if target_name not in all_deps:
                        raise ValueError(
                            "Target dependency \"{}\" was targeted with multiple mapping rules.".format(target_name))

                    # If we've made it this far, the dependency definitely exists.
                    actual_targets.append(annotations[base_dependency].__annotations__[dependency_attr])

                    all_deps.remove(target_name)

                are_plists = [issubclass(target, ParametrizedList) for target in actual_targets]
                if all(are_plists):
                    # We can do direct sets on everything but ParametrizedLists, so we'll only explicitly
                    # handle these here.
                    # TODO(cmikeh2): SPLIT, error if more than 1
                    glob_count = src_name.count("*")
                    if glob_count > 1:
                        raise ValueError(
                            "ParametrizedList index inference can only work with a single glob: {}".format(src_name))
                    elif glob_count == 0:
                        raise ValueError(
                            "Must have wildcard (*) in source name for ParametrizedList mapping: {}".format(src_name))

                    wildcard_idx = src_name.find("*")
                    prefix = src_name[:wildcard_idx]
                    suffix = src_name[wildcard_idx + 1:]
                    attrs[PLIST_HELPERS].append((prefix, suffix, target_or_targets))
                elif any(are_plists):
                    raise ValueError("Cannot mix ParametrizedLists and Tensors in a single mapping rule.")

            if len(all_deps) > 0:
                raise ValueError(
                    "A parameter mapping was provided for {}, but the following dependencies were not mapped: {}".
                    format(clsname, all_deps))

        attrs["finalization_callback"] = make_finalization_callback(all_names)

        new_obj = super().__new__(cls, clsname, bases, attrs)

        setattr(new_obj, "_n_params", len(all_names))
        setattr(new_obj, "_annotation_attrs", all_names)

        return new_obj

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        instance.__init__(*args, **kwargs)

        for name, annotation in instance.__annotations__.items():
            if issubclass(annotation, ParameterBase):
                # TODO(cmikeh2): Do we want to make this a property
                # It might also make sense to do this in the base class __init__
                # but since it is tied with the changes made in __new__ it feels
                # to me like it should be here.
                setattr(instance, name, annotation(instance.inference_model, instance))

        return instance


class LayerContainer(metaclass=LayerMetaclass):
    """
    Abstract base class for containing model parameters.

    This is primarily a guidance abstraction since we do not put any restrictions
    on how the parameters are stored.

    To use this class, annotate the class with `ParameterBase` subclasses and give them
    names. As a checkpoint is loaded into this container, the `ParameterBase` instances
    will be replaced with realized Tensors as soon as each of their dependencies are met.

    To enable automatic mapping, add a static attribute `PARAM_MAPPING` to the class
    definition. This should be a dictionary mapping from a source string to one or
    more dependencies.

    ```python
    class MyLayer(LayerContainer):
        PARAM_MAPPING = {
            "path.to.param.dependency", "container_param_1.dependency",
            "path.to.param2.dependency", "container_param_2.dependency",
            "path.to.param3.*.dependency", "container_param_3.list_dependency"
        }

        ...
    ```
    """

    def __init__(self, model: InferenceModel) -> None:
        """
        Initialization of the LayerContainer. This method does not need to be overridden
        for any children classes.

        Args:
            model (InferenceModel): Inference model that will be used to shard and transform
                parameters correctly, as well as provide specific information about the model
                for `ParameterizedList`s that may be part of one of the member `ParameterBase`s.
        """
        self.inference_model = model
        self._finalized_params = 0

    def _initialization_checker(self, check_device: bool = True) -> bool:
        """
        Returns whether or not all parameters have been initialized and transformed by
        the model. Once this returns True, all the `ParameterBase` instances will be
        torch.Tensors.
        """
        if self._finalized_params != self.n_params:
            return False

        for name in self._annotation_attrs:
            tensor = getattr(self, name)
            if tensor is None:
                continue
            elif not isinstance(tensor, InferenceParameter):
                raise ValueError("Layer should be finalized, but {} ({}) is neither InferenceParameter or None".format(
                    name, type(tensor)))
            elif check_device and tensor.device != torch.device(get_accelerator().current_device()):
                raise RuntimeError("Layer should be finalized, but {} is not on device {}".format(
                    name,
                    get_accelerator().current_device()))
        return True

    @property
    def is_populated(self) -> bool:
        """
        Returns whether or not all parameters have been populated by the checkpoint engine, but
        does not validat the parameters are on the correct device.
        """
        return self._initialization_checker(check_device=False)

    @property
    def is_initialized(self) -> bool:
        """
        Returns whether or not all parameters have been initialized and transformed by
        the model and are located on the appropriate device. Once this returns True, all
        the `ParameterBase` instances ``InferenceParameter``s or explicitly set to ``None``.
        """
        return self._initialization_checker()

    @property
    def n_params(self) -> int:
        """
        The number of parameters this container holds. This is a read-only value
        that is set by the metaclass.
        """
        return self._n_params

    @property
    def annotation_attrs(self) -> list:
        return self._annotation_attrs

    @property
    def mapping_params(self) -> dict:
        return getattr(self.__class__, MAPPING_KEY, {})

    @property
    def plist_helpers(self) -> list:
        return getattr(self.__class__, PLIST_HELPERS, [])

    def direct_injection(self, name: str, tensor: InferenceParameter) -> None:

        if name not in self._annotation_attrs:
            raise ValueError(f"Cannot directly inject {name}, not a valid parameter.")

        setattr(self, name, tensor)
        self._finalized_params += 1

    def set_dependency(self, dep_name: str, dep_value: torch.Tensor) -> None:
        """
        Set dependency can be used for managing dependencies when a mapping is provided
        in the class definition for the layer. The dep_name here should have any prefix
        for transformer layers removed (such as model.layers.*.attn.qkv.weight -> attn.qkv.weight).

        Args:
            dep_name (str): The name of the dependency to set.
            dep_value (torch.Tensor): The value to set the dependency to.
        """

        def get_dep_name_target(dep_name: str) -> str:
            """
            Helper method for getting the target name for a dependency from the
            mapping params. Tries to match exact string first, then looks for
            wildcards and attempts regex matching. Will return empty string if
            no match found.
            """
            if dep_name in self.mapping_params:
                # If we have an exact match, it's a direct mapping and we can
                # immediately set the value.
                return self.mapping_params[dep_name]

            matched_targets = []
            for key, target in self.mapping_params.items():
                regex_key = key.replace("*", ".*")
                if re.match(regex_key, dep_name):
                    matched_targets.append(target)
            if len(matched_targets) > 1:
                raise ValueError(f"Multiple targets matched for dependency {dep_name}: {matched_targets}")
            if matched_targets:
                return matched_targets[0]
            return ""

        if dep_name in self.mapping_params:
            # If we have an exact match, it's a direct mapping and we can immediately set
            # the value.
            target = self.mapping_params[dep_name]

            # Convert single targets to a list for consistency
            if isinstance(target, str):
                target = [target]

            for target_name in target:
                # Double setting doesn't set the attribute correctly, so we do a getattr then setattr
                target_param_name, target_dependency_name = target_name.split(".")
                target_param = getattr(self, target_param_name)
                setattr(target_param, target_dependency_name, dep_value)
            return

        # Otherwise we need to map to one of the parameter lists.
        for prefix, suffix, dests in self.plist_helpers:
            if dep_name.startswith(prefix) and dep_name.endswith(suffix):
                # We have a match, so we can set the value.
                target_idx = int(dep_name[len(prefix):-len(suffix)])

                # Convert single targets to a list for consistency
                if isinstance(dests, str):
                    dests = [dests]

                for dest in dests:
                    target_param_name, target_dependency_name = dest.split(".")
                    target_param = getattr(self, target_param_name)
                    target_dependency = getattr(target_param, target_dependency_name)
                    target_dependency[target_idx] = dep_value
                return

        # TODO: Refactor this with the help of cmikeh2
        # We should be able to combine this with the wildcard matching above.
        target = get_dep_name_target(dep_name)
        if target:
            # Convert single targets to a list for consistency
            if isinstance(target, str):
                target = [target]

            for target_name in target:
                # Double setting doesn't set the attribute correctly, so we do a getattr then setattr
                target_param_name, target_dependency_name = target_name.split(".")
                target_param = getattr(self, target_param_name)
                setattr(target_param, target_dependency_name, dep_value)
            return

        raise ValueError(
            "Could not find a mapping for dependency \"{}\". Check that it is included in the ``MAPPING_PARAMS``. See docstring for more on ``MAPPING_PARAMS``"
            .format(dep_name))
