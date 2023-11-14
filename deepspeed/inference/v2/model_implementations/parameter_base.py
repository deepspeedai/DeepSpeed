# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import weakref
from abc import abstractmethod
from typing import Type

import torch

# Currently have dependency loops for the type hints.
InferenceModel = Type["InferenceModel"]
LayerContainer = Type["LayerContainer"]

MAPPING_KEY = "PARAM_MAPPING"


def make_param_getter(clsname, param):
    """
    Normal getter implementation for a property.
    """

    def param_getter(self):
        return getattr(self, f"__{clsname}__{param}")

    return param_getter


def make_param_setter(clsname, param):
    """
    Setter implementation that will call complete component to potentially
    finalize the parameter.
    """

    def param_setter(self, value):
        setattr(self, f"__{clsname}__{param}", value)
        self.complete_component()

    return param_setter


def make_readonly_setter():
    """
    Setter implementation that will raise an error if called.
    """

    def paramlist_setter(self, value):
        raise ValueError("Cannot set a ParametrizedList directly.")

    return paramlist_setter


class ParameterMetaclass(type):
    """
    MetaClass for the ParameterBase base class. This class will parse the `src_params`
    attribute and create properties for each of the dependencies. A dependency can either
    be represented as a string, which is interpreted as a named Tensor, or a `ParametrizedList`
    subclass.
    """

    def __new__(cls, clsname, bases, attrs):

        annotations = attrs.get("__annotations__", {})
        dependencies = {
            name: annotation
            for name, annotation in annotations.items() if issubclass(annotation, (torch.Tensor, ParametrizedList))
        }
        n_dependencies = len(dependencies)

        # Create properties for each of our dependencies
        for d_name, d_type in dependencies.items():
            if issubclass(d_type, ParametrizedList):
                assert hasattr(
                    d_type, "count_attr"
                ), "ParametrizedList must have a count_attr attribute to access on the inference module."
                attrs[d_name] = property(make_param_getter(clsname, d_name), make_readonly_setter())
            else:  # torch.Tensor
                attrs[d_name] = property(make_param_getter(clsname, d_name), make_param_setter(clsname, d_name))

        new_cls = super().__new__(cls, clsname, bases, attrs)
        new_cls.n_dependencies = n_dependencies

        return new_cls

    def __call__(cls, *args, **kwargs):
        new_obj = super().__call__(*args, **kwargs)
        new_obj.__init__(*args, **kwargs)

        setattr(new_obj, "dest_param", None)

        # Initialize our dependences to None/empty `ParametrizedList`s
        for name, annotation in new_obj.__annotations__.items():
            if issubclass(annotation, ParametrizedList):
                #TODO(jeff): update assert with this, model implementation attribute does not align or missing wrt the ParametrizedList attributes
                assert hasattr(
                    new_obj.inference_model, annotation.count_attr
                ), f"new_obj={new_obj.__class__.__name__}, name={name}, annotation.count_attr={annotation.count_attr}"
                param_list = annotation(new_obj, getattr(new_obj.inference_model, annotation.count_attr))
                setattr(new_obj, f"__{new_obj.__class__.__name__}__{name}", param_list)
            else:  # torch.Tensor
                setattr(new_obj, f"__{new_obj.__class__.__name__}__{name}", None)

        return new_obj


class ParameterBase(metaclass=ParameterMetaclass):
    """
    A ParameterBase allows us to consolidate tracking the dependencies of loading a parameter from
    a checkpoint into a single object. This class should not be used directly, but rather subclassed
    and the `src_params` attribute set to a list of strings and/or `ParametrizedList`s.
    """

    # inference_model: InferenceModel
    """
    Inference model that will provide context on how to shard and transform the parameter.
    """

    #completed_components: int
    """
    How many of the layer dependencies have been met. This is used to determine when the parameter
    is ready to be finalized. A ParametrizedList counts as a single dependency for the purposes
    of this counter.
    """

    def __init__(self, model: InferenceModel, parent_container: LayerContainer) -> None:
        """
        Direct constructor. This should not be called from client code.

        Args:
            model (InferenceModel): Inference model that will be used to shard and transform the
                parameter in `finalize`.
            parent_container (LayerContainer): The parent container that this parameter is a member
                of. We will build a weakref to this container to call the finalization callback.
        """
        self.inference_model = model
        self.completed_components = 0
        self.parent_container = weakref.ref(parent_container)

    @abstractmethod
    def finalize(self) -> torch.Tensor:
        """
        Finalize the parameter after all of its source parameters have been set. This method
        will be automatically called when all inputs have been set. It should return the Tensor
        with all transformations performed on it.
        """
        pass

    def complete_component(self) -> None:
        """
        Mark a component as completed. This should be called by the relevant setter of a direct
        property or a ParametrizedList. This method will automatically call `finalize` when all
        dependencies have been met and then call the finalization callback on the parent container.

        Once the finalization callback has been called, the parameter will be replaced with the
        `dst_param` attribute on the parent container, and this instance will be destroyed.
        """
        self.completed_components += 1

        if self.completed_components != self.n_dependencies:
            return

        finalized_param = self.finalize()
        self.parent_container().finalization_callback(self, finalized_param)


class ParametrizedList:
    """
    A ParametrizedList is a list of parameters that are dependencies
    of a `ParameterBase` but may vary in length depending on the model
    configuration (rather than architecture). For example, a MoE layer
    may have different number of experts depending on the size of the model.

    This class is used to manage these lists and provide integer indexing
    of a single component rather than accessing names directly. For example,
    it tends to be more natural to access the 8th expert with `experts[8]`
    rather than a name like `expert_8`, especially as an attribute.

    To inherit from this class, set static variables `name` and `count_attr`.

    ```python
    class MyParametrizedList(ParametrizedList):
        count_attr: str = "my_list_count"
    ```

    In the above example, `my_list_count` should be an accessible attribute
    of the inference model (i.e. via `self.inference_model.my_list_count`).

    NOTE: There are some APIs in which this type cannot be used as if it is
    just a list of Tensors. For example, `torch.cat(param_list)` will not work.
    However, you can make it compatible with a tuple wrapper:
        `torch.cat(tuple(param_list))`
    """

    n_params: int
    """
    Number of params this list contains.
    """

    param: ParameterBase
    """
    WeakRef to the owning parameter.
    """

    def __init__(self, param: ParameterBase, n_params: int) -> None:
        """
        Constructor. Should not be called from client code.

        Args:
            param (ParameterBase): The owning parameter.
            n_params (int): The number of parameters this list contains. This should be
        """
        self.n_params = n_params
        self.set_params = 0
        self.param = weakref.ref(param)
        self._params = [None] * n_params

    def __getitem__(self, index):
        return self._params[index]

    def __setitem__(self, index, value):
        if self._params[index] is not None:
            raise ValueError("Cannot set a parameter twice.")

        self._params[index] = value
        self.set_params += 1

        if self.set_params != self.n_params:
            return

        self.param().complete_component()

    def __iter__(self):
        return iter(self._params)


def ParamList(attr: str):
    """
    Helper to create a subclass of ParametrizedList with the desired `count_attr`.

    In this manner, we can annotate the type of a Parameter dependency with the
    following:

    ```python
    class CustomParameter(ParameterBase):
        dependency_list: ParamList("dependencies_count_name")
    ```

    where "dependencies_count_name" is the name of the attribute on the inference model.
    """

    class ParametrizedListInstance(ParametrizedList):
        count_attr: str = attr

    return ParametrizedListInstance
