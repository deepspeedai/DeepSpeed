# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import List, Type


def _do_set_z3_leaf_modules(model: torch.nn.Module, leaf_module_classes: List[Type], flag: bool) -> None:
    assert all(isinstance(module_class, type) for module_class in leaf_module_classes), \
        f'leaf_module_classes must be a list of types, got {leaf_module_classes}'

    def _set_z3_leaf_flag(model: torch.nn.Module):
        if model.__class__ in leaf_module_classes:
            model._z3_leaf = flag

    model.apply(_set_z3_leaf_flag)


def set_z3_leaf_modules(model: torch.nn.Module, leaf_module_classes: List[Type]) -> None:
    """Sets a flag within a module in `model` to instruct ZeRO3 to stop setting hooks recursively when it encounters a module class listed in `leaf_module_classes`.
       This is particularly useful in the context of Mixture of Experts (MoE) models. In MoE models, the computation order of experts varies across forward passes. This variability can disrupt ZeRO3's functionality, as ZeRO3 relies on tracking the computation order of modules to prefetch parameters efficiently. By designating a module as a 'leaf' node, ZeRO3 will prefetch parameters for all child modules upon entering the module.
       Another scenario where this functionality is beneficial is in models with excessively fine-grained nested modules, where it helps to avoid the overhead associated with hooks.
        Args:
            model (torch.nn.Module): The model to which the leaf module flag will be applied.
            leaf_module_classes (List[Type]): A list of module classes that should be flagged as 'leaf' modules.
    """
    _do_set_z3_leaf_modules(model, leaf_module_classes, True)


def unset_z3_leaf_modules(model: torch.nn.Module, leaf_module_classes: List[Type]) -> None:
    """Unsets a flag within a module in `model` to instruct ZeRO3 to resume setting hooks recursively when it encounters a module class listed in `leaf_module_classes`.
        See `set_z3_leaf_modules` for more details.
        Args:
            model (torch.nn.Module): The model to which the leaf module flag will be applied.
            leaf_module_classes (List[Type]): A list of module classes that should be flagged as 'leaf' modules.
    """
    _do_set_z3_leaf_modules(model, leaf_module_classes, False)


def z3_leaf_module(model: torch.nn.Module) -> bool:
    """Returns whether a module in `model` has been flagged as a 'leaf' module.
        See `set_z3_leaf_modules` for more details.
        Args:
            model (torch.nn.Module): The model to which the leaf module flag will be applied.
    """
    return hasattr(model, '_z3_leaf') and model._z3_leaf
