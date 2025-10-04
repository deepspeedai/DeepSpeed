# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import List, Tuple, Type, Union, Optional, TYPE_CHECKING

from .logging import logger

if TYPE_CHECKING:
    from deepspeed.runtime.zero.leaf_module_config import DeepSpeedZeroLeafModuleConfig


def z3_leaf_module(model: torch.nn.Module) -> bool:
    """Returns whether a module in `model` has been flagged as a 'leaf' module.
        See `set_z3_leaf_modules` for more details.
        Args:
            model (torch.nn.Module): The model to which the leaf module flag will be applied.
        Returns:
            bool: Whether the module has been flagged as a 'leaf' module.
    """
    return hasattr(model, '_z3_leaf') and model._z3_leaf


def z3_leaf_parameter(model: torch.nn.Parameter) -> bool:
    """Returns whether a parameter belongs to a leaf module.
        See `set_z3_leaf_modules` for more details.
        Args:
            model (torch.nn.Parameter): The parameter to which the leaf module flag will be applied.
        Returns:
            bool: Whether the parameter belongs to a leaf module.
    """
    return hasattr(model, 'ds_z3_leaf_module')


def get_z3_leaf_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Returns a list of modules in `model` that have been flagged as 'leaf' modules.
        See `set_z3_leaf_modules` for more details.
        Args:
            model (torch.nn.Module): The model to which the leaf module flag will be applied.
        Returns:
            List[torch.nn.Module]: A list of modules that have been flagged as 'leaf' modules.
    """
    return [module for module in model.modules() if z3_leaf_module(module)]


def set_z3_leaf_module(model: torch.nn.Module, flag: bool):
    model._z3_leaf = flag


def _fully_qualified_class_name(module: torch.nn.Module) -> str:
    cls = module.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _do_set_z3_leaf_modules(model: torch.nn.Module,
                            leaf_module_classes: Union[List[Type], List[str]],
                            flag: bool,
                            raise_if_not_found: bool = True) -> List[torch.nn.Module]:
    assert all(isinstance(module_class, (type, str)) for module_class in leaf_module_classes), \
        f'leaf_module_classes must be a list of types or names, got {leaf_module_classes}'

    leaf_modules: List[torch.nn.Module] = []

    def _set_z3_leaf_flag(module_instance: torch.nn.Module):
        nonlocal leaf_modules
        for module in leaf_module_classes:
            if isinstance(module, type) and isinstance(module_instance, module):
                module_instance._z3_leaf = flag
                leaf_modules.append(module_instance)
                break

            if isinstance(module, str):
                if (module_instance.__class__.__name__ == module
                        or _fully_qualified_class_name(module_instance) == module):
                    module_instance._z3_leaf = flag
                    leaf_modules.append(module_instance)
                    break

    model.apply(_set_z3_leaf_flag)

    if len(leaf_modules) == 0 and raise_if_not_found:
        raise ValueError(f'No modules of type {leaf_module_classes} found in model {model}')

    return leaf_modules


def set_z3_leaf_modules_by_name(model: torch.nn.Module,
                                module_names: List[str],
                                flag: bool = True,
                                raise_if_not_found: bool = True) -> Tuple[List[torch.nn.Module], List[str]]:
    """Sets a leaf flag for modules referenced by their names in ``model.named_modules()``.
        Args:
            model (torch.nn.Module): The model containing the modules to update.
            module_names (List[str]): Module names as returned by ``named_modules()``.
            flag (bool): Desired flag state.
            raise_if_not_found (bool): Whether to raise when no module matches a provided name.
        Returns:
            Tuple[List[torch.nn.Module], List[str]]: Matched modules and missing module names.
    """
    modules_by_name = dict(model.named_modules())
    leaf_modules: List[torch.nn.Module] = []
    missing: List[str] = []

    for name in module_names:
        module = modules_by_name.get(name)
        if module is None:
            missing.append(name)
            continue
        module._z3_leaf = flag
        leaf_modules.append(module)

    if missing and raise_if_not_found:
        raise ValueError(f'No modules named {missing} found in model {model}')

    return leaf_modules, missing


def set_z3_leaf_modules_by_suffix(model: torch.nn.Module,
                                  module_name_suffixes: List[str],
                                  flag: bool = True,
                                  raise_if_not_found: bool = True) -> Tuple[List[torch.nn.Module], List[str]]:
    """Sets a leaf flag for modules referenced by suffixes of ``model.named_modules()`` names."""
    modules_by_name = dict(model.named_modules())
    leaf_modules: List[torch.nn.Module] = []
    missing: List[str] = []
    seen_ids = set()

    for suffix in module_name_suffixes:
        matched = False
        for name, module in modules_by_name.items():
            if name.endswith(suffix):
                module._z3_leaf = flag
                module_id = id(module)
                if module_id not in seen_ids:
                    seen_ids.add(module_id)
                    leaf_modules.append(module)
                matched = True
        if not matched:
            missing.append(suffix)

    if missing and raise_if_not_found:
        raise ValueError(f'No modules matching suffixes {missing} found in model {model}')

    return leaf_modules, missing


def set_z3_leaf_modules(model: torch.nn.Module,
                        leaf_module_classes: Union[List[Type], List[str]],
                        raise_if_not_found: bool = True) -> List[torch.nn.Module]:
    """Sets a flag within a module in `model` to instruct ZeRO3 to stop setting hooks recursively when it encounters a module class listed in `leaf_module_classes`.
       This is particularly useful in the context of Mixture of Experts (MoE) models. In MoE models, the computation order of experts varies across forward passes. This variability can disrupt ZeRO3's functionality, as ZeRO3 relies on tracking the computation order of modules to prefetch parameters efficiently. By designating a module as a 'leaf' node, ZeRO3 will prefetch parameters for all child modules upon entering the module.
       Another scenario where this functionality is beneficial is in models with excessively fine-grained nested modules, where it helps to avoid the overhead associated with hooks.
        Args:
            model (torch.nn.Module): The model to which the leaf module flag will be applied.
            leaf_module_classes (Union[List[Type], List[str]]): A list of module classes that should be flagged as 'leaf' modules.
            raise_if_not_found (bool): Whether to raise a ``ValueError`` when none of the provided classes
                match a module inside ``model``.
        Returns:
            List[torch.nn.Module]: A list of modules that match the module classes in `leaf_module_classes`.
    """
    return _do_set_z3_leaf_modules(model, leaf_module_classes, True, raise_if_not_found)


def unset_z3_leaf_modules(model: torch.nn.Module,
                          leaf_module_classes: List[Type],
                          raise_if_not_found: bool = True) -> List[torch.nn.Module]:
    """Unsets a flag within a module in `model` to instruct ZeRO3 to resume setting hooks recursively when it encounters a module class listed in `leaf_module_classes`.
        See `set_z3_leaf_modules` for more details.
        Args:
            model (torch.nn.Module): The model to which the leaf module flag will be applied.
            leaf_module_classes (Union[List[Type], List[str]]): A list of module classes that should be flagged as 'leaf' modules.
            raise_if_not_found (bool): Whether to raise a ``ValueError`` when none of the provided classes
                match a module inside ``model``.
        Returns:
            List[torch.nn.Module]: A list of modules that match the module classes in `leaf_module_classes`.
    """
    return _do_set_z3_leaf_modules(model, leaf_module_classes, False, raise_if_not_found)


def apply_zero_leaf_module_config(model: torch.nn.Module,
                                  leaf_cfg: Optional["DeepSpeedZeroLeafModuleConfig"]) -> List[torch.nn.Module]:
    """Apply ZeRO leaf module configuration to ``model``.

    Args:
        model (torch.nn.Module): Root module to update.
        leaf_cfg (DeepSpeedZeroLeafModuleConfig | None): Parsed configuration. If ``None``
            no changes are applied.

    Returns:
        List[torch.nn.Module]: Modules flagged as leaves.
    """
    if leaf_cfg is None:
        return []

    from deepspeed.runtime.zero.leaf_module_config import (
        DEFAULT_LEAF_MODULE_CLASSES,
        DEFAULT_LEAF_MODULE_NAMES,
        DEFAULT_LEAF_MODULE_NAME_SUFFIXES,
    )

    matched_modules: List[torch.nn.Module] = []
    matched_ids = set()

    customized_classes = leaf_cfg.classes != DEFAULT_LEAF_MODULE_CLASSES
    customized_names = leaf_cfg.names != DEFAULT_LEAF_MODULE_NAMES
    customized_suffixes = leaf_cfg.name_suffixes != DEFAULT_LEAF_MODULE_NAME_SUFFIXES

    if leaf_cfg.classes:
        class_matched = set_z3_leaf_modules(model, leaf_cfg.classes, raise_if_not_found=False)
        for module in class_matched:
            module_id = id(module)
            if module_id not in matched_ids:
                matched_ids.add(module_id)
                matched_modules.append(module)

    if leaf_cfg.names:
        name_matched, missing_names = set_z3_leaf_modules_by_name(model,
                                                                  leaf_cfg.names,
                                                                  flag=True,
                                                                  raise_if_not_found=False)
        for module in name_matched:
            module_id = id(module)
            if module_id not in matched_ids:
                matched_ids.add(module_id)
                matched_modules.append(module)

        if missing_names and customized_names:
            logger.warning(f"ZeRO leaf module configuration contains unknown module names: {missing_names}")

    if leaf_cfg.name_suffixes:
        suffix_matched, missing_suffixes = set_z3_leaf_modules_by_suffix(model,
                                                                         leaf_cfg.name_suffixes,
                                                                         flag=True,
                                                                         raise_if_not_found=False)
        for module in suffix_matched:
            module_id = id(module)
            if module_id not in matched_ids:
                matched_ids.add(module_id)
                matched_modules.append(module)

        if missing_suffixes and customized_suffixes:
            logger.warning(f"ZeRO leaf module configuration contains unmatched module suffixes: {missing_suffixes}")

    if not matched_modules and (customized_classes or customized_names or customized_suffixes):
        logger.warning("ZeRO leaf module configuration did not match any modules; hooks will be applied as usual")

    return matched_modules
