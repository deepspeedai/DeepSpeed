# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import contextlib
import functools
from collections import OrderedDict

import torch.nn as nn
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger

try:
    from torch.compiler import is_compiling as torch_is_compiling
except ImportError:
    try:
        from torch._dynamo.external_utils import is_compiling as torch_is_compiling
    except ImportError:
        # Torch does not have compiler support
        torch_is_compiling = lambda: False

try:
    if required_torch_version(min_version="2.6.0a"):
        from torch._dynamo.compiled_autograd import _enable as compiled_autograd_enable
    else:
        from torch._dynamo.compiled_autograd import enable as compiled_autograd_enable

    _COMPILED_AUTOGRAD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _COMPILED_AUTOGRAD_AVAILABLE = False


def is_compile_supported():
    return required_torch_version(min_version=2.1)


def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func


def enable(min_version=None):
    """
    Decorator factory to enable compiling of a function if the minimum PyTorch version requirement is met.

    Args:
        min_version (str, optional): Minimum PyTorch version required (e.g., "2.7.0").
            If None, the function is always enabled.

    Returns:
        Callable: A decorator that wraps the function.

    Examples:
        @enable("2.7.0")
        def my_function():
            pass

        @enable
        def another_function():
            pass
    """

    def decorator(func):
        if not is_compiling():
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if min_version is None or required_torch_version(min_version=min_version):
                return func(*args, **kwargs)
            return disable(func)(*args, **kwargs)

        return wrapper

    # Called with no arguments
    if callable(min_version):
        func = min_version
        min_version = None
        return decorator(func)

    return decorator


def is_compiling():
    return torch_is_compiling()


@contextlib.contextmanager
def compiled_autograd(enabled: bool, kwargs: dict):
    if not enabled or not _COMPILED_AUTOGRAD_AVAILABLE:
        yield
        return

    if torch_is_compiling():
        yield
        return

    compiler_fn = torch.compile(backend=get_accelerator().get_compile_backend(), **kwargs)

    with compiled_autograd_enable(compiler_fn):
        yield


def dummy_decorator(func):
    return func


# robust version of @torch.compile
def compile():
    if hasattr(torch, "compile"):
        return torch.compile
    else:
        return dummy_decorator


def compile_autoep_non_moe_regions(model: nn.Module, backend, compile_kwargs: dict) -> list[str]:
    """Compile decoder regions around AutoEP layers while keeping AutoEP eager."""
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    named_modules = dict(model.named_modules())
    autoep_modules = [(name, module) for name, module in named_modules.items() if isinstance(module, AutoEPMoELayer)]
    if not autoep_modules:
        raise ValueError("compile_mode='autoep_non_moe' requires at least one AutoEPMoELayer. "
                         "Enable expert_parallel and call compile() after deepspeed.initialize().")

    if compile_kwargs.get("fullgraph", False):
        raise ValueError("compile_mode='autoep_non_moe' requires fullgraph=False because AutoEP is an eager graph "
                         "break.")
    if compile_kwargs.get("dynamic", False):
        raise ValueError("compile_mode='autoep_non_moe' currently requires dynamic=False.")

    resolved_compile_kwargs = {
        "fullgraph": False,
        "dynamic": False,
        **compile_kwargs,
        "backend": backend,
    }
    regions: OrderedDict[str, nn.Module] = OrderedDict()
    for module_name, _ in autoep_modules:
        parent_name, separator, _ = module_name.rpartition(".")
        if not separator:
            raise ValueError("compile_mode='autoep_non_moe' cannot compile an AutoEPMoELayer at the model root.")
        parent = named_modules[parent_name]
        if type(parent).forward is nn.Module.forward:
            raise ValueError(f"AutoEP compile region '{parent_name}' has no forward implementation. "
                             "The MoE layer must be a direct child of a callable decoder block.")
        if getattr(parent, "_compiled_call_impl", None) is not None:
            raise ValueError(f"AutoEP compile region '{parent_name}' is already compiled.")
        regions.setdefault(parent_name, parent)

    original_forwards = {}
    original_compiled_calls = {}
    try:
        for module_name, module in autoep_modules:
            original_forwards[module] = module.__dict__.get("forward")
            module.forward = disable(module.forward)
            logger.debug("AutoEP regional compile: disabled compiler tracing for '%s'.", module_name)

        for region_name, region in regions.items():
            original_compiled_calls[region] = getattr(region, "_compiled_call_impl", None)
            region.compile(**resolved_compile_kwargs)
            logger.info("AutoEP regional compile: compiled '%s' with backend=%s.", region_name, backend)
    except BaseException:
        for module, original_forward in original_forwards.items():
            if original_forward is None:
                del module.forward
            else:
                module.forward = original_forward
        for region, compiled_call in original_compiled_calls.items():
            region._compiled_call_impl = compiled_call
        raise

    return list(regions)
