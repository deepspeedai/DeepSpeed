# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import contextlib
import functools
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator

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
