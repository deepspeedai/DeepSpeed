# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging
import sys

from packaging import version as pkg_version

import torch

_logger = logging.getLogger(__name__)


def required_torch_version(min_version=None, max_version=None):
    assert min_version or max_version, "Must provide a min_version or max_version argument"

    torch_version = pkg_version.parse(torch.__version__)

    if min_version and pkg_version.parse(str(min_version)) > torch_version:
        return False

    if max_version and pkg_version.parse(str(max_version)) < torch_version:
        return False

    return True


def register_grad_hook(param, hook):
    if required_torch_version(min_version=2.1):
        return param.register_post_accumulate_grad_hook(hook)
    else:
        param_tmp = param.expand_as(param)
        grad_acc = param_tmp.grad_fn.next_functions[0][0]
        return grad_acc.register_hook(hook)


def jit_script_compat(fn):
    fn_name = getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))

    can_try_compile = (required_torch_version(min_version=2.0) and hasattr(torch, "compile")
                       and not (sys.version_info >= (3, 12) and not required_torch_version(min_version=2.4)))

    if can_try_compile:
        try:
            return torch.compile(fn)
        except Exception:
            _logger.debug(
                "torch.compile failed for %s, falling back to torch.jit.script",
                fn_name,
                exc_info=True,
            )

    try:
        return torch.jit.script(fn)
    except Exception:
        _logger.debug(
            "torch.jit.script failed for %s, returning unmodified function",
            fn_name,
            exc_info=True,
        )
        return fn
