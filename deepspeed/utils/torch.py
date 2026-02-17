# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging

from packaging import version as pkg_version

import torch

logger = logging.getLogger(__name__)
_legacy_fallback_logged = False


def required_torch_version(min_version=None, max_version=None):
    assert min_version or max_version, "Must provide a min_version or max_version argument"

    torch_version = pkg_version.parse(torch.__version__)

    if min_version and pkg_version.parse(str(min_version)) > torch_version:
        return False

    if max_version and pkg_version.parse(str(max_version)) < torch_version:
        return False

    return True


def _log_legacy_grad_hook_fallback_once():
    global _legacy_fallback_logged
    if _legacy_fallback_logged:
        return

    logger.warning(
        "Falling back to param.register_hook for gradient hook registration "
        "because no grad accumulator node is available for this parameter."
    )
    _legacy_fallback_logged = True


def _get_grad_accumulator_for_legacy_hook(param):
    # On older torch versions we rely on traversing the autograd edge created
    # by expand_as to reach the parameter's AccumulateGrad node.
    try:
        param_tmp = param.expand_as(param)
    except Exception:
        return None

    grad_fn = getattr(param_tmp, "grad_fn", None)
    if grad_fn is None:
        return None

    next_functions = getattr(grad_fn, "next_functions", None)
    if not next_functions:
        return None

    return next_functions[0][0]


def _register_param_hook_fallback(param, hook):
    def wrapper(grad):
        hook(param)
        return grad

    return param.register_hook(wrapper)


def register_grad_hook(param, hook):
    if required_torch_version(min_version=2.1):
        return param.register_post_accumulate_grad_hook(hook)

    grad_acc = _get_grad_accumulator_for_legacy_hook(param)
    if grad_acc is None:
        _log_legacy_grad_hook_fallback_once()
        return _register_param_hook_fallback(param, hook)

    return grad_acc.register_hook(hook)


def jit_script_compat(fn):
    if required_torch_version(min_version=2.0) and hasattr(torch, "compile"):
        return torch.compile(fn)
    return torch.jit.script(fn)
