# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging
import sys
from itertools import chain

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


def is_functorch_transforming():
    """True when called under torch.func.grad / vmap / jacrev / etc."""
    # peek_interpreter_stack is private; getattr keeps this working if it moves.
    _functorch = getattr(torch._C, "_functorch", None)
    if _functorch is None:
        return False
    peek = getattr(_functorch, "peek_interpreter_stack", None)
    return peek is not None and peek() is not None


def register_grad_hook(param, hook):
    if required_torch_version(min_version=2.1):
        return param.register_post_accumulate_grad_hook(hook)
    else:
        param_tmp = param.expand_as(param)
        grad_acc = param_tmp.grad_fn.next_functions[0][0]
        return grad_acc.register_hook(hook)


def restore_optimizer_state_dtype(optimizer, state_dict, dtype):
    """Reload the floating-point states of ``state_dict`` as ``dtype`` after ``optimizer.load_state_dict(state_dict)``.

    ``torch.optim.Optimizer.load_state_dict`` casts them to the parameter dtype, which rounds fp32 states of
    low-precision parameters.
    """
    saved_ids = chain.from_iterable(group["params"] for group in state_dict["param_groups"])
    params = chain.from_iterable(group["params"] for group in optimizer.param_groups)
    for saved_id, param in zip(saved_ids, params):
        saved_state = state_dict["state"].get(saved_id, {})
        for key, saved_value in saved_state.items():
            # Like torch.optim, keep "step" as saved.
            if key == "step" or not torch.is_tensor(saved_value) or not saved_value.is_floating_point():
                continue
            if optimizer.state[param][key].dtype != dtype:
                optimizer.state[param][key] = saved_value.to(device=param.device, dtype=dtype)


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
