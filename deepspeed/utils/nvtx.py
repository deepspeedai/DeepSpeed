# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.compiler import is_compiling

enable_nvtx = True
DEEPSPEED_NVTX_DOMAIN = "DeepSpeed"


def _range_push(accelerator, msg):
    if getattr(accelerator, "supports_nvtx_domain", False):
        return accelerator.range_push(msg, domain=DEEPSPEED_NVTX_DOMAIN)
    return accelerator.range_push(msg)


def _range_pop(accelerator):
    if getattr(accelerator, "supports_nvtx_domain", False):
        return accelerator.range_pop(domain=DEEPSPEED_NVTX_DOMAIN)
    return accelerator.range_pop()


def instrument_w_nvtx(func):
    """Decorator that records an NVTX range for the duration of the function call.
       Skips NVTX instrumentation when torch.compile is active to avoid graph breaks.
    """

    def wrapped_fn(*args, **kwargs):
        if enable_nvtx and not is_compiling():
            _range_push(get_accelerator(), func.__qualname__)
        ret_val = func(*args, **kwargs)
        if enable_nvtx and not is_compiling():
            _range_pop(get_accelerator())
        return ret_val

    return wrapped_fn
