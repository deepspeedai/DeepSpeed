# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.compiler import is_compiling

enable_nvtx = True


def instrument_w_nvtx(func):
    """Decorator that records an NVTX range for the duration of the function call.
       Skips NVTX instrumentation when torch.compile is active to avoid graph breaks.
    """

    def wrapped_fn(*args, **kwargs):
        if enable_nvtx and not is_compiling():
            get_accelerator().range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        if enable_nvtx and not is_compiling():
            get_accelerator().range_pop()
        return ret_val

    return wrapped_fn
