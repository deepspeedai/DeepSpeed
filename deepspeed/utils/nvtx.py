# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.accelerator import get_accelerator

enable_nvtx = True


def instrument_w_nvtx(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""

    def wrapped_fn(*args, **kwargs):
        if enable_nvtx:
            get_accelerator().range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        if enable_nvtx:
            get_accelerator().range_pop()
        return ret_val

    return wrapped_fn
