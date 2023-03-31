# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.accelerator import get_accelerator


def instrument_w_nvtx(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""

    def wrapped_fn(*args, **kwargs):
        get_accelerator().range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        get_accelerator().range_pop()
        return ret_val

    return wrapped_fn
