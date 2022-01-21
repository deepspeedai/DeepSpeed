import torch


def instrument_w_nvtx(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""
    if hasattr(torch.cuda.nvtx, "range"):

        def wrapped_fn(*args, **kwargs):
            with torch.cuda.nvtx.range(func.__qualname__):
                return func(*args, **kwargs)

        return wrapped_fn
    else:
        return func
