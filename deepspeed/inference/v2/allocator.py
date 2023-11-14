# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from functools import reduce
from typing import Iterable

import torch

from deepspeed.accelerator import get_accelerator


def empty_from(tensor: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    shape_size = reduce(lambda x, y: x * y, shape)
    if shape_size == 0:
        raise ValueError("Cannot create empty tensor with size 0")
    return tensor.flatten()[:shape_size].view(shape)


def on_device(method) -> torch.Tensor:
    """
    Wraps a method to ensure the returned tensor is on the current device.
    """

    def wrapped(self, *args, **kwargs):
        tensor = method(self, *args, **kwargs)
        if isinstance(tensor, torch.Tensor):
            return tensor.to(get_accelerator().current_device())
        return tensor

    return wrapped
