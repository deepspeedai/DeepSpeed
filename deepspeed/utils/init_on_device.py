# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Callable
from torch import Tensor
from packaging import version as pkg_version


class OnDevice(object):
    """
    Create modules/tensors w. specific devices and dtypes. Examples:

    Create MyModule which consists of many different sub-modules and parameters. In this case we can create
    MyModule as a collection of 'meta' tensors by passing `device='meta'` or we can create the module _directly_
    on a CUDA device by passing `device=f'cuda:{local_rank}'` (where `local_rank` is the local GPU id.

    with OnDevice(dtype=torch.float16, device='meta'):
        model = MyModel()

    with OnDevice(dtype=torch.float16, device=f'cuda:{local_rank}'):
        model = MyModel()

    """

    _orig_torch_empty = torch.empty
    _orig_torch_zeros = torch.zeros
    _orig_torch_ones = torch.ones
    _orig_torch_full = torch.full

    def __init__(self, dtype, device="meta", enabled=True):
        self.dtype = dtype
        self.enabled = enabled
        self.device = device

        if device == "meta":
            if pkg_version.parse('1.10') > pkg_version.parse(torch.__version__):
                raise NotImplementedError("Meta tensor support is not available, please upgrade to torch 1.10+")

    def fp_tensor_constructor(self, fn: Callable, target_fp_dtype: torch.dtype) -> Callable:

        def wrapped_fn(*args, **kwargs) -> Tensor:
            if kwargs.get("device", None) is None:
                kwargs['device'] = self.device
            tensor: Tensor = fn(*args, **kwargs)
            if tensor.is_floating_point():
                tensor = tensor.to(target_fp_dtype)
            return tensor

        return wrapped_fn

    def get_new_tensor_fn_for_dtype(self, dtype: torch.dtype) -> Callable:

        def new_tensor(cls, *args) -> Tensor:
            tensor = OnDevice._orig_torch_empty(0, device=self.device).new_empty(*args)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype)
            return tensor

        return new_tensor

    def __enter__(self):
        if not self.enabled:
            return
        torch.Tensor.__old_new__ = torch.Tensor.__new__
        torch.Tensor.__new__ = self.get_new_tensor_fn_for_dtype(self.dtype)
        torch.empty = self.fp_tensor_constructor(self._orig_torch_empty, self.dtype)
        torch.zeros = self.fp_tensor_constructor(self._orig_torch_zeros, self.dtype)
        torch.ones = self.fp_tensor_constructor(self._orig_torch_ones, self.dtype)
        torch.full = self.fp_tensor_constructor(self._orig_torch_full, self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        torch.Tensor.__new__ = torch.Tensor.__old_new__
        torch.empty = self._orig_torch_empty
        torch.zeros = self._orig_torch_zeros
        torch.ones = self._orig_torch_ones
        torch.full = self._orig_torch_full
