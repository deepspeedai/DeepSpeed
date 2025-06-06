# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import numpy
import torch
from dataclasses import dataclass


@dataclass
class serialize_details:
    obj: object
    dtype: torch.dtype
    size: int
    nbytes: int


def tensor_to_bytes(tensor):
    return tensor.numpy().tobytes()


def bytes_to_tensor(buffer):
    return torch.from_numpy(numpy.array(numpy.frombuffer(buffer, dtype=numpy.uint8)))


def required_minimum_torch_version(major_version, minor_version):
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])

    if TORCH_MAJOR < major_version:
        return False

    return TORCH_MAJOR > major_version or TORCH_MINOR >= minor_version


# torch < 1.12
def _legacy_obj_serialization_details(storage_obj):
    nbytes = storage_obj.element_size() * storage_obj.size()
    return serialize_details(obj=storage_obj, dtype=storage_obj.dtype, size=nbytes, nbytes=nbytes)


# torch >= 1.12
def _new_obj_serialization_details(storage_obj):
    obj, dtype = storage_obj
    return serialize_details(obj=obj,
                             dtype=dtype,
                             size=obj.size() // torch._utils._element_size(dtype),
                             nbytes=obj.size())


def obj_serialization_details():
    if required_minimum_torch_version(1, 12):
        return _new_obj_serialization_details

    return _legacy_obj_serialization_details
