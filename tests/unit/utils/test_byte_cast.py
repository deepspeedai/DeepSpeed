# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.ops.op_builder import UtilsBuilder
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest

if not deepspeed.ops.__compatible_ops__[UtilsBuilder.NAME]:
    pytest.skip(f'Skip tests since {UtilsBuilder.NAME} is not compatible', allow_module_level=True)


def _validate_tensor_cast_properties(typed_tensor, byte_tensor):
    assert byte_tensor.dtype == torch.uint8
    assert byte_tensor.numel() == typed_tensor.numel() * typed_tensor.element_size()
    assert byte_tensor.data_ptr() == typed_tensor.data_ptr()


def _byte_cast_single_tensor(typed_tensor):
    util_ops = UtilsBuilder().load()
    byte_tensor = util_ops.cast_to_byte_tensor(typed_tensor)

    _validate_tensor_cast_properties(typed_tensor=typed_tensor, byte_tensor=byte_tensor)


def _byte_cast_multiple_tensors(typed_tensor_list):
    util_ops = UtilsBuilder().load()
    byte_tensor_list = util_ops.cast_to_byte_tensor(typed_tensor_list)

    assert len(typed_tensor_list) == len(byte_tensor_list)

    for typed_tensor, byte_tensor in zip(typed_tensor_list, byte_tensor_list):
        _validate_tensor_cast_properties(typed_tensor=typed_tensor, byte_tensor=byte_tensor)


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.half, torch.bfloat16, torch.float64, torch.int32, torch.short, torch.int64],
)
class TestCastSingleTensor(DistributedTest):
    world_size = 1

    def test_byte_cast_accelerator_tensor(self, dtype):
        numel = 1024
        typed_tensor = torch.empty(numel, dtype=dtype).to(get_accelerator().device_name())
        _byte_cast_single_tensor(typed_tensor)

    @pytest.mark.parametrize("pinned_memory", [True, False])
    def test_byte_cast_cpu_tensor(self, dtype, pinned_memory):
        numel = 1024
        typed_tensor = torch.empty(numel, dtype=dtype, device='cpu')
        if pinned_memory:
            typed_tensor = typed_tensor.pin_memory()

        _byte_cast_single_tensor(typed_tensor)


@pytest.mark.parametrize('tensor_count', [1, 8, 15])
class TestCastTensorList(DistributedTest):
    world_size = 1

    def test_byte_cast_accelerator_tensor_list(self, tensor_count):
        typed_tensor_list = [torch.empty(1024, dtype=torch.half).to(get_accelerator().device_name())] * tensor_count
        _byte_cast_multiple_tensors(typed_tensor_list)

    def test_byte_cast_cpu_tensor_list(self, tensor_count):
        typed_tensor_list = [torch.empty(1024, dtype=torch.half, device='cpu')] * tensor_count
        _byte_cast_multiple_tensors(typed_tensor_list)
