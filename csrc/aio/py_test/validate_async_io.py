"""
Copyright 2021 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder.builder_names import AsyncIOBuilder
assert get_accelerator().create_op_builder(AsyncIOBuilder).is_compatible()
