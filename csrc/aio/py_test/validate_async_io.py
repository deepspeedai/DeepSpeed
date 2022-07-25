"""
Copyright 2021 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""
from deepspeed.ops.aio import AsyncIOBuilder
assert AsyncIOBuilder().is_compatible()
