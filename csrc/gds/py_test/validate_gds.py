# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""
from deepspeed.ops.op_builder import GDSBuilder
assert GDSBuilder().is_compatible(True)
assert GDSBuilder().load(True)
