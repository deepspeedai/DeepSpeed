# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex, commit a109f85
"""


class MultiTensorApply(object):

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
