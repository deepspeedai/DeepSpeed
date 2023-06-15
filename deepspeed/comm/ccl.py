# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''
Copyright 2021 The Microsoft DeepSpeed Team
'''

import torch
from deepspeed.accelerator import get_accelerator
from .reduce_op import ReduceOp
from .torch import TorchBackend


def build_ccl_op():
    builder = get_accelerator().create_op_builder("CCLCommBuilder")
    if builder == None:
        return None
    ccl_cpp_module = builder.load()
    print(f'DeepSpeed {builder.absolute_name()} built successfully')
    return ccl_cpp_module


class CCLBackend(TorchBackend):

    def __init__(self, name='ccl', rank=-1, world_size=-1, mpu=None, timeout=None, init_method=None):
        self.ccl_comm_op = build_ccl_op()
        if self.ccl_comm_op == None:
            # set CCLBackend to uninitialized state if CCLCommBuilder cannot be loaded
            self.initialized = False
            return
        super(CCLBackend, self).__init__(backend='ccl',
                                         name='torch',
                                         rank=rank,
                                         world_size=world_size,
                                         timeout=timeout,
                                         init_method=init_method)
        self.name = 'ccl'
        size = self.get_world_size()
        rank = self.get_rank()
        main_kvs = self.ccl_comm_op.get_kvs_addr(rank)
        main_kvs = torch.tensor(main_kvs).to(torch.uint8)
        super(CCLBackend, self).broadcast(main_kvs, 0)
        self.ccl_comm_op.initialize(size, rank, main_kvs)
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def broadcast(self, tensor, src, group=None, async_op=False):
        self.ccl_comm_op.broadcast(tensor, src, group, async_op)

    def all_reduce(self, tensor, op=ReduceOp.SUM, group=None, async_op=False):
        use_caching = False
        if use_caching:
            match_id = f"{tensor.size()}-{op}"
            self.ccl_comm_op.all_reduce_caching(tensor, op, match_id, group, async_op)
        else:
            self.ccl_comm_op.all_reduce(tensor, op, group, async_op)

    def barrier(self, group=None, async_op=False):
        self.ccl_comm_op.barrier(group, async_op)
