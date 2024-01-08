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
    if builder is None:
        return None
    ccl_cpp_module = builder.load()
    print(f'DeepSpeed {builder.absolute_name()} built successfully')
    return ccl_cpp_module


class CCLHandler():

    def __init__(self, ccl_comm_op=None):
        self.ccl_comm_op = ccl_comm_op

    def wait(self):
        # backend covered it
        pass


class CCLBackend(TorchBackend):

    def __init__(self, name='ccl', rank=-1, world_size=-1, mpu=None, timeout=None, init_method=None):
        self.ccl_comm_op = build_ccl_op()
        if self.ccl_comm_op is None:
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
        main_kvs = torch.tensor(main_kvs).to(torch.uint8).to(get_accelerator().current_device_name())
        super(CCLBackend, self).broadcast(main_kvs, 0)
        self.ccl_comm_op.initialize(size, rank, main_kvs)
        self.initialized = True
        self.groups = [tuple(range(self.get_world_size()))]
        self.available_coll = self.ccl_comm_op.get_available_coll()

    def is_initialized(self):
        return self.initialized

    def run_collective(self, name, **kwargs):
        if name in self.available_coll:
            if 'group' in kwargs:
                kwargs['group'] = self.get_all_ranks_from_group(kwargs['group'])
            if 'dst' in kwargs:
                kwargs['dst'] = kwargs['group'].index(kwargs['dst'])
            if 'src' in kwargs:
                kwargs['src'] = kwargs['group'].index(kwargs['src'])
            func = "self.ccl_comm_op." + name
            eval(func)(*(kwargs.values()))
            return CCLHandler(self.ccl_comm_op)
        else:
            func = "super(CCLBackend, self)." + name
            eval(func)(*(kwargs.values()))
            return CCLHandler(self.ccl_comm_op)

    def all_reduce(self, tensor, op=ReduceOp.SUM, group=None, async_op=False):
        use_caching = False
        if use_caching:
            match_id = f"{tensor.size()}-{op}"
            name = "all_reduce_caching"
            if name in self.available_coll:
                group = self.get_all_ranks_from_group(group)
                return self.ccl_comm_op.all_reduce_caching(tensor, op, match_id, group, async_op)
            else:
                return self.run_collective(name=name,
                                           tensor=tensor,
                                           op=op,
                                           match_id=match_id,
                                           group=group,
                                           async_op=async_op)
        else:
            name = "all_reduce"
            if name in self.available_coll:
                group = self.get_all_ranks_from_group(group)
                return self.ccl_comm_op.all_reduce(tensor, op, group, async_op)
            else:
                return self.run_collective(name=name, tensor=tensor, op=op, group=group, async_op=async_op)

    def inference_all_reduce(self, tensor, op=ReduceOp.SUM, group=None, async_op=False):
        name = "inference_all_reduce"
        if name in self.available_coll:
            return self.ccl_comm_op.inference_all_reduce(tensor, op, async_op)
        else:
            return self.run_collective(name=name, tensor=tensor, op=op, group=None, async_op=async_op)

    def broadcast(self, tensor, src, group=None, async_op=False):
        return self.run_collective(name="broadcast", tensor=tensor, src=src, group=group, async_op=async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        return self.run_collective(name="all_gather",
                                   tensor_list=tensor_list,
                                   tensor=tensor,
                                   group=group,
                                   async_op=async_op)

    def reduce_scatter_tensor(self, output_tensor, input_tensor, op, group=None, async_op=False):
        return self.run_collective(name="reduce_scatter_tensor",
                                   output_tensor=output_tensor,
                                   input_tensor=input_tensor,
                                   op=op,
                                   group=group)

    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False):
        return self.run_collective(name="all_gather_into_tensor",
                                   output_tensor=output_tensor,
                                   input_tensor=input_tensor,
                                   group=group)

    def all_to_all_single(self, output, input, output_split_sizes, input_split_sizes, group=None, async_op=False):
        return self.run_collective(name="all_to_all_single",
                                   output=output,
                                   input=input,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes,
                                   group=group)

    def send(self, tensor, dst, group=None, tag=0):
        return self.run_collective(name="send", tensor=tensor, dst=dst, group=group, tag=tag)

    def recv(self, tensor, src, group=None, tag=0):
        return self.run_collective(name="recv", tensor=tensor, src=src, group=group, tag=tag)

    def gather(self, tensor, gather_list, dst, group=None, async_op=False):
        return self.run_collective(name="gather", tensor=tensor, gather_list=gather_list, dst=dst, group=group)

    def scatter(self, tensor, gather_list, dst, group=None, async_op=False):
        return self.run_collective(name="scatter", tensor=tensor, gather_list=gather_list, dst=dst, group=group)

    def barrier(self, group=None, async_op=False):
        return self.run_collective(name="barrier", group=group, async_op=async_op)

    def monitored_barrier(self, group=None, timeout=None, wait_all_ranks=False):
        return self.run_collective(name="monitored_barrier", group=group)

    def reduce_scatter(self, output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
        return self.run_collective(name="reduce_scatter",
                                   output=output,
                                   input_list=input_list,
                                   op=op,
                                   group=group,
                                   async_op=async_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
        return self.run_collective(name="reduce", tensor=tensor, dst=dst, op=op, group=group, async_op=async_op)

    def new_group(self, ranks):
        return super(CCLBackend, self).new_group(ranks)

    def _new_group(self, ranks, group):
        size = len(ranks)
        rank = self.get_rank()
        sub_main_kvs = self.ccl_comm_op.get_sub_kvs_addr(rank == ranks[0])
        sub_main_kvs = torch.tensor(sub_main_kvs).to(torch.uint8).to(get_accelerator().current_device_name())
        super(CCLBackend, self).broadcast(sub_main_kvs, ranks[0], group)
        self.ccl_comm_op.initialize_sub_comm(size, ranks.index(rank), sub_main_kvs, ranks)
        self.groups.append(tuple(ranks))

    def get_all_ranks_from_group(self, group):
        if group is None:
            return list(range(self.get_world_size()))
        rank = 0
        results = []
        try:
            while True:
                results.append(super(CCLBackend, self).get_global_rank(group, rank))
                rank += 1
        except (ValueError, RuntimeError):
            pass
        if tuple(results) not in self.groups:
            self._new_group(results, group)
        return results
