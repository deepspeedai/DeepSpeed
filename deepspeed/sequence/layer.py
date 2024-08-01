# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator


def post_all2all(transpose, res_shape):

    def post_func(input):
        if transpose:
            input = input.transpose(0, 2).contiguous()
        input = input.reshape(res_shape)
        return input

    return post_func


def single_all_to_all(input, scatter_idx, gather_idx, group, async_op=False, handle=None, type=None):
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    work = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    res_shape=( inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:])
    transpose = True if scatter_idx < 2 else False
    post_all2all_fun = post_all2all(transpose, res_shape)

    if async_op:
        if type in ('dq', 'dk'):
            handle[type + '_work'] = work
            handle[type + '_grad'] = output
            handle[type + '_post_all2all_func'] = post_all2all_fun
            return output.view(res_shape)

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
                group: dist.ProcessGroup,
                input: Tensor,
                scatter_idx: int,
                gather_idx: int,
                stream=None,
                handle=None,
                type=None,
                is_fwd=True) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.stream = stream
        ctx.handle = handle
        ctx.type = type
        if ctx.handle is None:
            res = single_all_to_all(input, scatter_idx, gather_idx, group, False)

        else:
            # overlap communication path
            if not is_fwd and type == 'o':
                assert ctx.stream != None
                res = single_all_to_all(input, scatter_idx, gather_idx, group, False)
                get_accelerator().current_stream().wait_stream(ctx.stream)
                del ctx.stream.activation_buffer_list
                # The computation of d o_weight can overlap with the communication of d o_input

            elif not is_fwd and type in ('q', 'k'):
                # Achieve communication overlap by pipelining the matrix computation and communication of dq, dk, and dv
                type = 'd' + type
                res = single_all_to_all(input, scatter_idx, gather_idx, group, True, handle, type)

            elif is_fwd and type in ('q', 'k'):
                # Achieve communication overlap by pipelining the matrix computation and communication of q, k, and v
                type = 'fwd_' + type
                res = single_all_to_all(input, scatter_idx, gather_idx, group, False, handle, type)

            else:
                res = single_all_to_all(input, scatter_idx, gather_idx, group, False)

        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:

        return (None,
                _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.stream, ctx.handle,
                                   ctx.type, False), None, None, None, None, None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 0,
        sp_stream=None,
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.sp_overlap_comm = False
        self.overlap_handles = None
        self.sp_stream = sp_stream
        if sp_stream is not None:
            self.overlap_handles = {}
            self.sp_overlap_comm = True
            self.dafult_stream = get_accelerator().default_stream()

    def layer_sync(self, layer):
        if self.sp_overlap_comm and hasattr(layer, 'done_event'):
            self.dafult_stream.wait_event(layer.done_event)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        #in shape : e.g.,  [s/p:h:]

        def bwd_hook(layer_type):

            def pre_hook_fun(grad):
                type = 'd' + layer_type
                self.overlap_handles[type + '_work'].wait()
                self.sp_stream.wait_stream(self.dafult_stream)
                all2all_output = self.overlap_handles[type + '_grad']
                grad = list(grad)
                grad[0] = self.overlap_handles[type + '_post_all2all_func'](all2all_output)
                grad = tuple(grad)

            return pre_hook_fun

        self.layer_sync(query)
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx, None,
                                         self.overlap_handles, 'q')
        self.layer_sync(key)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx, None, self.overlap_handles,
                                       'k')
        if self.sp_overlap_comm:
            self.dafult_stream.wait_stream(self.sp_stream)

        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx, None,
                                         self.overlap_handles, 'v')

        if self.sp_overlap_comm:
            # Register a hook to synchronize dq and dk after the all-to-all
            # operation when the gradient data is used.
            # Place this logic after the q, k, v all-to-all operation to
            # improve interpreter speed to
            # call and launch of the forward all-to-all communication.
            grad_fn_q = query.grad_fn.next_functions[0][0]
            grad_fn_q.register_prehook(bwd_hook(layer_type='q'))
            grad_fn_k = key.grad_fn.next_functions[0][0]
            grad_fn_k.register_prehook(bwd_hook(layer_type='k'))

        #out shape : e.g., [s:h/p:]

        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)

        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx, self.sp_stream,
                                    self.overlap_handles, 'o')

        #out e.g., [s/p::h]
        return output
