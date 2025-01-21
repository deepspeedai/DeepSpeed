# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

from einops import rearrange

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import get_shard_size_list, set_num_kv_heads, get_num_kv_heads
from deepspeed.utils import groups


def _generate_layout_params(scatter_idx, batch_dim_idx, seq_world_size, input):
    """
    This function generates the parameters required for `permute` and `reshape` operations,
    which are used to process data before and after `all2all` communication.
    """
    if batch_dim_idx == 0:
        if scatter_idx < 2:
            bs, global_seq_len, num_local_head, head_dim = input.shape
            pre_all2all_inp_shape = [bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]
            pre_all2all_permute_idx = (1, 0, 2, 3, 4)

            post_all2all_permute_idx = (1, 2, 0, 3, 4)
            post_all2all_res_shape = [bs, global_seq_len // seq_world_size, seq_world_size * num_local_head, head_dim]
        else:
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            pre_all2all_inp_shape = [bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]
            pre_all2all_permute_idx = (2, 0, 1, 3, 4)

            post_all2all_permute_idx = (1, 0, 2, 3, 4)
            post_all2all_res_shape = [bs, seq_world_size * local_seq_len, num_total_head // seq_world_size, head_dim]
    else:
        if scatter_idx < 2:
            global_seq_len, bs, num_local_head, head_dim = input.shape
            pre_all2all_inp_shape = [seq_world_size, global_seq_len // seq_world_size, bs, num_local_head, head_dim]
            pre_all2all_permute_idx = None

            post_all2all_permute_idx = (1, 2, 0, 3, 4)
            post_all2all_res_shape = [bs, seq_world_size * global_seq_len, num_local_head // seq_world_size, head_dim]
        else:
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            pre_all2all_inp_shape = [local_seq_len, bs, seq_world_size, num_total_head // seq_world_size, head_dim]
            pre_all2all_permute_idx = (2, 0, 1, 3, 4)
            post_all2all_permute_idx = None
            post_all2all_res_shape = [local_seq_len * seq_world_size, bs, num_total_head // seq_world_size, head_dim]

    return pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape


def post_all2all(permute_idx, res_shape):
    """
    Post-processing function for `all2all` communication.
    """

    def post_func(input):
        if permute_idx is not None:
            input = input.permute(permute_idx).contiguous()
        output = input.reshape(res_shape).contiguous()

        return output

    return post_func


def pre_all2all_fun(permute_idx, inp_shape, input):
    """
    Pre-processing function for `all2all` communication.
    """
    input_t = input.reshape(inp_shape).contiguous()
    if permute_idx is not None:
        input_t = input_t.permute(permute_idx).contiguous()
    return input_t


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs_cos, freqs_sin):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs_cos.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs_cos) + (_rotate_half(t) * freqs_sin)

    res = t if t_pass.shape[-1] == 0 else torch.cat((t, t_pass), dim=-1)
    return res


def uneven_heads_all2all(input, scatter_idx, gather_idx, batch_dim_idx, group):
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    assert batch_dim_idx in [0, 1], "batch_dim_idx must be either 0 or 1"

    if not (scatter_idx < 2):
        input_splits = get_shard_size_list(inp_shape[scatter_idx], seq_world_size)
        input = input.transpose(0, scatter_idx).contiguous()
        local_heads = input_splits[groups._get_sequence_parallel_rank()]
        output_splits = [local_heads] * seq_world_size

        output_buffer_shape = [seq_world_size * local_heads] + list(input.shape[1:])
        output = torch.empty(output_buffer_shape, device=input.device, dtype=input.dtype)
        dist.all_to_all_single(output,input,output_split_sizes=output_splits,\
            input_split_sizes=input_splits,group=group)
        ###[seq_ws*local_heads, ...] to [seq_ws, local_heads, ...]
        output = output.view(seq_world_size, local_heads, *output.shape[1:])
        ###[seq_ws,local_heads,b,seq_len,...] to [seq_ws,seq_len,b,local_heads,...]

        ### batch_dim_idx=0 [seq_ws,local_heads,seq_len,b,...] to [b, seq_ws, seq_len, local_heads ...]
        ### batch_dim_idx=1 [seq_ws,local_heads,b,seq_len,...] to [seq_ws,seq_len,b,local_heads,...]
        if batch_dim_idx == 0:
            order = [3, 0, 2, 1] + list(range(4, len(output.shape)))
            output = output.permute(order).contiguous()
            ###[b, seq_ws*local_seq_len, local_heads,...]
            output = output.view(output.shape[0], inp_shape[gather_idx] * seq_world_size,
                                 *output.shape[3:]).contiguous()
        elif batch_dim_idx == 1:
            output = output.transpose(1, 3).contiguous()
            ###[seq_ws*local_seq_len, b, local_heads,...]
            output = output.view(inp_shape[gather_idx] * seq_world_size, *output.shape[2:]).contiguous()
    else:
        # The compatibility handling of 4D and 3D tensors, standardizing to 3D.
        input = input.reshape(input.shape[0], input.shape[1], -1)

        if batch_dim_idx == 0:  #b,s,h
            input = input.permute(1, 2, 0).contiguous()  #s,h,b
        elif batch_dim_idx == 1:  #s,b,h
            input = input.transpose(1, 2).contiguous()  #s,h,b
        seq_len, h, batch_size = input.shape
        num_local_heads_list = get_shard_size_list(get_num_kv_heads(), seq_world_size)
        local_heads = num_local_heads_list[groups._get_sequence_parallel_rank()]
        h_dim = h // local_heads
        local_seq_len = seq_len // seq_world_size

        input = input.view(seq_len * h, batch_size)
        local_seq_len_with_heads = int(input.shape[0] / seq_world_size)  # dim size of local_seq_len*local_heads*hdim
        input_splits = [local_seq_len_with_heads] * seq_world_size
        coeff = local_seq_len_with_heads // local_heads  #per head: dim size of local_seq_len*hdim

        #uneven seq_world_size coeff,  total_heads/local_heads.
        heads_scale_coeff = get_num_kv_heads() / local_heads

        output_splits = [num_local_heads * coeff for num_local_heads in num_local_heads_list]
        output_buff_d1_size = int(heads_scale_coeff * local_seq_len_with_heads)
        total_h = int(inp_shape[gather_idx] * heads_scale_coeff)
        output = torch.empty(output_buff_d1_size, input.shape[1], device=input.device, dtype=input.dtype)
        dist.all_to_all_single(output,input,output_split_sizes=output_splits, \
            input_split_sizes=input_splits,group=group)
        ##################
        #suppose 7 heads divide into 4 ranks [2,2,2,1]
        #chunk_num_heads_small=floor(7/4)=1
        #chunk_num_heads_large=ceil(7/4)=2
        #num_chunk_heads_large=len([2,2,2])=3, all2all_buffer_counts
        #num_chunk_heads_small=len([1])=1, all2all_buffer_counts
        #total_num_large_heads=sum([2,2,2])=7
        #total_num_small_heads=sum([1])=1

        chunk_num_heads_small = get_num_kv_heads() // seq_world_size  # even heads compatible
        chunk_num_heads_large = chunk_num_heads_small + 1
        num_chunk_heads_large = get_num_kv_heads() % seq_world_size
        num_chunk_heads_small = seq_world_size - num_chunk_heads_large
        total_num_large_heads = num_chunk_heads_large * chunk_num_heads_large
        total_num_small_heads = num_chunk_heads_small * chunk_num_heads_small

        heads_large_combine_size = coeff * total_num_large_heads
        heads_small_combine_size = coeff * total_num_small_heads
        heads_large_chunk, heads_small_chunk = output.split([heads_large_combine_size, heads_small_combine_size],
                                                            dim=0)
        heads_large_chunk = heads_large_chunk.view(num_chunk_heads_large, local_seq_len, chunk_num_heads_large, h_dim,
                                                   batch_size)
        heads_small_chunk = heads_small_chunk.view(num_chunk_heads_small, local_seq_len, chunk_num_heads_small, h_dim,
                                                   batch_size)
        if batch_dim_idx == 0:
            #[all2all_buffer_counts, local_seq_len, n_heads,dim,batch]->[batch,local_seq_len,all2all_buffer_counts*n_heads,dim]
            order = [4, 1, 0, 2, 3]
            heads_large_chunk = heads_large_chunk.permute(order).contiguous().view(batch_size, local_seq_len,
                                                                                   total_num_large_heads, h_dim)
            heads_small_chunk = heads_small_chunk.permute(order).contiguous().view(batch_size, local_seq_len,
                                                                                   total_num_small_heads, h_dim)
        elif batch_dim_idx == 1:
            #[all2all_buffer_counts, local_seq_len, n_heads,dim,batch]->[local_seq_len,batch,all2all_buffer_counts*n_heads,dim]
            order = [1, 4, 0, 2, 3]
            heads_large_chunk = heads_large_chunk.permute(order).contiguous().view(local_seq_len, batch_size,
                                                                                   total_num_large_heads, h_dim)
            heads_small_chunk = heads_small_chunk.permute(order).contiguous().view(local_seq_len, batch_size,
                                                                                   total_num_small_heads, h_dim)

        output = torch.cat([heads_large_chunk, heads_small_chunk], dim=2).contiguous()

        inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
        output_shape=  inp_shape[: gather_idx] + \
            [total_h,] + \
            inp_shape[gather_idx + 1:]

        output = output.view(output_shape)

    return output


def single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, async_op=False, handle=None, type=None):
    seq_world_size = dist.get_world_size(group)
    # we only need num_heads once
    num_heads = input.shape[2]

    if get_num_kv_heads() is not None or (num_heads % seq_world_size != 0 and not scatter_idx < 2):
        # Assuming here that the number of heads for q is consistent with kv
        # If not, additional logic is required for cases like GQA
        if get_num_kv_heads() is None:
            assert num_heads > seq_world_size, f"Number of heads ({num_heads}) must be larger than sequence parallel size ({seq_world_size})"
            # set heads at first call by num_total_heads.
            # then use ``get_num_kv_heads() is not None`` to re-entry uneven path.
            set_num_kv_heads(num_heads)
        assert async_op == False, "uneven head sp does not support async op"
        return uneven_heads_all2all(input, scatter_idx, gather_idx, batch_dim_idx, group)

    pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape = _generate_layout_params(
        scatter_idx, batch_dim_idx, seq_world_size, input)

    input_t = pre_all2all_fun(pre_all2all_permute_idx, pre_all2all_inp_shape, input)

    post_all2all_fun = post_all2all(post_all2all_permute_idx, post_all2all_res_shape)
    output = torch.empty_like(input_t)
    work = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    if async_op:
        if type in ('dq', 'dk'):
            handle[type + '_work'] = work
            handle[type + '_grad'] = output
            handle[type + '_post_all2all_func'] = post_all2all_fun
            return output.view(post_all2all_res_shape)

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
                group: dist.ProcessGroup,
                input: Tensor,
                scatter_idx: int,
                gather_idx: int,
                batch_dim_idx: int,
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
        ctx.batch_dim_idx = batch_dim_idx
        if ctx.handle is None:
            res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)

        else:
            # overlap communication path
            if not is_fwd and type == 'o':
                assert ctx.stream != None
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)
                get_accelerator().current_stream().wait_stream(ctx.stream)
                # The computation of d o_weight can overlap with the communication of d o_input

            elif not is_fwd and type in ('q', 'k'):
                # Achieve communication overlap by pipelining the matrix computation and communication of dq, dk, and dv
                type = 'd' + type
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, True, handle, type)

            elif is_fwd and type in ('q', 'k'):
                # Achieve communication overlap by pipelining the matrix computation and communication of q, k, and v
                type = 'fwd_' + type
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False, handle, type)

            else:
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)

        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:

        return (None,
                _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.batch_dim_idx,
                                   ctx.stream, ctx.handle, ctx.type, False), None, None, None, None, None, None, None)


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

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                batch_dim_idx: int,
                rotary_pos_emb=None,
                *args: Any,
                **kwargs) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            batch_dim_idx (int): indicating which dim is batch
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
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx, batch_dim_idx, None,
                                         self.overlap_handles, 'q')
        self.layer_sync(key)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx, batch_dim_idx, None,
                                       self.overlap_handles, 'k')
        if self.sp_overlap_comm:
            self.dafult_stream.wait_stream(self.sp_stream)

        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx, batch_dim_idx, None,
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
        if rotary_pos_emb is not None:
            pos_emb_cos, pos_emb_sin = rotary_pos_emb[0].permute(1, 0, 2, 3), rotary_pos_emb[1].permute(1, 0, 2, 3)
            query_layer = apply_rotary_pos_emb(query_layer, pos_emb_cos, pos_emb_sin)
            key_layer = apply_rotary_pos_emb(key_layer, pos_emb_cos, pos_emb_sin)

        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)

        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx, batch_dim_idx,
                                    self.sp_stream, self.overlap_handles, 'o')

        #out e.g., [s/p::h]
        return output
