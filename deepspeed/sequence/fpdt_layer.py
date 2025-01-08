# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Optional, Any, Tuple
from torch import Tensor
from packaging import version
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    flash_attn_version = version.parse(flash_attn.__version__)
except ImportError:
    _flash_attn_forward = None
    _flash_attn_backward = None

from einops import rearrange
from .layer import single_all_to_all, apply_rotary_pos_emb


def _rotate_half_backward(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((x2, -x1), dim=-1)


def apply_rotary_pos_emb_backward(grad_output, freqs_cos, freqs_sin):
    rot_dim = freqs_cos.shape[-1]
    grad, grad_pass = grad_output[..., :rot_dim], grad_output[..., rot_dim:]
    grad_t = (grad * freqs_cos) + (_rotate_half_backward(grad * freqs_sin))
    grad = grad_t if grad_pass.shape[-1] == 0 else torch.cat((grad_t, grad_pass), dim=-1)
    return grad


def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    new_lse = lse + torch.log1p(torch.exp(block_lse - lse))

    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.permute(0, 2, 1).contiguous().unsqueeze(dim=-1).contiguous()
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


class FPDT_InputConstruct(torch.nn.Module):

    def __init__(self, tokens, labels, loss_mask, attention_mask, position_ids, args, sp_size, sp_rank) -> None:

        super(FPDT_InputConstruct, self).__init__()
        self.tokens = tokens
        self.labels = labels
        self.loss_mask = loss_mask
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        global_seq_len = tokens.shape[1]
        batch_size = tokens.shape[0]
        assert global_seq_len % sp_size == 0
        assert global_seq_len % args.ds_sequence_parallel_fpdt_chunk_size == 0
        num_chunk_per_gpu = global_seq_len // args.ds_sequence_parallel_fpdt_chunk_size
        local_seq_len = global_seq_len // sp_size
        assert local_seq_len % num_chunk_per_gpu == 0

        self.num_chunk_per_gpu = num_chunk_per_gpu
        self.chunk_size = local_seq_len // num_chunk_per_gpu
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self.global_seq_len = global_seq_len
        self.local_seq_len = local_seq_len
        self.batch_size = batch_size
        self.device = tokens.device

    def generate(self):
        device = self.device
        totalChunks = self.global_seq_len // self.chunk_size
        token_chunk_idx = torch.arange(self.global_seq_len, device=device, dtype=torch.int) // self.chunk_size
        chunk_to_gpu = torch.arange(totalChunks, device=device, dtype=torch.int)
        chunk_to_gpu = chunk_to_gpu.reshape(self.num_chunk_per_gpu, -1).t().contiguous()

        gather_chunk = chunk_to_gpu.flatten().unsqueeze(1).contiguous()
        mask = gather_chunk == token_chunk_idx

        indices = mask.nonzero(as_tuple=False)
        gather_indices = indices[:, 0]
        token_chunk_indices = indices[:, 1]
        indices = torch.cat([token_chunk_indices[gather_indices == i] for i in range(gather_chunk.shape[0])])
        load_balanced_loss_mask = self.loss_mask[:, indices] if self.loss_mask is not None else self.loss_mask

        indices = indices.reshape(-1, self.chunk_size)[self.num_chunk_per_gpu * self.sp_rank:self.num_chunk_per_gpu *
                                                       (self.sp_rank + 1)].flatten().contiguous()
        load_balanced_tokens = self.tokens[:, indices]
        load_balanced_labels = self.labels[:, indices] if self.labels is not None else self.labels

        load_balanced_attention_mask = self.attention_mask if self.attention_mask is not None else self.attention_mask
        load_balanced_position_ids = self.position_ids[:,
                                                       indices] if self.position_ids is not None else self.position_ids

        return load_balanced_tokens, load_balanced_labels, load_balanced_loss_mask, load_balanced_attention_mask, load_balanced_position_ids


class _FPDTGPUAttentionImpl_(torch.autograd.Function):
    generate_vmap_rule = False

    @staticmethod
    def forward(ctx: Any,
                layernorm_output,
                attention_mask,
                inference_params,
                rotary_pos_emb,
                spg,
                scatter_idx,
                gather_idx,
                hidden_size,
                projection_size,
                hidden_size_per_attention_head,
                kv_projection_size,
                qkv_linear_weight,
                qkv_linear_bias,
                dropout,
                num_chunks=8,
                cpu_offloading=True):

        do_save = layernorm_output.requires_grad

        if rotary_pos_emb is not None:
            pos_emb_cos, pos_emb_sin = rotary_pos_emb[0].permute(1, 0, 2, 3), rotary_pos_emb[1].permute(1, 0, 2, 3)
            ctx.pos_emb_cos = pos_emb_cos
            ctx.pos_emb_sin = pos_emb_sin
        else:
            ctx.pos_emb_cos = None
            ctx.pos_emb_sin = None

        with torch.no_grad():
            per_gpu_seq_len = layernorm_output.shape[0]
            chunk_size = per_gpu_seq_len // num_chunks
            assert chunk_size * num_chunks == per_gpu_seq_len
            assert attention_mask is None
            ctx.num_chunks = num_chunks
            ctx.cpu_offloading = cpu_offloading
            ctx.spg = spg
            ctx.scatter_idx = scatter_idx
            ctx.gather_idx = gather_idx

            device = get_accelerator().current_device_name()
            ctx.device = device
            ctx.dtype = layernorm_output.dtype
            ctx.projection_size = projection_size
            ctx.kv_projection_size = kv_projection_size

            global_q = []
            global_k = []
            global_v = []

            ctx.softmax_scale = hidden_size_per_attention_head**(-0.5)

            ctx.dropout_p = dropout
            ctx.window_size = (-1, -1)
            ctx.alibi_slopes = None

            batch_size = layernorm_output.shape[1]

            global_o = [None for _ in range(num_chunks)]
            global_lse = [None for _ in range(num_chunks)]

            for i in range(num_chunks):

                st = chunk_size * i
                ed = st + chunk_size

                qkv_chunk = torch.matmul(layernorm_output[st:ed], qkv_linear_weight.t()) + qkv_linear_bias

                q_chunk = qkv_chunk[:, :, :projection_size].contiguous().reshape(
                    qkv_chunk.shape[0], qkv_chunk.shape[1], -1,
                    hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()  # b, l, nh, hd
                q_chunk = single_all_to_all(q_chunk, scatter_idx, gather_idx, 0, spg)
                global_q_chunk_len = q_chunk.shape[1]
                if rotary_pos_emb is not None:
                    q_chunk = apply_rotary_pos_emb(q_chunk,
                                                   pos_emb_cos[:, global_q_chunk_len * i:global_q_chunk_len * (i + 1)],
                                                   pos_emb_sin[:, global_q_chunk_len * i:global_q_chunk_len * (i + 1)])
                global_q.append(q_chunk)

                k_chunk = qkv_chunk[:, :, projection_size:projection_size + kv_projection_size].contiguous().reshape(
                    qkv_chunk.shape[0], qkv_chunk.shape[1], -1,
                    hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()  # b, l, nh, hd
                k_chunk = single_all_to_all(k_chunk, scatter_idx, gather_idx, 0, spg)
                if rotary_pos_emb is not None:
                    k_chunk = apply_rotary_pos_emb(k_chunk,
                                                   pos_emb_cos[:, global_q_chunk_len * i:global_q_chunk_len * (i + 1)],
                                                   pos_emb_sin[:, global_q_chunk_len * i:global_q_chunk_len * (i + 1)])
                global_k.append(k_chunk)

                v_chunk = qkv_chunk[:, :, projection_size + kv_projection_size:].contiguous().reshape(
                    qkv_chunk.shape[0], qkv_chunk.shape[1], -1,
                    hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()  # b, l, nh, hd
                v_chunk = single_all_to_all(v_chunk, scatter_idx, gather_idx, 0, spg)
                global_v.append(v_chunk)

                for k_i in range(len(global_k)):
                    causal_chunk = i == k_i
                    if flash_attn_version >= version.parse("2.6.0"):
                        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(global_q[i],
                                                                                     global_k[k_i],
                                                                                     global_v[k_i],
                                                                                     ctx.dropout_p,
                                                                                     ctx.softmax_scale,
                                                                                     causal=causal_chunk,
                                                                                     window_size=ctx.window_size,
                                                                                     softcap=0.0,
                                                                                     alibi_slopes=ctx.alibi_slopes,
                                                                                     return_softmax=False)
                    else:
                        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(global_q[i],
                                                                                     global_k[k_i],
                                                                                     global_v[k_i],
                                                                                     ctx.dropout_p,
                                                                                     ctx.softmax_scale,
                                                                                     causal=causal_chunk,
                                                                                     window_size=ctx.window_size,
                                                                                     alibi_slopes=ctx.alibi_slopes,
                                                                                     return_softmax=False)

                    global_o[i], global_lse[i] = update_out_and_lse(global_o[i], global_lse[i], block_out, block_lse)

                global_o[i] = global_o[i].to(q_chunk.dtype)

            output = [None for i in range(num_chunks)]

            for i in range(num_chunks):
                global_lse[i] = global_lse[i][:, :, :, 0].permute(0, 2, 1).contiguous()
                output[i] = single_all_to_all(global_o[i].to(ctx.dtype).contiguous(), gather_idx, scatter_idx, 0, spg)
            output = torch.cat(output, dim=1)

            head_dim = output.shape[-1]

            if do_save:
                ctx.save_for_backward(layernorm_output)
                ctx.global_q = global_q
                ctx.global_k = global_k
                ctx.global_v = global_v
                ctx.attn_output = global_o
                ctx.attn_lse = global_lse
                ctx.head_dim = head_dim
                ctx.batch_size = batch_size

                ctx.qkv_linear_weight = qkv_linear_weight
                ctx.qkv_linear_bias = qkv_linear_bias

        return output

    @staticmethod
    def backward(ctx, grad_output):

        num_chunks = ctx.num_chunks
        device = ctx.device
        dtype = ctx.dtype
        spg = ctx.spg
        scatter_idx = ctx.scatter_idx
        gather_idx = ctx.gather_idx
        softmax_scale = ctx.softmax_scale
        dropout_p = ctx.dropout_p
        window_size = ctx.window_size
        alibi_slopes = ctx.alibi_slopes

        projection_size = ctx.projection_size
        kv_projection_size = ctx.kv_projection_size

        layernorm_output = ctx.saved_tensors[0]

        global_q = ctx.global_q
        global_k = ctx.global_k
        global_v = ctx.global_v
        attn_output = ctx.attn_output
        lse = ctx.attn_lse

        qkv_linear_weight = ctx.qkv_linear_weight
        qkv_linear_bias = ctx.qkv_linear_bias

        input_chunk_size = layernorm_output.shape[0] // num_chunks
        grad_layernorm_output = [
            torch.zeros((input_chunk_size, layernorm_output.shape[1], layernorm_output.shape[2]),
                        device=device,
                        dtype=dtype) for _ in range(num_chunks)
        ]

        grad_global_attn_output = []
        chunk_size = grad_output.shape[1] // num_chunks

        for i in range(num_chunks):
            st = chunk_size * i
            ed = st + chunk_size
            grad_global_attn_output.append(
                single_all_to_all(grad_output[:, st:ed].contiguous(), scatter_idx, gather_idx, 0, spg))

        del grad_output

        dq = [torch.zeros(global_q[0].shape, dtype=torch.float, device=device) for _ in range(num_chunks)]
        dk = [torch.zeros(global_q[0].shape, dtype=torch.float, device=device) for _ in range(num_chunks)]
        dv = [torch.zeros(global_q[0].shape, dtype=torch.float, device=device) for _ in range(num_chunks)]

        grad_qkv_linear_weight = torch.zeros(qkv_linear_weight.shape,
                                             device=qkv_linear_weight.device,
                                             dtype=torch.float)
        grad_qkv_linear_bias = torch.zeros(qkv_linear_bias.shape, device=qkv_linear_weight.device, dtype=torch.float)

        for i in range(num_chunks):
            k_chunk = global_k[i]
            v_chunk = global_v[i]

            for q_i in range(num_chunks):
                no_computation = q_i < i
                if no_computation:
                    continue

                causal_chunk = q_i == i

                q_chunk = global_q[q_i]
                attn_output_chunk = attn_output[q_i]
                lse_chunk = lse[q_i]
                d_out = grad_global_attn_output[q_i]

                dq_this = torch.zeros(global_q[0].shape, dtype=dtype, device=device)
                dk_this = torch.zeros(global_k[0].shape, dtype=dtype, device=device)
                dv_this = torch.zeros(global_v[0].shape, dtype=dtype, device=device)

                if flash_attn_version >= version.parse("2.6.0"):
                    _flash_attn_backward(d_out,
                                         q_chunk,
                                         k_chunk,
                                         v_chunk,
                                         attn_output_chunk,
                                         lse_chunk,
                                         dq_this,
                                         dk_this,
                                         dv_this,
                                         dropout_p,
                                         softmax_scale,
                                         causal_chunk,
                                         window_size,
                                         softcap=0.0,
                                         alibi_slopes=alibi_slopes,
                                         deterministic=False,
                                         rng_state=None)
                else:
                    _flash_attn_backward(d_out,
                                         q_chunk,
                                         k_chunk,
                                         v_chunk,
                                         attn_output_chunk,
                                         lse_chunk,
                                         dq_this,
                                         dk_this,
                                         dv_this,
                                         dropout_p,
                                         softmax_scale,
                                         causal_chunk,
                                         window_size,
                                         alibi_slopes=alibi_slopes,
                                         deterministic=False,
                                         rng_state=None)

                dq[q_i].add_(dq_this.to(torch.float))
                dk[i].add_(dk_this.to(torch.float))
                dv[i].add_(dv_this.to(torch.float))

            dk_seq_len = dk[i].shape[1]

            if ctx.pos_emb_cos is not None:
                dk[i] = apply_rotary_pos_emb_backward(dk[i].to(dtype),
                                                      ctx.pos_emb_cos[:, dk_seq_len * i:dk_seq_len * (i + 1)],
                                                      ctx.pos_emb_sin[:, dk_seq_len * i:dk_seq_len * (i + 1)])
            else:
                dk[i] = dk[i].to(dtype)
            dv[i] = dv[i].to(dtype)
            dk[i] = single_all_to_all(dk[i].contiguous(), gather_idx, scatter_idx, 0, spg)
            dv[i] = single_all_to_all(dv[i].contiguous(), gather_idx, scatter_idx, 0, spg)

            input_st = i * input_chunk_size
            input_ed = input_st + input_chunk_size

            input_chunk = layernorm_output[input_st:input_ed].reshape(-1, layernorm_output.shape[-1])

            dk[i] = dk[i].flatten(2).permute(1, 0, 2)
            dv[i] = dv[i].flatten(2).permute(1, 0, 2)
            l, b = dk[i].shape[0], dk[i].shape[1]
            grad_qkv_linear_weight[projection_size:projection_size + kv_projection_size].add_(
                torch.matmul(dk[i].reshape(l * b, -1).t(), input_chunk))
            grad_qkv_linear_weight[projection_size + kv_projection_size:].add_(
                torch.matmul(dv[i].reshape(l * b, -1).t(), input_chunk))
            grad_qkv_linear_bias[projection_size:projection_size + kv_projection_size].add_(dk[i].sum(0).sum(0))
            grad_qkv_linear_bias[projection_size + kv_projection_size:].add_(dv[i].sum(0).sum(0))

            grad_layernorm_output[i].add_(
                torch.matmul(dk[i], qkv_linear_weight[projection_size:projection_size + kv_projection_size]))
            grad_layernorm_output[i].add_(torch.matmul(dv[i],
                                                       qkv_linear_weight[projection_size + kv_projection_size:]))

            dk[i] = None
            dv[i] = None

        for i in range(num_chunks):
            dq_seq_len = dq[i].shape[1]
            if ctx.pos_emb_cos is not None:
                dq[i] = apply_rotary_pos_emb_backward(dq[i].to(dtype),
                                                      ctx.pos_emb_cos[:, dq_seq_len * i:dq_seq_len * (i + 1)],
                                                      ctx.pos_emb_sin[:, dq_seq_len * i:dq_seq_len * (i + 1)])
            else:
                dq[i] = dq[i].to(dtype)
            dq[i] = single_all_to_all(dq[i].to(dtype).contiguous(), gather_idx, scatter_idx, 0, spg)

            input_chunk = layernorm_output[:input_chunk_size].reshape(-1, layernorm_output.shape[-1])
            layernorm_output = layernorm_output[input_chunk_size:]

            dq[i] = dq[i].flatten(2).permute(1, 0, 2)
            l, b = dq[i].shape[0], dq[i].shape[1]
            grad_qkv_linear_weight[:projection_size].add_(torch.matmul(dq[i].reshape(l * b, -1).t(), input_chunk))
            grad_qkv_linear_bias[:projection_size].add_(dq[i].sum(0).sum(0))

            grad_layernorm_output[i].add_(torch.matmul(dq[i], qkv_linear_weight[:projection_size]))

            dq[i] = None

        return torch.cat(
            grad_layernorm_output,
            dim=0).to(dtype), None, None, None, None, None, None, None, None, None, None, grad_qkv_linear_weight.to(
                dtype), grad_qkv_linear_bias.to(dtype), None, None, None


class SequenceChunk:

    def __init__(self, chunk: torch.Tensor, device=None, is_in_use=False):

        self.chunk_shape = chunk.shape
        self.chunk_dtype = chunk.dtype
        self.device = chunk.device if device is None else device

        cpu_chunk = torch.empty(chunk.shape, dtype=chunk.dtype, device='cpu', pin_memory=True)

        if get_accelerator().on_accelerator(chunk):
            cpu_chunk.copy_(chunk, non_blocking=True)
        else:
            cpu_chunk = chunk

        self.cpu_chunk = cpu_chunk

        self.gpu_chunk = chunk if is_in_use else None

    def load_to_gpu(self):
        assert self.gpu_chunk is None
        if self.gpu_chunk is not None:
            pass
        else:
            gpu_chunk = torch.empty(self.chunk_shape, device=self.device, dtype=self.chunk_dtype)
            gpu_chunk.copy_(self.cpu_chunk, non_blocking=True)
            self.gpu_chunk = gpu_chunk

    def get_gpu_chunk(self):
        assert self.gpu_chunk is not None and self.gpu_chunk.device == self.device
        return self.gpu_chunk

    def check_gpu_chunk(self, ):
        assert (self.gpu_chunk is not None) and (
            self.gpu_chunk.device == self.device
        ), f"gpu_chunk {self.gpu_chunk is not None} shound be on {self.device}, but it is now on {self.gpu_chunk.device}"
        return True

    def offload(self):
        assert self.gpu_chunk is not None and self.gpu_chunk.device == self.device
        del self.gpu_chunk
        self.gpu_chunk = None

    def overwrite_to_cpu(self):
        assert self.gpu_chunk is not None and self.gpu_chunk.device == self.device
        self.cpu_chunk.copy_(self.gpu_chunk, non_blocking=True)


class _FPDTGPUOffloadingAttentionImpl_(torch.autograd.Function):
    generate_vmap_rule = False

    @staticmethod
    def forward(ctx: Any,
                layernorm_output,
                attention_mask,
                inference_params,
                rotary_pos_emb,
                spg,
                scatter_idx,
                gather_idx,
                hidden_size,
                projection_size,
                hidden_size_per_attention_head,
                kv_projection_size,
                qkv_linear_weight,
                qkv_linear_bias,
                dropout,
                num_chunks=8,
                cpu_offloading=True):

        do_save = layernorm_output.requires_grad

        if rotary_pos_emb is not None:
            pos_emb_cos, pos_emb_sin = rotary_pos_emb[0].permute(1, 0, 2, 3), rotary_pos_emb[1].permute(1, 0, 2, 3)
            ctx.pos_emb_cos = pos_emb_cos
            ctx.pos_emb_sin = pos_emb_sin
        else:
            ctx.pos_emb_cos = None
            ctx.pos_emb_sin = None
        with torch.no_grad():
            per_gpu_seq_len = layernorm_output.shape[0]
            chunk_size = per_gpu_seq_len // num_chunks
            assert chunk_size * num_chunks == per_gpu_seq_len
            assert attention_mask is None
            ctx.num_chunks = num_chunks
            ctx.cpu_offloading = cpu_offloading
            ctx.spg = spg
            ctx.scatter_idx = scatter_idx
            ctx.gather_idx = gather_idx

            ctx.chunk_size = chunk_size
            device = get_accelerator().current_device_name()
            ctx.device = device
            ctx.dtype = layernorm_output.dtype
            ctx.projection_size = projection_size
            ctx.kv_projection_size = kv_projection_size

            global_q = []
            global_k = []
            global_v = []

            ctx.softmax_scale = hidden_size_per_attention_head**(-0.5)

            ctx.dropout_p = dropout
            ctx.window_size = (-1, -1)
            ctx.alibi_slopes = None

            batch_size = layernorm_output.shape[1]

            global_o = []
            global_lse = []

            layernorm_output_cpu = []
            final_output = []

            offload_stream = get_accelerator().Stream()
            general_offload_stream = get_accelerator().Stream()
            compute_stream = get_accelerator().default_stream()

            q_compute_chunk_idx = 0
            kv_compute_chunk_idx = 0
            for i in range(num_chunks):

                qkv_chunk = torch.matmul(layernorm_output[:chunk_size],
                                         qkv_linear_weight.t()) + qkv_linear_bias  # torch.Size([18126, 1, 12288])

                with get_accelerator().stream(general_offload_stream):
                    layernorm_output_cpu.append(SequenceChunk(layernorm_output[:chunk_size]))

                layernorm_output = layernorm_output[chunk_size:]

                q_chunk = qkv_chunk[:, :, :projection_size].contiguous().reshape(
                    qkv_chunk.shape[0], qkv_chunk.shape[1], -1,
                    hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()  # b, l, nh, hd
                q_chunk = single_all_to_all(q_chunk, scatter_idx, gather_idx, 0, spg)
                global_q_chunk_len = q_chunk.shape[1]

                k_chunk = qkv_chunk[:, :, projection_size:projection_size + kv_projection_size].contiguous().reshape(
                    qkv_chunk.shape[0], qkv_chunk.shape[1], -1,
                    hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()  # b, l, nh, hd
                k_chunk = single_all_to_all(k_chunk, scatter_idx, gather_idx, 0, spg)

                v_chunk = qkv_chunk[:, :, projection_size + kv_projection_size:].contiguous().reshape(
                    qkv_chunk.shape[0], qkv_chunk.shape[1], -1,
                    hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()  # b, l, nh, hd
                v_chunk = single_all_to_all(v_chunk, scatter_idx, gather_idx, 0, spg)

                dist.barrier()

                if ctx.pos_emb_cos is not None:
                    pos_emb_cos_chunk = pos_emb_cos[:, global_q_chunk_len * i:global_q_chunk_len * (i + 1)]
                    pos_emb_sin_chunk = pos_emb_sin[:, global_q_chunk_len * i:global_q_chunk_len * (i + 1)]

                    q_chunk = apply_rotary_pos_emb(q_chunk, pos_emb_cos_chunk, pos_emb_sin_chunk)
                    k_chunk = apply_rotary_pos_emb(k_chunk, pos_emb_cos_chunk, pos_emb_sin_chunk)

                compute_stream.wait_stream(offload_stream)
                compute_stream.synchronize()
                with get_accelerator().stream(offload_stream):
                    global_q.append(SequenceChunk(q_chunk, is_in_use=True))
                    global_k.append(SequenceChunk(k_chunk, is_in_use=True))
                    global_v.append(SequenceChunk(v_chunk, is_in_use=True))

                del qkv_chunk

                cur_attn_output = None
                cur_attn_lse = None
                for k_i in range(len(global_k)):
                    causal_chunk = i == k_i
                    with get_accelerator().stream(compute_stream):
                        if flash_attn_version >= version.parse("2.6.0"):
                            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                                global_q[q_compute_chunk_idx].get_gpu_chunk(),
                                global_k[kv_compute_chunk_idx].get_gpu_chunk(),
                                global_v[kv_compute_chunk_idx].get_gpu_chunk(),
                                ctx.dropout_p,
                                ctx.softmax_scale,
                                causal=causal_chunk,
                                window_size=ctx.window_size,
                                softcap=0.0,
                                alibi_slopes=ctx.alibi_slopes,
                                return_softmax=False)
                        else:
                            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                                global_q[q_compute_chunk_idx].get_gpu_chunk(),
                                global_k[kv_compute_chunk_idx].get_gpu_chunk(),
                                global_v[kv_compute_chunk_idx].get_gpu_chunk(),
                                ctx.dropout_p,
                                ctx.softmax_scale,
                                causal=causal_chunk,
                                window_size=ctx.window_size,
                                alibi_slopes=ctx.alibi_slopes,
                                return_softmax=False)
                        cur_attn_output, cur_attn_lse = update_out_and_lse(cur_attn_output, cur_attn_lse, block_out,
                                                                           block_lse)

                    can_offload_kv = True
                    if k_i != (len(global_k) - 1) or i != (num_chunks - 1):
                        if k_i != (len(global_k) - 1):
                            next_kv_compute_chunk_idx = k_i + 1
                        else:
                            next_kv_compute_chunk_idx = 0

                        if next_kv_compute_chunk_idx == kv_compute_chunk_idx:
                            can_offload_kv = False
                        else:
                            if next_kv_compute_chunk_idx != (len(global_k) - 1):
                                with get_accelerator().stream(offload_stream):
                                    global_k[next_kv_compute_chunk_idx].load_to_gpu()
                                    global_v[next_kv_compute_chunk_idx].load_to_gpu()

                    if i == num_chunks - 1 and k_i == num_chunks - 1:
                        with get_accelerator().stream(offload_stream):
                            global_q[0].load_to_gpu()
                            global_k[0].load_to_gpu()
                            global_v[0].load_to_gpu()
                            global_o[0].load_to_gpu()
                            global_lse[0].load_to_gpu()

                    compute_stream.wait_stream(offload_stream)
                    compute_stream.synchronize()

                    if can_offload_kv:
                        global_k[kv_compute_chunk_idx].offload()
                        global_v[kv_compute_chunk_idx].offload()
                    kv_compute_chunk_idx = next_kv_compute_chunk_idx

                global_q[q_compute_chunk_idx].offload()
                q_compute_chunk_idx += 1

                all2all_output = single_all_to_all(
                    cur_attn_output.to(ctx.dtype).contiguous(), gather_idx, scatter_idx, 0, spg)
                final_output.append(all2all_output)
                with get_accelerator().stream(general_offload_stream):
                    global_o.append(SequenceChunk(cur_attn_output.to(ctx.dtype)))
                    global_lse.append(SequenceChunk(cur_attn_lse[:, :, :, 0].permute(0, 2, 1).contiguous()))

            compute_stream.wait_stream(general_offload_stream)
            compute_stream.synchronize()

            final_output = torch.cat(final_output, dim=1)

            head_dim = final_output.shape[-1]

        if do_save:
            ctx.layernorm_output = layernorm_output_cpu
            ctx.global_q = global_q
            ctx.global_k = global_k
            ctx.global_v = global_v
            ctx.attn_output = global_o
            ctx.attn_lse = global_lse
            ctx.head_dim = head_dim
            ctx.batch_size = batch_size

            ctx.qkv_linear_weight = qkv_linear_weight
            ctx.qkv_linear_bias = qkv_linear_bias

        return final_output

    @staticmethod
    def backward(ctx, grad_output):
        num_chunks = ctx.num_chunks
        device = grad_output.device
        dtype = ctx.dtype
        spg = ctx.spg
        scatter_idx = ctx.scatter_idx
        gather_idx = ctx.gather_idx
        softmax_scale = ctx.softmax_scale
        dropout_p = ctx.dropout_p
        window_size = ctx.window_size
        alibi_slopes = ctx.alibi_slopes

        projection_size = ctx.projection_size
        kv_projection_size = ctx.kv_projection_size

        layernorm_output = ctx.layernorm_output

        global_q = ctx.global_q
        global_k = ctx.global_k
        global_v = ctx.global_v
        attn_output = ctx.attn_output
        lse = ctx.attn_lse

        qkv_linear_weight = ctx.qkv_linear_weight
        qkv_linear_bias = ctx.qkv_linear_bias

        offload_stream = get_accelerator().Stream()
        general_offload_stream = get_accelerator().Stream()
        compute_stream = get_accelerator().default_stream()

        chunk_size = grad_output.shape[1] // num_chunks
        assert chunk_size == layernorm_output[0].cpu_chunk.shape[0]

        grad_layernorm_output = [
            torch.zeros(layernorm_output[0].chunk_shape, device=device, dtype=dtype) for _ in range(num_chunks)
        ]

        grad_global_attn_output = [None for _ in range(num_chunks)]

        q_compute_chunk_idx = 0
        kv_compute_chunk_idx = 0
        last_q_accum_idx = 0

        with get_accelerator().stream(general_offload_stream):
            layernorm_output[0].load_to_gpu()
            grad_qkv_linear_weight = torch.zeros(qkv_linear_weight.shape,
                                                 device=qkv_linear_weight.device,
                                                 dtype=torch.float)
            grad_qkv_linear_bias = torch.zeros(qkv_linear_bias.shape,
                                               device=qkv_linear_weight.device,
                                               dtype=torch.float)

        grad_global_attn_output_chunk = single_all_to_all(grad_output[:, :chunk_size].contiguous(), scatter_idx,
                                                          gather_idx, 0, spg)
        get_accelerator().synchronize()
        grad_output = grad_output[:, chunk_size:]

        with get_accelerator().stream(offload_stream):
            grad_global_attn_output[0] = SequenceChunk(grad_global_attn_output_chunk, is_in_use=True)
            dq = [
                SequenceChunk(torch.zeros(global_q[0].chunk_shape, dtype=torch.float, device=device), is_in_use=True)
            ] + [
                SequenceChunk(torch.zeros(global_q[0].chunk_shape, dtype=torch.float, device='cpu', pin_memory=True),
                              device) for _ in range(num_chunks - 1)
            ]
            dk_accum = torch.zeros(global_k[0].chunk_shape, dtype=torch.float, device=device)
            dv_accum = torch.zeros(global_v[0].chunk_shape, dtype=torch.float, device=device)

        for i in range(num_chunks):
            for q_i in range(num_chunks):
                no_computation = q_i < i
                if no_computation:
                    continue

                causal_chunk = q_i == i

                dq_this = torch.zeros(global_q[0].chunk_shape, dtype=dtype, device=device)
                dk_this = torch.zeros(global_k[0].chunk_shape, dtype=dtype, device=device)
                dv_this = torch.zeros(global_v[0].chunk_shape, dtype=dtype, device=device)

                with get_accelerator().stream(compute_stream):
                    if flash_attn_version >= version.parse("2.6.0"):
                        _flash_attn_backward(grad_global_attn_output[q_compute_chunk_idx].get_gpu_chunk(),
                                             global_q[q_compute_chunk_idx].get_gpu_chunk(),
                                             global_k[kv_compute_chunk_idx].get_gpu_chunk(),
                                             global_v[kv_compute_chunk_idx].get_gpu_chunk(),
                                             attn_output[q_compute_chunk_idx].get_gpu_chunk(),
                                             lse[q_compute_chunk_idx].get_gpu_chunk(),
                                             dq_this,
                                             dk_this,
                                             dv_this,
                                             dropout_p,
                                             softmax_scale,
                                             causal_chunk,
                                             window_size,
                                             softcap=0.0,
                                             alibi_slopes=alibi_slopes,
                                             deterministic=False,
                                             rng_state=None)
                    else:
                        _flash_attn_backward(grad_global_attn_output[q_compute_chunk_idx].get_gpu_chunk(),
                                             global_q[q_compute_chunk_idx].get_gpu_chunk(),
                                             global_k[kv_compute_chunk_idx].get_gpu_chunk(),
                                             global_v[kv_compute_chunk_idx].get_gpu_chunk(),
                                             attn_output[q_compute_chunk_idx].get_gpu_chunk(),
                                             lse[q_compute_chunk_idx].get_gpu_chunk(),
                                             dq_this,
                                             dk_this,
                                             dv_this,
                                             dropout_p,
                                             softmax_scale,
                                             causal_chunk,
                                             window_size,
                                             alibi_slopes=alibi_slopes,
                                             deterministic=False,
                                             rng_state=None)

                if i != (len(global_k) - 1):
                    if q_i != (len(global_q) - 1):
                        next_q_compute_chunk_idx = q_i + 1
                    else:
                        next_q_compute_chunk_idx = i + 1

                can_offload_q = True

                if next_q_compute_chunk_idx == q_compute_chunk_idx:
                    can_offload_q = False
                else:
                    with get_accelerator().stream(offload_stream):
                        if i > 0 or q_i > 0:
                            if can_offload_q and last_q_accum_idx != i:  # the first q chunk calculate in the loop will be sent out, therefore we do not offload it
                                dq[last_q_accum_idx].offload()
                        dq[next_q_compute_chunk_idx].load_to_gpu()
                        global_q[next_q_compute_chunk_idx].load_to_gpu()
                        attn_output[next_q_compute_chunk_idx].load_to_gpu()
                        lse[next_q_compute_chunk_idx].load_to_gpu()
                        if grad_global_attn_output[next_q_compute_chunk_idx] is not None:
                            grad_global_attn_output[next_q_compute_chunk_idx].load_to_gpu()

                        if grad_global_attn_output[next_q_compute_chunk_idx] is None:
                            grad_global_attn_output_chunk = single_all_to_all(grad_output[:, :chunk_size].contiguous(),
                                                                              scatter_idx, gather_idx, 0, spg)
                            dist.barrier()
                            grad_output = grad_output[:, chunk_size:]
                            grad_global_attn_output[next_q_compute_chunk_idx] = SequenceChunk(
                                grad_global_attn_output_chunk, is_in_use=True)

                compute_stream.wait_stream(offload_stream)
                compute_stream.synchronize()

                with get_accelerator().stream(compute_stream):
                    dq[q_compute_chunk_idx].check_gpu_chunk()
                    dq[q_compute_chunk_idx].gpu_chunk.add_(dq_this)
                    dk_accum.add_(dk_this)
                    dv_accum.add_(dv_this)

                offload_stream.wait_stream(compute_stream)
                with get_accelerator().stream(offload_stream):
                    dq[q_compute_chunk_idx].overwrite_to_cpu()

                if can_offload_q:
                    global_q[q_compute_chunk_idx].offload()
                    attn_output[q_compute_chunk_idx].offload()
                    lse[q_compute_chunk_idx].offload()
                    grad_global_attn_output[q_compute_chunk_idx].offload()

                last_q_accum_idx = q_compute_chunk_idx
                q_compute_chunk_idx = next_q_compute_chunk_idx

            compute_stream.wait_stream(offload_stream)
            compute_stream.synchronize()

            dk_seq_len = dk_accum.shape[1]

            if ctx.pos_emb_cos is not None:
                dq_accum = apply_rotary_pos_emb_backward(dq[kv_compute_chunk_idx].get_gpu_chunk().to(dtype),
                                                         ctx.pos_emb_cos[:, dk_seq_len * i:dk_seq_len * (i + 1)],
                                                         ctx.pos_emb_sin[:, dk_seq_len * i:dk_seq_len * (i + 1)])
                dk_accum = apply_rotary_pos_emb_backward(dk_accum.to(dtype),
                                                         ctx.pos_emb_cos[:, dk_seq_len * i:dk_seq_len * (i + 1)],
                                                         ctx.pos_emb_sin[:, dk_seq_len * i:dk_seq_len * (i + 1)])
            else:
                dq_accum = dq[kv_compute_chunk_idx].get_gpu_chunk().to(dtype)
                dk_accum = dk_accum.to(dtype)
            dv_accum = dv_accum.to(dtype)

            dq_accum = single_all_to_all(dq_accum.contiguous(), gather_idx, scatter_idx, 0, spg)
            dk_accum = single_all_to_all(dk_accum.contiguous(), gather_idx, scatter_idx, 0, spg)
            dv_accum = single_all_to_all(dv_accum.contiguous(), gather_idx, scatter_idx, 0, spg)

            general_offload_stream.synchronize()
            compute_stream.wait_stream(general_offload_stream)
            dist.barrier()

            with get_accelerator().stream(compute_stream):
                input_chunk = layernorm_output[i].get_gpu_chunk().reshape(-1, layernorm_output[i].chunk_shape[-1])

                dq_accum = dq_accum.flatten(2).permute(1, 0, 2)
                dk_accum = dk_accum.flatten(2).permute(1, 0, 2)
                dv_accum = dv_accum.flatten(2).permute(1, 0, 2)

                l, b = dk_accum.shape[0], dk_accum.shape[1]

                grad_qkv_linear_weight[:projection_size].add_(
                    torch.matmul(dq_accum.reshape(l * b, -1).t(), input_chunk))
                grad_qkv_linear_weight[projection_size:projection_size + kv_projection_size].add_(
                    torch.matmul(dk_accum.reshape(l * b, -1).t(), input_chunk))
                grad_qkv_linear_weight[projection_size + kv_projection_size:].add_(
                    torch.matmul(dv_accum.reshape(l * b, -1).t(), input_chunk))

                grad_qkv_linear_bias[:projection_size].add_(dq_accum.sum(0).sum(0))
                grad_qkv_linear_bias[projection_size:projection_size + kv_projection_size].add_(dk_accum.sum(0).sum(0))
                grad_qkv_linear_bias[projection_size + kv_projection_size:].add_(dv_accum.sum(0).sum(0))

                grad_layernorm_output[i].add_(torch.matmul(dq_accum, qkv_linear_weight[:projection_size]))
                grad_layernorm_output[i].add_(
                    torch.matmul(dk_accum, qkv_linear_weight[projection_size:projection_size + kv_projection_size]))
                grad_layernorm_output[i].add_(
                    torch.matmul(dv_accum, qkv_linear_weight[projection_size + kv_projection_size:]))

                del dq_accum, dk_accum, dv_accum
                dk_accum = torch.zeros(global_k[i].chunk_shape, dtype=torch.float, device=device)
                dv_accum = torch.zeros(global_v[i].chunk_shape, dtype=torch.float, device=device)
                dq[kv_compute_chunk_idx].offload()
                dq[kv_compute_chunk_idx] = None

            if i != (len(global_k) - 1):
                next_kv_compute_chunk_idx = kv_compute_chunk_idx + 1
                with get_accelerator().stream(offload_stream):
                    global_k[next_kv_compute_chunk_idx].load_to_gpu()
                    global_v[next_kv_compute_chunk_idx].load_to_gpu()

                with get_accelerator().stream(general_offload_stream):
                    layernorm_output[next_kv_compute_chunk_idx].load_to_gpu()

            compute_stream.wait_stream(offload_stream)
            compute_stream.synchronize()

            layernorm_output[kv_compute_chunk_idx].offload()
            global_k[kv_compute_chunk_idx].offload()
            global_v[kv_compute_chunk_idx].offload()
            kv_compute_chunk_idx = next_kv_compute_chunk_idx

        return torch.cat(
            grad_layernorm_output,
            dim=0).to(dtype), None, None, None, None, None, None, None, None, None, None, grad_qkv_linear_weight.to(
                dtype), grad_qkv_linear_bias.to(dtype), None, None, None


class FPDT_Attention(torch.nn.Module):

    def __init__(self,
                 config,
                 first_weight,
                 first_bias,
                 second_weight,
                 second_bias,
                 sequence_process_group,
                 gather_idx: int = 0,
                 scatter_idx: int = 2,
                 return_bias=True,
                 chunk_size=65536,
                 enable_offloading=True) -> None:

        super(FPDT_Attention, self).__init__()
        if _flash_attn_forward is None or _flash_attn_backward is None:
            raise ImportError(
                "DeepSpeed FPDT requires flash-attn 2.6.3. Please install it with `pip install flash-attn --no-build-isolation`."
            )

        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.config = config

        self.projection_size = config.kv_channels * config.num_attention_heads
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.kv_projection_size = config.kv_channels * config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.qkv_linear_weight = first_weight
        self.qkv_linear_bias = first_bias
        self.qkv_dense_weight = second_weight
        self.qkv_dense_bias = second_bias

        self.reture_bias = return_bias
        self.dropout = config.attention_dropout

        self.chunk_size = chunk_size
        self.double_buffer = enable_offloading

    def forward(self,
                layernorm_output,
                attention_mask,
                inference_params,
                rotary_pos_emb,
                cpu_offloading=True) -> Tensor:
        self.num_chunks_attn = layernorm_output.shape[0] * dist.get_world_size(self.spg) // self.chunk_size

        if not cpu_offloading or self.num_chunks_attn == 1:
            output = _FPDTGPUAttentionImpl_.apply(layernorm_output, attention_mask, inference_params, rotary_pos_emb,
                                                  self.spg, self.scatter_idx, self.gather_idx, self.hidden_size,
                                                  self.projection_size, self.hidden_size_per_attention_head,
                                                  self.kv_projection_size, self.qkv_linear_weight,
                                                  self.qkv_linear_bias, self.dropout, self.num_chunks_attn,
                                                  cpu_offloading)
        else:
            output = _FPDTGPUOffloadingAttentionImpl_.apply(
                layernorm_output, attention_mask, inference_params, rotary_pos_emb, self.spg, self.scatter_idx,
                self.gather_idx, self.hidden_size, self.projection_size, self.hidden_size_per_attention_head,
                self.kv_projection_size, self.qkv_linear_weight, self.qkv_linear_bias, self.dropout,
                self.num_chunks_attn, cpu_offloading)

        output = output.flatten(2).permute(1, 0, 2).contiguous()

        output = torch.matmul(output, self.qkv_dense_weight.t())
        if not self.reture_bias:
            output += self.qkv_dense_bias
        return output, self.qkv_dense_bias if self.reture_bias else None


@torch.jit.script
def bias_gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@torch.jit.script
def bias_gelu_back(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class FPDT_FFN(torch.autograd.Function):
    generate_vmap_rule = False

    @staticmethod
    def forward(ctx: Any, x, w1, b1, w2, b2, add_bias, chunk_size):
        do_save = x.requires_grad
        ctx.add_bias = add_bias
        device = x.device

        with torch.no_grad():
            num_chunk = x.shape[0] // chunk_size
            ctx.num_chunk = num_chunk
            result = torch.empty(x.shape, device=device, dtype=x.dtype)
            assert chunk_size * num_chunk == x.shape[0]
            for i in range(num_chunk):
                st = i * chunk_size
                ed = st + chunk_size
                x_ = torch.matmul(x[st:ed], w1.t()) + b1
                x_ = bias_gelu(x_)
                if add_bias:
                    result[st:ed] = torch.matmul(x_, w2.t()) + b2
                else:
                    result[st:ed] = torch.matmul(x_, w2.t())

                del x_

            if do_save:
                ctx.device = device
                ctx.dtype = x.dtype
                ctx.save_for_backward(x, w1, b1, w2, b2)
                ctx.grad_x_shape = x.shape
        return result.to(x.dtype), b2 if not add_bias else None

    @staticmethod
    def backward(ctx, grad_output, grad_bias):
        x, w1, b1, w2, b2 = ctx.saved_tensors
        device = ctx.device
        dtype = ctx.dtype
        add_bias = ctx.add_bias

        num_chunk = ctx.num_chunk
        chunk_size = x.shape[0] // num_chunk
        assert chunk_size * num_chunk == grad_output.shape[0]

        grad_w2 = torch.zeros(w2.shape, device=device, dtype=torch.float)
        grad_b2 = torch.zeros(b2.shape, device=device, dtype=torch.float)
        grad_w1 = torch.zeros(w1.shape, device=device, dtype=torch.float)
        grad_b1 = torch.zeros(b1.shape, device=device, dtype=torch.float)

        for i in range(num_chunk):
            st = i * chunk_size
            ed = st + chunk_size
            x_chunk = x[st:ed]

            before_act = (torch.matmul(x_chunk, w1.t()) + b1)
            before_act_2 = before_act**2
            tanh_out = torch.tanh(0.79788456 * before_act * (1 + 0.044715 * before_act_2))
            ff = 0.5 * before_act * ((1 - tanh_out * tanh_out) *
                                     (0.79788456 + 0.1070322243 * before_act_2)) + 0.5 * (1 + tanh_out)
            grad_w2.add_(
                torch.matmul(grad_output[st:ed].reshape(-1, grad_output.shape[2]).t(),
                             (before_act * 0.5 * (1 + tanh_out)).reshape(-1, before_act.shape[2])))
            del before_act, before_act_2, tanh_out

            grad_inter = torch.matmul(grad_output[st:ed], w2) * ff
            del ff

            grad_w1.add_(torch.matmul(
                grad_inter.reshape(-1, grad_inter.shape[2]).t(), x_chunk.reshape(-1, x.shape[2])))
            grad_b1.add_(grad_inter.sum(0).sum(0))

            x[st:ed].copy_(torch.matmul(grad_inter, w1))

            del grad_inter

            if add_bias:
                grad_b2.add_(grad_output[st:ed].sum(0).sum(0))

        return x, grad_w1.to(dtype), grad_b1.to(dtype), grad_w2.to(dtype), grad_b2.to(dtype), None, None


class FPDT_LogitsLoss(torch.autograd.Function):
    generate_vmap_rule = False

    @staticmethod
    def forward(ctx: Any, lm_output, labels, logit_weights, rank, spg_size, spg, num_chunk):
        labels = labels.t()
        chunk_size = lm_output.shape[0] // num_chunk
        assert chunk_size * num_chunk == lm_output.shape[0]
        batch_size, local_seq_len = lm_output.shape[1], lm_output.shape[0]
        loss = torch.empty((batch_size, local_seq_len), dtype=torch.float, device=lm_output.device)

        ctx.num_chunk = num_chunk
        ctx.chunk_size = chunk_size
        ctx.device = lm_output.device
        ctx.dtype = lm_output.dtype

        ctx.rank = rank
        ctx.local_seq_len = local_seq_len
        with torch.no_grad():
            for i in range(num_chunk):
                st = i * chunk_size
                ed = st + chunk_size
                logits_chunk = torch.matmul(lm_output[st:ed], logit_weights.t()).float()

                vocab_size = logits_chunk.size(2)
                # nll
                softmax = torch.nn.functional.softmax(logits_chunk, dim=-1)
                loss_chunk = torch.nn.functional.nll_loss(softmax.log().reshape(-1, vocab_size).contiguous(),
                                                          labels[st:ed, :].reshape(-1).contiguous(),
                                                          reduction='none')
                loss[:, st:ed] = loss_chunk.reshape(chunk_size, batch_size).t()

                del logits_chunk
            ctx.save_for_backward(lm_output.to('cpu'), labels)
            ctx.logit_weights = logit_weights

        seqlen = local_seq_len * spg_size
        batch_size = loss.size(0)
        loss = loss.t().contiguous()
        loss_all = torch.empty(seqlen, batch_size, dtype=loss.dtype, device=loss.device).contiguous()

        dist.allgather_fn(loss_all, loss, group=spg)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        lm_output, labels = ctx.saved_tensors
        logit_weights = ctx.logit_weights
        device = ctx.device
        dtype = ctx.dtype
        num_chunk = ctx.num_chunk
        chunk_size = ctx.chunk_size

        rank = ctx.rank
        local_seq_len = ctx.local_seq_len

        grad_output = grad_output[rank * local_seq_len:(rank + 1) * local_seq_len]
        grad_lm_output = [None for _ in range(num_chunk)]
        grad_logit_weights = torch.zeros(logit_weights.shape, device=grad_output.device, dtype=torch.float)
        for i in range(num_chunk):
            st = i * chunk_size
            ed = st + chunk_size
            lm_output_chunk = lm_output[st:ed].to(device)
            logits_chunk = torch.matmul(lm_output_chunk, logit_weights.t()).float()

            # nll
            softmax = torch.nn.functional.softmax(logits_chunk, dim=-1)
            vocab_size = logits_chunk.size(2)

            grad_input = softmax
            grad_2d = grad_input.reshape(-1, vocab_size).contiguous()
            arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=device)

            grad_2d[arange_1d, labels[st:ed, :].reshape(-1).contiguous()] -= 1
            grad_input.mul_(grad_output[:chunk_size, :].unsqueeze(dim=-1))
            grad_input = grad_input.to(dtype)

            grad_output = grad_output[chunk_size:].contiguous()

            grad_lm_output_chunk = torch.matmul(grad_input, logit_weights)
            grad_lm_output[i] = grad_lm_output_chunk

            grad_logit_weights.add_(
                torch.matmul(
                    grad_input.reshape(-1, grad_input.shape[2]).t(),
                    lm_output_chunk.reshape(-1, lm_output_chunk.shape[2])))

        return torch.cat(grad_lm_output, dim=0).to(dtype), None, grad_logit_weights.to(dtype), None, None, None, None
