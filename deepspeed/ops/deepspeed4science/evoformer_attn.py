# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import numpy as np
from deepspeed.ops.op_builder import EvoformerAttnBuilder
from deepspeed.accelerator import get_accelerator

kernel_ = None


def _attention(Q, K, V, bias1, bias2):
    assert Q.shape[-3] > 16, "seq_len must be greater than 16"
    O = torch.empty_like(Q, dtype=Q.dtype)
    assert get_accelerator().on_accelerator(Q), "Q must be on cuda"
    assert get_accelerator().on_accelerator(K), "K must be on cuda"
    assert get_accelerator().on_accelerator(V), "V must be on cuda"
    assert get_accelerator().on_accelerator(bias1), "bias1 must be on cuda"
    assert get_accelerator().on_accelerator(bias2), "bias2 must be on cuda"
    global kernel_
    if kernel_ is None:
        kernel_ = EvoformerAttnBuilder().load()
    nheads = Q.shape[-2]
    nq = (Q.shape[-3] + 31) // 32 * 32
    nb = np.prod(Q.shape[:-3])
    lse = torch.empty((nb, nheads, nq), dtype=torch.float32, device=Q.device)
    kernel_.attention(Q, K, V, bias1, bias2, O, lse)
    return O, lse


def attention_bwd(dO, Q, K, V, O, lse, bias1, bias2):
    assert max(Q.shape[-1], V.shape[-1]) <= 64, "Hidden size is too large. Need to change kMax to a larger value"
    dQ = torch.empty_like(Q, dtype=Q.dtype)
    dK = torch.empty_like(K, dtype=K.dtype)
    dV = torch.empty_like(V, dtype=V.dtype)
    assert get_accelerator().on_accelerator(dO), "dO must be on cuda"
    assert get_accelerator().on_accelerator(Q), "Q must be on cuda"
    assert get_accelerator().on_accelerator(K), "K must be on cuda"
    assert get_accelerator().on_accelerator(V), "V must be on cuda"
    assert get_accelerator().on_accelerator(O), "O must be on cuda"
    global kernel_
    if kernel_ is None:
        kernel_ = EvoformerAttnBuilder().load()
    delta = torch.empty_like(lse)
    dB1 = torch.zeros_like(bias1, dtype=torch.float32)
    dB2 = torch.zeros_like(bias2, dtype=torch.float32)
    kernel_.attention_bwd(dO, Q, K, V, O, lse, delta, bias1, bias2, dQ, dK, dV, dB1, dB2)
    return dQ, dK, dV, dB1.to(dO.dtype), dB2.to(dO.dtype)


class EvoformerFusedAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias1=None, bias2=None):
        """
        q, k, v: are in shape [*, L, H, D]
        """
        bias1_ = bias1.contiguous() if bias1 is not None else torch.tensor([], dtype=q.dtype, device=q.device)
        bias2_ = bias2.contiguous() if bias2 is not None else torch.tensor([], dtype=q.dtype, device=q.device)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o, lse = _attention(q, k, v, bias1_, bias2_)
        ctx.save_for_backward(q, k, v, o, lse, bias1_, bias2_)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, bias1, bias2 = ctx.saved_tensors
        dQ, dK, dV, dB1, dB2 = attention_bwd(grad_output, q, k, v, o, lse, bias1, bias2)
        if bias1.numel() == 0:
            dB1 = None
        if bias2.numel() == 0:
            dB2 = None
        return dQ, dK, dV, dB1, dB2


def DS4Sci_EvoformerAttention(Q, K, V, biases):
    assert len(biases) <= 2

    if (len(biases) == 0):
        biases.append(None)

    if (len(biases) == 1):
        biases.append(None)

    bias_1_shape = lambda x: (x.shape[0], x.shape[1], 1, 1, x.shape[2])
    bias_2_shape = lambda x: (x.shape[0], 1, x.shape[3], x.shape[2], x.shape[2])

    if biases[0] is not None:
        assert biases[0].shape == bias_1_shape(Q)
    else:
        biases[0] = Q.new_zeros(bias_1_shape(Q))

    if biases[1] is not None:
        assert biases[1].shape == bias_2_shape(Q)
    else:
        biases[1] = Q.new_zeros(bias_2_shape(Q))

    return EvoformerFusedAttention.apply(Q, K, V, biases[0], biases[1])
