# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Reference implementations (executable specifications) for segment-KI kernels.

Each function here defines the mathematically correct output for its
corresponding interface method (the m.def entries on the loaded extension).
Custom kernels (CUDA today, SYCL/XPU in the future) must produce results
equivalent to these functions within bf16 tolerance. Three roles in one:

1. **Specification**: an agent porting kernels to a new backend reads these
   to understand what each interface method must compute.
2. **Runtime fallback**: the forward paths in segment_ki call these when no
   custom kernel is available (e.g. CPU-only, XPU without compiled kernels).
3. **Test oracle**: unit tests compare custom kernel output against these
   functions on identical inputs.

All implementations use only standard PyTorch ops — no custom kernels — so
they are correct by construction and run on any device.
"""

import math

import torch
import torch.nn.functional as F


def grouped_linear(x, weights, glu_epilogue=False):
    """Unified projection: matmul against each weight, optional GLU epilogue.

    The portable interface's single entry point for all GEMV/GEMM paths
    (flat single-row, small-batch grouped, large-batch composite). Returns
    a list of per-weight outputs; the GLU variant folds silu(gate)*up into
    the first output.
    """
    outs = [F.linear(x, w) for w in weights]
    if glu_epilogue:
        outs = [F.silu(outs[0]) * outs[1]]
    return outs


def gdn_gates(a, b, a_log, dt_bias):
    """GDN gating: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)."""
    beta = torch.sigmoid(b)
    g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float())
    return beta, g


def decode_attn(q, K, V, pos):
    """Single-query attention over valid KV positions, batched.

    q: [b, num_heads, head_dim]
    K, V: [b, num_kv_heads, max_len, head_dim]
    pos: int or LongTensor [b] (valid positions, 0-indexed inclusive;
         an int applies to every row)

    Returns [b, num_heads, head_dim] with GQA (K/V heads shared across query
    heads).
    """
    num_q_heads = q.shape[-2]
    num_kv_heads = K.shape[-3]
    head_dim = q.shape[-1]
    n_rep = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    lens = (pos + 1).tolist() if torch.is_tensor(pos) else [pos + 1] * q.shape[0]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for r in range(q.shape[0]):
        K_valid = K[r, :, :lens[r], :]  # [nkv, len, hd]
        V_valid = V[r, :, :lens[r], :]
        if n_rep > 1:
            K_valid = K_valid.repeat_interleave(n_rep, dim=0)
            V_valid = V_valid.repeat_interleave(n_rep, dim=0)
        scores = torch.einsum('hd,hld->hl', q[r].float(), K_valid.float()) * scale
        weights = torch.softmax(scores, dim=-1)
        out[r] = torch.einsum('hl,hld->hd', weights, V_valid.float())
    return out.to(q.dtype)


def silu_mul(gu, n):
    """GLU epilogue on a [M, 2n] buffer: silu(g) * u."""
    return F.silu(gu[:, :n]) * gu[:, n:]


def repeat_qk(q, k, ratio):
    """Head-repeat for the GDN scan's v-head layout: out[...,hv,:] = in[...,hv/ratio,:]."""
    return q.repeat_interleave(ratio, dim=-2), k.repeat_interleave(ratio, dim=-2)


def gated_norm(x, z, weight, eps=1e-5):
    """GDN z-gated output norm: (weight * rms(x)) * silu(z), weight direct."""
    xf = x.float()
    normalized = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    weighted = normalized * weight
    return (weighted.float() * torch.nn.functional.silu(z.float())).to(x.dtype)


def fused_add_norm(hidden, residual, weight, eps=1e-5):
    """Residual add + RMSNorm: rms_norm(hidden + residual) * weight."""
    combined = (hidden + residual).float()
    rms = torch.rsqrt(combined.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (combined * rms * weight.float()).to(hidden.dtype)
