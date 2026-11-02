# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Reference implementations (executable specifications) for segment-KI kernels.

Each function here defines the mathematically correct output for its
corresponding custom kernel.  Custom kernels (CUDA today, SYCL/XPU in the
future) must produce results equivalent to these functions within bf16
tolerance.  Three roles in one:

1. **Specification**: an agent porting kernels to a new backend reads these
   to understand what the kernel must compute.
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


def dual_gemv_silu_mul(hidden, gate_w, up_w):
    """silu(hidden @ gate_w.T) * (hidden @ up_w.T).

    Reads gate and up weight matrices directly (no concatenation), so
    gradients flow to the original Parameters.  Works for any batch size.
    """
    gate_out = torch.matmul(hidden, gate_w.t())
    up_out = torch.matmul(hidden, up_w.t())
    return F.silu(gate_out) * up_out


def gdn_gates(a, b, a_log, dt_bias):
    """GDN gating: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)."""
    beta = torch.sigmoid(b)
    g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float())
    return beta, g.to(b.dtype)


def gdn_input_proj(hidden, w_qkv, w_z, w_b, w_a):
    """GDN input projections: cat(h @ Wqkv.T, h @ Wz.T, h @ Wb.T, h @ Wa.T)."""
    parts = [torch.matmul(hidden, w.t()) for w in (w_qkv, w_z, w_b, w_a)]
    return torch.cat(parts, dim=-1)


def triple_gemv(hidden, q_w, k_w, v_w):
    """QKV projection: (hidden @ q_w.T, hidden @ k_w.T, hidden @ v_w.T).

    Shape-agnostic: the spec for both the b=1 triple_gemv kernel and the
    batched triple_gemm grouped kernel (same math, any input shape)."""
    q = torch.matmul(hidden, q_w.t())
    k = torch.matmul(hidden, k_w.t())
    v = torch.matmul(hidden, v_w.t())
    return q, k, v


# Batched aliases — explicit specs for the grouped-GEMM kernels so the
# kernel-to-reference mapping is nameable. They delegate to the
# shape-agnostic originals above (identical math).


def triple_gemm(x, q_w, k_w, v_w):
    """Spec for the triple_gemm kernel (decode batches 16..32)."""
    return triple_gemv(x, q_w, k_w, v_w)


def quad_gemm(x, w_qkv, w_z, w_b, w_a):
    """Spec for the quad_gemm kernel: concatenated GDN input projections."""
    return gdn_input_proj(x, w_qkv, w_z, w_b, w_a)


def dual_gemm_silu_mul(x, gate_w, up_w):
    """Spec for the dual_gemm_silu_mul kernel: silu(x@g.T) * (x@u.T)."""
    return dual_gemv_silu_mul(x, gate_w, up_w)


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


def gated_norm(x, z, weight, eps=1e-5):
    """GDN z-gated output norm: (weight * rms(x)) * silu(z), weight direct."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float() * torch.nn.functional.silu(z.float())).to(x.dtype)


def fused_add_norm(hidden, residual, weight, eps=1e-5):
    """Residual add + RMSNorm: rms_norm(hidden + residual) * weight."""
    combined = hidden.float() + residual.float()
    rms = torch.rsqrt(combined.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (combined * rms * weight.float()).to(hidden.dtype)
