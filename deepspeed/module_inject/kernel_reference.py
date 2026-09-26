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


def triple_gemv(hidden, q_w, k_w, v_w):
    """QKV projection: (hidden @ q_w.T, hidden @ k_w.T, hidden @ v_w.T)."""
    q = torch.matmul(hidden, q_w.t())
    k = torch.matmul(hidden, k_w.t())
    v = torch.matmul(hidden, v_w.t())
    return q, k, v


def decode_attn(q, K, V, pos):
    """Single-query attention over valid KV positions.

    q: [num_heads, head_dim]
    K, V: [num_kv_heads, max_len, head_dim]
    pos: int (number of valid positions, 0-indexed inclusive)

    Returns [num_heads, head_dim] with GQA (K/V heads shared across query heads).
    """
    num_q_heads = q.shape[0]
    num_kv_heads = K.shape[0]
    head_dim = q.shape[1]
    n_rep = num_q_heads // num_kv_heads

    K_valid = K[:, :pos + 1, :]  # [nkv, pos+1, hd]
    V_valid = V[:, :pos + 1, :]

    if n_rep > 1:
        K_valid = K_valid.repeat_interleave(n_rep, dim=0)
        V_valid = V_valid.repeat_interleave(n_rep, dim=0)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum('hd,hld->hl', q.float(), K_valid.float()) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum('hl,hld->hd', weights, V_valid.float())
    return out.to(q.dtype)


def fused_add_norm(hidden, residual, weight, eps=1e-5):
    """Residual add + RMSNorm: rms_norm(hidden + residual) * weight."""
    combined = hidden.float() + residual.float()
    rms = torch.rsqrt(combined.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (combined * rms * weight.float()).to(hidden.dtype)


def decode_step(logits, vocab_size):
    """Argmax over logits (greedy decode)."""
    return torch.argmax(logits).item()
