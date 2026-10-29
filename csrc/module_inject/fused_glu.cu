// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// Native CUDA kernel for the segment-KI fused_glu op: computes
// out = silu(hidden[:, :k]) * hidden[:, k:2k] on the fused gate|up GEMM
// output without materializing chunked views. hidden is contiguous
// [N, 2k]; out is contiguous [N, k]. Activation math runs in fp32 with a
// single rounding to the storage dtype (torch opmath convention).

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <algorithm>

#define SILU(x) ((x) / (1.0f + expf(-(x))))

// GDN gating for the segment-KI fused_gdn op: per-token, per-value-head
//   beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)
// computed in fp32 regardless of storage dtype (matches the native forward's
// .float() upcast, which keeps A from reaching -inf in fp16/bf16).
__global__ void gdn_gates_kernel(const __nv_bfloat16* __restrict__ a,
                                 const __nv_bfloat16* __restrict__ b,
                                 const float* __restrict__ a_log,
                                 const float* __restrict__ dt_bias,
                                 __nv_bfloat16* __restrict__ beta_out,
                                 __nv_bfloat16* __restrict__ g_out,
                                 int64_t total,
                                 int num_heads,
                                 int64_t row_stride)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int64_t tok = i / num_heads;
        int h = (int)(i - tok * num_heads);
        // a/b arrive as [tokens, num_heads] slices of the fused GEMM output,
        // contiguous within each row but with a fused-width row stride.
        int64_t off = tok * row_stride + h;
        float av = __bfloat162float(a[off]);
        float bv = __bfloat162float(b[off]);
        float sp_x = av + dt_bias[h];
        // Numerically stable softplus: 1+exp(x) rounds to 1 in fp32 for
        // strongly negative x (log(1+exp(-20)) == 0), so use log1p there;
        // for large x, log(1+exp(x)) == x to fp32 precision.
        float sp = (sp_x > 20.0f)    ? sp_x
                   : (sp_x < -10.0f) ? log1pf(expf(sp_x))
                                     : logf(1.0f + expf(sp_x));
        beta_out[i] = __float2bfloat16(1.0f / (1.0f + expf(-bv)));
        g_out[i] = __float2bfloat16(-expf(a_log[h]) * sp);
    }
}

std::vector<at::Tensor> gdn_gates(at::Tensor a, at::Tensor b, at::Tensor a_log, at::Tensor dt_bias)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "gdn_gates is CUDA-only");
    TORCH_CHECK(a.stride(-1) == 1 && b.stride(-1) == 1, "a/b must be unit-stride in the last dim");
    auto beta = at::empty_like(a);
    auto g = at::empty_like(b);
    int64_t total = a.numel();
    if (total == 0) return {beta, g};
    int heads = (int)dt_bias.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    gdn_gates_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
        a_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
        total,
        heads,
        (int64_t)a.stride(-2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {beta, g};
}

// Fused decode step: argmax(logits) -> write token -> advance write_pos ->
// reveal mask -> record token in output buffer. Eliminates 6 Python->CUDA
// dispatches per decode step into one kernel call.
__global__ void decode_step_graph_kernel(const __nv_bfloat16* __restrict__ logits,
                                         int64_t* __restrict__ token_out,
                                         int64_t* __restrict__ write_pos,
                                         bool* __restrict__ mask,
                                         int64_t* __restrict__ out_buf,
                                         int vocab_size,
                                         int max_len,
                                         int pos_stride)
{
    // One block per row: each block argmaxes its row's logits and advances
    // its own write_pos entry (pos_stride 1); a scalar pos (stride 0) is
    // only valid with a single block.
    __shared__ int s_idx[1024];
    __shared__ float s_val[1024];

    const int row = blockIdx.x;
    const __nv_bfloat16* lg = logits + (long)row * vocab_size;
    int tid = threadIdx.x;
    int local_idx = 0;
    float local_val = -INFINITY;

    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = __bfloat162float(lg[v]);
        if (val > local_val) {
            local_val = val;
            local_idx = v;
        }
    }
    s_idx[tid] = local_idx;
    s_val[tid] = local_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // torch.argmax picks the lowest index on ties.
            if (s_val[tid + stride] > s_val[tid] ||
                (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid])) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t best = (int64_t)s_idx[0];
        token_out[row] = best;
        // The output token becomes the NEXT step's input; write it one
        // position ahead so it does not overwrite the current position
        // (which holds the token being processed this step).
        int64_t new_pos = write_pos[row * pos_stride] + 1;
        if (pos_stride != 0 || row == 0) write_pos[row * pos_stride] = new_pos;
        out_buf[(long)row * max_len + new_pos] = best;
        // new_pos is the slot the token just written will occupy during the
        // NEXT replay, matching the Python loop's reveal-before-replay
        // convention; revealing new_pos+1 would unmask an unwritten slot
        // and mask the current one for layers still on native SDPA.
        if (new_pos < max_len) { mask[(long)row * max_len + new_pos] = true; }
    }
}

void decode_step_graph(at::Tensor logits,
                       at::Tensor token_out,
                       at::Tensor write_pos,
                       at::Tensor mask,
                       at::Tensor out_buf)
{
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::ScalarType::BFloat16,
                "logits must be CUDA bf16");
    TORCH_CHECK(logits.dim() == 2, "logits must be [batch, vocab]");
    const int batch = (int)logits.size(0);
    int vocab = (int)logits.size(1);
    int max_len = (int)mask.size(-1);
    TORCH_CHECK(token_out.numel() == batch, "token_out must have one entry per row");
    TORCH_CHECK(mask.numel() == (int64_t)batch * max_len, "mask must be [batch, ..., max_len]");
    TORCH_CHECK(out_buf.numel() == (int64_t)batch * max_len, "out_buf must be [batch, max_len]");
    TORCH_CHECK(write_pos.numel() == batch || (write_pos.numel() == 1 && batch == 1),
                "write_pos must have one entry per row (a scalar only for batch 1)");
    const int pos_stride = write_pos.numel() == batch ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_step_graph_kernel<<<batch, 1024, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        token_out.data_ptr<int64_t>(),
        write_pos.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        out_buf.data_ptr<int64_t>(),
        vocab,
        max_len,
        pos_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ─── Dual-weight GEMV (b=1 decode) ───
// Reads gate.weight and up.weight directly — no concat, no weight copies.
// Each warp handles one output feature: lanes cooperatively compute two
// dot products (gate row and up row share the same hidden vector read),
// then applies silu(a)*b.  Memory-bound at b=1; coalesced reads achieve
// near-peak HBM bandwidth with zero tiling or tensor-core complexity.

// Lane-local partial dot product of hidden[0..n) with a weight row;
// 8 bf16 per 16-byte load per lane (n and the pointers are multiples of
// 8 / 16B-aligned, enforced by the wrappers). The caller reduces across
// the warp.
__device__ __forceinline__ float dot_partial_v8(const __nv_bfloat16* __restrict__ hidden,
                                                const __nv_bfloat16* __restrict__ w_row,
                                                int n,
                                                int lane)
{
    float acc = 0.0f;
    for (int i = lane * 8; i < n; i += 256) {
        uint4 hv = *reinterpret_cast<const uint4*>(hidden + i);
        uint4 wv = *reinterpret_cast<const uint4*>(w_row + i);
        const __nv_bfloat16* h8 = reinterpret_cast<const __nv_bfloat16*>(&hv);
        const __nv_bfloat16* w8 = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
        for (int e = 0; e < 8; e++) acc += __bfloat162float(h8[e]) * __bfloat162float(w8[e]);
    }
    return acc;
}

__global__ void dual_gemv_silu_mul_kernel(const __nv_bfloat16* __restrict__ hidden,
                                          const __nv_bfloat16* __restrict__ gate_w,
                                          const __nv_bfloat16* __restrict__ up_w,
                                          __nv_bfloat16* __restrict__ out,
                                          int out_features,
                                          int in_features)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= out_features) return;

    const __nv_bfloat16* gate_row = gate_w + (long)warp_id * in_features;
    const __nv_bfloat16* up_row = up_w + (long)warp_id * in_features;

    // 8 bf16 per lane per 16-byte load; 32 lanes cover 256 elements/iter
    float gate_acc = dot_partial_v8(hidden, gate_row, in_features, lane);
    float up_acc = dot_partial_v8(hidden, up_row, in_features, lane);

    // Warp-level reduction for both dot products simultaneously.
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_down_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffff, up_acc, offset);
    }

    if (lane == 0) {
        float activated = SILU(gate_acc);
        out[warp_id] = __float2bfloat16_rn(activated * up_acc);
    }
}

void dual_gemv_silu_mul(at::Tensor hidden, at::Tensor gate_w, at::Tensor up_w, at::Tensor out)
{
    TORCH_CHECK(hidden.size(-1) % 8 == 0, "in_features must be a multiple of 8");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(hidden.data_ptr()) & 15) == 0, "hidden must be 16B aligned");
    TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::ScalarType::BFloat16,
                "hidden must be CUDA bf16");
    TORCH_CHECK(gate_w.is_contiguous() && up_w.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(gate_w.size(0) == up_w.size(0) && gate_w.size(1) == up_w.size(1),
                "gate|up weight shapes must match");
    int out_f = (int)gate_w.size(0);
    int in_f = (int)gate_w.size(1);
    int warps_per_block = 8;  // 256 threads
    int blocks = (out_f + warps_per_block - 1) / warps_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    dual_gemv_silu_mul_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        out_f,
        in_f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ─── Quad-weight GEMV (b=1 GDN input projections) ───
// Reads the four original GDN projection weights directly (qkv|z|b|a) — no
// concat, no weight copies, gradient-safe for train/generate co-location.
// Same warp-per-row scheme as the dual-weight GEMV: the global row index
// selects which weight matrix the row lives in; output is one contiguous
// [rows0+rows1+rows2+rows3] buffer that the caller slices.

__global__ void quad_gemv_kernel(const __nv_bfloat16* __restrict__ hidden,
                                 const __nv_bfloat16* __restrict__ w0,
                                 const __nv_bfloat16* __restrict__ w1,
                                 const __nv_bfloat16* __restrict__ w2,
                                 const __nv_bfloat16* __restrict__ w3,
                                 __nv_bfloat16* __restrict__ out,
                                 int rows0,
                                 int rows1,
                                 int rows2,
                                 int rows3,
                                 int in_features)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total = rows0 + rows1 + rows2 + rows3;
    if (warp_id >= total) return;

    const __nv_bfloat16* row;
    int row_idx;
    if (warp_id < rows0) {
        row = w0 + (long)warp_id * in_features;
        row_idx = warp_id;
    } else if (warp_id < rows0 + rows1) {
        row_idx = warp_id - rows0;
        row = w1 + (long)row_idx * in_features;
    } else if (warp_id < rows0 + rows1 + rows2) {
        row_idx = warp_id - rows0 - rows1;
        row = w2 + (long)row_idx * in_features;
    } else {
        row_idx = warp_id - rows0 - rows1 - rows2;
        row = w3 + (long)row_idx * in_features;
    }

    float acc = dot_partial_v8(hidden, row, in_features, lane);
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) { out[warp_id] = __float2bfloat16_rn(acc); }
}

void quad_gemv(at::Tensor hidden,
               at::Tensor w0,
               at::Tensor w1,
               at::Tensor w2,
               at::Tensor w3,
               at::Tensor out)
{
    TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::ScalarType::BFloat16,
                "hidden must be CUDA bf16");
    TORCH_CHECK(
        w0.is_contiguous() && w1.is_contiguous() && w2.is_contiguous() && w3.is_contiguous(),
        "weights must be contiguous");
    TORCH_CHECK(w0.size(1) % 8 == 0, "in_features must be a multiple of 8");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(hidden.data_ptr()) & 15) == 0, "hidden must be 16B aligned");
    int rows0 = (int)w0.size(0), rows1 = (int)w1.size(0);
    int rows2 = (int)w2.size(0), rows3 = (int)w3.size(0);
    int in_f = (int)w0.size(1);
    int total = rows0 + rows1 + rows2 + rows3;
    int warps_per_block = 8;  // 256 threads
    int blocks = (total + warps_per_block - 1) / warps_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    quad_gemv_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(w0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(w3.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        rows0,
        rows1,
        rows2,
        rows3,
        in_f);
}

// ─── Custom decode attention (batched, graph-compatible, no mask) ───
// Streams softmax(q @ K[0:pos]^T / sqrt(d)) @ V[0:pos] with GQA, for any
// batch size. The valid KV length comes from a GPU-resident write_pos
// tensor (shared scalar or per-row), so the kernel is CUDA-graph
// replayable (addresses fixed, only the values change per step).
// One block per KV head; its 8 warps split as (query head, KV chunk):
// warp w handles query head w % q_per_kv over KV positions strided by
// split = 8 / q_per_kv, then per-head partial online-softmax states are
// merged through shared memory. Each lane owns a contiguous 16B slice of
// the head dimension (HEAD_DIM/32 elements) for coalesced K/V row loads.

template <int HEAD_DIM>
__global__ void __launch_bounds__(512)
    decode_attn_kernel(const __nv_bfloat16* __restrict__ q,
                       __nv_bfloat16* __restrict__ K,
                       __nv_bfloat16* __restrict__ V,
                       const int64_t* __restrict__ write_pos,
                       __nv_bfloat16* __restrict__ out,
                       int num_q_heads,
                       int num_kv_heads,
                       long q_hs,
                       int max_len,
                       float scale,
                       int pos_stride,
                       const float* __restrict__ q_norm_w,      // [HEAD_DIM], pre-scaled (1+w)
                       const float* __restrict__ cos_t,         // [max_len, rope_dim]
                       const float* __restrict__ sin_t,         // [max_len, rope_dim]
                       const __nv_bfloat16* __restrict__ gate,   // q layout
                       const __nv_bfloat16* __restrict__ k_raw,    // this step's K, pre-norm/rope
                       const __nv_bfloat16* __restrict__ v_raw,    // this step's V
                       const float* __restrict__ k_norm_w,         // [HEAD_DIM], pre-scaled (1+w)
                       float eps,
                       int rope_dim)
{
    constexpr int EPL = HEAD_DIM / 32;  // head-dim elements owned per lane
    const int q_per_kv = num_q_heads / num_kv_heads;
    const int warps_per_block = blockDim.x >> 5;
    const int split = warps_per_block;  // KV chunks per query head

    // One block per (row, query head): all the block's warps split that
    // head's KV range, giving enough independent streams to cover HBM latency.
    const int row = blockIdx.x / num_q_heads;
    const int q_head = blockIdx.x % num_q_heads;
    const int kv_head = q_head / q_per_kv;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int s = warp_id;

    // valid KV positions: 0..pos inclusive; a shared scalar when pos_stride
    // is 0, one per row otherwise
    const int pos = (int)write_pos[row * pos_stride];

    __nv_bfloat16* Kg = K + ((long)(row * num_kv_heads + kv_head)) * max_len * HEAD_DIM;
    __nv_bfloat16* Vg = V + ((long)(row * num_kv_heads + kv_head)) * max_len * HEAD_DIM;
    const __nv_bfloat16* q_row = q + (long)row * num_q_heads * q_hs + (long)q_head * q_hs;
    const __nv_bfloat16* g_row = gate + (long)row * num_q_heads * q_hs + (long)q_head * q_hs;

    float q_reg[EPL];
#pragma unroll
    for (int e = 0; e < EPL; e++) q_reg[e] = __bfloat162float(q_row[lane * EPL + e]);

    if (q_norm_w != nullptr) {
        // Fused q-side preprocessing (the eager path's biggest elementwise
        // contributors): per-head RMSNorm with the (1+w) weight, then
        // rotate-half RoPE at this row's position. Both live in registers;
        // the RoPE pairing exchanges EPL values with lane^16 (element j
        // pairs with j +/- HEAD_DIM/2, always 16 lanes away for EPL=H/32).
        float ss = 0.0f;
#pragma unroll
        for (int e = 0; e < EPL; e++) ss += q_reg[e] * q_reg[e];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
        const float inv_norm = rsqrtf(__shfl_sync(0xffffffff, ss, 0) / HEAD_DIM + eps);
#pragma unroll
        for (int e = 0; e < EPL; e++) q_reg[e] = q_reg[e] * inv_norm * q_norm_w[lane * EPL + e];
        // Partial RoPE (HF convention): rotate_half only over [0, rope_dim),
        // elements past rope_dim pass through. Lanes holding rope elements
        // pair with the lane half a rope region away.
        // All lanes execute the shuffle (a warp-wide sync op cannot sit in a
        // divergent branch); non-rope lanes swap with themselves harmlessly.
        const int j0 = lane * EPL;
        const int half_lanes = rope_dim > 0 ? (rope_dim / 2) / EPL : 0;
        const int partner = (j0 < rope_dim / 2) ? lane + half_lanes
                            : (j0 < rope_dim)   ? lane - half_lanes
                                                : lane;
        float pair[EPL];
#pragma unroll
        for (int e = 0; e < EPL; e++) pair[e] = __shfl_sync(0xffffffff, q_reg[e], partner);
        if (j0 < rope_dim) {
            const bool first_half = j0 < rope_dim / 2;
            const float* crow = cos_t + (long)pos * rope_dim + j0;
            const float* srow = sin_t + (long)pos * rope_dim + j0;
#pragma unroll
            for (int e = 0; e < EPL; e++)
                q_reg[e] = q_reg[e] * crow[e] + (first_half ? -pair[e] : pair[e]) * srow[e];
        }
    }

    if (k_norm_w != nullptr) {
        // Fused K-side preprocessing (mirror of the q side): per-head RMSNorm,
        // partial RoPE, and the cache-slot write for this position. Each lane
        // writes exactly the elements it reads back at idx == pos, so
        // same-thread visibility holds without fences; the sibling q-head
        // blocks of this kv head redundantly write identical values.
        float k_reg[EPL];
        const __nv_bfloat16* k_in_row = k_raw + ((long)(row * num_kv_heads + kv_head)) * HEAD_DIM;
#pragma unroll
        for (int e = 0; e < EPL; e++) k_reg[e] = __bfloat162float(k_in_row[lane * EPL + e]);
        float kss = 0.0f;
#pragma unroll
        for (int e = 0; e < EPL; e++) kss += k_reg[e] * k_reg[e];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) kss += __shfl_down_sync(0xffffffff, kss, off);
        const float k_inv = rsqrtf(__shfl_sync(0xffffffff, kss, 0) / HEAD_DIM + eps);
#pragma unroll
        for (int e = 0; e < EPL; e++) k_reg[e] = k_reg[e] * k_inv * k_norm_w[lane * EPL + e];
        const int kj0 = lane * EPL;
        const int khalf_lanes = rope_dim > 0 ? (rope_dim / 2) / EPL : 0;
        const int kpartner = (kj0 < rope_dim / 2) ? lane + khalf_lanes
                            : (kj0 < rope_dim)   ? lane - khalf_lanes
                                                 : lane;
        float kpair[EPL];
#pragma unroll
        for (int e = 0; e < EPL; e++) kpair[e] = __shfl_sync(0xffffffff, k_reg[e], kpartner);
        if (kj0 < rope_dim) {
            const bool kfirst = kj0 < rope_dim / 2;
            const float* kcrow = cos_t + (long)pos * rope_dim + kj0;
            const float* ksrow = sin_t + (long)pos * rope_dim + kj0;
#pragma unroll
            for (int e = 0; e < EPL; e++)
                k_reg[e] = k_reg[e] * kcrow[e] + (kfirst ? -kpair[e] : kpair[e]) * ksrow[e];
        }
        __nv_bfloat16* k_slot = Kg + (long)pos * HEAD_DIM;
        __nv_bfloat16* v_slot = Vg + (long)pos * HEAD_DIM;
        const __nv_bfloat16* v_in_row = v_raw + ((long)(row * num_kv_heads + kv_head)) * HEAD_DIM;
#pragma unroll
        for (int e = 0; e < EPL; e++) {
            k_slot[lane * EPL + e] = __float2bfloat16_rn(k_reg[e]);
            v_slot[lane * EPL + e] = v_in_row[lane * EPL + e];
        }
    }

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_reg[EPL];
#pragma unroll
    for (int e = 0; e < EPL; e++) out_reg[e] = 0.0f;

    // UNROLL positions in flight so their K loads overlap and hide HBM latency.
    constexpr int UNROLL = 4;
    for (int base = s; base <= pos; base += UNROLL * split) {
        float k_reg[UNROLL][EPL];
        int idx[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            idx[u] = base + u * split;
            if (idx[u] <= pos) {
                const __nv_bfloat16* krow = Kg + (long)idx[u] * HEAD_DIM;
#pragma unroll
                for (int e = 0; e < EPL; e++) k_reg[u][e] = __bfloat162float(krow[lane * EPL + e]);
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            if (idx[u] > pos) continue;  // uniform across the warp
            float dot = 0.0f;
#pragma unroll
            for (int e = 0; e < EPL; e++) dot += q_reg[e] * k_reg[u][e];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffff, dot, off);
            float score = __shfl_sync(0xffffffff, dot, 0) * scale;

            float new_max = fmaxf(max_score, score);
            float rescale = expf(max_score - new_max);  // 0 when max_score is still -inf
            float weight = expf(score - new_max);
            sum_exp = sum_exp * rescale + weight;
#pragma unroll
            for (int e = 0; e < EPL; e++) out_reg[e] *= rescale;
            const __nv_bfloat16* vrow = Vg + (long)idx[u] * HEAD_DIM;
#pragma unroll
            for (int e = 0; e < EPL; e++)
                out_reg[e] += weight * __bfloat162float(vrow[lane * EPL + e]);
            max_score = new_max;
        }
    }

    // Merge the per-chunk partial softmax states (max, sum, weighted V) in
    // shared memory; an empty chunk contributes 0 via exp(-inf - M).
    __shared__ float s_max[16];
    __shared__ float s_sum[16];
    __shared__ float s_out[16][HEAD_DIM];

    s_max[warp_id] = max_score;
    s_sum[warp_id] = sum_exp;
#pragma unroll
    for (int e = 0; e < EPL; e++) s_out[warp_id][lane * EPL + e] = out_reg[e];
    __syncthreads();

    if (s == 0) {  // the first chunk's warp combines all chunks for its head
        float M = -INFINITY;
        for (int t = 0; t < split; t++) M = fmaxf(M, s_max[t]);
        float S = 0.0f;
        float acc[EPL];
#pragma unroll
        for (int e = 0; e < EPL; e++) acc[e] = 0.0f;
        for (int t = 0; t < split; t++) {
            float wm = expf(s_max[t] - M);
            S += s_sum[t] * wm;
#pragma unroll
            for (int e = 0; e < EPL; e++) acc[e] += s_out[t][lane * EPL + e] * wm;
        }
        float inv = 1.0f / S;
#pragma unroll
        for (int e = 0; e < EPL; e++) {
            float v = acc[e] * inv;
            if (gate != nullptr)
                v *= 1.0f / (1.0f + expf(-__bfloat162float(g_row[lane * EPL + e])));
            out[((long)(row * num_q_heads + q_head)) * HEAD_DIM + lane * EPL + e] =
                __float2bfloat16_rn(v);
        }
    }
}

void decode_attn(at::Tensor q,
                 at::Tensor K,
                 at::Tensor V,
                 at::Tensor write_pos,
                 at::Tensor out,
                 int64_t num_q_heads,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 int64_t max_len,
                 at::Tensor q_norm_w,
                 at::Tensor cos_t,
                 at::Tensor sin_t,
                 at::Tensor gate,
                 at::Tensor k_raw,
                 at::Tensor v_raw,
                 at::Tensor k_norm_w,
                 double eps,
                 int64_t rope_dim)
{
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::ScalarType::BFloat16, "q must be CUDA bf16");
    // accept packed or chunked-[q|gate] layouts: last dim dense, uniform
    // head stride (the kernel indexes through q_hs)
    TORCH_CHECK(q.stride(-1) == 1, "q last dim must be dense");
    const long q_hs = q.stride(-2);
    TORCH_CHECK(gate.defined() ? (gate.stride(-1) == 1 && gate.stride(-2) == q_hs) : true,
                "gate layout must match q");
    TORCH_CHECK(K.is_contiguous() && V.is_contiguous(), "K/V must be contiguous");
    TORCH_CHECK(write_pos.is_cuda() && write_pos.scalar_type() == at::ScalarType::Long,
                "write_pos must be a CUDA int64 tensor");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "GQA requires num_q_heads divisible by num_kv_heads");
    int q_per_kv = (int)(num_q_heads / num_kv_heads);
    TORCH_CHECK(head_dim % 32 == 0, "head_dim must be a multiple of 32, got ", head_dim);
    // 16 warps/block (512 threads) so each query head gets a 4-way KV split
    // — the per-warp stream is otherwise latency-bound at these sizes. More
    // warps than this exhaust registers with __launch_bounds__(512).
    const int warps_per_block = 16;
    TORCH_CHECK(warps_per_block % q_per_kv == 0,
                "q heads per KV head (",
                q_per_kv,
                ") must divide ",
                warps_per_block);

    // Any leading batch dims flatten to rows: q [b, hq, d], K/V [b, hkv, L, d].
    const int64_t batch = q.numel() / (num_q_heads * head_dim);
    TORCH_CHECK(batch >= 1 && q.numel() == batch * num_q_heads * head_dim, "q shape mismatch");
    TORCH_CHECK(K.numel() == batch * num_kv_heads * max_len * head_dim, "K shape mismatch");
    TORCH_CHECK(V.numel() == batch * num_kv_heads * max_len * head_dim, "V shape mismatch");
    TORCH_CHECK(out.numel() == q.numel(), "out shape mismatch");
    TORCH_CHECK(write_pos.numel() == 1 || write_pos.numel() == batch,
                "write_pos must be a scalar or one entry per row");
    const int pos_stride = write_pos.numel() == 1 ? 0 : 1;

    // Optional fused q-side preprocessing: q_norm_w [head_dim] fp32
    // (pre-scaled 1+w), cos/sin tables [max_len, head_dim] fp32, gate with
    // q's layout. Nullptr tensors disable each piece.
    const bool fuse_q = q_norm_w.defined() && q_norm_w.numel() == head_dim;
    const bool use_gate = gate.defined() && gate.numel() == q.numel();
    const bool fuse_k = k_raw.defined() && k_raw.numel() > 0 && k_norm_w.defined();
    TORCH_CHECK(!fuse_k || (k_raw.is_contiguous() && v_raw.is_contiguous()),
                "k_raw/v_raw must be contiguous");
    TORCH_CHECK(!fuse_k || (fuse_q), "fused K requires the fused Q path (shared cos/sin)");
    // gate rides the same strided [b,H,hd] layout as q (checked above)
    TORCH_CHECK(
        !fuse_q || (cos_t.defined() && sin_t.defined() &&
                    cos_t.numel() == (int64_t)max_len * rope_dim && sin_t.numel() == cos_t.numel()),
        "fused q path needs cos/sin tables of [max_len, rope_dim]");
    TORCH_CHECK(!fuse_q || (rope_dim % (2 * (head_dim / 32)) == 0),
                "rope_dim/2 must be a multiple of the per-lane element count");

    float scale = 1.0f / sqrtf((float)head_dim);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    auto k_ptr = reinterpret_cast<__nv_bfloat16*>(K.data_ptr<at::BFloat16>());
    auto v_ptr = reinterpret_cast<__nv_bfloat16*>(V.data_ptr<at::BFloat16>());
    auto o_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    auto k_raw_ptr = fuse_k ? reinterpret_cast<const __nv_bfloat16*>(k_raw.data_ptr<at::BFloat16>()) : nullptr;
    auto v_raw_ptr = fuse_k ? reinterpret_cast<const __nv_bfloat16*>(v_raw.data_ptr<at::BFloat16>()) : nullptr;
    auto k_norm_ptr = fuse_k ? k_norm_w.data_ptr<float>() : nullptr;
    auto qn_ptr = fuse_q ? q_norm_w.data_ptr<float>() : nullptr;
    auto cos_ptr = fuse_q ? cos_t.data_ptr<float>() : nullptr;
    auto sin_ptr = fuse_q ? sin_t.data_ptr<float>() : nullptr;
    auto g_ptr = use_gate ? reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>())
                          : nullptr;

    switch (head_dim) {
        case 64:
            decode_attn_kernel<64><<<(int)(batch * num_q_heads), warps_per_block * 32, 0, stream>>>(
                q_ptr,
                k_ptr,
                v_ptr,
                write_pos.data_ptr<int64_t>(),
                o_ptr,
                (int)num_q_heads,
                (int)num_kv_heads,
                q_hs,
                (int)max_len,
                scale,
                pos_stride,
                qn_ptr,
                cos_ptr,
                sin_ptr,
                g_ptr,
                k_raw_ptr,
                v_raw_ptr,
                k_norm_ptr,
                (float)eps,
                (int)rope_dim);
            break;
        case 128:
            decode_attn_kernel<128>
                <<<(int)(batch * num_q_heads), warps_per_block * 32, 0, stream>>>(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    write_pos.data_ptr<int64_t>(),
                    o_ptr,
                    (int)num_q_heads,
                    (int)num_kv_heads,
                    q_hs,
                    (int)max_len,
                    scale,
                    pos_stride,
                    qn_ptr,
                    cos_ptr,
                    sin_ptr,
                    g_ptr,
                    k_raw_ptr,
                    v_raw_ptr,
                    k_norm_ptr,
                    (float)eps,
                    (int)rope_dim);
            break;
        case 256:
            decode_attn_kernel<256>
                <<<(int)(batch * num_q_heads), warps_per_block * 32, 0, stream>>>(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    write_pos.data_ptr<int64_t>(),
                    o_ptr,
                    (int)num_q_heads,
                    (int)num_kv_heads,
                    q_hs,
                    (int)max_len,
                    scale,
                    pos_stride,
                    qn_ptr,
                    cos_ptr,
                    sin_ptr,
                    g_ptr,
                    k_raw_ptr,
                    v_raw_ptr,
                    k_norm_ptr,
                    (float)eps,
                    (int)rope_dim);
            break;
        default: TORCH_CHECK(false, "Unsupported head_dim: ", head_dim);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ─── Fused residual add + RMSNorm (b=1 decode) ───
// One kernel replaces the two PyTorch ops per norm site (hidden += residual,
// then RMSNorm(hidden) * weight). hidden is updated in-place to hold the new
// residual stream; the normalized output is written to a separate tensor that
// feeds the next sublayer. One thread block covers the whole row so the
// sum-of-squares reduction stays block-local.
__global__ void fused_add_norm_kernel(__nv_bfloat16* __restrict__ hidden,
                                      const __nv_bfloat16* __restrict__ residual,
                                      const __nv_bfloat16* __restrict__ weight,
                                      __nv_bfloat16* __restrict__ out,
                                      float eps,
                                      int dim)
{
    extern __shared__ float s_data[];
    __shared__ float s_reduce[32];

    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    // one block per row (the original kernel indexed a single row only;
    // multi-row launches silently left rows 1+ of `out` uninitialized)
    const long row = blockIdx.x;
    __nv_bfloat16* hrow = hidden + row * dim;
    const __nv_bfloat16* rrow = residual + row * dim;
    __nv_bfloat16* orow = out + row * dim;

    float local_ss = 0.0f;
    for (int i = tid; i < dim; i += n_threads) {
        float h = __bfloat162float(hrow[i]) + __bfloat162float(rrow[i]);
        s_data[i] = h;
        hrow[i] = __float2bfloat16_rn(h);
        local_ss += h * h;
    }

    // Warp then cross-warp reduction of the sum of squares.
    for (int offset = 16; offset > 0; offset >>= 1)
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
    int warp_id = tid >> 5;
    if ((tid & 31) == 0) s_reduce[warp_id] = local_ss;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (n_threads >> 5)) ? s_reduce[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (tid == 0) s_reduce[0] = rsqrtf(v / dim + eps);
    }
    __syncthreads();
    float rms = s_reduce[0];

    for (int i = tid; i < dim; i += n_threads)
        orow[i] = __float2bfloat16_rn(s_data[i] * rms * __bfloat162float(weight[i]));
}

__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + expf(-v)); }

// GDN z-gated output norm: y = (w * x * rsqrt(mean(x^2)+eps)) * silu(z),
// weight applied directly (no 1+w here, no residual stream).
__global__ void gated_norm_kernel(const __nv_bfloat16* __restrict__ x,
                                  const __nv_bfloat16* __restrict__ z,
                                  const __nv_bfloat16* __restrict__ w,
                                  __nv_bfloat16* __restrict__ out,
                                  float eps,
                                  int dim)
{
    extern __shared__ float s_data[];
    __shared__ float s_reduce[32];

    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    const long row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * dim;
    const __nv_bfloat16* zr = z + row * dim;
    __nv_bfloat16* orow = out + row * dim;

    float local_ss = 0.0f;
    for (int i = tid; i < dim; i += n_threads) {
        float v = __bfloat162float(xr[i]);
        s_data[i] = v;
        local_ss += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_ss += __shfl_down_sync(0xffffffff, local_ss, offset);
    int warp_id = tid >> 5;
    if ((tid & 31) == 0) s_reduce[warp_id] = local_ss;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (n_threads >> 5)) ? s_reduce[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (tid == 0) s_reduce[0] = rsqrtf(v / dim + eps);
    }
    __syncthreads();
    float rms = s_reduce[0];
    for (int i = tid; i < dim; i += n_threads)
        orow[i] = __float2bfloat16_rn(s_data[i] * rms * __bfloat162float(w[i]) *
                                      silu_f(__bfloat162float(zr[i])));
}

at::Tensor fused_add_norm(at::Tensor hidden, at::Tensor residual, at::Tensor weight, double eps)
{
    TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::ScalarType::BFloat16,
                "hidden/residual/weight must be CUDA bf16");
    TORCH_CHECK(hidden.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
                "inputs must be contiguous");
    int64_t rows = hidden.numel() / hidden.size(-1);
    int dim = (int)hidden.size(-1);
    TORCH_CHECK(residual.numel() == hidden.numel() && weight.numel() == dim, "shape mismatch");
    TORCH_CHECK((int64_t)dim * sizeof(float) <= 48 * 1024,
                "dim too large for one-block shared memory: ",
                dim);
    auto out = at::empty_like(hidden);
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = std::min(1024, ((dim + 31) / 32) * 32);
    if (threads == 0) threads = 32;
    fused_add_norm_kernel<<<(int)rows, threads, (size_t)dim * sizeof(float), stream>>>(
        reinterpret_cast<__nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        (float)eps,
        dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ─── Triple-weight QKV GEMV (b=1 decode) ───
// Reads q_proj/k_proj/v_proj weights directly — no concat, no weight copies
// (gradient-safe). One warp per output row across all three
// matrices, so a single launch covers the whole QKV projection.
__global__ void triple_gemv_kernel(const __nv_bfloat16* __restrict__ hidden,
                                   const __nv_bfloat16* __restrict__ q_w,
                                   const __nv_bfloat16* __restrict__ k_w,
                                   const __nv_bfloat16* __restrict__ v_w,
                                   __nv_bfloat16* __restrict__ q_out,
                                   __nv_bfloat16* __restrict__ k_out,
                                   __nv_bfloat16* __restrict__ v_out,
                                   int q_out_f,
                                   int kv_out_f,
                                   int in_f)
{
    int total_rows = q_out_f + 2 * kv_out_f;
    for (int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; warp_id < total_rows;
         warp_id += (gridDim.x * blockDim.x) >> 5) {
        int lane = threadIdx.x & 31;
        const __nv_bfloat16* w_ptr;
        __nv_bfloat16* out_ptr;
        int row;
        if (warp_id < q_out_f) {
            w_ptr = q_w;
            out_ptr = q_out;
            row = warp_id;
        } else if (warp_id < q_out_f + kv_out_f) {
            w_ptr = k_w;
            out_ptr = k_out;
            row = warp_id - q_out_f;
        } else {
            w_ptr = v_w;
            out_ptr = v_out;
            row = warp_id - q_out_f - kv_out_f;
        }

        float acc = dot_partial_v8(hidden, w_ptr + (long)row * in_f, in_f, lane);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        if (lane == 0) out_ptr[row] = __float2bfloat16_rn(acc);
    }
}

std::vector<at::Tensor> triple_gemv(at::Tensor hidden,
                                    at::Tensor q_w,
                                    at::Tensor k_w,
                                    at::Tensor v_w)
{
    TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::ScalarType::BFloat16,
                "hidden must be CUDA bf16");
    TORCH_CHECK(q_w.is_contiguous() && k_w.is_contiguous() && v_w.is_contiguous(),
                "weights must be contiguous");
    TORCH_CHECK(q_w.size(1) == k_w.size(1) && k_w.size(1) == v_w.size(1),
                "in_features must match across q/k/v");
    TORCH_CHECK(q_w.size(1) % 8 == 0, "in_features must be a multiple of 8");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(hidden.data_ptr()) & 15) == 0, "hidden must be 16B aligned");
    int q_out_f = (int)q_w.size(0);
    int kv_out_f = (int)k_w.size(0);
    TORCH_CHECK(v_w.size(0) == kv_out_f, "v/k weight shapes must match");
    int in_f = (int)q_w.size(1);
    auto q_out = at::empty({q_out_f}, hidden.options());
    auto k_out = at::empty({kv_out_f}, hidden.options());
    auto v_out = at::empty({kv_out_f}, hidden.options());
    int total_rows = q_out_f + 2 * kv_out_f;
    int warps_per_block = 8;
    int blocks = (total_rows + warps_per_block - 1) / warps_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    triple_gemv_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr<at::BFloat16>()),
        q_out_f,
        kv_out_f,
        in_f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_out, k_out, v_out};
}



// Grouped linear for small batches (2..32): one launch covers up to four
// weight matrices, each weight element streamed exactly once (the
// activation tile rides in shared memory). Replaces the separate cuBLAS
// GEMMs the composite fallback issues per matrix; no weight copies.
static constexpr int GRP_MT = 32;   // max batch rows
static constexpr int GRP_NT = 64;        // Ada default (2 WMMA tiles per warp)
static constexpr int GRP_NT_AMPERE = 128;  // A100: more smem (164KB) + bandwidth
static constexpr int GRP_KT = 128;  // k-chunk staged through smem
static constexpr int GRP_TPB = 256;      // threads per block (8 warps, one 16x16 tile each)
static constexpr int GRP_KT_GLU = 64;   // GLU stages gate+up tiles: smaller KT fits 99KB

constexpr int GRP_SMEM_BYTES = (2 * GRP_MT * (GRP_KT + 8) + 2 * GRP_NT * (GRP_KT + 8)) * 2 +
                               GRP_MT * (GRP_NT + 8) * 4;
constexpr int GRP_SMEM_BYTES_AMPERE = (2 * GRP_MT * (GRP_KT + 8) + 2 * GRP_NT_AMPERE * (GRP_KT + 8)) * 2 +
                                      GRP_MT * (GRP_NT_AMPERE + 8) * 4;
constexpr int GRP_GLU_SMEM_BYTES = (2 * GRP_MT * (GRP_KT_GLU + 8) + 4 * GRP_NT * (GRP_KT_GLU + 8)) * 2 +
                                   2 * GRP_MT * (GRP_NT + 8) * 4;

struct GrpSeg {
    int mat;  // which weight matrix
    int lrow; // row within that matrix
};

__device__ __forceinline__ const __nv_bfloat16* wgs_ptr(int mat,
                                                        const __nv_bfloat16* w0,
                                                        const __nv_bfloat16* w1,
                                                        const __nv_bfloat16* w2,
                                                        const __nv_bfloat16* w3)
{
    return mat == 0 ? w0 : mat == 1 ? w1 : mat == 2 ? w2 : w3;
}

__device__ __forceinline__ GrpSeg grp_map_row(int row, int f0, int f1, int f2, int f3, int n_mat)
{
    GrpSeg s{0, row};
    int fs[4] = {f0, f1, f2, f3};
    for (int i = 0; i < n_mat && i < 4; i++) {
        if (s.lrow < fs[i]) { s.mat = i; return s; }
        s.lrow -= fs[i];
    }
    s.mat = 0;  // unreachable for well-formed launches
    return s;
}

template<int NT_TILE>
__global__ void grouped_linear_kernel(const __nv_bfloat16* __restrict__ x,
                                      const __nv_bfloat16* __restrict__ w0,
                                      const __nv_bfloat16* __restrict__ w1,
                                      const __nv_bfloat16* __restrict__ w2,
                                      const __nv_bfloat16* __restrict__ w3,
                                      __nv_bfloat16* __restrict__ o0,
                                      __nv_bfloat16* __restrict__ o1,
                                      __nv_bfloat16* __restrict__ o2,
                                      __nv_bfloat16* __restrict__ o3,
                                      int f0,
                                      int f1,
                                      int f2,
                                      int f3,
                                      int M,
                                      int K,
                                      int total_out,
                                      int concat_mode)
{
    using namespace nvcuda;
    // dynamic smem: xs[2][MT][KT+8] | ws[2][NT][KT+8] | out_s[MT][NT+8]
    extern __shared__ char smem_raw[];
    const int XS_STRIDE = GRP_KT + 8;
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* ws = xs + 2 * GRP_MT * XS_STRIDE;
    float* out_s = reinterpret_cast<float*>(ws + 2 * NT_TILE * XS_STRIDE);
    __shared__ GrpSeg seg_tab[NT_TILE];

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int n0 = blockIdx.x * NT_TILE;
    if (n0 >= total_out) return;

    for (int c = tid; c < NT_TILE; c += GRP_TPB) {
        seg_tab[c] = (n0 + c < total_out) ? grp_map_row(n0 + c, f0, f1, f2, f3, 4) : GrpSeg{0, 0};
    }
    __syncthreads();

    int m0w = (warp >> 2) * (GRP_MT / 2);
    int n0w = (warp & 3) * 16;  // one 16x16 tile per warp (8 warps = 2x4 grid)

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;

    const int nchunks = (K + GRP_KT - 1) / GRP_KT;
    // async-stage chunk c into buffer slot c&1; compute overlaps the next load
    auto stage = [&](int c) {
        int k0 = c * GRP_KT;
        int kn = min(GRP_KT, K - k0);
        int kn8 = kn >> 3;
        __nv_bfloat16* xbuf = xs + (long)(c & 1) * GRP_MT * XS_STRIDE;
        __nv_bfloat16* wbuf = ws + (long)(c & 1) * NT_TILE * XS_STRIDE;
        for (int idx = tid; idx < M * kn8; idx += GRP_TPB) {
            int m = idx / kn8, j = idx % kn8;
            __pipeline_memcpy_async(xbuf + (long)m * XS_STRIDE + j * 8, x + (long)m * K + k0 + (long)j * 8, 16);
        }
        for (int idx = tid; idx < NT_TILE * kn8; idx += GRP_TPB) {
            int r = idx / kn8, j = idx % kn8;
            if (n0 + r < total_out) {
                GrpSeg g = seg_tab[r];
                __pipeline_memcpy_async(wbuf + (long)r * XS_STRIDE + j * 8,
                                        wgs_ptr(g.mat, w0, w1, w2, w3) + (long)g.lrow * K + k0 + (long)j * 8, 16);
            }
        }
        __pipeline_commit();
    };

    stage(0);
    for (int c = 0; c < nchunks; c++) {
        if (c + 1 < nchunks) stage(c + 1);
        __pipeline_wait_prior(c + 1 < nchunks ? 1 : 0);
        __syncthreads();
        int kn = min(GRP_KT, K - c * GRP_KT);
        if (kn < GRP_KT) {
            // partial final chunk: zero the tail so wmma sees zeros
            __nv_bfloat16* xbuf = xs + (long)(c & 1) * GRP_MT * XS_STRIDE;
            __nv_bfloat16* wbuf = ws + (long)(c & 1) * NT_TILE * XS_STRIDE;
            for (int idx = tid; idx < GRP_MT * GRP_KT; idx += GRP_TPB) {
                int r = idx / GRP_KT, cc = idx % GRP_KT;
                if (cc >= kn && r < M) xbuf[r * XS_STRIDE + cc] = __float2bfloat16_rn(0.f);
            }
            for (int idx = tid; idx < NT_TILE * GRP_KT; idx += GRP_TPB) {
                int r = idx / GRP_KT, cc = idx % GRP_KT;
                if (cc >= kn && n0 + r < total_out) wbuf[r * XS_STRIDE + cc] = __float2bfloat16_rn(0.f);
            }
            __syncthreads();
        }
        __nv_bfloat16* xbuf = xs + (long)(c & 1) * GRP_MT * XS_STRIDE;
        __nv_bfloat16* wbuf = ws + (long)(c & 1) * NT_TILE * XS_STRIDE;
#pragma unroll 4
        for (int kk = 0; kk < GRP_KT; kk += 16) {
            wmma::load_matrix_sync(fa, xbuf + m0w * XS_STRIDE + kk, XS_STRIDE);
            wmma::load_matrix_sync(fb, wbuf + n0w * XS_STRIDE + kk, XS_STRIDE);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(out_s + m0w * (NT_TILE + 8) + n0w, acc, NT_TILE + 8, wmma::mem_row_major);
    __syncthreads();

    __nv_bfloat16* ogs[4] = {o0, o1, o2, o3};
    int fs2[4] = {f0, f1, f2, f3};
    for (int idx = tid; idx < M * NT_TILE; idx += GRP_TPB) {
        int m = idx / NT_TILE, c = idx % NT_TILE;
        if (n0 + c >= total_out) continue;
        GrpSeg g = seg_tab[c];
        float v = out_s[m * (NT_TILE + 8) + c];
        __nv_bfloat16* o = ogs[g.mat];
        if (concat_mode) {
            int base = 0;
            for (int i = 0; i < g.mat; i++) base += fs2[i];
            o[(long)m * total_out + base + g.lrow] = __float2bfloat16_rn(v);
        } else {
            o[(long)m * fs2[g.mat] + g.lrow] = __float2bfloat16_rn(v);
        }
    }
}

// Paired-matrix GLU GEMM: out = silu(x@g^T) * (x@u^T). cp.async tile
// stream: per k-chunk two weight tiles (gate, up) flow through a double
// buffer while the x tile rides its own pair; compute overlaps loads.
template<int NT_TILE>
__global__ void glu_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ g_w,
                                const __nv_bfloat16* __restrict__ u_w,
                                __nv_bfloat16* __restrict__ out,
                                int M,
                                int N,
                                int K)
{
    using namespace nvcuda;
    // dynamic smem: xs[2][MT][KT+8] | ws[2][NT][KT+8] | out_s, out_u [MT][NT+8]
    extern __shared__ char smem_raw[];
    const int XS_STRIDE = GRP_KT_GLU + 8;
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* ws = xs + 2 * GRP_MT * XS_STRIDE;
    // ws holds 2 slots x (gate,up) tiles = 4*NT rows before the outputs
    float* out_s = reinterpret_cast<float*>(ws + 4 * NT_TILE * XS_STRIDE);
    float* out_u = out_s + GRP_MT * (NT_TILE + 8);

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int n0 = blockIdx.x * NT_TILE;
    if (n0 >= N) return;
    int m0w = (warp >> 2) * (GRP_MT / 2);
    int n0w = (warp & 3) * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accg, accu;
    wmma::fill_fragment(accg, 0.f);
    wmma::fill_fragment(accu, 0.f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;

    const int nchunks = (K + GRP_KT_GLU - 1) / GRP_KT_GLU;
    // stage chunk c: x into xs[c&1], gate tile into ws[0], up tile into ws[1]
    // (both weight tiles for the chunk issue together; the slot pair rotates
    // per chunk so chunk c+1 lands while chunk c computes)
    auto stage = [&](int c) {
        int k0 = c * GRP_KT_GLU;
        int kn = min(GRP_KT_GLU, K - k0);
        int kn8 = kn >> 3;
        __nv_bfloat16* xbuf = xs + (long)(c & 1) * GRP_MT * XS_STRIDE;
        __nv_bfloat16* gw_buf = ws + (long)(c & 1) * 2 * NT_TILE * XS_STRIDE;
        __nv_bfloat16* uw_buf = gw_buf + (long)NT_TILE * XS_STRIDE;
        for (int idx = tid; idx < M * kn8; idx += GRP_TPB) {
            int m = idx / kn8, j = idx % kn8;
            __pipeline_memcpy_async(xbuf + (long)m * XS_STRIDE + j * 8, x + (long)m * K + k0 + (long)j * 8, 16);
        }
        for (int idx = tid; idx < NT_TILE * kn8; idx += GRP_TPB) {
            int r = idx / kn8, j = idx % kn8;
            if (n0 + r < N) {
                __pipeline_memcpy_async(gw_buf + (long)r * XS_STRIDE + j * 8,
                                        g_w + (long)(n0 + r) * K + k0 + (long)j * 8, 16);
                __pipeline_memcpy_async(uw_buf + (long)r * XS_STRIDE + j * 8,
                                        u_w + (long)(n0 + r) * K + k0 + (long)j * 8, 16);
            }
        }
        if (kn < GRP_KT_GLU) {
            // partial chunk: zero tails (plain stores; synced below)
            for (int idx = tid; idx < GRP_MT * GRP_KT_GLU; idx += GRP_TPB) {
                int r = idx / GRP_KT_GLU, cc = idx % GRP_KT_GLU;
                if (cc >= kn && r < M) xbuf[r * XS_STRIDE + cc] = __float2bfloat16_rn(0.f);
            }
            for (int idx = tid; idx < 2 * NT_TILE * GRP_KT_GLU; idx += GRP_TPB) {
                int r = idx / GRP_KT_GLU, cc = idx % GRP_KT_GLU;
                if (cc >= kn && n0 + (r % NT_TILE) < N) gw_buf[r * XS_STRIDE + cc] = __float2bfloat16_rn(0.f);
            }
        }
        __pipeline_commit();
    };

    stage(0);
    for (int c = 0; c < nchunks; c++) {
        if (c + 1 < nchunks) stage(c + 1);
        __pipeline_wait_prior(c + 1 < nchunks ? 1 : 0);
        __syncthreads();
        __nv_bfloat16* xbuf = xs + (long)(c & 1) * GRP_MT * XS_STRIDE;
        __nv_bfloat16* gw_buf = ws + (long)(c & 1) * 2 * NT_TILE * XS_STRIDE;
        __nv_bfloat16* uw_buf = gw_buf + (long)NT_TILE * XS_STRIDE;
#pragma unroll 4
        for (int kk = 0; kk < GRP_KT_GLU; kk += 16) {
            wmma::load_matrix_sync(fa, xbuf + m0w * XS_STRIDE + kk, XS_STRIDE);
            wmma::load_matrix_sync(fb, gw_buf + n0w * XS_STRIDE + kk, XS_STRIDE);
            wmma::mma_sync(accg, fa, fb, accg);
        }
#pragma unroll 4
        for (int kk = 0; kk < GRP_KT_GLU; kk += 16) {
            wmma::load_matrix_sync(fa, xbuf + m0w * XS_STRIDE + kk, XS_STRIDE);
            wmma::load_matrix_sync(fb, uw_buf + n0w * XS_STRIDE + kk, XS_STRIDE);
            wmma::mma_sync(accu, fa, fb, accu);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(out_s + m0w * (NT_TILE + 8) + n0w, accg, NT_TILE + 8, wmma::mem_row_major);
    wmma::store_matrix_sync(out_u + m0w * (NT_TILE + 8) + n0w, accu, NT_TILE + 8, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < M * NT_TILE; idx += GRP_TPB) {
        int m = idx / NT_TILE, c = idx % NT_TILE;
        if (n0 + c < N) {
            float v = silu_f(out_s[m * (NT_TILE + 8) + c]) * out_u[m * (NT_TILE + 8) + c];
            out[(long)m * N + n0 + c] = __float2bfloat16_rn(v);
        }
    }
}

std::vector<at::Tensor> grouped_gemm_impl(at::Tensor x,
                                          at::Tensor w0,
                                          at::Tensor w1,
                                          at::Tensor w2,
                                          at::Tensor w3,
                                          int64_t n_mat,
                                          bool want_separate)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16, "x must be CUDA bf16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2 && x.size(0) > 1 && x.size(0) <= GRP_MT, "batch must be 2..32");
    int M = (int)x.size(0), K = (int)x.size(1);
    at::Tensor ws[4] = {w0, w1, w2, w3};
    // the vectorized smem staging does 16-byte transactions
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "x pointer must be 16B aligned");
    for (int i = 0; i < n_mat; i++) {
        TORCH_CHECK(ws[i].is_cuda() && ws[i].scalar_type() == at::ScalarType::BFloat16, "weights must be CUDA bf16");
        TORCH_CHECK(ws[i].is_contiguous(), "weights must be contiguous");
        TORCH_CHECK(ws[i].size(1) == K, "weight K mismatch");
        TORCH_CHECK((reinterpret_cast<uintptr_t>(ws[i].data_ptr()) & 15) == 0, "weight pointer must be 16B aligned");
    }
    int fs[4] = {0, 0, 0, 0};
    int total = 0;
    for (int i = 0; i < n_mat; i++) {
        fs[i] = (int)ws[i].size(0);
        total += fs[i];
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    int blocks = (total + GRP_NT - 1) / GRP_NT;  // grid covers both variants
    auto opts = x.options();
    std::vector<at::Tensor> outs;
    at::Tensor o0, o1, o2, o3;
    if (want_separate) {
        for (int i = 0; i < n_mat; i++) outs.push_back(at::empty({M, fs[i]}, opts));
        o0 = n_mat > 0 ? outs[0] : at::Tensor();
        o1 = n_mat > 1 ? outs[1] : at::Tensor();
        o2 = n_mat > 2 ? outs[2] : at::Tensor();
        o3 = n_mat > 3 ? outs[3] : at::Tensor();
    } else {
        o0 = at::empty({M, total}, opts);
        outs.push_back(o0);
    }
    at::Tensor dummy = o0.defined() ? o0 : x;  // unused slots point at a live buffer
        static bool grp_attr_set = false;
    if (!grp_attr_set) {
        cudaFuncSetAttribute(grouped_linear_kernel<GRP_NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, GRP_SMEM_BYTES);
        grp_attr_set = true;
    }
grouped_linear_kernel<GRP_NT><<<blocks, GRP_TPB, GRP_SMEM_BYTES, stream.stream()>>>(
        (const __nv_bfloat16*)x.data_ptr(),
        (const __nv_bfloat16*)(n_mat > 0 ? w0 : dummy).data_ptr(),
        (const __nv_bfloat16*)(n_mat > 1 ? w1 : dummy).data_ptr(),
        (const __nv_bfloat16*)(n_mat > 2 ? w2 : dummy).data_ptr(),
        (const __nv_bfloat16*)(n_mat > 3 ? w3 : dummy).data_ptr(),
        (__nv_bfloat16*)(want_separate && o0.defined() ? o0 : o0).data_ptr(),
        (__nv_bfloat16*)(want_separate && o1.defined() ? o1 : o0).data_ptr(),
        (__nv_bfloat16*)(want_separate && o2.defined() ? o2 : o0).data_ptr(),
        (__nv_bfloat16*)(want_separate && o3.defined() ? o3 : o0).data_ptr(),
        fs[0], fs[1], fs[2], fs[3], M, K, total, want_separate ? 0 : 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outs;
}


// Single-token gated-delta-rule recurrent update, IN PLACE on the cache
// state. Mirrors FLA's fused_recurrent_gated_delta_rule for T=1 with
// precomputed beta/g and in-kernel qk-l2norm; the state rides in shared
// memory so the four passes (FLA read+write, HF copy-back read+write)
// collapse to one read + one write.
__global__ void gdn_recurrent_step_kernel(const __nv_bfloat16* __restrict__ q,
                                          const __nv_bfloat16* __restrict__ k,
                                          const __nv_bfloat16* __restrict__ v,
                                          const __nv_bfloat16* __restrict__ g,
                                          const __nv_bfloat16* __restrict__ beta,
                                          float* __restrict__ state,
                                          __nv_bfloat16* __restrict__ out,
                                          int K,
                                          int V)
{
    extern __shared__ float smem[];  // state [K*V], then q[K], k[K], v[V]
    float* h = smem;
    float* qs = h + (long)K * V;
    float* ks = qs + K;
    float* vs = ks + K;

    const int bh = blockIdx.x;  // (batch, v-head) flattened
    const int tid = threadIdx.x;
    const float scale = rsqrtf((float)K);
    float* g_state = state + (long)bh * K * V;
    for (int idx = tid; idx < K * V; idx += blockDim.x) h[idx] = g_state[idx];
    for (int i = tid; i < K; i += blockDim.x) {
        qs[i] = __bfloat162float(q[(long)bh * K + i]);
        ks[i] = __bfloat162float(k[(long)bh * K + i]);
    }
    for (int i = tid; i < V; i += blockDim.x) vs[i] = __bfloat162float(v[(long)bh * V + i]);
    __syncthreads();

    // l2-normalize q (scaled) and k; eps matches FLA
    __shared__ float s_red[2];
    if (tid == 0) {
        s_red[0] = 0.f;
        s_red[1] = 0.f;
    }
    __syncthreads();
    float sq = 0.f, sk = 0.f;
    for (int i = tid; i < K; i += blockDim.x) {
        sq += qs[i] * qs[i];
        sk += ks[i] * ks[i];
    }
    atomicAdd(&s_red[0], sq);
    atomicAdd(&s_red[1], sk);
    __syncthreads();
    const float inv_qq = rsqrtf(s_red[0] + 1e-6f) * scale;
    const float inv_kk = rsqrtf(s_red[1] + 1e-6f);
    __syncthreads();

    const float decay = __expf(__bfloat162float(g[bh]));
    const float b_beta = __bfloat162float(beta[bh]);

    // one thread per state column j: kv = k^T (decay*h), then the
    // outer-add update and the output dot in a second pass
    if (tid < V) {
        const int j = tid;
        float kv = 0.f;
        for (int i = 0; i < K; i++) kv += ks[i] * h[i * V + j];
        kv *= decay * inv_kk;
        const float bvj = b_beta * (vs[j] - kv);
        float o = 0.f;
        for (int i = 0; i < K; i++) {
            const float khat = ks[i] * inv_kk;
            const float hij = decay * h[i * V + j] + khat * bvj;
            h[i * V + j] = hij;
            o += qs[i] * inv_qq * hij;
        }
        out[(long)bh * V + j] = __float2bfloat16_rn(o);
    }
    __syncthreads();
    for (int idx = tid; idx < K * V; idx += blockDim.x) g_state[idx] = h[idx];
}

void gdn_recurrent_step(at::Tensor q,
                        at::Tensor k,
                        at::Tensor v,
                        at::Tensor g,
                        at::Tensor beta,
                        at::Tensor state,
                        at::Tensor out)
{
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::ScalarType::BFloat16, "q must be CUDA bf16");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous");
    TORCH_CHECK(state.is_contiguous() && state.scalar_type() == at::ScalarType::Float, "state must be fp32 contiguous");
    const int B = (int)q.size(0), H = (int)q.size(1), K = (int)q.size(2), V = (int)v.size(2);
    TORCH_CHECK(k.sizes() == q.sizes() && v.size(0) == B && v.size(1) == H && v.size(2) == V, "shape mismatch");
    TORCH_CHECK(state.size(0) == B && state.size(1) == H && state.size(2) == K && state.size(3) == V,
                "state must be [B,H,K,V]");
    TORCH_CHECK(V <= 1024, "V must fit one-thread-per-column");
    const int smem = (K * V + 2 * K + V) * 4;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(gdn_recurrent_step_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    gdn_recurrent_step_kernel<<<B * H, (V + 31) / 32 * 32, smem, stream.stream()>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(),
        (const __nv_bfloat16*)v.data_ptr(), (const __nv_bfloat16*)g.data_ptr(),
        (const __nv_bfloat16*)beta.data_ptr(), (float*)state.data_ptr(),
        (__nv_bfloat16*)out.data_ptr(), K, V);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


py::dict segki_capabilities()
{
    py::dict d;
    d["version"] = 1;
    d["dtypes"] = py::make_tuple("bf16");
    d["max_grouped_batch"] = GRP_MT;  // above this the dispatcher routes to aten
    d["fused_glu_epilogue"] = true;
    d["fused_rope_q"] = true;
    d["fused_rope_k"] = true;
    d["fused_cache_write"] = true;
    d["inplace_recurrent"] = true;
    return d;
}

at::Tensor single_gemm(at::Tensor x, at::Tensor w)
{
    return grouped_gemm_impl(x, w, w, w, w, 1, true)[0];
}

std::vector<at::Tensor> triple_gemm(at::Tensor x, at::Tensor qw, at::Tensor kw, at::Tensor vw)
{
    return grouped_gemm_impl(x, qw, kw, vw, qw, 3, true);
}

at::Tensor quad_gemm(at::Tensor x, at::Tensor w_qkv, at::Tensor w_z, at::Tensor w_b, at::Tensor w_a)
{
    return grouped_gemm_impl(x, w_qkv, w_z, w_b, w_a, 4, false)[0];
}

at::Tensor dual_gemm_silu_mul(at::Tensor x, at::Tensor gate_w, at::Tensor up_w)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16, "x must be CUDA bf16");
    TORCH_CHECK(x.is_contiguous() && gate_w.is_contiguous() && up_w.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2 && x.size(0) > 1 && x.size(0) <= GRP_MT, "batch must be 2..32");
    int M = (int)x.size(0), K = (int)x.size(1), N = (int)gate_w.size(0);
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0
                    && (reinterpret_cast<uintptr_t>(gate_w.data_ptr()) & 15) == 0
                    && (reinterpret_cast<uintptr_t>(up_w.data_ptr()) & 15) == 0,
                "inputs must be 16B aligned");
    TORCH_CHECK(up_w.size(0) == N && gate_w.size(1) == K && up_w.size(1) == K, "shape mismatch");
    auto out = at::empty({M, N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    int blocks = (N + GRP_NT - 1) / GRP_NT;
        static bool glu_attr_set = false;
    if (!glu_attr_set) {
        cudaFuncSetAttribute(glu_gemm_kernel<GRP_NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, GRP_GLU_SMEM_BYTES);
        glu_attr_set = true;
    }
    glu_gemm_kernel<GRP_NT><<<blocks, GRP_TPB, GRP_GLU_SMEM_BYTES, stream.stream()>>>((const __nv_bfloat16*)x.data_ptr(),
                                                             (const __nv_bfloat16*)gate_w.data_ptr(),
                                                             (const __nv_bfloat16*)up_w.data_ptr(),
                                                             (__nv_bfloat16*)out.data_ptr(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Arch-adaptive batch routing threshold: on Ada the WMMA grouped kernel
// beats cuBLAS for M<=32 (launch overhead dominates); on Ampere cuBLAS
// is fast enough to win at any M>1 (profiled on A100 SXM4: our WMMA at
// 39.7% of step vs cuBLAS composite scaling better). Same interface,
// same kernels — only the crossover point moves.
static int64_t grouped_wmma_threshold()
{
    static int64_t thr = -1;
    if (thr < 0) {
        int dev;
        cudaGetDevice(&dev);
        int smem;
        cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        thr = (smem > 100 * 1024) ? 1 : GRP_MT;  // Ampere -> 1, Ada -> GRP_MT
    }
    return thr;
}

// Portable segKI interface: one dispatcher routes by weight count and
// batch size to this backend's private kernels (gemv / WMMA grouped /
// aten composite), so other accelerators can expose the same entry with
// an entirely different internal split.
std::vector<at::Tensor> grouped_linear(at::Tensor x, std::vector<at::Tensor> weights, bool glu_epilogue)
{
    const size_t n = weights.size();
    TORCH_CHECK(n >= 1 && n <= 6, "grouped_linear takes 1..6 weight matrices");
    if (n == 6) {
        // TP block-sliced GDN in_proj [q, k, v, z, b, a]: route through this
        // interface so TP=1 (fused 4-mat) and TP=2 (block-sliced 6-mat)
        // share one dispatch path in the forward
        auto qkv = grouped_linear(x, {weights[0], weights[1], weights[2]}, false);
        auto zba = grouped_linear(x, {weights[3], weights[4], weights[5]}, false);
        at::Tensor fused = at::cat({qkv[0], qkv[1], qkv[2], zba[0], zba[1], zba[2]}, -1);
        return {fused};
    }
    const int64_t K = weights[0].size(1);
    // M == 1 rides the flat GEMV kernels (1-D input or a single row)
    if (x.numel() == K) {
        at::Tensor flat = x.reshape({-1});
        if (glu_epilogue && n == 2) {
            auto out = at::empty({weights[0].size(0)}, x.options());
            dual_gemv_silu_mul(flat, weights[0], weights[1], out);
            return {out};
        }
        if (!glu_epilogue && n == 3) return triple_gemv(flat, weights[0], weights[1], weights[2]);
        if (!glu_epilogue && n == 4) {
            auto out = at::empty({weights[0].size(0) + weights[1].size(0) + weights[2].size(0) + weights[3].size(0)},
                                 x.options());
            quad_gemv(flat, weights[0], weights[1], weights[2], weights[3], out);
            return {out};
        }
        // n == 1 or n == 2 without epilogue: aten composite
        std::vector<at::Tensor> outs;
        for (auto& w : weights) outs.push_back(at::matmul(flat, w.t()));
        if (glu_epilogue && n == 2) {
            at::Tensor g = at::silu(outs[0].to(at::kFloat)).to(outs[0].scalar_type());
            outs = {g * outs[1]};
        }
        if (!glu_epilogue && n == 4) outs = {at::cat(outs, -1)};  // quad contract: concatenated
        return outs;
    }
    at::Tensor x2 = x.reshape({-1, K});
    const int64_t M = x2.size(0);
    // Fused GLU rides the WMMA kernel on EVERY architecture up to the tile
    // row limit: the in-register silu epilogue saves three launches plus an
    // fp32 round-trip vs composite (loop 3 regression: composite -2.3%).
    // M in (GRP_MT, 2*GRP_MT] chunks into two fused calls — still beats
    // the 4-kernel composite (bs=64 decodes hit this).
    if (glu_epilogue && n == 2 && M <= GRP_MT) return {dual_gemm_silu_mul(x2, weights[0], weights[1])};
    if (glu_epilogue && n == 2 && M <= 2 * GRP_MT) {
        const int64_t half = GRP_MT;
        at::Tensor out = at::empty({M, weights[0].size(0)}, x2.options());
        auto o1 = dual_gemm_silu_mul(x2.narrow(0, 0, half), weights[0], weights[1]);
        auto o2 = dual_gemm_silu_mul(x2.narrow(0, half, M - half), weights[0], weights[1]);
        out.narrow(0, 0, half).copy_(o1);
        out.narrow(0, half, M - half).copy_(o2);
        return {out};
    }
    if (M > grouped_wmma_threshold()) {
        // large batches: cuBLAS/aten composite (this backend's measured choice)
        std::vector<at::Tensor> outs;
        for (auto& w : weights) outs.push_back(at::matmul(x2, w.t()));
        if (glu_epilogue && n == 2) {
            at::Tensor g = at::silu(outs[0].to(at::kFloat)).to(outs[0].scalar_type());
            outs = {g * outs[1]};
        }
        if (!glu_epilogue && n == 4) outs = {at::cat(outs, -1)};  // quad contract: concatenated
        return outs;
    }
    TORCH_CHECK(x2.is_contiguous(), "grouped_linear needs a contiguous [M,K] view");
    if (glu_epilogue && n == 2) return {dual_gemm_silu_mul(x2, weights[0], weights[1])};
    if (!glu_epilogue && n == 3) return triple_gemm(x2, weights[0], weights[1], weights[2]);
    if (!glu_epilogue && n == 4) return {quad_gemm(x2, weights[0], weights[1], weights[2], weights[3])};
    if (!glu_epilogue && n == 1) return {single_gemm(x2, weights[0])};
    // glu with n != 2 or multi-output without epilogue at n == 2: composite
    std::vector<at::Tensor> outs;
    for (auto& w : weights) outs.push_back(at::matmul(x2, w.t()));
    if (glu_epilogue && n == 2) {
            at::Tensor g = at::silu(outs[0].to(at::kFloat)).to(outs[0].scalar_type());
            outs = {g * outs[1]};
        }
    return outs;
}


namespace {
// Head-repeat for the GDN scan's v-head layout: out[b,s,hv,d] = in[b,s,hv/ratio,d].
// One launch covers BOTH q and k (idx space 2*total), replacing two eager
// repeat_interleave copies per layer.
__global__ void repeat_qk_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 __nv_bfloat16* __restrict__ q_out,
                                 __nv_bfloat16* __restrict__ k_out,
                                 long long src_rows,
                                 int dk,
                                 int hv,
                                 int ratio,
                                 long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const bool is_k = idx >= total;
    const long long i = is_k ? idx - total : idx;
    if (i >= total) return;
    const int d = (int)(i % dk);
    long long t = i / dk;
    const int h = (int)(t % hv);
    t /= hv;  // flat [B*S] row
    const long long src = (t * src_rows + h / ratio) * dk + d;
    if (is_k)
        k_out[i] = k[src];
    else
        q_out[i] = q[src];
}
}  // namespace

std::vector<at::Tensor> repeat_qk(at::Tensor q, at::Tensor k, int64_t ratio)
{
    const auto sz = q.sizes();
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "repeat_qk needs contiguous q/k");
    const long long B = sz[0], S = sz[1], Hk = sz[2];
    const int dk = (int)sz[3];
    const int hv = (int)(Hk * ratio);
    auto q_out = at::empty({B, S, (long long)hv, (long long)dk}, q.options());
    auto k_out = at::empty_like(q_out);
    const long long total = B * S * hv * dk;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)((2 * total + 255) / 256);
    repeat_qk_kernel<<<blocks, 256, 0, stream.stream()>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(),
        (__nv_bfloat16*)q_out.data_ptr(), (__nv_bfloat16*)k_out.data_ptr(), B * S * Hk, dk, hv, (int)ratio,
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_out, k_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // two explicit overloads: at::Tensor() pybind defaults proved fragile
    // in this toolchain (the 9-arg call failed to bind)
    m.def(
        "decode_attn",
        [](at::Tensor q,
           at::Tensor K,
           at::Tensor V,
           at::Tensor write_pos,
           at::Tensor out,
           int64_t num_q_heads,
           int64_t num_kv_heads,
           int64_t head_dim,
           int64_t max_len) {
            return decode_attn(q,
                               K,
                               V,
                               write_pos,
                               out,
                               num_q_heads,
                               num_kv_heads,
                               head_dim,
                               max_len,
                               {},
                               {},
                               {},
                               {},
                               {},
                               {},
                               {},
                               1e-6,
                               0);
        },
        "batched decode attention (GQA, valid-length)",
        py::arg("q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("write_pos"),
        py::arg("out"),
        py::arg("num_q_heads"),
        py::arg("num_kv_heads"),
        py::arg("head_dim"),
        py::arg("max_len"));
    m.def("grouped_linear",
          &grouped_linear,
          "portable segKI projection: 1..4 original weight matrices, one "
          "launch per group, GLU epilogue optional; routes internally by "
          "batch to gemv / grouped WMMA / aten composite",
          py::arg("x"),
          py::arg("weights"),
          py::arg("glu_epilogue") = false);
    m.def("repeat_qk", &repeat_qk, "fused q/k head-repeat for the scan (one launch)");
    m.def("capabilities", &segki_capabilities, "segKI backend capability report");
    m.def("gdn_recurrent_step",
          &gdn_recurrent_step,
          "in-place single-token gated-delta-rule recurrent update",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("g"),
          py::arg("beta"),
          py::arg("state"),
          py::arg("out"));
    m.def("single_gemm",
          &single_gemm,
          "single-matrix GEMM on the grouped kernel (small batches)",
          py::arg("x"),
          py::arg("w"));
    m.def("triple_gemm",
          &triple_gemm,
          "grouped q/k/v GEMM for small batches (2..32), no weight copies",
          py::arg("x"),
          py::arg("q_w"),
          py::arg("k_w"),
          py::arg("v_w"));
    m.def("quad_gemm",
          &quad_gemm,
          "grouped GDN input-projection GEMM (qkv|z|b|a) -> concat, small batches",
          py::arg("x"),
          py::arg("w_qkv"),
          py::arg("w_z"),
          py::arg("w_b"),
          py::arg("w_a"));
    m.def("dual_gemm_silu_mul",
          &dual_gemm_silu_mul,
          "grouped gate/up GEMM with silu(g)*u epilogue, small batches",
          py::arg("x"),
          py::arg("gate_w"),
          py::arg("up_w"));
    m.def("decode_attn",
          &decode_attn,
          "batched decode attention (GQA, valid-length, fused q-norm/rope/gate)",
          py::arg("q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("write_pos"),
          py::arg("out"),
          py::arg("num_q_heads"),
          py::arg("num_kv_heads"),
          py::arg("head_dim"),
          py::arg("max_len"),
          py::arg("q_norm_w"),
          py::arg("cos_t"),
          py::arg("sin_t"),
          py::arg("gate"),
          py::arg("k_raw"),
          py::arg("v_raw"),
          py::arg("k_norm_w"),
          py::arg("eps"),
          py::arg("rope_dim"));
    m.def("decode_step_graph", &decode_step_graph, "graph-capturable decode step (CUDA)");
    m.def("gdn_gates", &gdn_gates, "fused GDN beta/g gating (CUDA)");
    m.def("quad_gemv", &quad_gemv, "b=1 GDN 4-weight input GEMV (qkv|z|b|a), no concat (CUDA)");
    m.def("dual_gemv_silu_mul",
          &dual_gemv_silu_mul,
          "b=1 GEMV reading gate|up weights separately: silu(h*Wg)*(h*Wu) (CUDA)");
    m.def("fused_add_norm",
          &fused_add_norm,
          "fused residual add + RMSNorm, in-place residual update (CUDA)");
    m.def(
        "gated_norm",
        [](at::Tensor x, at::Tensor z, at::Tensor w, at::Tensor out, double eps) {
            TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16,
                        "x must be CUDA bf16");
            TORCH_CHECK(
                x.is_contiguous() && z.is_contiguous() && w.is_contiguous() && out.is_contiguous(),
                "inputs must be contiguous");
            TORCH_CHECK(
                z.numel() == x.numel() && out.numel() == x.numel() && w.numel() == x.size(-1),
                "shape mismatch");
            int64_t rows = x.numel() / x.size(-1);
            int dim = (int)x.size(-1);
            TORCH_CHECK((int64_t)dim * sizeof(float) <= 48 * 1024,
                        "dim too large for shared memory");
            auto stream = at::cuda::getCurrentCUDAStream();
            int threads = std::min(1024, ((dim + 31) / 32) * 32);
            if (threads == 0) threads = 32;
            gated_norm_kernel<<<(int)rows, threads, (size_t)dim * sizeof(float), stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(z.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                (float)eps,
                dim);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        },
        "GDN z-gated output norm: (w * rms(x)) * silu(z)",
        py::arg("x"),
        py::arg("z"),
        py::arg("w"),
        py::arg("out"),
        py::arg("eps"));
    m.def("triple_gemv",
          &triple_gemv,
          "b=1 QKV GEMV reading q/k/v weights directly, no concat (CUDA)");
}