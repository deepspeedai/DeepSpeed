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
__global__ void decode_step_kernel(const __nv_bfloat16* __restrict__ logits,
                                   int64_t* __restrict__ token_out,
                                   int64_t* __restrict__ write_pos,
                                   bool* __restrict__ mask,
                                   int64_t* __restrict__ out_buf,
                                   int step,
                                   int vocab_size,
                                   int max_len)
{
    __shared__ int s_idx[1024];
    __shared__ float s_val[1024];

    int tid = threadIdx.x;
    int local_idx = 0;
    float local_val = -INFINITY;

    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = __bfloat162float(logits[v]);
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
        token_out[0] = best;
        out_buf[step] = best;
        int64_t new_pos = write_pos[0] + 1;
        write_pos[0] = new_pos;
        if (new_pos < max_len) { mask[new_pos] = true; }
    }
}

void decode_step(at::Tensor logits,
                 at::Tensor token_out,
                 at::Tensor write_pos,
                 at::Tensor mask,
                 at::Tensor out_buf,
                 int64_t step)
{
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::ScalarType::BFloat16,
                "logits must be CUDA bf16");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    int vocab = (int)logits.numel();
    int max_len = (int)mask.size(-1);
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_step_kernel<<<1, 1024, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        token_out.data_ptr<int64_t>(),
        write_pos.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        out_buf.data_ptr<int64_t>(),
        (int)step,
        vocab,
        max_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Graph-capturable variant: no host-side step parameter. The token output
// index is derived from the GPU-resident write_pos, which the kernel itself
// advances, making the whole step self-contained for CUDA graph replay.
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

    // Each lane accumulates every 32nd element (coalesced across lanes).
    float gate_acc = 0.0f, up_acc = 0.0f;
    for (int i = lane; i < in_features; i += 32) {
        float h = __bfloat162float(hidden[i]);
        gate_acc += h * __bfloat162float(gate_row[i]);
        up_acc += h * __bfloat162float(up_row[i]);
    }

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

    float acc = 0.0f;
    for (int i = lane; i < in_features; i += 32) {
        acc += __bfloat162float(hidden[i]) * __bfloat162float(row[i]);
    }
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
                       const __nv_bfloat16* __restrict__ K,
                       const __nv_bfloat16* __restrict__ V,
                       const int64_t* __restrict__ write_pos,
                       __nv_bfloat16* __restrict__ out,
                       int num_q_heads,
                       int num_kv_heads,
                       int max_len,
                       float scale,
                       int pos_stride,
                       const float* __restrict__ q_norm_w,      // [HEAD_DIM], pre-scaled (1+w)
                       const float* __restrict__ cos_t,         // [max_len, rope_dim]
                       const float* __restrict__ sin_t,         // [max_len, rope_dim]
                       const __nv_bfloat16* __restrict__ gate,  // q layout
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

    const __nv_bfloat16* Kg = K + ((long)(row * num_kv_heads + kv_head)) * max_len * HEAD_DIM;
    const __nv_bfloat16* Vg = V + ((long)(row * num_kv_heads + kv_head)) * max_len * HEAD_DIM;
    const __nv_bfloat16* q_row = q + ((long)(row * num_q_heads + q_head)) * HEAD_DIM;
    const __nv_bfloat16* g_row = gate + ((long)(row * num_q_heads + q_head)) * HEAD_DIM;

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
                 double eps,
                 int64_t rope_dim)
{
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::ScalarType::BFloat16, "q must be CUDA bf16");
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
    TORCH_CHECK(!use_gate || gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(
        !fuse_q || (cos_t.defined() && sin_t.defined() &&
                    cos_t.numel() == (int64_t)max_len * rope_dim && sin_t.numel() == cos_t.numel()),
        "fused q path needs cos/sin tables of [max_len, rope_dim]");
    TORCH_CHECK(!fuse_q || (rope_dim % (2 * (head_dim / 32)) == 0),
                "rope_dim/2 must be a multiple of the per-lane element count");

    float scale = 1.0f / sqrtf((float)head_dim);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    auto k_ptr = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
    auto v_ptr = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
    auto o_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
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
                (int)max_len,
                scale,
                pos_stride,
                qn_ptr,
                cos_ptr,
                sin_ptr,
                g_ptr,
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
                    (int)max_len,
                    scale,
                    pos_stride,
                    qn_ptr,
                    cos_ptr,
                    sin_ptr,
                    g_ptr,
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
                    (int)max_len,
                    scale,
                    pos_stride,
                    qn_ptr,
                    cos_ptr,
                    sin_ptr,
                    g_ptr,
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

        float acc = 0.0f;
        for (int i = lane; i < in_f; i += 32)
            acc += __bfloat162float(hidden[i]) * __bfloat162float(w_ptr[(long)row * in_f + i]);
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
          py::arg("eps"),
          py::arg("rope_dim"));
    m.def("decode_step", &decode_step, "fused decode step update (CUDA)");
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
