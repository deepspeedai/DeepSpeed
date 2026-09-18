// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// Native CUDA kernel for the segment-KI fused_glu op: computes
// out = silu(hidden[:, :k]) * hidden[:, k:2k] on the fused gate/up GEMM
// output without materializing chunked views. hidden is contiguous
// [N, 2k]; out is contiguous [N, k]. Activation math runs in fp32 with a
// single rounding to the storage dtype (torch opmath convention).

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define SILU(x) ((x) / (1.0f + expf(-(x))))

__global__ void silu_mul_halves_fp32(const float* __restrict__ hidden,
                                     float* __restrict__ out,
                                     int64_t total,
                                     int64_t k)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int64_t row = i / k;
        int64_t col = i - row * k;
        int64_t base = row * (2 * k);
        out[i] = SILU(hidden[base + col]) * hidden[base + k + col];
    }
}

__global__ void silu_mul_halves_bf16(const __nv_bfloat16* __restrict__ hidden,
                                     __nv_bfloat16* __restrict__ out,
                                     int64_t total,
                                     int64_t k)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int64_t row = i / k;
        int64_t col = i - row * k;
        int64_t base = row * (2 * k);
        float g = __bfloat162float(hidden[base + col]);
        float u = __bfloat162float(hidden[base + k + col]);
        out[i] = __float2bfloat16(SILU(g) * u);
    }
}

at::Tensor fused_silu_mul_halves(at::Tensor hidden)
{
    TORCH_CHECK(hidden.is_cuda(), "fused_silu_mul_halves is CUDA-only");
    TORCH_CHECK(hidden.dim() >= 1 && hidden.size(-1) % 2 == 0,
                "last dim must be even (gate/up layout)");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    auto sizes = hidden.sizes().vec();
    sizes.back() /= 2;
    auto out = at::empty(sizes, hidden.options());
    int64_t k = sizes.back();
    int64_t total = out.numel();
    if (total == 0) return out;

    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    TORCH_CHECK(blocks <= INT32_MAX, "fused_silu_mul_halves: tensor too large");

    if (hidden.scalar_type() == at::ScalarType::Float) {
        silu_mul_halves_fp32<<<blocks, threads, 0, stream>>>(
            hidden.data_ptr<float>(), out.data_ptr<float>(), total, k);
    } else if (hidden.scalar_type() == at::ScalarType::BFloat16) {
        silu_mul_halves_bf16<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            total,
            k);
    } else {
        TORCH_CHECK(false, "fused_silu_mul_halves supports float32 and bfloat16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

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
        // Numerically stable softplus: for large x, log(1+exp(x)) == x to fp32
        // precision and the naive form loses significant digits.
        float sp = (sp_x > 20.0f) ? sp_x : logf(1.0f + expf(sp_x));
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
            if (s_val[tid + stride] > s_val[tid]) {
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
        if (new_pos + 1 < max_len) { mask[new_pos + 1] = true; }
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
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t best = (int64_t)s_idx[0];
        token_out[0] = best;
        int64_t pos = write_pos[0];
        out_buf[pos] = best;
        int64_t new_pos = pos + 1;
        write_pos[0] = new_pos;
        if (new_pos + 1 < max_len) { mask[new_pos + 1] = true; }
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
    int vocab = (int)logits.numel();
    int max_len = (int)mask.size(-1);
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_step_graph_kernel<<<1, 1024, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        token_out.data_ptr<int64_t>(),
        write_pos.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        out_buf.data_ptr<int64_t>(),
        vocab,
        max_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decode_step", &decode_step, "fused decode step update (CUDA)");
    \n m.def("decode_step_graph", &decode_step_graph, "graph-capturable decode step (CUDA)");
    m.def("gdn_gates", &gdn_gates, "fused GDN beta/g gating (CUDA)");
    m.def("fused_silu_mul_halves",
          &fused_silu_mul_halves,
          "fused silu(gate)*up on [N, 2k] hidden (CUDA)");
}
