// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// Native CUDA kernel for the segment-KI fused_glu op: computes
// out = silu(hidden[:, :k]) * hidden[:, k:2k] on the fused gate|up GEMM
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
                "last dim must be even (gate|up layout)");
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
                                 int num_heads)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int64_t tok = i / num_heads;
        int h = (int)(i - tok * num_heads);
        float av = __bfloat162float(a[i]);
        float bv = __bfloat162float(b[i]);
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
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a/b must be contiguous");
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
        heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {beta, g};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gdn_gates", &gdn_gates, "fused GDN beta/g gating (CUDA)");
    m.def("fused_silu_mul_halves",
          &fused_silu_mul_halves,
          "fused silu(gate)*up on [N, 2k] hidden (CUDA)");
}
