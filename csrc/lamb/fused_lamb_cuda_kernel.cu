// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
// #include "ATen/Type.h"
#include "ATen/AccumulateType.h"

#include <iostream>

// #include <helper_functions.h>
#if defined(__HIP_PLATFORM_HCC__) && HIP_VERSION > 305
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
#include <cuda_runtime_api.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template <typename T>
struct SharedMemory {
    // Ensure that we won't compile any un-specialized types
    __device__ inline operator T*()
    {
#ifndef _WIN32
        extern __device__ void error(void);
        error();
#endif
        return NULL;
    }
};

template <>
struct SharedMemory<float> {
    __device__ inline operator float*()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory<double> {
    __device__ inline operator double*()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};
}  // namespace

#include "type_shim.h"

typedef enum {
    ADAM_MODE_0 = 0,  // eps under square root
    ADAM_MODE_1 = 1   // eps outside square root
} adamMode_t;

// s_a and s_b are in shared memory
// g_a and g_b are in shared memory
template <typename T, int blockSize>
__device__ void reduce_block_in_shared_memory(T* s_a, T* s_b, T* g_a, T* g_b)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // perform block reduction in shared memory,
    unsigned int tid = cta.thread_rank();

    T a_sum = s_a[tid];
    T b_sum = s_b[tid];

    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 256];
        s_b[tid] = b_sum = b_sum + s_b[tid + 256];
    }

    cg::sync(cta);

    if ((blockSize >= 256) && (tid < 128)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 128];
        s_b[tid] = b_sum = b_sum + s_b[tid + 128];
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid < 64)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 64];
        s_b[tid] = b_sum = b_sum + s_b[tid + 64];
    }

    cg::sync(cta);

#if (__CUDA_ARCH__ >= 300) || (defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 502)
    if (tid < 32) {
        cg::coalesced_group active = cg::coalesced_threads();

        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64) {
            a_sum = a_sum + s_a[tid + 32];
            b_sum = b_sum + s_b[tid + 32];
        }

        // Reduce final warp using shuffle
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            a_sum += active.shfl_down(a_sum, offset);
            b_sum += active.shfl_down(b_sum, offset);
        }
    }
#else
    if ((blockSize >= 64) && (tid < 32)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 32];
        s_b[tid] = b_sum = b_sum + s_b[tid + 32];
    }

    cg::sync(cta);

    if ((blockSize >= 32) && (tid < 16)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 16];
        s_b[tid] = b_sum = b_sum + s_b[tid + 16];
    }

    cg::sync(cta);

    if ((blockSize >= 16) && (tid < 8)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 8];
        s_b[tid] = b_sum = b_sum + s_b[tid + 8];
    }

    cg::sync(cta);

    if ((blockSize >= 8) && (tid < 4)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 4];
        s_b[tid] = b_sum = b_sum + s_b[tid + 4];
    }

    cg::sync(cta);

    if ((blockSize >= 4) && (tid < 2)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 2];
        s_b[tid] = b_sum = b_sum + s_b[tid + 2];
    }

    cg::sync(cta);

    if ((blockSize >= 2) && (tid < 1)) {
        s_a[tid] = a_sum = a_sum + s_a[tid + 1];
        s_b[tid] = b_sum = b_sum + s_b[tid + 1];
    }

    cg::sync(cta);

#endif

    // write result for this block to global mem
    if (tid == 0) {
        g_a[blockIdx.x] = (T)a_sum;
        g_b[blockIdx.x] = (T)b_sum;
    }
}

template <typename T, int blockSize>
__device__ void reduce_two_vectors_in_register(T a, T b, T* g_a, T* g_b)
{
    const int threadIdInBlock = cg::this_thread_block().thread_rank();

    T* s_a = SharedMemory<T>();
    T* s_b = SharedMemory<T>() + cg::this_thread_block().size();

    s_a[threadIdInBlock] = a;
    s_b[threadIdInBlock] = b;

    reduce_block_in_shared_memory<T, blockSize>(s_a, s_b, g_a, g_b);
}

template <typename T, typename GRAD_T, int blockSize>
__global__ void lamb_cuda_kernel_part1(
    T* __restrict__ p,
    GRAD_T* __restrict__ p_copy,  // For mixed precision training, pass NULL if not needed
    T* __restrict__ m,
    T* __restrict__ v,
    const GRAD_T* __restrict__ g,
    const float b1,
    const float b2,
    const float eps,
    const float grad_scale,
    const float step_size,
    const size_t tsize,
    adamMode_t mode,
    const float decay,
    T* __restrict__ w_l2_i,
    T* __restrict__ u_l2_i)
{
    // Assuming 2D grids and 2D blocks
    const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int threadIdInBlock = cg::this_thread_block().thread_rank();
    const int i = (blockId * threadsPerBlock + threadIdInBlock);
    const int totThreads = gridDim.x * gridDim.y * threadsPerBlock;

    T reg_w = 0;
    T reg_u = 0;

    for (int j = i; j < tsize; j += totThreads) {
        T scaled_grad = g[j] / grad_scale;
        T pj = p[j];
        m[j] = b1 * m[j] + (1 - b1) * scaled_grad;
        v[j] = b2 * v[j] + (1 - b2) * scaled_grad * scaled_grad;
        float denom;
        if (mode == ADAM_MODE_0)
            denom = sqrtf(v[j] + eps);
        else  // Mode 1
            denom = sqrtf(v[j]) + eps;
        T update = (m[j] / denom) + (decay * p[j]);

        reg_u += update * update;
        reg_w += pj * pj;
    }

    reduce_two_vectors_in_register<T, blockSize>(reg_w, reg_u, w_l2_i, u_l2_i);
}

template <typename T, typename GRAD_T, int blockSize>
__global__ void lamb_cuda_kernel_part2(const size_t tsize, T* __restrict__ g_a, T* __restrict__ g_b)
{
    T* s_a = SharedMemory<T>();
    T* s_b = SharedMemory<T>() + cg::this_thread_block().size();

    const int threadIdInBlock = cg::this_thread_block().thread_rank();

    s_a[threadIdInBlock] = g_a[threadIdInBlock];
    s_b[threadIdInBlock] = g_b[threadIdInBlock];

    if (threadIdInBlock >= tsize) {
        s_a[threadIdInBlock] = 0.0;
        s_b[threadIdInBlock] = 0.0;
    }

    reduce_block_in_shared_memory<T, blockSize>(s_a, s_b, g_a, g_b);
}

template <typename T, typename GRAD_T>
__global__ void lamb_cuda_kernel_part3(
    T* __restrict__ p,
    GRAD_T* __restrict__ p_copy,  // For mixed precision training, pass NULL if not needed
    T* __restrict__ m,
    T* __restrict__ v,
    const GRAD_T* __restrict__ g,
    const float b1,
    const float b2,
    const float max_coeff,
    const float min_coeff,
    const float eps,
    const float grad_scale,
    const float step_size,
    const size_t tsize,
    adamMode_t mode,
    const float decay,
    T* __restrict__ w_l2_i,
    T* __restrict__ u_l2_i,
    T* __restrict__ lamb_coeff_val)
{
    // Assuming 2D grids and 2D blocks
    const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int threadIdInBlock = cg::this_thread_block().thread_rank();
    const int i = (blockId * threadsPerBlock + threadIdInBlock);
    const int totThreads = gridDim.x * gridDim.y * threadsPerBlock;

    T reg_w = sqrtf(w_l2_i[0]);
    T reg_u = sqrtf(u_l2_i[0]);

    float lamb_coeff = 1.0;

    if (reg_w != 0 && reg_u != 0) {
        lamb_coeff = reg_w / reg_u;
        if (lamb_coeff > max_coeff) { lamb_coeff = max_coeff; }
        if (lamb_coeff < min_coeff) { lamb_coeff = min_coeff; }
    }

    if (blockId == 0 && threadIdInBlock == 0) {
        lamb_coeff_val[0] = lamb_coeff;
        // printf("Cuda Lamb Coeff is %.6f \n",lamb_coeff);
    }

    for (int j = i; j < tsize; j += totThreads) {
        T pj = (float)p[j];
        T mj = m[j];
        T vj = v[j];
        float denom;
        if (mode == ADAM_MODE_0)
            denom = sqrtf(vj + eps);
        else  // Mode 1
            denom = sqrtf(vj) + eps;
        T update = (mj / denom) + (decay * pj);

        pj = pj - (step_size * lamb_coeff * update);
        p[j] = pj;
        if (p_copy != NULL) p_copy[j] = (GRAD_T)pj;
    }
}

void fused_lamb_cuda(at::Tensor& p,
                     at::Tensor& p_copy,
                     at::Tensor& m,
                     at::Tensor& v,
                     at::Tensor& g,
                     float lr,
                     float beta1,
                     float beta2,
                     float max_coeff,
                     float min_coeff,
                     float eps,
                     float grad_scale,
                     int step,
                     int mode,
                     int bias_correction,
                     float decay,
                     at::Tensor& w_l2_i,
                     at::Tensor& u_l2_i,
                     at::Tensor& lamb_coeff)
{
    //        using namespace at;

    // Get tensor size
    int tsize = p.numel();
    // Determine #threads and #blocks
    const int threadsPerBlock = 512;
    int num_blocks = (tsize + threadsPerBlock - 1) / threadsPerBlock;
    if (num_blocks > 512) num_blocks = 512;

    int smemsize = 0;

    if (p.type().scalarType() == at::ScalarType::Double)
        smemsize = 2 * threadsPerBlock * sizeof(double);
    else
        smemsize = 2 * threadsPerBlock * sizeof(float);

    const dim3 blocks(num_blocks);
    const dim3 threads(threadsPerBlock);

    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
               "parameter tensor is too large to be indexed with int32");
    // Constants
    float step_size = 0;
    if (bias_correction == 1) {
        const float bias_correction1 = 1 - std::pow(beta1, step);
        const float bias_correction2 = 1 - std::pow(beta2, step);
        step_size = lr * std::sqrt(bias_correction2) / bias_correction1;
    } else {
        step_size = lr;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (g.type().scalarType() == at::ScalarType::Half) {
        // all other values should be fp32 for half gradients
        AT_ASSERTM(p.type().scalarType() == at::ScalarType::Float,
                   "expected parameter to be of float type");
        // dispatch is done on the gradient type
        using namespace at;  // prevents "toString is undefined" errors
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            g.scalar_type(), "lamb_cuda_kernel", ([&] {
                using accscalar_t = at::acc_type<scalar_t, true>;

                lamb_cuda_kernel_part1<accscalar_t, scalar_t, threadsPerBlock>
                    <<<blocks, threadsPerBlock, smemsize, stream>>>(
                        p.data<accscalar_t>(),
                        p_copy.numel() ? p_copy.data<scalar_t>() : NULL,
                        m.data<accscalar_t>(),
                        v.data<accscalar_t>(),
                        g.data<scalar_t>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t)mode,
                        decay,
                        w_l2_i.data<accscalar_t>(),
                        u_l2_i.data<accscalar_t>());

                lamb_cuda_kernel_part2<accscalar_t, scalar_t, threadsPerBlock>
                    <<<1, threadsPerBlock, smemsize, stream>>>(
                        num_blocks, w_l2_i.data<accscalar_t>(), u_l2_i.data<accscalar_t>());

                lamb_cuda_kernel_part3<accscalar_t, scalar_t>
                    <<<blocks, threadsPerBlock, smemsize, stream>>>(
                        p.data<accscalar_t>(),
                        p_copy.numel() ? p_copy.data<scalar_t>() : NULL,
                        m.data<accscalar_t>(),
                        v.data<accscalar_t>(),
                        g.data<scalar_t>(),
                        beta1,
                        beta2,
                        max_coeff,
                        min_coeff,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t)mode,
                        decay,
                        w_l2_i.data<accscalar_t>(),
                        u_l2_i.data<accscalar_t>(),
                        lamb_coeff.data<accscalar_t>());
            }));
    } else {
        using namespace at;
        AT_DISPATCH_FLOATING_TYPES(
            g.scalar_type(), "lamb_cuda_kernel", ([&] {
                lamb_cuda_kernel_part1<scalar_t, scalar_t, threadsPerBlock>
                    <<<blocks, threadsPerBlock, smemsize, stream>>>(
                        p.data<scalar_t>(),
                        NULL,  // don't output p_copy for fp32, it's wasted write
                        m.data<scalar_t>(),
                        v.data<scalar_t>(),
                        g.data<scalar_t>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t)mode,
                        decay,
                        w_l2_i.data<scalar_t>(),
                        u_l2_i.data<scalar_t>());

                lamb_cuda_kernel_part2<scalar_t, scalar_t, threadsPerBlock>
                    <<<1, threadsPerBlock, smemsize, stream>>>(
                        num_blocks, w_l2_i.data<scalar_t>(), u_l2_i.data<scalar_t>());

                lamb_cuda_kernel_part3<scalar_t, scalar_t>
                    <<<blocks, threadsPerBlock, smemsize, stream>>>(
                        p.data<scalar_t>(),
                        NULL,  // don't output p_copy for fp32, it's wasted write
                        m.data<scalar_t>(),
                        v.data<scalar_t>(),
                        g.data<scalar_t>(),
                        beta1,
                        beta2,
                        max_coeff,
                        min_coeff,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t)mode,
                        decay,
                        w_l2_i.data<scalar_t>(),
                        u_l2_i.data<scalar_t>(),
                        lamb_coeff.data<scalar_t>());
            }));
    }
    C10_CUDA_CHECK(cudaGetLastError());
}

// template __device__ void reduce_two_vectors_in_register<float,512>(float a, float b, float* g_a,
// float* g_b, cg::grid_group &cgg);
