// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "custom_cuda_layers.h"

namespace cg = cooperative_groups;

/*
Fused bias add, residual (elementwise) add, and normalization layer.

For FP16, this kernel does not promote to FP32 in order to utilize the 2x throughput for
__half2 instructions, and avoid the conversion overhead (1/8 of __hal2 arithmetic).

For specific launch constraints, see the launch functions.
*/

#define NORM_REG (MAX_REGISTERS / 4)

__global__ void fused_bias_residual_layer_norm(float* vals,
                                               const float* residual,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               float* vars,
                                               float* means,
                                               int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id / WARP_SIZE;

    float vals_arr[NORM_REG];
    __shared__ float shr[MAX_WARP_NUM];

    residual += (row * row_stride);
    vals += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = residual[i * iteration_stride + id];
        sum += vals_arr[i];
    }
    if (high_index < row_stride) {
        vals_arr[iterations] = residual[high_index];
        sum += vals_arr[iterations];
        iterations++;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> WARP_SIZE_BITS)) sum = shr[g.thread_rank()];

#if !defined(__STOCHASTIC_MODE__) || __CUDA_ARCH__ < 700
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> WARP_SIZE_BITS); i *= 2) {
        sum += g.shfl_down(sum, i);
    }

    sum = g.shfl(sum, 0);
    float mean = sum / row_stride;
    if (training)
        if (threadIdx.x == 0) means[row] = mean;
    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] -= mean;
        variance += vals_arr[i] * vals_arr[i];
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> WARP_SIZE_BITS)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> WARP_SIZE_BITS); i *= 2) {
        variance += g.shfl_down(variance, i);
    }
    variance = g.shfl(variance, 0);
    variance /= row_stride;
    variance += epsilon;
    if (training)
        if (threadIdx.x == 0) vars[row] = variance;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = vals_arr[i] * rsqrtf(variance);
        vals_arr[i] =
            vals_arr[i] * gamma[i * iteration_stride + id] + beta[i * iteration_stride + id];
        vals[i * iteration_stride + id] = vals_arr[i];
    }
    if ((high_index) < row_stride) {
        vals_arr[iterations] = vals_arr[iterations] * rsqrtf(variance);
        vals_arr[iterations] = vals_arr[iterations] * gamma[high_index] + beta[high_index];
        vals[high_index] = vals_arr[iterations];
    }
}

__global__ void fused_bias_residual_layer_norm(__half* vals,
                                               const __half* residual,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               __half* vars,
                                               __half* means,
                                               int row_stride)
{
#ifdef HALF_PRECISION_AVAILABLE
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> WARP_SIZE_BITS;

    float2 vals_f[NORM_REG];
    __shared__ float shr[MAX_WARP_NUM];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);

    residual_cast += (row * row_stride);
    vals_cast += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_f[i] = __half22float2(residual_cast[i * iteration_stride + id]);
        sum += vals_f[i].x;
        sum += vals_f[i].y;
    }
    if ((high_index) < row_stride) {
        vals_f[iterations] = __half22float2(residual_cast[high_index]);
        sum += vals_f[iterations].x;
        sum += vals_f[iterations].y;
        iterations++;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> WARP_SIZE_BITS)) sum = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> WARP_SIZE_BITS); i *= 2) {
        sum += g.shfl_down(sum, i);
    }
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride * 2);

    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_f[i].x -= mean;
        vals_f[i].y -= mean;
        variance += vals_f[i].x * vals_f[i].x;
        variance += vals_f[i].y * vals_f[i].y;
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> WARP_SIZE_BITS)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> WARP_SIZE_BITS); i *= 2) {
        variance += g.shfl_down(variance, i);
    }
    variance = g.shfl(variance, 0);
    variance /= (row_stride * 2);
    variance += epsilon;

    __half2 variance_h = __float2half2_rn(variance);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);

    if (training && threadIdx.x == 0) {
        vars[row] = __float2half(variance);
        means[row] = __float2half(mean);
    }
    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        __half2 vals_arr = __float22half2_rn(vals_f[i]);
        vals_arr = vals_arr * h2rsqrt(variance_h);
        vals_arr =
            vals_arr * gamma_cast[i * iteration_stride + id] + beta_cast[i * iteration_stride + id];
        vals_cast[i * iteration_stride + id] = vals_arr;
    }
    if ((high_index) < row_stride) {
        __half2 vals_arr = __float22half2_rn(vals_f[iterations]);
        vals_arr = vals_arr * h2rsqrt(variance_h);
        vals_arr = vals_arr * gamma_cast[high_index] + beta_cast[high_index];
        vals_cast[high_index] = vals_arr;
    }
#endif
}

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars,
                                     T* means);

template <>
void launch_bias_residual_layer_norm<float>(float* vals,
                                            const float* residual,
                                            const float* gamma,
                                            const float* beta,
                                            float epsilon,
                                            int batch_size,
                                            int hidden_dim,
                                            cudaStream_t stream,
                                            bool preLayerNorm,
                                            bool training,
                                            float* vars,
                                            float* means)
{
    int threads = THREADS;

    dim3 grid_dim(batch_size);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim(threads);

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means, hidden_dim);
}

template <>
void launch_bias_residual_layer_norm<__half>(__half* vals,
                                             const __half* residual,
                                             const __half* gamma,
                                             const __half* beta,
                                             float epsilon,
                                             int batch_size,
                                             int hidden_dim,
                                             cudaStream_t stream,
                                             bool preLayerNorm,
                                             bool training,
                                             __half* vars,
                                             __half* means)
{
    int threads = 128;

    dim3 grid_dim(batch_size);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim(threads);

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means, hidden_dim / 2);
}

__global__ void fused_bias_residual_layer_norm(float* vals,
                                               const float* residual,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               float* vars,
                                               int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id / 32;

    float vals_arr[NORM_REG];
    __shared__ float shr[MAX_WARP_NUM];

    residual += (row * row_stride);
    vals += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = residual[i * iteration_stride + id];
        sum += vals_arr[i];
    }
    if ((high_index) < row_stride) {
        vals_arr[iterations] = residual[high_index];
        sum += vals_arr[iterations];
        iterations++;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> WARP_SIZE_BITS)) sum = shr[g.thread_rank()];

#if !defined(__STOCHASTIC_MODE__) || __CUDA_ARCH__ < 700
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> WARP_SIZE_BITS); i *= 2) {
        sum += g.shfl_down(sum, i);
    }

    sum = g.shfl(sum, 0);
    float mean = sum / row_stride;
    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] -= mean;
        variance += vals_arr[i] * vals_arr[i];
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> WARP_SIZE_BITS)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> WARP_SIZE_BITS); i *= 2) {
        variance += g.shfl_down(variance, i);
    }
    variance = g.shfl(variance, 0);
    variance /= row_stride;
    variance += epsilon;
    if (training)
        if (threadIdx.x == 0) vars[row] = variance;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = vals_arr[i] * rsqrtf(variance);
        vals_arr[i] =
            vals_arr[i] * gamma[i * iteration_stride + id] + beta[i * iteration_stride + id];
        vals[i * iteration_stride + id] = vals_arr[i];
    }
    if ((high_index) < row_stride) {
        vals_arr[iterations] = vals_arr[iterations] * rsqrtf(variance);
        vals_arr[iterations] = vals_arr[iterations] * gamma[high_index] + beta[high_index];
        vals[high_index] = vals_arr[iterations];
    }
}

__global__ void fused_bias_residual_layer_norm(__half* vals,
                                               const __half* residual,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               __half* vars,
                                               int row_stride)
{
#ifdef HALF_PRECISION_AVAILABLE

    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> WARP_SIZE_BITS;

    float2 vals_f[NORM_REG];
    __shared__ float shr[MAX_WARP_NUM];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);

    residual_cast += (row * row_stride);
    vals_cast += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_f[i] = __half22float2(residual_cast[i * iteration_stride + id]);
        sum += vals_f[i].x;
        sum += vals_f[i].y;
    }
    if ((high_index) < row_stride) {
        vals_f[iterations] = __half22float2(residual_cast[high_index]);
        sum += vals_f[iterations].x;
        sum += vals_f[iterations].y;
        iterations++;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> WARP_SIZE_BITS)) sum = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> WARP_SIZE_BITS); i *= 2) {
        sum += g.shfl_down(sum, i);
    }
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride * 2);

    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_f[i].x -= mean;
        vals_f[i].y -= mean;
        variance += vals_f[i].x * vals_f[i].x;
        variance += vals_f[i].y * vals_f[i].y;
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> WARP_SIZE_BITS)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> WARP_SIZE_BITS); i *= 2) {
        variance += g.shfl_down(variance, i);
    }
    variance = g.shfl(variance, 0);
    variance /= (row_stride * 2);
    variance += epsilon;

    __half2 variance_h = __float2half2_rn(variance);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);

    if (training && threadIdx.x == 0) vars[row] = __float2half(variance);

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        __half2 vals_arr = __float22half2_rn(vals_f[i]);
        vals_arr = vals_arr * h2rsqrt(variance_h);
        vals_arr =
            vals_arr * gamma_cast[i * iteration_stride + id] + beta_cast[i * iteration_stride + id];
        vals_cast[i * iteration_stride + id] = vals_arr;
    }
    if ((high_index) < row_stride) {
        __half2 vals_arr = __float22half2_rn(vals_f[iterations]);
        vals_arr = vals_arr * h2rsqrt(variance_h);
        vals_arr = vals_arr * gamma_cast[high_index] + beta_cast[high_index];
        vals_cast[high_index] = vals_arr;
    }
#endif
}

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars);

/*
To tune this launch the following restrictions must be met:

For float:
row_stride == hidden_size
threads * iterations == row_stride
threads is in [32, 64, 128, 256, 512, 1024]

For half:
row_stride == hidden_size / 2
threads * iterations == row_stride
threads is in [32, 64, 128, 256, 512, 1024]

*/

template <>
void launch_bias_residual_layer_norm<float>(float* vals,
                                            const float* residual,
                                            const float* gamma,
                                            const float* beta,
                                            float epsilon,
                                            int batch_size,
                                            int hidden_dim,
                                            cudaStream_t stream,
                                            bool preLayerNorm,
                                            bool training,
                                            float* vars)
{
    int threads = THREADS;

    dim3 grid_dim(batch_size);

    // There are some limitations to call below functions, now just enumerate the situations.

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim(threads);

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, hidden_dim);
}

template <>
void launch_bias_residual_layer_norm<__half>(__half* vals,
                                             const __half* residual,
                                             const __half* gamma,
                                             const __half* beta,
                                             float epsilon,
                                             int batch_size,
                                             int hidden_dim,
                                             cudaStream_t stream,
                                             bool preLayerNorm,
                                             bool training,
                                             __half* vars)
{
    int threads = 128;

    dim3 grid_dim(batch_size);

    // There are some limitations to call below functions, now just enumerate the situations.

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim(threads);
    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, hidden_dim / 2);
}

/* Normalize Gamma & Betta gradients
 * Compute gradients using either X_hat or
 * normalize input (invertible).
 * Combine transpose with gradients computation.
 */

template <typename T>
__global__ void LayerNormBackward1(const T* __restrict__ out_grad,
                                   const T* __restrict__ vals_hat,
                                   const T* __restrict__ gamma,
                                   const T* __restrict__ betta,
                                   T* __restrict__ gamma_grad,
                                   T* __restrict__ betta_grad,
                                   int rows,
                                   int width,
                                   bool invertible)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    float betta_reg = (invertible ? (float)betta[idx] : 0.0f);
    float gamma_reg = (float)gamma[idx];

    // Loop across matrix height
    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad[offset];
        float val = (invertible ? ((float)vals_hat[offset] - betta_reg) / gamma_reg
                                : (float)vals_hat[offset]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        int pos = blockIdx.x * TILE_DIM + threadIdx.y;
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

/* Normalize Gamma & Betta gradients
 * Compute gradients using the input to
 * the normalize.
 * Combine transpose with gradients computation.
 */

template <typename T>
__global__ void LayerNormBackward1(const T* __restrict__ out_grad,
                                   const T* __restrict__ X_data,
                                   const T* __restrict__ vars,
                                   const T* __restrict__ means,
                                   T* __restrict__ gamma_grad,
                                   T* __restrict__ betta_grad,
                                   int rows,
                                   int width)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = blockIdx.x * TILE_DIM + threadIdx.y;
    // Loop across matrix height

    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad[offset];
        float val = (float)X_data[offset];
        val = (val - (float)means[r]) * rsqrtf((float)vars[r]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}
/*

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is invertible!
 * We do the backward using the X_hat (X - u) / sqrt(variance) or the output of Normalization.
 */

__global__ void LayerNormBackward2(const float* out_grad,
                                   const float* vals_hat,
                                   const float* gamma,
                                   const float* betta,
                                   const float* vars,
                                   float* inp_grad,
                                   bool invertible,
                                   int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = iteration_stride >> WARP_SIZE_BITS;
    __shared__ float partialSum[MAX_WARP_NUM];

    out_grad += (row * row_stride);
    vals_hat += (row * row_stride);
    inp_grad += (row * row_stride);

    float vals_arr[NORM_REG];
    float vals_hat_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] =
            (invertible ? (vals_hat[i * iteration_stride + id] - betta[i * iteration_stride + id]) /
                              gamma_reg
                        : vals_hat[i * iteration_stride + id]);
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] =
            (invertible ? (vals_hat[high_index] - betta[high_index]) / gamma_reg
                        : vals_hat[high_index]);
        iterations++;
    }

    float var_reg = vars[row];

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += vals_hat_arr[i] * vals_arr[i] *
               sqrtf(var_reg);           // dval_hat = gamma * (x - u) * out_grad
        vals_arr[i] *= rsqrtf(var_reg);  // dvar_inv = gamma * out_grad / sqrt(var)
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) { vals_arr[i] += ((-sum * vals_hat_arr[i]) / var_reg); }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) inp_grad[i * iteration_stride + id] = (vals_arr[i] - sum);
    if ((high_index) < row_stride) inp_grad[high_index] = (vals_arr[iterations] - sum);
}

__global__ void LayerNormBackward2(const __half* out_grad,
                                   const __half* vals_hat,
                                   const __half* gamma,
                                   const __half* betta,
                                   const __half* vars,
                                   __half* inp_grad,
                                   bool invertible,
                                   int row_stride)
{
#ifdef HALF_PRECISION_AVAILABLE
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = iteration_stride >> WARP_SIZE_BITS;
    __shared__ float partialSum[MAX_WARP_NUM];

    __half2 vals_arr[NORM_REG];
    float2 vals_arr_f[NORM_REG];
    __half2 vals_hat_arr[NORM_REG];

    __half2* inp_grad_h = reinterpret_cast<__half2*>(inp_grad);
    const __half2* out_grad_h = reinterpret_cast<const __half2*>(out_grad);
    const __half2* vals_hat_h = reinterpret_cast<const __half2*>(vals_hat);

    inp_grad_h += (row * row_stride);
    out_grad_h += (row * row_stride);
    vals_hat_h += (row * row_stride);

    const __half2* gamma_h = reinterpret_cast<const __half2*>(gamma);
    const __half2* betta_h = (invertible ? reinterpret_cast<const __half2*>(betta) : nullptr);
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        __half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] =
            (invertible
                 ? (vals_hat_h[i * iteration_stride + id] - betta_h[i * iteration_stride + id]) /
                       gamma_reg
                 : vals_hat_h[i * iteration_stride + id]);
    }
    if ((high_index) < row_stride) {
        __half2 gamma_reg = gamma_h[high_index];
        vals_arr[iterations] = out_grad_h[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] =
            (invertible ? (vals_hat_h[high_index] - betta_h[high_index]) / gamma_reg
                        : vals_hat_h[high_index]);
        iterations++;
    }
    __half var_h = vars[row];
    __half2 var_reg = __halves2half2(var_h, var_h);

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        __half2 result_h = (vals_hat_arr[i] * vals_arr[i] * h2sqrt(var_reg));
        float2 result_f = __half22float2(result_h);
        sum += result_f.x;
        sum += result_f.y;
        vals_arr[i] *= h2rsqrt(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);
    __half2 sum_h = __float2half2_rn(sum);

    for (int i = 0; i < iterations; i++) {
        __half2 temp = ((-sum_h * vals_hat_arr[i]) / (var_reg));
        vals_arr_f[i] = __half22float2(vals_arr[i]);
        float2 temp_f = __half22float2(temp);
        vals_arr_f[i].x += temp_f.x;
        vals_arr_f[i].y += temp_f.y;
    }
    sum = 0.f;

    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x);
        sum += (vals_arr_f[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x -= sum;
        vals_arr_f[i].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[i]);

        inp_grad_h[i * iteration_stride + id] = temp;
    }
    if ((high_index) < row_stride) {
        vals_arr_f[iterations].x -= sum;
        vals_arr_f[iterations].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[iterations]);

        inp_grad_h[high_index] = temp;
    }
#endif
}

template <>
void launch_layerNorm_backward<float>(const float* out_grad,
                                      const float* vals_hat,
                                      const float* vars,
                                      const float* gamma,
                                      float* gamma_grad,
                                      float* betta_grad,
                                      float* inp_grad,
                                      int batch,
                                      int hidden_dim,
                                      cudaStream_t stream[2],
                                      bool invertible,
                                      const float* betta)
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads);

    LayerNormBackward2<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible, hidden_dim);
}

template <>
void launch_layerNorm_backward<__half>(const __half* out_grad,
                                       const __half* vals_hat,
                                       const __half* vars,
                                       const __half* gamma,
                                       __half* gamma_grad,
                                       __half* betta_grad,
                                       __half* inp_grad,
                                       int batch,
                                       int hidden_dim,
                                       cudaStream_t stream[2],
                                       bool invertible,
                                       const __half* betta)
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    // LayerNormBackward1<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
    //    out_grad, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads / 2);

    LayerNormBackward2<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible, hidden_dim / 2);
}

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is not invertible!
 * We do the backward using the input (X)
 */

__global__ void LayerNormBackward2(const float* out_grad,
                                   const float* X_vals,
                                   const float* gamma,
                                   const float* vars,
                                   const float* means,
                                   float* inp_grad,
                                   int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id >> WARP_SIZE_BITS;
    int warp_num = iteration_stride >> WARP_SIZE_BITS;
    __shared__ float partialSum[MAX_WARP_NUM];

    out_grad += (row * row_stride);
    X_vals += (row * row_stride);
    inp_grad += (row * row_stride);

    float vals_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad[high_index];
        vals_arr[iterations] *= gamma_reg;
        iterations++;
    }

    float var_reg = vars[row];
    float mean_reg = means[row];

    float sum = 0;
    float xu[NORM_REG];
    for (int i = 0; i < iterations; i++) {
        xu[i] = (X_vals[i * iteration_stride + id] - mean_reg);
        sum += vals_arr[i] * xu[i];
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] += (-sum * xu[i] * rsqrtf(var_reg) / (var_reg));
    }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) inp_grad[i * iteration_stride + id] = (vals_arr[i] - sum);
    if ((high_index) < row_stride) inp_grad[high_index] = (vals_arr[iterations] - sum);
}

__global__ void LayerNormBackward2(const __half* out_grad,
                                   const __half* X_vals,
                                   const __half* gamma,
                                   const __half* vars,
                                   const __half* means,
                                   __half* inp_grad,
                                   int row_stride)
{
#ifdef HALF_PRECISION_AVAILABLE
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id >> WARP_SIZE_BITS;
    int warp_num = iteration_stride >> WARP_SIZE_BITS;

    __shared__ float partialSum[MAX_WARP_NUM];

    __half2 vals_arr[NORM_REG];
    float2 vals_arr_f[NORM_REG];
    __half2 xu[NORM_REG];

    __half2* inp_grad_h = reinterpret_cast<__half2*>(inp_grad);
    const __half2* out_grad_h = reinterpret_cast<const __half2*>(out_grad);
    const __half2* vals_hat_h = reinterpret_cast<const __half2*>(X_vals);

    inp_grad_h += (row * row_stride);
    out_grad_h += (row * row_stride);
    vals_hat_h += (row * row_stride);

    const __half2* gamma_h = reinterpret_cast<const __half2*>(gamma);
    int high_index = iterations * iteration_stride + id;

    __half mean_h = means[row];
    __half2 mean_reg = __halves2half2(mean_h, mean_h);
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        __half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;  // out_grad * gamma
        xu[i] = (vals_hat_h[i * iteration_stride + id] - mean_reg);
    }
    if ((high_index) < row_stride) {
        __half2 gamma_reg = gamma_h[high_index];
        vals_arr[iterations] = out_grad_h[high_index];
        vals_arr[iterations] *= gamma_reg;  // out_grad * gamma
        xu[iterations] = (vals_hat_h[high_index] - mean_reg);
        iterations++;
    }
    __half var_h = vars[row];
    __half2 var_reg = __halves2half2(var_h, var_h);

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        __half2 result_h = (xu[i] * vals_arr[i]);
        float2 result_f = __half22float2(result_h);
        sum += result_f.x;
        sum += result_f.y;
        vals_arr[i] *= h2rsqrt(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);
    __half2 sum_h = __float2half2_rn(sum);

    for (int i = 0; i < iterations; i++) {
        __half2 xu_grad = ((-sum_h * xu[i] * h2rsqrt(var_reg)) / (var_reg));
        vals_arr_f[i] = __half22float2(vals_arr[i]);
        float2 xu_grad_f = __half22float2(xu_grad);
        vals_arr_f[i].x += xu_grad_f.x;
        vals_arr_f[i].y += xu_grad_f.y;
    }

    sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x);
        sum += (vals_arr_f[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x -= sum;
        vals_arr_f[i].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[i]);
        inp_grad_h[i * iteration_stride + id] = temp;
    }
    if ((high_index) < row_stride) {
        vals_arr_f[iterations].x -= sum;
        vals_arr_f[iterations].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[iterations]);
        inp_grad_h[high_index] = temp;
    }
#endif
}

template <>
void launch_layerNorm_backward<float>(const float* out_grad,
                                      const float* X_data,
                                      const float* vars,
                                      const float* means,
                                      const float* gamma,
                                      float* gamma_grad,
                                      float* betta_grad,
                                      float* inp_grad,
                                      int batch,
                                      int hidden_dim,
                                      cudaStream_t stream[2])
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads);
    LayerNormBackward2<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad, X_data, gamma, vars, means, inp_grad, hidden_dim);
}

template <>
void launch_layerNorm_backward<__half>(const __half* out_grad,
                                       const __half* X_data,
                                       const __half* vars,
                                       const __half* means,
                                       const __half* gamma,
                                       __half* gamma_grad,
                                       __half* betta_grad,
                                       __half* inp_grad,
                                       int batch,
                                       int hidden_dim,
                                       cudaStream_t stream[2])
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads / 2);
    LayerNormBackward2<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad, X_data, gamma, vars, means, inp_grad, hidden_dim / 2);
}

template <typename T>
__global__ void LayerNormBackward1_fused_add(const T* __restrict__ out_grad1,
                                             const T* __restrict__ out_grad2,
                                             const T* __restrict__ vals_hat,
                                             const T* __restrict__ gamma,
                                             const T* __restrict__ betta,
                                             T* __restrict__ gamma_grad,
                                             T* __restrict__ betta_grad,
                                             int rows,
                                             int width,
                                             bool invertible)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    float betta_reg = (invertible ? (float)betta[idx] : 0.0f);
    float gamma_reg = (float)gamma[idx];

    // Loop across matrix height
    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad1[offset] + (float)out_grad2[offset];
        float val = (invertible ? ((float)vals_hat[offset] - betta_reg) / gamma_reg
                                : (float)vals_hat[offset]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        int pos = blockIdx.x * TILE_DIM + threadIdx.y;
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

template <typename T>
__global__ void LayerNormBackward1_fused_add(const T* __restrict__ out_grad1,
                                             const T* __restrict__ out_grad2,
                                             const T* __restrict__ X_data,
                                             const T* __restrict__ vars,
                                             const T* __restrict__ means,
                                             T* __restrict__ gamma_grad,
                                             T* __restrict__ betta_grad,
                                             int rows,
                                             int width)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = blockIdx.x * TILE_DIM + threadIdx.y;
    // Loop across matrix height

    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad1[offset] + (float)out_grad2[offset];
        float val = (float)X_data[offset];
        val = (val - (float)means[r]) * rsqrtf((float)vars[r]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

__global__ void LayerNormBackward2_fused_add(const float* out_grad1,
                                             const float* out_grad2,
                                             const float* vals_hat,
                                             const float* gamma,
                                             const float* betta,
                                             const float* vars,
                                             float* inp_grad,
                                             bool invertible,
                                             int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = iteration_stride >> WARP_SIZE_BITS;
    __shared__ float partialSum[MAX_WARP_NUM];

    out_grad1 += (row * row_stride);
    out_grad2 += (row * row_stride);
    vals_hat += (row * row_stride);
    inp_grad += (row * row_stride);

    float vals_arr[NORM_REG];
    float vals_hat_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad1[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] =
            (invertible ? (vals_hat[i * iteration_stride + id] - betta[i * iteration_stride + id]) /
                              gamma_reg
                        : vals_hat[i * iteration_stride + id]);
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad1[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] =
            (invertible ? (vals_hat[high_index] - betta[high_index]) / gamma_reg
                        : vals_hat[high_index]);
        iterations++;
    }

    float var_reg = vars[row];

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += vals_hat_arr[i] * vals_arr[i] * sqrtf(var_reg);
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) { vals_arr[i] += ((-sum * vals_hat_arr[i]) / var_reg); }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++)
        inp_grad[i * iteration_stride + id] =
            (vals_arr[i] - sum) + out_grad2[i * iteration_stride + id];
    if ((high_index) < row_stride)
        inp_grad[high_index] = (vals_arr[iterations] - sum) + out_grad2[high_index];
}

__global__ void LayerNormBackward2_fused_add(const __half* out_grad1,
                                             const __half* out_grad2,
                                             const __half* vals_hat,
                                             const __half* gamma,
                                             const __half* betta,
                                             const __half* vars,
                                             __half* inp_grad,
                                             bool invertible,
                                             int row_stride)
{
#ifdef HALF_PRECISION_AVAILABLE
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = iteration_stride >> WARP_SIZE_BITS;
    __shared__ float partialSum[MAX_WARP_NUM];

    __half2 vals_arr[NORM_REG];
    float2 vals_arr_f[NORM_REG];
    __half2 vals_hat_arr[NORM_REG];

    // float2 result[iterations];

    __half2* inp_grad_h = reinterpret_cast<__half2*>(inp_grad);
    const __half2* out_grad_h1 = reinterpret_cast<const __half2*>(out_grad1);
    const __half2* out_grad_h2 = reinterpret_cast<const __half2*>(out_grad2);
    const __half2* vals_hat_h = reinterpret_cast<const __half2*>(vals_hat);

    inp_grad_h += (row * row_stride);
    out_grad_h1 += (row * row_stride);
    out_grad_h2 += (row * row_stride);
    vals_hat_h += (row * row_stride);

    const __half2* gamma_h = reinterpret_cast<const __half2*>(gamma);
    const __half2* betta_h = (invertible ? reinterpret_cast<const __half2*>(betta) : nullptr);
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        __half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h1[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;  // out_grad * gamma
        vals_hat_arr[i] =
            (invertible
                 ? (vals_hat_h[i * iteration_stride + id] - betta_h[i * iteration_stride + id]) /
                       gamma_reg
                 : vals_hat_h[i * iteration_stride + id]);
    }
    if ((high_index) < row_stride) {
        __half2 gamma_reg = gamma_h[high_index];
        vals_arr[iterations] = out_grad_h1[high_index];
        vals_arr[iterations] *= gamma_reg;  // out_grad * gamma
        vals_hat_arr[iterations] =
            (invertible ? (vals_hat_h[high_index] - betta_h[high_index]) / gamma_reg
                        : vals_hat_h[high_index]);
        iterations++;
    }
    __half var_h = vars[row];
    __half2 var_reg = __halves2half2(var_h, var_h);

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        __half2 result_h = (vals_hat_arr[i] * vals_arr[i] * h2sqrt(var_reg));
        float2 result_f = __half22float2(result_h);
        sum += result_f.x;
        sum += result_f.y;
        vals_arr[i] *= h2rsqrt(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);
    __half2 sum_h = __float2half2_rn(sum);

    for (int i = 0; i < iterations; i++) {
        __half2 temp = ((-sum_h * vals_hat_arr[i]) / (var_reg));
        vals_arr_f[i] = __half22float2(vals_arr[i]);
        float2 temp_f = __half22float2(temp);
        vals_arr_f[i].x += temp_f.x;
        vals_arr_f[i].y += temp_f.y;
    }
    sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x);
        sum += (vals_arr_f[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x -= sum;
        vals_arr_f[i].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[i]);

        inp_grad_h[i * iteration_stride + id] = temp + out_grad_h2[i * iteration_stride + id];
    }
    if ((high_index) < row_stride) {
        vals_arr_f[iterations].x -= sum;
        vals_arr_f[iterations].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[iterations]);

        inp_grad_h[high_index] = temp + out_grad_h2[high_index];
    }
#endif
}

template <>
void launch_layerNorm_backward_fused_add<float>(const float* out_grad1,
                                                const float* out_grad2,
                                                const float* vals_hat,
                                                const float* vars,
                                                const float* gamma,
                                                float* gamma_grad,
                                                float* betta_grad,
                                                float* inp_grad,
                                                int batch,
                                                int hidden_dim,
                                                cudaStream_t stream[2],
                                                bool invertible,
                                                const float* betta)
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);
    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads);
    LayerNormBackward2_fused_add<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible, hidden_dim);
}

template <>
void launch_layerNorm_backward_fused_add<__half>(const __half* out_grad1,
                                                 const __half* out_grad2,
                                                 const __half* vals_hat,
                                                 const __half* vars,
                                                 const __half* gamma,
                                                 __half* gamma_grad,
                                                 __half* betta_grad,
                                                 __half* inp_grad,
                                                 int batch,
                                                 int hidden_dim,
                                                 cudaStream_t stream[2],
                                                 bool invertible,
                                                 const __half* betta)
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads / 2);
    LayerNormBackward2_fused_add<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible, hidden_dim / 2);
}

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is not invertible!
 * We do the backward using the input (X)
 */

__global__ void LayerNormBackward2_fused_add(const float* out_grad1,
                                             const float* out_grad2,
                                             const float* X_vals,
                                             const float* gamma,
                                             const float* vars,
                                             const float* means,
                                             float* inp_grad,
                                             int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = iteration_stride >> WARP_SIZE_BITS;
    __shared__ float partialSum[MAX_WARP_NUM];

    float vals_arr[NORM_REG];
    float vals_hat_arr[NORM_REG];

    out_grad1 += (row * row_stride);
    out_grad2 += (row * row_stride);
    X_vals += (row * row_stride);
    inp_grad += (row * row_stride);
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad1[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] = X_vals[i * iteration_stride + id];
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad1[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] = X_vals[high_index];
        iterations++;
    }

    float var_reg = vars[row];
    float mean_reg = means[row];

    float sum = 0;
    float xu[NORM_REG];
    for (int i = 0; i < iterations; i++) {
        xu[i] = (vals_hat_arr[i] - mean_reg);
        sum += vals_arr[i] * xu[i];
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] += (-sum * xu[i] * rsqrtf(var_reg) / (var_reg));
    }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++)
        inp_grad[i * iteration_stride + id] =
            (vals_arr[i] - sum) + out_grad2[i * iteration_stride + id];
    if ((high_index) < row_stride)
        inp_grad[high_index] = (vals_arr[iterations] - sum) + out_grad2[high_index];
}

__global__ void LayerNormBackward2_fused_add(const __half* out_grad1,
                                             const __half* out_grad2,
                                             const __half* X_vals,
                                             const __half* gamma,
                                             const __half* vars,
                                             const __half* means,
                                             __half* inp_grad,
                                             int row_stride)
{
#ifdef HALF_PRECISION_AVAILABLE
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = iteration_stride >> WARP_SIZE_BITS;

    __shared__ float partialSum[MAX_WARP_NUM];

    __half2 vals_arr[NORM_REG];
    float2 vals_arr_f[NORM_REG];
    __half2 vals_hat_arr[NORM_REG];

    __half2* inp_grad_h = reinterpret_cast<__half2*>(inp_grad);
    const __half2* out_grad_h1 = reinterpret_cast<const __half2*>(out_grad1);
    const __half2* out_grad_h2 = reinterpret_cast<const __half2*>(out_grad2);
    const __half2* vals_hat_h = reinterpret_cast<const __half2*>(X_vals);

    out_grad_h1 += (row * row_stride);
    out_grad_h2 += (row * row_stride);
    inp_grad_h += (row * row_stride);
    vals_hat_h += (row * row_stride);

    const __half2* gamma_h = reinterpret_cast<const __half2*>(gamma);
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        __half2 gamma_reg = gamma_h[i * iteration_stride + id];
        vals_arr[i] = out_grad_h1[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;  // out_grad * gamma
        vals_hat_arr[i] = vals_hat_h[i * iteration_stride + id];
    }
    if ((high_index) < row_stride) {
        __half2 gamma_reg = gamma_h[high_index];
        vals_arr[iterations] = out_grad_h1[high_index];
        vals_arr[iterations] *= gamma_reg;  // out_grad * gamma
        vals_hat_arr[iterations] = vals_hat_h[high_index];
        iterations++;
    }

    __half mean_h = means[row];
    __half var_h = vars[row];
    __half2 var_reg = __halves2half2(var_h, var_h);
    __half2 mean_reg = __halves2half2(mean_h, mean_h);
    __half2 xu[NORM_REG];

    float sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        xu[i] = (vals_hat_arr[i] - mean_reg);
        __half2 result_h = (xu[i] * vals_arr[i]);
        float2 result_f = __half22float2(result_h);
        sum += result_f.x;
        sum += result_f.y;
        vals_arr[i] *= h2rsqrt(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);
    __half2 sum_h = __float2half2_rn(sum);

    for (int i = 0; i < iterations; i++) {
        __half2 xu_grad = ((-sum_h * xu[i] * h2rsqrt(var_reg)) / (var_reg));
        vals_arr_f[i] = __half22float2(vals_arr[i]);
        float2 xu_grad_f = __half22float2(xu_grad);
        vals_arr_f[i].x += xu_grad_f.x;
        vals_arr_f[i].y += xu_grad_f.y;
    }

    sum = 0.f;
    for (int i = 0; i < iterations; i++) {
        sum += (vals_arr_f[i].x);
        sum += (vals_arr_f[i].y);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= (2 * row_stride);

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr_f[i].x -= sum;
        vals_arr_f[i].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[i]);
        inp_grad_h[i * iteration_stride + id] = temp + out_grad_h2[i * iteration_stride + id];
    }
    if ((high_index) < row_stride) {
        vals_arr_f[iterations].x -= sum;
        vals_arr_f[iterations].y -= sum;
        __half2 temp = __float22half2_rn(vals_arr_f[iterations]);
        inp_grad_h[high_index] = temp + out_grad_h2[high_index];
    }
#endif
}

template <>
void launch_layerNorm_backward_fused_add<float>(const float* out_grad1,
                                                const float* out_grad2,
                                                const float* X_data,
                                                const float* vars,
                                                const float* means,
                                                const float* gamma,
                                                float* gamma_grad,
                                                float* betta_grad,
                                                float* inp_grad,
                                                int batch,
                                                int hidden_dim,
                                                cudaStream_t stream[2])
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads);
    LayerNormBackward2_fused_add<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad, hidden_dim);
}

template <>
void launch_layerNorm_backward_fused_add<__half>(const __half* out_grad1,
                                                 const __half* out_grad2,
                                                 const __half* X_data,
                                                 const __half* vars,
                                                 const __half* means,
                                                 const __half* gamma,
                                                 __half* gamma_grad,
                                                 __half* betta_grad,
                                                 __half* inp_grad,
                                                 int batch,
                                                 int hidden_dim,
                                                 cudaStream_t stream[2])
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads / 2);
    LayerNormBackward2_fused_add<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad, hidden_dim / 2);
}
