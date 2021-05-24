#include <limits>
#include "custom_cuda_layers.h"

#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define NORM_REG (MAX_REGISTERS)

namespace cg = cooperative_groups;

__global__ void fused_bias_residual_layer_norm(float* output,
                                               const float* vals,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    float inp_reg[NORM_REG];

    int k = 0;
    float sum = 0;
    int input_id = id;
    while (input_id < row_stride) {
        inp_reg[k] = vals[input_id + row * row_stride];
        sum += inp_reg[k++];
        input_id += iteration_stride;
    }

    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);

    __shared__ float shr[MAX_WARP_NUM];

    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();

    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();

    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);

    float mean = sum / (row_stride);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        inp_reg[f] -= mean;
        sum += inp_reg[f] * inp_reg[f];
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();

    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();

    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f] = inp_reg[f] * sum;
        inp_reg[f] = inp_reg[f] * gamma[out_id] + beta[out_id];
        output[out_id + row * row_stride] = inp_reg[f];
    }
}

__global__ void fused_bias_residual_layer_norm(__half* output,
                                               const __half* vals,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               int row_stride)
{
#if __CUDA_ARCH__ >= 700
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    __half2 inp_reg[NORM_REG];

    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    __half2* out_cast = reinterpret_cast<__half2*>(output);

    int k = 0;
    int input_id = id;
    while (input_id < row_stride) {
        inp_reg[k++] = vals_cast[input_id + row * row_stride];
        input_id += iteration_stride;
    }
    float sum = 0;
    for (int f = k - 1; f >= 0; f--) {
        float2 inp_f = __half22float2(inp_reg[f]);
        sum += inp_f.x + inp_f.y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride << 1);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        float2 inp_f = __half22float2(inp_reg[f]);
        inp_f.x -= mean;
        inp_f.y -= mean;
        inp_reg[f] = __float22half2_rn(inp_f);
        sum += inp_f.x * inp_f.x;
        sum += inp_f.y * inp_f.y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride << 1);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    __half2 variance_h = __float2half2_rn(sum);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f] = inp_reg[f] * variance_h;
        inp_reg[f] = inp_reg[f] * gamma_cast[out_id] + beta_cast[out_id];
        out_cast[out_id + row * row_stride] = inp_reg[f];
    }
#endif
}

template <typename T>
void launch_layer_norm(T* out,
                       T* vals,
                       const T* gamma,
                       const T* beta,
                       float epsilon,
                       int batch_size,
                       int hidden_dim,
                       cudaStream_t stream);

template <>
void launch_layer_norm<float>(float* out,
                              float* vals,
                              const float* gamma,
                              const float* beta,
                              float epsilon,
                              int batch_size,
                              int hidden_dim,
                              cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);

    dim3 block_dim(threads);

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        out, vals, gamma, beta, epsilon, hidden_dim);
}

template <>
void launch_layer_norm<__half>(__half* out,
                               __half* vals,
                               const __half* gamma,
                               const __half* beta,
                               float epsilon,
                               int batch_size,
                               int hidden_dim,
                               cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        out, vals, gamma, beta, epsilon, hidden_dim / 2);
}

__global__ void fused_residual_layer_norm(float* norm,
                                          float* res_add,
                                          float* vals,
                                          float* residual,
                                          const float* bias,
                                          const float* gamma,
                                          const float* beta,
                                          float epsilon,
                                          int row_stride,
                                          bool preLN)
{
    int iteration_stride = blockDim.x;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    float inp_reg[NORM_REG];

    int k = 0;
    int input_id = id;

    float sum = 0;
    while (input_id < row_stride) {
        inp_reg[k] = vals[input_id + row * row_stride];
        float res_f = (residual[input_id + row * row_stride]);
        float bias_f = (bias[input_id]);
        inp_reg[k] += res_f + bias_f;
        if (preLN) res_add[input_id + row * row_stride] = inp_reg[k];
        sum += inp_reg[k++];
        input_id += iteration_stride;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);

    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();

    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();

    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        inp_reg[f] -= mean;
        sum += inp_reg[f] * inp_reg[f];
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();

    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();

    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride);
    sum += epsilon;
    sum = __frsqrt_rn(sum);

    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f] = inp_reg[f] * sum;
        inp_reg[f] = inp_reg[f] * gamma[out_id] + beta[out_id];
        norm[out_id + row * row_stride] = inp_reg[f];
    }
}

__global__ void fused_residual_layer_norm(__half* norm,
                                          __half* res_add,
                                          __half* vals,
                                          __half* residual,
                                          const __half* bias,
                                          const __half* gamma,
                                          const __half* beta,
                                          float epsilon,
                                          int row_stride,
                                          bool preLN)
{
#if __CUDA_ARCH__ >= 700
    int iteration_stride = blockDim.x;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    __half2 inp_reg[NORM_REG];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    __half2* norm_cast = reinterpret_cast<__half2*>(norm);
    __half2* res_add_cast = reinterpret_cast<__half2*>(res_add);
    __half2* residual_cast = reinterpret_cast<__half2*>(residual);
    const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);

    int k = 0;
    int input_id = id;

    float sum = 0;
    while (input_id < row_stride) {
        inp_reg[k] = vals_cast[input_id + row * row_stride];
        float2 inp_f = __half22float2(inp_reg[k]);
        float2 res_f = __half22float2(residual_cast[input_id + row * row_stride]);
        float2 bias_f = __half22float2(bias_cast[input_id]);
        inp_f.x += res_f.x + bias_f.x;
        inp_f.y += res_f.y + bias_f.y;
        inp_reg[k] = __float22half2_rn(inp_f);

        if (preLN) res_add_cast[input_id + row * row_stride] = inp_reg[k];
        sum += inp_f.x + inp_f.y;
        input_id += iteration_stride;
        k++;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride << 1);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        float2 inp_f = __half22float2(inp_reg[f]);
        inp_f.x -= mean;
        inp_f.y -= mean;
        inp_reg[f] = __float22half2_rn(inp_f);
        sum += inp_f.x * inp_f.x;
        sum += inp_f.y * inp_f.y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride << 1);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    __half2 variance_h = __float2half2_rn(sum);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f] = inp_reg[f] * variance_h;
        inp_reg[f] = inp_reg[f] * gamma_cast[out_id] + beta_cast[out_id];
        norm_cast[out_id + row * row_stride] = inp_reg[f];
    }
#endif
}

template <typename T>
void launch_residual_layer_norm(T* norm,
                                T* res_add,
                                T* vals,
                                T* residual,
                                const T* bias,
                                const T* gamma,
                                const T* beta,
                                float epsilon,
                                int batch_size,
                                int hidden_dim,
                                bool preLN,
                                cudaStream_t stream);

template <>
void launch_residual_layer_norm<float>(float* norm,
                                       float* res_add,
                                       float* vals,
                                       float* residual,
                                       const float* bias,
                                       const float* gamma,
                                       const float* beta,
                                       float epsilon,
                                       int batch_size,
                                       int hidden_dim,
                                       bool preLN,
                                       cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);

    dim3 block_dim(threads);

    fused_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        norm, res_add, vals, residual, bias, gamma, beta, epsilon, hidden_dim, preLN);
}

template <>
void launch_residual_layer_norm<__half>(__half* norm,
                                        __half* res_add,
                                        __half* vals,
                                        __half* residual,
                                        const __half* bias,
                                        const __half* gamma,
                                        const __half* beta,
                                        float epsilon,
                                        int batch_size,
                                        int hidden_dim,
                                        bool preLN,
                                        cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);

    fused_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        norm, res_add, vals, residual, bias, gamma, beta, epsilon, hidden_dim / 2, preLN);
}
