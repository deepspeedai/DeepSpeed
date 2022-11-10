/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "conversion_utils.h"
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"

namespace cg = cooperative_groups;
using rop = reduce::ROpType;

namespace ln {
constexpr int granularity = 16;
constexpr int max_threads = 512;
constexpr int max_warps = max_threads / hw_warp_size;

constexpr int internal_unroll = 4;
}  // namespace ln

/*
Primary layer norm implementation. Assumes elems_per_row % 8
is equal to 0.

Args:
    output: buffer for output data
    vals: buffer for input data
    gamma: gain for normalization
    beta: bias for normalization
    epsilon: numeric stability
    elems_per_row: number of elements each block will normalize
*/
template <typename T, int UNROLL>
__global__ void fused_ln(T* output,
                         const T* vals,
                         const T* gamma,
                         const T* beta,
                         float epsilon,
                         int elems_per_row)
{
    constexpr int T_per_load = ln::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // X-dimension of the block
    const int block_offset = tb.group_index().x * elems_per_row;
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.size() * T_per_load;

    // TODO(cmikeh2): refactor to reduction utility library
    float sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;
    T local_buffer[UNROLL * ln::internal_unroll * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        T* iteration_buffer = local_buffer + i * ln::internal_unroll * T_per_load;

#pragma unroll
        for (int j = 0; j < ln::internal_unroll; j++) {
            const int iteration = i * ln::internal_unroll + j;
            mem_access::load_global<ln::granularity>(
                iteration_buffer + j * T_per_load,
                input_base + iteration * stride,
                thread_offset + iteration * stride < elems_per_row);
        }

#pragma unroll
        for (int j = 0; j < ln::internal_unroll * T_per_load; j++) {
            float up_cast = conversion::to<float>(iteration_buffer[j]);
            sum = reduce::element<rop::Add>(sum, up_cast);
        }
    }

    reduce::block<rop::Add, ln::max_warps>(tb, warp, sum);
    const float mean = sum / elems_per_row;

    float mean_diff = reduce::init<rop::Add, float>();

#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                mean_diff = reduce::element<rop::Add>(mean_diff, diff * diff);
            }
        }
    }

    reduce::block<rop::Add, ln::max_warps>(tb, warp, mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    const T mean_compute = conversion::to<T>(mean);
    const T denom_compute = conversion::to<T>(denom);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = (iteration_buffer[j] - mean_compute) * denom_compute;
            iteration_buffer[j] = iteration_buffer[j] * gamma_local[j] + beta_local[j];
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

#define LAUNCH_FUSED_LN(unroll_factor) \
    fused_ln<T, unroll_factor>         \
        <<<grid, block, 0, stream>>>(output, vals, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_ln(T* output,
                     const T* vals,
                     const T* gamma,
                     const T* beta,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);
    // 32 for __half, 16 for float
    constexpr int T_per_thread_unroll = T_per_load * ln::internal_unroll;
    // 1024 for __half, 512 for float
    constexpr int T_per_warp_unroll = T_per_thread_unroll * hw_warp_size;

    int32_t unroll = 1;
    while (T_per_warp_unroll * ln::max_warps * unroll < elems_per_row) { unroll++; }

    const int sched_warps =
        (elems_per_row + unroll * T_per_warp_unroll - 1) / (unroll * T_per_warp_unroll);

    const int warps = (unroll > 1) ? ln::max_warps : sched_warps;

    dim3 grid(rows);
    dim3 block(warps * hw_warp_size);

    // This should match the max_unroll constexpr
    if (unroll == 1) {
        LAUNCH_FUSED_LN(1);
    } else if (unroll == 2) {
        LAUNCH_FUSED_LN(2);
    } else if (unroll == 3) {
        LAUNCH_FUSED_LN(3);
    } else if (unroll == 4) {
        LAUNCH_FUSED_LN(4);
    }
}

template void launch_fused_ln(__half*,
                              const __half*,
                              const __half*,
                              const __half*,
                              float,
                              int,
                              int,
                              cudaStream_t);
template void
launch_fused_ln(float*, const float*, const float*, const float*, float, int, int, cudaStream_t);

/*
Fused resiual + bias + layer norm implementation. Assumes elems_per_row % 8
is equal to 0.

TODO(cmikeh2): Goal is to deprecate this implementation. The bias + residual
need to be fused into compute-bound producer operations.

Args:
    output: buffer for output data
    res_output: output of residual addition
    vals: buffer for input data
    residual: residual data
    bias: bias of of input data
    gamma: gain for normalization
    beta: bias for normalization
    epsilon: numeric stability
    elems_per_row: number of elements each block will normalize
Template arg:
    StoreResidual: controls whether the residual calculation is stored
        or not. When set to false, the input `res_output` is unused.
*/
template <typename T, int UNROLL, bool StoreResidual>
__global__ void fused_residual_ln(T* output,
                                  T* res_output,
                                  const T* vals,
                                  const T* residual,
                                  const T* bias,
                                  const T* gamma,
                                  const T* beta,
                                  float epsilon,
                                  int elems_per_row)
{
    constexpr int T_per_load = ln::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // X-dimension of the block
    const int block_offset = tb.group_index().x * elems_per_row;
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.size() * T_per_load;

    float sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;
    const T* residual_base = residual + base_offset;
    const T* bias_base = bias + thread_offset;

    T local_buffer[UNROLL * ln::internal_unroll * T_per_load];

    // Unlike a vanilla layernorm, since we're fusing the two adds as well
    // an inner unroll seems to be less valuable. If anything, a double unroll
    // makes the most sense if we find we are having performance issues.
#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        T residual_buffer[T_per_load];
        T bias_buffer[T_per_load];

        mem_access::load_global<ln::granularity>(
            iteration_buffer, input_base + i * stride, thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(residual_buffer,
                                                 residual_base + i * stride,
                                                 thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(
            bias_buffer, bias_base + i * stride, thread_offset + i * stride < elems_per_row);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            float vals_up_cast = conversion::to<float>(iteration_buffer[j]);
            float res_up_cast = conversion::to<float>(residual_buffer[j]);
            float bias_up_cast = conversion::to<float>(bias_buffer[j]);
            vals_up_cast += res_up_cast + bias_up_cast;
            sum = reduce::element<rop::Add>(sum, vals_up_cast);
            iteration_buffer[j] = conversion::to<T>(vals_up_cast);
        }

        if (StoreResidual && (thread_offset + i * stride < elems_per_row)) {
            mem_access::store_global<ln::granularity>(res_output + base_offset + i * stride,
                                                      iteration_buffer);
        }
    }

    reduce::block<rop::Add, ln::max_warps>(tb, warp, sum);
    const float mean = sum / elems_per_row;

    float mean_diff = reduce::init<rop::Add, float>();
#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                mean_diff = reduce::element<rop::Add>(mean_diff, diff * diff);
            }
        }
    }

    reduce::block<rop::Add, ln::max_warps>(tb, warp, mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    const T mean_compute = conversion::to<T>(mean);
    const T denom_compute = conversion::to<T>(denom);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = (iteration_buffer[j] - mean_compute) * denom_compute;
            iteration_buffer[j] = iteration_buffer[j] * gamma_local[j] + beta_local[j];
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

// TODO(cmikeh2): There's a bunch of redundancy here that needs to be removed/simplified.
#define LAUNCH_FUSED_RES_LN(unroll_factor)                                  \
    fused_residual_ln<T, unroll_factor, false><<<grid, block, 0, stream>>>( \
        output, nullptr, vals, residual, bias, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_residual_ln(T* output,
                              const T* vals,
                              const T* residual,
                              const T* bias,
                              const T* gamma,
                              const T* beta,
                              float epsilon,
                              int rows,
                              int elems_per_row,
                              cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);
    // 32 for __half, 16 for float
    constexpr int T_per_thread_unroll = T_per_load * ln::internal_unroll;
    // 1024 for __half, 512 for float
    constexpr int T_per_warp_unroll = T_per_thread_unroll * hw_warp_size;

    int32_t unroll = 1;
    while (T_per_warp_unroll * ln::max_warps * unroll < elems_per_row) { unroll++; }

    const int warps =
        (elems_per_row + unroll * T_per_warp_unroll - 1) / (unroll * T_per_warp_unroll);

    dim3 grid(rows);
    dim3 block(warps * hw_warp_size);

    // This should match the max_unroll constexpr
    if (unroll == 1) {
        LAUNCH_FUSED_RES_LN(1);
    } else if (unroll == 2) {
        LAUNCH_FUSED_RES_LN(2);
    } else if (unroll == 3) {
        LAUNCH_FUSED_RES_LN(3);
    } else if (unroll == 4) {
        LAUNCH_FUSED_RES_LN(4);
    }
}

#define LAUNCH_FUSED_RES_LN_STORE(unroll_factor)                           \
    fused_residual_ln<T, unroll_factor, true><<<grid, block, 0, stream>>>( \
        norm_output, res_output, vals, residual, bias, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_residual_ln_store(T* norm_output,
                                    T* res_output,
                                    const T* vals,
                                    const T* residual,
                                    const T* bias,
                                    const T* gamma,
                                    const T* beta,
                                    float epsilon,
                                    int rows,
                                    int elems_per_row,
                                    cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);
    // 32 for __half, 16 for float
    constexpr int T_per_thread_unroll = T_per_load * ln::internal_unroll;
    // 1024 for __half, 512 for float
    constexpr int T_per_warp_unroll = T_per_thread_unroll * hw_warp_size;

    int32_t unroll = 1;
    while (T_per_warp_unroll * ln::max_warps * unroll < elems_per_row) { unroll++; }

    const int warps =
        (elems_per_row + unroll * T_per_warp_unroll - 1) / (unroll * T_per_warp_unroll);

    dim3 grid(rows);
    dim3 block(warps * hw_warp_size);

    // This should match the max_unroll constexpr
    if (unroll == 1) {
        LAUNCH_FUSED_RES_LN_STORE(1);
    } else if (unroll == 2) {
        LAUNCH_FUSED_RES_LN_STORE(2);
    } else if (unroll == 3) {
        LAUNCH_FUSED_RES_LN_STORE(3);
    } else if (unroll == 4) {
        LAUNCH_FUSED_RES_LN_STORE(4);
    }
}

// No-store specializations
template void launch_fused_residual_ln(__half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       float,
                                       int,
                                       int,
                                       cudaStream_t);

template void launch_fused_residual_ln(float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       float,
                                       int,
                                       int,
                                       cudaStream_t);

// Store specializations
template void launch_fused_residual_ln_store(__half*,
                                             __half*,
                                             const __half*,
                                             const __half*,
                                             const __half*,
                                             const __half*,
                                             const __half*,
                                             float,
                                             int,
                                             int,
                                             cudaStream_t);

template void launch_fused_residual_ln_store(float*,
                                             float*,
                                             const float*,
                                             const float*,
                                             const float*,
                                             const float*,
                                             const float*,
                                             float,
                                             int,
                                             int,
                                             cudaStream_t);
