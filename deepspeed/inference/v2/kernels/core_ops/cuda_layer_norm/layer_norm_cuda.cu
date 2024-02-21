// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"

namespace cg = cooperative_groups;
using rop = reduce::ROpType;

namespace ln {
constexpr int granularity = 16;
}  // namespace ln

/*
Regular layer norm implementation. Assumes elems_per_row % 8
is equal to 0.

Args:
    output: buffer for output data
    vals: buffer for input data
    gamma: gain for normalization
    beta: bias for normalization
    epsilon: numeric stability
    elems_per_row: number of elements each block will normalize
*/
template <typename T, int unRoll, int threadsPerGroup, int maxThreads>
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
    const int block_offset = (tb.group_index().x * (maxThreads / threadsPerGroup) * elems_per_row) +
                             (tb.thread_index().y * elems_per_row);
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = blockDim.x * T_per_load;

    float sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;

    T local_buffer[unRoll * T_per_load];

#pragma unRoll
    for (int i = 0; i < unRoll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;

        mem_access::load_global<ln::granularity>(
            iteration_buffer, input_base + i * stride, thread_offset + i * stride < elems_per_row);

#pragma unRoll
        for (int j = 0; j < T_per_load; j++) {
            float vals_up_cast = conversion::to<float>(iteration_buffer[j]);
            sum = reduce::element<rop::Add>(sum, vals_up_cast);
        }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, sum);
    const float mean = sum / elems_per_row;

    float mean_diff = reduce::init<rop::Add, float>();

#pragma unRoll
    for (int i = 0; i < unRoll; i++) {
#pragma unRoll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                mean_diff = reduce::element<rop::Add>(mean_diff, diff * diff);
            }
        }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    T* block_output = output + block_offset;

#pragma unRoll
    for (int i = 0; i < unRoll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unRoll
        for (int j = 0; j < T_per_load; j++) {
            float val = conversion::to<float>(iteration_buffer[j]);
            val = (val - mean) * denom;
            val =
                val * conversion::to<float>(gamma_local[j]) + conversion::to<float>(beta_local[j]);
            iteration_buffer[j] = conversion::to<T>(val);
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

#define LAUNCH_FUSED_LN(unRollFactor, threadsPerGroup, maxThreads) \
    fused_ln<T, unRollFactor, threadsPerGroup, maxThreads>         \
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

    constexpr int maxThreads = 256;

    // For Flaoat, unRoll 4, for __half, unRoll 2
    constexpr int internal_unRoll = sizeof(T) == 4 ? 4 : 2;

    const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
    const int h_per_step = is_subblock_schedule ? T_per_load : T_per_load * internal_unRoll;

    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
    const int threadsPerGroup = (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

    const int groups_per_block_max =
        is_subblock_schedule ? (maxThreads + threadsPerGroup - 1) / threadsPerGroup : 1;
    const int groups_per_block = (rows < groups_per_block_max) ? rows : groups_per_block_max;
    const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

    dim3 block(threadsPerGroup, groups_per_block);
    dim3 grid(groups_launch);

    const int elems_per_step = threadsPerGroup * h_per_step;
    const int external_unRoll = (elems_per_row + elems_per_step - 1) / elems_per_step;

    if (is_subblock_schedule) {
        // <=128
        if (threadsPerGroup == 1) {
            LAUNCH_FUSED_LN(1, 1, maxThreads);
        } else if (threadsPerGroup == 2) {
            LAUNCH_FUSED_LN(1, 2, maxThreads);
        } else if (threadsPerGroup == 4) {
            LAUNCH_FUSED_LN(1, 4, maxThreads);
        } else if (threadsPerGroup == 8) {
            LAUNCH_FUSED_LN(1, 8, maxThreads);
        } else if (threadsPerGroup == 16) {
            LAUNCH_FUSED_LN(1, 16, maxThreads);
        }
    } else if (external_unRoll == 1) {
        // 129 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_FUSED_LN(1 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 2) {
        // 4097 - 8192 elems
        LAUNCH_FUSED_LN(2 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 3) {
        // 8193 - 12288 elems
        LAUNCH_FUSED_LN(3 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 4) {
        // 12289 - 16384 elems
        LAUNCH_FUSED_LN(4 * internal_unRoll, maxThreads, maxThreads);
    }
}

#define INSTANTIATE_FUSED_LN(T) \
    template void launch_fused_ln(T*, const T*, const T*, const T*, float, int, int, cudaStream_t);

INSTANTIATE_FUSED_LN(__half);
#ifdef BF16_AVAILABLE
INSTANTIATE_FUSED_LN(__nv_bfloat16);
#endif
INSTANTIATE_FUSED_LN(float);

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
template <typename T, int unRoll, int threadsPerGroup, int maxThreads, bool preLnResidual>
__global__ void fused_residual_ln(T* output,
                                  T* res_output,
                                  const T* vals,
                                  const T* residual,
                                  const T* gamma,
                                  const T* beta,
                                  float epsilon,
                                  int elems_per_row)
{
    constexpr int T_per_load = ln::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // X-dimension of the block
    const int block_offset = (tb.group_index().x * (maxThreads / threadsPerGroup) * elems_per_row) +
                             (tb.thread_index().y * elems_per_row);
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.size() * T_per_load;

    float sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;
    const T* residual_base = residual + base_offset;

    T local_buffer[unRoll * T_per_load];

    // Unlike a vanilla layernorm, since we're fusing the two adds as well
    // an inner unRoll seems to be less valuable. If anything, a double unRoll
    // makes the most sense if we find we are having performance issues.
#pragma unRoll
    for (int i = 0; i < unRoll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        T residual_buffer[T_per_load];
        T bias_buffer[T_per_load];

        mem_access::load_global<ln::granularity>(
            iteration_buffer, input_base + i * stride, thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(residual_buffer,
                                                 residual_base + i * stride,
                                                 thread_offset + i * stride < elems_per_row);

#pragma unRoll
        for (int j = 0; j < T_per_load; j++) {
            float vals_up_cast = conversion::to<float>(iteration_buffer[j]);
            float res_up_cast = conversion::to<float>(residual_buffer[j]);
            vals_up_cast += res_up_cast;
            sum = reduce::element<rop::Add>(sum, vals_up_cast);
            iteration_buffer[j] = conversion::to<T>(vals_up_cast);
        }

        if (preLnResidual && (thread_offset + i * stride < elems_per_row)) {
            mem_access::store_global<ln::granularity>(res_output + base_offset + i * stride,
                                                      iteration_buffer);
        }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, sum);
    const float mean = sum / elems_per_row;

    float mean_diff = reduce::init<rop::Add, float>();
#pragma unRoll
    for (int i = 0; i < unRoll; i++) {
#pragma unRoll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                mean_diff = reduce::element<rop::Add>(mean_diff, diff * diff);
            }
        }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    T* block_output = output + block_offset;

#pragma unRoll
    for (int i = 0; i < unRoll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unRoll
        for (int j = 0; j < T_per_load; j++) {
            float val = conversion::to<float>(iteration_buffer[j]);
            val = (val - mean) * denom;
            val =
                val * conversion::to<float>(gamma_local[j]) + conversion::to<float>(beta_local[j]);
            iteration_buffer[j] = conversion::to<T>(val);
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

// TODO(cmikeh2): There's a bunch of redundancy here that needs to be removed/simplified.
#define LAUNCH_FUSED_RES_LN(unRollFactor, threadsPerGroup, maxThreads)     \
    fused_residual_ln<T, unRollFactor, threadsPerGroup, maxThreads, false> \
        <<<grid, block, 0, stream>>>(                                      \
            output, nullptr, vals, residual, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_post_ln(T* output,
                          const T* vals,
                          const T* residual,
                          const T* gamma,
                          const T* beta,
                          float epsilon,
                          int rows,
                          int elems_per_row,
                          cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);

    constexpr int maxThreads = 256;

    // For Flaoat, unRoll 4, for __half, unRoll 2
    constexpr int internal_unRoll = sizeof(T) == 4 ? 4 : 2;

    const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
    const int h_per_step = is_subblock_schedule ? T_per_load : T_per_load * internal_unRoll;

    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
    const int threadsPerGroup = (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

    const int groups_per_block_max =
        is_subblock_schedule ? (maxThreads + threadsPerGroup - 1) / threadsPerGroup : 1;
    const int groups_per_block = (rows < groups_per_block_max) ? rows : groups_per_block_max;
    const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

    dim3 block(threadsPerGroup, groups_per_block);
    dim3 grid(groups_launch);

    const int elems_per_step = threadsPerGroup * h_per_step;
    const int external_unRoll = (elems_per_row + elems_per_step - 1) / elems_per_step;

    if (is_subblock_schedule) {
        // <=128
        if (threadsPerGroup == 1) {
            LAUNCH_FUSED_RES_LN(1, 1, maxThreads);
        } else if (threadsPerGroup == 2) {
            LAUNCH_FUSED_RES_LN(1, 2, maxThreads);
        } else if (threadsPerGroup == 4) {
            LAUNCH_FUSED_RES_LN(1, 4, maxThreads);
        } else if (threadsPerGroup == 8) {
            LAUNCH_FUSED_RES_LN(1, 8, maxThreads);
        } else if (threadsPerGroup == 16) {
            LAUNCH_FUSED_RES_LN(1, 16, maxThreads);
        }
    } else if (external_unRoll == 1) {
        // 129 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_FUSED_RES_LN(1 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 2) {
        // 4097 - 8192 elems
        LAUNCH_FUSED_RES_LN(2 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 3) {
        // 8193 - 12288 elems
        LAUNCH_FUSED_RES_LN(3 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 4) {
        // 12289 - 16384 elems
        LAUNCH_FUSED_RES_LN(4 * internal_unRoll, maxThreads, maxThreads);
    }
}

#define LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(unRollFactor, threadsPerGroup, maxThreads) \
    fused_residual_ln<T, unRollFactor, threadsPerGroup, maxThreads, true>               \
        <<<grid, block, 0, stream>>>(                                                   \
            norm_output, res_output, vals, residual, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_pre_ln(T* norm_output,
                         T* res_output,
                         const T* vals,
                         const T* residual,
                         const T* gamma,
                         const T* beta,
                         float epsilon,
                         int rows,
                         int elems_per_row,
                         cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);

    constexpr int maxThreads = 256;

    // For Flaoat, unRoll 4, for __half, unRoll 2
    constexpr int internal_unRoll = sizeof(T) == 4 ? 4 : 2;

    const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
    const int h_per_step = is_subblock_schedule ? T_per_load : T_per_load * internal_unRoll;

    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
    const int threadsPerGroup = (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

    const int groups_per_block_max =
        is_subblock_schedule ? (maxThreads + threadsPerGroup - 1) / threadsPerGroup : 1;
    const int groups_per_block = (rows < groups_per_block_max) ? rows : groups_per_block_max;
    const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

    dim3 block(threadsPerGroup, groups_per_block);
    dim3 grid(groups_launch);

    const int elems_per_step = threadsPerGroup * h_per_step;
    const int external_unRoll = (elems_per_row + elems_per_step - 1) / elems_per_step;

    if (is_subblock_schedule) {
        // <=128
        if (threadsPerGroup == 1) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 1, maxThreads);
        } else if (threadsPerGroup == 2) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 2, maxThreads);
        } else if (threadsPerGroup == 4) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 4, maxThreads);
        } else if (threadsPerGroup == 8) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 8, maxThreads);
        } else if (threadsPerGroup == 16) {
            LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1, 16, maxThreads);
        }
    } else if (external_unRoll == 1) {
        // 129 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(1 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 2) {
        // 4097 - 8192 elems
        LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(2 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 3) {
        // 8193 - 12288 elems
        LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(3 * internal_unRoll, maxThreads, maxThreads);
    } else if (external_unRoll == 4) {
        // 12289 - 16384 elems
        LAUNCH_FUSED_RES_LN_STORE_PRE_LN_RES(4 * internal_unRoll, maxThreads, maxThreads);
    }
}

#define INSTANTIATE_RES_LN(T)              \
    template void launch_fused_post_ln<T>( \
        T*, const T*, const T*, const T*, const T*, float, int, int, cudaStream_t);

#define INSTANTIATE_PRE_LN_RES(T)         \
    template void launch_fused_pre_ln<T>( \
        T*, T*, const T*, const T*, const T*, const T*, float, int, int, cudaStream_t);

INSTANTIATE_RES_LN(__half);
INSTANTIATE_RES_LN(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_RES_LN(__nv_bfloat16);
#endif

INSTANTIATE_PRE_LN_RES(__half);
INSTANTIATE_PRE_LN_RES(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_PRE_LN_RES(__nv_bfloat16);
#endif
