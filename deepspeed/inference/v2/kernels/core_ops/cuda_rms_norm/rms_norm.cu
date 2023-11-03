// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"

namespace cg = cooperative_groups;
using rop = reduce::ROpType;

namespace rms {
constexpr int granularity = 16;
}  // namespace rms

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
__global__ void rms_norm(T* output, const T* vals, const T* gamma, float epsilon, int elems_per_row)
{
    constexpr int T_per_load = rms::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // X-dimension of the block
    const int block_offset = (tb.group_index().x * (maxThreads / threadsPerGroup) * elems_per_row) +
                             (tb.thread_index().y * elems_per_row);
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = blockDim.x * T_per_load;

    float var_sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        T* iteration_buffer = local_buffer + (i * T_per_load);

        mem_access::load_global<rms::granularity>(iteration_buffer,
                                                  input_base + (i * stride),
                                                  thread_offset + (i * stride) < elems_per_row);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            float up_cast = conversion::to<float>(iteration_buffer[j]);
            float sq_val = up_cast * up_cast;
            var_sum = reduce::element<rop::Add, float>(var_sum, sq_val);
        }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, var_sum);
    const float var = var_sum / elems_per_row;
    const T denom = conversion::to<T>(__frsqrt_rn(var + epsilon));

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        T* iteration_buffer = local_buffer + (i * T_per_load);
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = (iter_idx < elems_per_row);

        T gamma_local[T_per_load];

        mem_access::load_global<rms::granularity>(gamma_local, gamma + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] *= denom;
            iteration_buffer[j] *= gamma_local[j];
        }

        if (do_loads) {
            mem_access::store_global<rms::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
__global__ void pre_rms_norm(T* output,
                             T* res_out,
                             const T* vals,
                             const T* residual,
                             const T* gamma,
                             float epsilon,
                             int elems_per_row)
{
    constexpr int T_per_load = rms::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // X-dimension of the block
    const int block_offset = (tb.group_index().x * (maxThreads / threadsPerGroup) * elems_per_row) +
                             (tb.thread_index().y * elems_per_row);
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = blockDim.x * T_per_load;

    float var_sum = reduce::init<rop::Add, float>();

    const T* input_base = vals + base_offset;
    const T* residual_base = residual + base_offset;
    T* res_output = res_out + base_offset;

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        T* iteration_buffer = local_buffer + (i * T_per_load);
        T residual_buffer[T_per_load];

        const int iter_offset = i * stride + thread_offset;
        const bool do_loads = (iter_offset < elems_per_row);

        mem_access::load_global<rms::granularity>(
            iteration_buffer, input_base + (i * stride), do_loads);
        mem_access::load_global<rms::granularity>(
            residual_buffer, residual_base + (i * stride), do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] += residual_buffer[j];
            float vals_up_cast = conversion::to<float>(iteration_buffer[j]);

            var_sum = reduce::element<rop::Add, float>(var_sum, vals_up_cast * vals_up_cast);
        }

        if (do_loads) {
            mem_access::store_global<rms::granularity>(res_output + i * stride, iteration_buffer);
        }
    }

    reduce::partitioned_block<rop::Add, threadsPerGroup>(tb, warp, var_sum);
    const float var = var_sum / elems_per_row;
    const T denom = conversion::to<T>(__frsqrt_rn(var + epsilon));

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        T* iteration_buffer = local_buffer + (i * T_per_load);
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = (iter_idx < elems_per_row);

        T gamma_local[T_per_load];

        mem_access::load_global<rms::granularity>(gamma_local, gamma + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] *= denom;
            iteration_buffer[j] *= gamma_local[j];
        }

        if (do_loads) {
            mem_access::store_global<rms::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

#define LAUNCH_RMS_NORM(UNROLL, threadsPerGroup, maxThreads) \
    rms_norm<T, UNROLL, threadsPerGroup, maxThreads>         \
        <<<grid, block, 0, stream>>>(norm_output, vals, gamma, epsilon, elems_per_row);

#define LAUNCH_PRE_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)                      \
    pre_rms_norm<T, UNROLL, threadsPerGroup, maxThreads><<<grid, block, 0, stream>>>( \
        norm_output, res_output, vals, residual, gamma, epsilon, elems_per_row);

#define LAUNCH_ALL_RMS_NORM(UNROLL, threadsPerGroup, maxThreads) \
    if (pre_norm) {                                              \
        LAUNCH_PRE_RMS_NORM(UNROLL, threadsPerGroup, maxThreads) \
    } else {                                                     \
        LAUNCH_RMS_NORM(UNROLL, threadsPerGroup, maxThreads)     \
    }

template <typename T>
void launch_rms_norm(T* norm_output,
                     T* res_output,
                     const T* vals,
                     const T* residual,
                     const T* gamma,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     cudaStream_t stream)
{
    // 8 for __half, 4 for float
    constexpr int T_per_load = rms::granularity / sizeof(T);
    constexpr int maxThreads = 256;
    constexpr int internalUnroll = sizeof(T) == 4 ? 4 : 2;

    const bool is_subblock_schedule = (elems_per_row <= 128) ? true : false;
    const int h_per_step = is_subblock_schedule ? T_per_load : T_per_load * internalUnroll;

    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads = next_pow2((elems_per_row + h_per_step - 1) / h_per_step);
    const int threads_per_group = (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

    const int groups_per_block_max =
        is_subblock_schedule ? (maxThreads + threads_per_group - 1) / threads_per_group : 1;
    const int groups_per_block = (rows < groups_per_block_max) ? rows : groups_per_block_max;
    const int groups_launch = (groups_per_block + rows - 1) / groups_per_block;

    dim3 block(threads_per_group, groups_per_block);
    dim3 grid(groups_launch);

    const int elems_per_step = threads_per_group * h_per_step;
    const int external_unRoll = (elems_per_row + elems_per_step - 1) / elems_per_step;

    bool pre_norm = (residual == nullptr) ? false : true;

    if (is_subblock_schedule) {
        // <=128
        if (threads_per_group == 1) {
            LAUNCH_ALL_RMS_NORM(1, 1, maxThreads);
        } else if (threads_per_group == 2) {
            LAUNCH_ALL_RMS_NORM(1, 2, maxThreads);
        } else if (threads_per_group == 4) {
            LAUNCH_ALL_RMS_NORM(1, 4, maxThreads);
        } else if (threads_per_group == 8) {
            LAUNCH_ALL_RMS_NORM(1, 8, maxThreads);
        } else if (threads_per_group == 16) {
            LAUNCH_ALL_RMS_NORM(1, 16, maxThreads);
        }
    } else if (external_unRoll == 1) {
        // 129 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_ALL_RMS_NORM(1 * internalUnroll, maxThreads, maxThreads);
    } else if (external_unRoll == 2) {
        // 4097 - 8192 elems
        LAUNCH_ALL_RMS_NORM(2 * internalUnroll, maxThreads, maxThreads);
    } else if (external_unRoll == 3) {
        // 8193 - 12288 elems
        LAUNCH_ALL_RMS_NORM(3 * internalUnroll, maxThreads, maxThreads);
    } else if (external_unRoll == 4) {
        // 12289 - 16384 elems
        LAUNCH_ALL_RMS_NORM(4 * internalUnroll, maxThreads, maxThreads);
    }
}

#define INSTANTIATE_LAUNCH_RMS_NORM(T)                  \
    template void launch_rms_norm<T>(T * norm_output,   \
                                     T * res_output,    \
                                     const T* vals,     \
                                     const T* residual, \
                                     const T* gamma,    \
                                     float epsilon,     \
                                     int rows,          \
                                     int elems_per_row, \
                                     cudaStream_t stream);

INSTANTIATE_LAUNCH_RMS_NORM(float)
INSTANTIATE_LAUNCH_RMS_NORM(__half)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_RMS_NORM(__nv_bfloat16)
#endif
