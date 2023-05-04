// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <limits>
#include "conversion_utils.h"
#include "inference_cuda_layers.h"

#ifndef __HIP_PLATFORM_HCC__
#include <cuda_profiler_api.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define MAX_REG_SIZE 8

#define minus_infinity -10000.0

void CheckCudaErrorAux(const char* file, unsigned line)
{
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) return;
    std::cerr << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line
              << std::endl;
    throw std::runtime_error("CUDA ERROR!!!\n");
}

#define CUDA_CHECK_ERROR() CheckCudaErrorAux(__FILE__, __LINE__)

namespace cg = cooperative_groups;

template <typename T, int iterations>
__global__ void attn_softmax_v2(T* vals,
                                T* mask,
                                T* alibi,
                                float layer_scale,
                                bool triangular,
                                bool recompute,
                                bool local_attention,
                                int window_size,
                                int total_count,
                                int heads,
                                int sequence_length,
                                int num_seq,
                                int head_offset,
                                int mask_stride,
                                int mp_size,
                                int reduceWidth)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    float2 low_data[MAX_REG_SIZE];
    float2 high_data[MAX_REG_SIZE];
    const T zero_h = conversion::to<T>(0.f);

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = threadIdx.x % reduceWidth;

    __shared__ float partialSum[MAX_WARP_NUM];

    int iter_offset = blockIdx.x * (warp_num / reduce_blocks) + (wid / reduce_blocks);
    int batch_idx = iter_offset / (num_seq * heads);
    int alibi_offset = batch_idx * heads * mp_size + head_offset;
    int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);

    if (iter_offset < total_count) {
        vals += (iter_offset * sequence_length);

        alibi_offset = (alibi_offset + ((iter_offset / num_seq) % heads)) * sequence_length;
        mask_offset = mask_offset * sequence_length;
        int seq_id = iter_offset % num_seq;

        int real_seq_id = seq_id + (num_seq == sequence_length ? 0 : sequence_length);
        int window_stride4 = (local_attention && (real_seq_id >> 2) > (window_size >> 2))
                                 ? (real_seq_id >> 2) - (window_size >> 2)
                                 : 0;
        int window_stride =
            (local_attention && real_seq_id >= window_size) ? real_seq_id - window_size : -1;

        float max_val = minus_infinity;
        // if (lane == 0) printf("%d, %d: %d \n", wid, blockIdx.x, mask_offset);
        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            bool check = (data_id >> 2) >= window_stride4;
            bool low_x_check = check && (data_id < sequence_length) &&
                               (!triangular || (data_id <= seq_id)) && (data_id > window_stride);
            bool low_y_check = check && ((data_id + reduceWidth) < sequence_length) &&
                               (!triangular || ((data_id + reduceWidth) <= seq_id)) &&
                               ((data_id + reduceWidth) > window_stride);
            bool high_x_check = check && ((data_id + reduceWidth * 2) < sequence_length) &&
                                (!triangular || ((data_id + reduceWidth * 2) <= seq_id)) &&
                                ((data_id + reduceWidth * 2) > window_stride);
            bool high_y_check = check && ((data_id + reduceWidth * 3) < sequence_length) &&
                                (!triangular || ((data_id + reduceWidth * 3) <= seq_id)) &&
                                ((data_id + reduceWidth * 3) > window_stride);

            if (mask && alibi) {
                low_data[i].x = low_x_check
                                    ? conversion::to<float>(vals[data_id]) * layer_scale +
                                          (conversion::to<float>(alibi[data_id + alibi_offset])) +
                                          (conversion::to<float>(mask[data_id + mask_offset]))
                                    : minus_infinity;
                low_data[i].y =
                    low_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale +
                              (conversion::to<float>(alibi[data_id + alibi_offset + reduceWidth])) +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth]))
                        : minus_infinity;
                high_data[i].x =
                    high_x_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 2]) * layer_scale +
                              (conversion::to<float>(
                                  alibi[data_id + alibi_offset + reduceWidth * 2])) +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth * 2]))
                        : minus_infinity;
                high_data[i].y =
                    high_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 3]) * layer_scale +
                              (conversion::to<float>(
                                  alibi[data_id + alibi_offset + reduceWidth * 3])) +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth * 3]))
                        : minus_infinity;
            } else if (mask) {
                low_data[i].x = low_x_check
                                    ? conversion::to<float>(vals[data_id]) * layer_scale +
                                          (conversion::to<float>(mask[data_id + mask_offset]))
                                    : minus_infinity;
                low_data[i].y =
                    low_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth]))
                        : minus_infinity;
                high_data[i].x =
                    high_x_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 2]) * layer_scale +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth * 2]))
                        : minus_infinity;
                high_data[i].y =
                    high_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 3]) * layer_scale +
                              (conversion::to<float>(mask[data_id + mask_offset + reduceWidth * 3]))
                        : minus_infinity;
            } else if (alibi) {
                low_data[i].x = low_x_check
                                    ? conversion::to<float>(vals[data_id]) * layer_scale +
                                          (conversion::to<float>(alibi[data_id + alibi_offset]))
                                    : minus_infinity;
                low_data[i].y =
                    low_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale +
                              (conversion::to<float>(alibi[data_id + alibi_offset + reduceWidth]))
                        : minus_infinity;
                high_data[i].x =
                    high_x_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 2]) * layer_scale +
                              (conversion::to<float>(
                                  alibi[data_id + alibi_offset + reduceWidth * 2]))
                        : minus_infinity;
                high_data[i].y =
                    high_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 3]) * layer_scale +
                              (conversion::to<float>(
                                  alibi[data_id + alibi_offset + reduceWidth * 3]))
                        : minus_infinity;
            } else {
                low_data[i].x = low_x_check ? conversion::to<float>(vals[data_id]) * layer_scale
                                            : minus_infinity;
                low_data[i].y =
                    low_y_check ? conversion::to<float>(vals[data_id + reduceWidth]) * layer_scale
                                : minus_infinity;
                high_data[i].x =
                    high_x_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 2]) * layer_scale
                        : minus_infinity;
                high_data[i].y =
                    high_y_check
                        ? conversion::to<float>(vals[data_id + reduceWidth * 3]) * layer_scale
                        : minus_infinity;
            }

            // if(lane == 0) printf("%f , %d, %d \n", low_data[i].x, data_id, seq_id);
            max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
            max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
            max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
            max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = max_val;
            b.sync();

            if (lane < warp_num) max_val = partialSum[lane];

            b.sync();

            for (int i = 1; i < reduce_blocks; i *= 2) {
                auto temp = g.shfl_xor(max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            max_val = g.shfl(max_val, threadIdx.x / WARP_SIZE);
        }
        float sum = 0;
        for (int i = 0; i < iterations; i++) {
            low_data[i].x = __expf(low_data[i].x - max_val);
            low_data[i].y = __expf(low_data[i].y - max_val);
            high_data[i].x = __expf(high_data[i].x - max_val);
            high_data[i].y = __expf(high_data[i].y - max_val);

            sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) sum += g.shfl_xor(sum, i);

        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = sum;
            b.sync();

            if (lane < warp_num) sum = partialSum[lane];

            b.sync();

            for (int i = 1; i < reduce_blocks; i *= 2) { sum += g.shfl_xor(sum, i); }

            sum = g.shfl(sum, threadIdx.x / WARP_SIZE);
        }
        sum += 1e-6;
        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            if (data_id < sequence_length) {
                vals[data_id] = conversion::to<T>(low_data[i].x / sum);
                if ((data_id + reduceWidth) < sequence_length)
                    vals[data_id + reduceWidth] = conversion::to<T>(low_data[i].y / sum);
                if ((data_id + reduceWidth * 2) < sequence_length)
                    vals[data_id + reduceWidth * 2] = conversion::to<T>(high_data[i].x / sum);
                if ((data_id + reduceWidth * 3) < sequence_length)
                    vals[data_id + reduceWidth * 3] = conversion::to<T>(high_data[i].y / sum);
            }
        }
    }
}

template <int iterations>
__global__ void attn_softmax_v2(float* vals,
                                float* attn_mask,
                                float* alibi,
                                float layer_scale,
                                bool triangular,
                                bool recompute,
                                bool local_attention,
                                int window_size,
                                int total_count,
                                int heads,
                                int sequence_length,
                                int num_seq,
                                int head_offset,
                                int mask_stride,
                                int mp_size,
                                int reduceWidth)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    float4 data[MAX_REG_SIZE];

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    int reduce_blocks = reduceWidth >> 5;
    int seq_lane = threadIdx.x % reduceWidth;

    __shared__ float partialSum[MAX_WARP_NUM];

    int iter_offset = blockIdx.x * (warp_num / reduce_blocks) + (wid / reduce_blocks);
    if (iter_offset < total_count) {
        vals += (iter_offset * sequence_length);

        int batch_idx = iter_offset / (num_seq * heads);
        int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);
        mask_offset = mask_offset * sequence_length;
        int seq_id = iter_offset % num_seq;

        int real_seq_id = seq_id + (num_seq == sequence_length ? 0 : sequence_length);
        int window_stride4 = (local_attention && (real_seq_id >> 2) > (window_size >> 2))
                                 ? (real_seq_id >> 2) - (window_size >> 2)
                                 : 0;
        int window_stride =
            (local_attention && real_seq_id >= window_size) ? real_seq_id - window_size : -1;

        float max_val = minus_infinity;

        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            bool check = (data_id >> 2) >= window_stride4;
            bool x_check = check && (data_id < sequence_length) &&
                           (!triangular || (data_id <= seq_id)) && (data_id > window_stride);
            bool y_check = check && ((data_id + reduceWidth) < sequence_length) &&
                           (!triangular || ((data_id + reduceWidth) <= seq_id)) &&
                           ((data_id + reduceWidth) > window_stride);
            bool z_check = check && ((data_id + reduceWidth * 2) < sequence_length) &&
                           (!triangular || ((data_id + reduceWidth * 2) <= seq_id)) &&
                           ((data_id + reduceWidth * 2) > window_stride);
            bool w_check = check && ((data_id + reduceWidth * 3) < sequence_length) &&
                           (!triangular || ((data_id + reduceWidth * 3) <= seq_id)) &&
                           ((data_id + reduceWidth * 3) > window_stride);

            if (attn_mask) {
                data[i].x = x_check ? vals[data_id] + attn_mask[data_id + mask_offset]
                                    : minus_infinity;
                data[i].y = y_check ? vals[data_id + reduceWidth] +
                                          attn_mask[data_id + mask_offset + reduceWidth]
                                    : minus_infinity;
                data[i].z = z_check ? vals[data_id + reduceWidth * 2] +
                                          attn_mask[data_id + mask_offset + reduceWidth * 2]
                                    : minus_infinity;
                data[i].w = w_check ? vals[data_id + reduceWidth * 3] +
                                          attn_mask[data_id + mask_offset + reduceWidth * 3]
                                    : minus_infinity;
            } else {
                data[i].x = x_check ? vals[data_id] : minus_infinity;
                data[i].y = y_check ? vals[data_id + reduceWidth] : minus_infinity;
                data[i].z = z_check ? vals[data_id + reduceWidth * 2] : minus_infinity;
                data[i].w = w_check ? vals[data_id + reduceWidth * 3] : minus_infinity;
            }

            max_val = (data[i].x > max_val ? data[i].x : max_val);
            max_val = (data[i].y > max_val ? data[i].y : max_val);
            max_val = (data[i].z > max_val ? data[i].z : max_val);
            max_val = (data[i].w > max_val ? data[i].w : max_val);
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = max_val;
            b.sync();

            if (lane < warp_num) max_val = partialSum[lane];

            b.sync();

            for (int i = 1; i < reduce_blocks; i *= 2) {
                auto temp = g.shfl_xor(max_val, i);
                max_val = (temp > max_val ? temp : max_val);
            }

            max_val = g.shfl(max_val, threadIdx.x / WARP_SIZE);
        }

        float sum = 0;
        for (int i = 0; i < iterations; i++) {
            data[i].x = __expf(data[i].x - max_val);
            data[i].y = __expf(data[i].y - max_val);
            data[i].z = __expf(data[i].z - max_val);
            data[i].w = __expf(data[i].w - max_val);

            sum += (data[i].x + data[i].y + data[i].z + data[i].w);
        }

        for (int i = 1; i < WARP_SIZE; i *= 2) sum += g.shfl_xor(sum, i);

        if (reduceWidth > WARP_SIZE) {
            if (lane == 0) partialSum[wid] = sum;
            b.sync();

            if (lane < warp_num) sum = partialSum[lane];

            b.sync();

            for (int i = 1; i < reduce_blocks; i *= 2) { sum += g.shfl_xor(sum, i); }

            sum = g.shfl(sum, threadIdx.x / WARP_SIZE);
        }
        sum += 1e-6;

        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane);
            if (data_id < sequence_length) {
                vals[data_id] = data[i].x / sum;
                if ((data_id + reduceWidth) < sequence_length)
                    vals[data_id + reduceWidth] = data[i].y / sum;
                if ((data_id + reduceWidth * 2) < sequence_length)
                    vals[data_id + reduceWidth * 2] = data[i].z / sum;
                if ((data_id + reduceWidth * 3) < sequence_length)
                    vals[data_id + reduceWidth * 3] = data[i].w / sum;
            }
        }
    }
}

#define LAUNCH_ATTN_SOFTMAX_V2(iterations)                                      \
    attn_softmax_v2<T, iterations><<<grid, block, 0, stream>>>(vals,            \
                                                               mask,            \
                                                               alibi,           \
                                                               layer_scale,     \
                                                               triangular,      \
                                                               recompute,       \
                                                               local_attention, \
                                                               window_size,     \
                                                               total_count,     \
                                                               heads,           \
                                                               sequence_length, \
                                                               num_seq,         \
                                                               head_offset,     \
                                                               mask_stride,     \
                                                               mp_size,         \
                                                               reduce_width);

template <typename T>
void launch_attn_softmax_v2(T* vals,
                            T* mask,
                            T* alibi,
                            float layer_scale,
                            bool triangular,
                            bool recompute,
                            bool local_attention,
                            int window_size,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            int head_offset,
                            int mask_stride,
                            int mp_size,
                            cudaStream_t stream)
{
    const int total_count = batch_size * heads * num_seq;

    // Scheduling Overview
    // 4 element unroll with power of 2 `reduce_width` threads to a ceiling of `attn_threads`
    // Each block should be partitioned into as many `reduce_width` blocks
    // as can be fit.
    constexpr int attn_threads = 256;
    constexpr int min_reduce_width = hw_warp_size;
    constexpr int internal_unroll = 4;

    // Handle internal unroll then round to next power of 2. Bump up to minimum granularity.
    const int thread_steps_rounded =
        next_pow2((sequence_length + internal_unroll - 1) / internal_unroll);
    const int thread_steps_schedule =
        (thread_steps_rounded < min_reduce_width) ? min_reduce_width : thread_steps_rounded;
    // Bound reduce width to the number of threads
    const int reduce_width = (thread_steps_schedule < attn_threads) ? thread_steps_schedule
                                                                    : attn_threads;
    // Scale for the excess
    const int iterations = thread_steps_schedule / reduce_width;
    // Should be safe since reduce_width is capped to attn_threads
    const int partitions = attn_threads / reduce_width;

    // Launch params
    dim3 grid((total_count + partitions - 1) / partitions);
    dim3 block(attn_threads);

    if (sequence_length <= 32768) {
        if (iterations == 1) {
            LAUNCH_ATTN_SOFTMAX_V2(1);
        } else if (iterations == 2) {
            LAUNCH_ATTN_SOFTMAX_V2(2);
        } else if (iterations == 4) {
            LAUNCH_ATTN_SOFTMAX_V2(4);
        } else if (iterations == 8) {
            LAUNCH_ATTN_SOFTMAX_V2(8);
        } else if (iterations == 16) {
            LAUNCH_ATTN_SOFTMAX_V2(16);
        } else if (iterations == 32) {
            LAUNCH_ATTN_SOFTMAX_V2(32);
        } else if (iterations == 64) {
            LAUNCH_ATTN_SOFTMAX_V2(64);
        }
    } else
        throw std::runtime_error("Unsupport Seq_Length!");
}

#define INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(T)                  \
    template void launch_attn_softmax_v2(T* vals,              \
                                         T* mask,              \
                                         T* alibi,             \
                                         float layer_scale,    \
                                         bool triangular,      \
                                         bool recompute,       \
                                         bool local_attention, \
                                         int window_size,      \
                                         int batch_size,       \
                                         int heads,            \
                                         int num_seq,          \
                                         int sequence_length,  \
                                         int head_offset,      \
                                         int mask_stride,      \
                                         int mp_size,          \
                                         cudaStream_t stream);

INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(float);
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(__nv_bfloat16);
#endif
INSTANTIATE_LAUNCH_ATTN_SOFTMAX_V2(__half);

#define DEF_ATTN_SOFTMAX_V2_HALF(_iter)                                           \
    template __global__ void attn_softmax_v2<__half, _iter>(__half * vals,        \
                                                            __half * mask,        \
                                                            __half * alibi,       \
                                                            float layer_scale,    \
                                                            bool triangular,      \
                                                            bool recompute,       \
                                                            bool local_attention, \
                                                            int window_size,      \
                                                            int total_count,      \
                                                            int heads,            \
                                                            int sequence_length,  \
                                                            int num_seq,          \
                                                            int head_offset,      \
                                                            int mask_stride,      \
                                                            int mp_size,          \
                                                            int reduceWidth)

#define DEF_ATTN_SOFTMAX_V2_BF16(_iter)                                                   \
    template __global__ void attn_softmax_v2<__nv_bfloat16, _iter>(__nv_bfloat16 * vals,  \
                                                                   __nv_bfloat16 * mask,  \
                                                                   __nv_bfloat16 * alibi, \
                                                                   float layer_scale,     \
                                                                   bool triangular,       \
                                                                   bool recompute,        \
                                                                   bool local_attention,  \
                                                                   int window_size,       \
                                                                   int total_count,       \
                                                                   int heads,             \
                                                                   int sequence_length,   \
                                                                   int num_seq,           \
                                                                   int head_offset,       \
                                                                   int mask_stride,       \
                                                                   int mp_size,           \
                                                                   int reduceWidth)

#define FOREACH_ITERATIONS(cb) \
    cb(1);                     \
    cb(2);                     \
    cb(4);                     \
    cb(8);                     \
    cb(16);                    \
    cb(32);                    \
    cb(64)

FOREACH_ITERATIONS(DEF_ATTN_SOFTMAX_V2_HALF);
#ifdef BF16_AVAILABLE
FOREACH_ITERATIONS(DEF_ATTN_SOFTMAX_V2_BF16);
#endif
