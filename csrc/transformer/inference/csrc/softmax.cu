/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include <limits>
#include "inference_cuda_layers.h"

#ifndef __HIP_PLATFORM_HCC__
#include <cuda_profiler_api.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define ATTN_THREADS 256
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

__global__ void attn_softmax_v2(__half* vals,
                                __half* mask,
                                __half* alibi,
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
                                int iterations,
                                int reduceWidth)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    float2 low_data[MAX_REG_SIZE];
    float2 high_data[MAX_REG_SIZE];
    const __half zero_h = __float2half(0.f);

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
        int seq_id4 = seq_id >> 2;

        int real_seq_id = seq_id + (num_seq == sequence_length ? 0 : sequence_length);
        int window_stride4 = (local_attention && (real_seq_id >> 2) > (window_size >> 2))
                                 ? (real_seq_id >> 2) - (window_size >> 2)
                                 : 0;
        int window_stride =
            (local_attention && real_seq_id >= window_size) ? real_seq_id - window_size : -1;

        float max_val = minus_infinity;
        // if (lane == 0) printf("%d, %d: %d \n", wid, blockIdx.x, mask_offset);
        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane << 2);
            if ((!triangular || ((data_id >> 2) <= seq_id4)) && (data_id >> 2) >= window_stride4 &&
                data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    low_data[i].x = data_id > window_stride
                                        ? __half2float(vals[data_id]) * layer_scale
                                        : minus_infinity;
                    low_data[i].y = ((!triangular || ((data_id + 1) <= seq_id)) &&
                                     (data_id + 1) > window_stride)
                                        ? __half2float(vals[data_id + 1]) * layer_scale
                                        : minus_infinity;
                    high_data[i].x = ((!triangular || ((data_id + 2) <= seq_id)) &&
                                      (data_id + 2) > window_stride)
                                         ? __half2float(vals[data_id + 2]) * layer_scale
                                         : minus_infinity;
                    high_data[i].y = ((!triangular || ((data_id + 3) <= seq_id)) &&
                                      (data_id + 3) > window_stride)
                                         ? __half2float(vals[data_id + 3]) * layer_scale
                                         : minus_infinity;
                    if (alibi) {
                        low_data[i].x = low_data[i].x + __half2float(alibi[data_id + alibi_offset]);
                        low_data[i].y =
                            low_data[i].y + __half2float(alibi[data_id + alibi_offset + 1]);
                        high_data[i].x =
                            high_data[i].x + __half2float(alibi[data_id + alibi_offset + 2]);
                        high_data[i].y =
                            high_data[i].y + __half2float(alibi[data_id + alibi_offset + 3]);
                    }
                    if (mask) {
                        low_data[i].x += __half2float(mask[data_id + mask_offset]);
                        low_data[i].y += __half2float(mask[data_id + mask_offset + 1]);
                        high_data[i].x += __half2float(mask[data_id + mask_offset + 2]);
                        high_data[i].y += __half2float(mask[data_id + mask_offset + 3]);
                    }
                } else {
                    low_data[i].x = data_id > window_stride
                                        ? __half2float(vals[data_id]) * layer_scale
                                        : minus_infinity;
                    low_data[i].y = (((!triangular || (data_id + 1) <= seq_id) &&
                                      (data_id + 1) > window_stride) &&
                                     (data_id + 1) < sequence_length)
                                        ? __half2float(vals[data_id + 1]) * layer_scale
                                        : minus_infinity;
                    high_data[i].x = (((!triangular || (data_id + 2) <= seq_id) &&
                                       (data_id + 2) > window_stride) &&
                                      (data_id + 2) < sequence_length)
                                         ? __half2float(vals[data_id + 2]) * layer_scale
                                         : minus_infinity;
                    if (alibi) {
                        low_data[i].x = low_data[i].x + __half2float(alibi[data_id + alibi_offset]);
                        if ((data_id + 1) < sequence_length)
                            low_data[i].y =
                                low_data[i].y + __half2float(alibi[data_id + alibi_offset + 1]);
                        if ((data_id + 2) < sequence_length)
                            high_data[i].x =
                                high_data[i].x + __half2float(alibi[data_id + alibi_offset + 2]);
                    }
                    high_data[i].y = minus_infinity;
                    if (mask) {
                        low_data[i].x += __half2float(mask[data_id + mask_offset]);
                        if ((data_id + 1) < sequence_length)
                            low_data[i].y += __half2float(mask[data_id + mask_offset + 1]);
                        if ((data_id + 2) < sequence_length)
                            high_data[i].x += __half2float(mask[data_id + mask_offset + 2]);
                    }
                }
                // if(lane == 0) printf("%f , %d, %d \n", low_data[i].x, data_id, seq_id);
                max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
                max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
                max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
                max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
            } else {
                low_data[i].x = minus_infinity;
                low_data[i].y = minus_infinity;
                high_data[i].x = minus_infinity;
                high_data[i].y = minus_infinity;
            }
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
            int data_id = i * (reduceWidth << 2) + (seq_lane << 2);

            if (data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    vals[data_id] = __float2half(low_data[i].x / sum);
                    vals[data_id + 1] = __float2half(low_data[i].y / sum);
                    vals[data_id + 2] = __float2half(high_data[i].x / sum);
                    vals[data_id + 3] = __float2half(high_data[i].y / sum);
                } else {
                    vals[data_id] = __float2half(low_data[i].x / sum);
                    if ((data_id + 1) < sequence_length)
                        vals[data_id + 1] = __float2half(low_data[i].y / sum);
                    if ((data_id + 2) < sequence_length)
                        vals[data_id + 2] = __float2half(high_data[i].x / sum);
                }
            }
        }
    }
}

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
                                int iterations,
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
        int alibi_offset = batch_idx * heads * mp_size + head_offset;
        int mask_offset = batch_idx * mask_stride + (iter_offset % mask_stride);
        mask_offset = mask_offset * sequence_length;
        int seq_id = iter_offset % num_seq;
        int seq_id4 = seq_id >> 2;

        int real_seq_id = seq_id + (num_seq == sequence_length ? 0 : sequence_length);
        int window_stride4 = (local_attention && (real_seq_id >> 2) > (window_size >> 2))
                                 ? (real_seq_id >> 2) - (window_size >> 2)
                                 : 0;
        int window_stride =
            (local_attention && real_seq_id >= window_size) ? real_seq_id - window_size : -1;

        float max_val = minus_infinity;

        for (int i = 0; i < iterations; i++) {
            int data_id = i * (reduceWidth << 2) + (seq_lane << 2);
            if ((!triangular || ((data_id >> 2) <= seq_id4)) && (data_id >> 2) >= window_stride4 &&
                data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    data[i].x = (data_id > window_stride ? vals[data_id] : minus_infinity);
                    data[i].y = ((!triangular || ((data_id + 1) <= seq_id)) &&
                                 (data_id + 1) > window_stride)
                                    ? vals[data_id + 1]
                                    : minus_infinity;
                    data[i].z = ((!triangular || ((data_id + 2) <= seq_id)) &&
                                 (data_id + 2) > window_stride)
                                    ? vals[data_id + 2]
                                    : minus_infinity;
                    data[i].w = ((!triangular || ((data_id + 3) <= seq_id)) &&
                                 (data_id + 3) > window_stride)
                                    ? vals[data_id + 3]
                                    : minus_infinity;
                    if (attn_mask) {
                        data[i].x += attn_mask[data_id + mask_offset];
                        data[i].y += attn_mask[data_id + mask_offset + 1];
                        data[i].z += attn_mask[data_id + mask_offset + 2];
                        data[i].w += attn_mask[data_id + mask_offset + 3];
                    }
                } else {
                    data[i].x = data_id > window_stride ? vals[data_id] : minus_infinity;
                    data[i].y = (((!triangular || (data_id + 1) <= seq_id)) &&
                                 (data_id + 1) > window_stride && (data_id + 1) < sequence_length)
                                    ? (vals[data_id + 1])
                                    : minus_infinity;
                    data[i].z = (((!triangular || (data_id + 2) <= seq_id)) &&
                                 (data_id + 2) > window_stride && (data_id + 2) < sequence_length)
                                    ? (vals[data_id + 2])
                                    : minus_infinity;
                    data[i].w = minus_infinity;
                    if (attn_mask) {
                        data[i].x += attn_mask[data_id + mask_offset];
                        if ((data_id + 1) < sequence_length)
                            data[i].y += attn_mask[data_id + mask_offset + 1];
                        if ((data_id + 2) < sequence_length)
                            data[i].z += attn_mask[data_id + mask_offset + 2];
                    }
                }
                max_val = (data[i].x > max_val ? data[i].x : max_val);
                max_val = (data[i].y > max_val ? data[i].y : max_val);
                max_val = (data[i].z > max_val ? data[i].z : max_val);
                max_val = (data[i].w > max_val ? data[i].w : max_val);
            } else {
                data[i].x = minus_infinity;
                data[i].y = minus_infinity;
                data[i].z = minus_infinity;
                data[i].w = minus_infinity;
            }
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
            int data_id = i * (reduceWidth << 2) + (seq_lane << 2);

            if (data_id < sequence_length) {
                if ((sequence_length - data_id) >= 4) {
                    vals[data_id] = data[i].x / sum;
                    vals[data_id + 1] = data[i].y / sum;
                    vals[data_id + 2] = data[i].z / sum;
                    vals[data_id + 3] = data[i].w / sum;
                } else {
                    vals[data_id] = data[i].x / sum;
                    if ((data_id + 1) < sequence_length) vals[data_id + 1] = data[i].y / sum;
                    if ((data_id + 2) < sequence_length) vals[data_id + 2] = data[i].z / sum;
                }
            }
        }
    }
}

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
    int total_count = batch_size * heads * num_seq;
    int warp_num = ATTN_THREADS / WARP_SIZE;
    int reduce_width = ((sequence_length - 1) / ATTN_THREADS + 1);
    reduce_width = (int)pow(2.0, floor(log2((float)(reduce_width)))) * WARP_SIZE;
    dim3 grid_dim((total_count - 1) / (ATTN_THREADS / reduce_width) + 1);
    dim3 block_dim(ATTN_THREADS);

    const int iterations = (sequence_length - 1) / (reduce_width << 2) + 1;

    if (sequence_length <= 32768)
        attn_softmax_v2<<<grid_dim, block_dim, 0, stream>>>(vals,
                                                            mask,
                                                            alibi,
                                                            layer_scale,
                                                            triangular,
                                                            recompute,
                                                            local_attention,
                                                            window_size,
                                                            total_count,
                                                            heads,
                                                            sequence_length,
                                                            num_seq,
                                                            head_offset,
                                                            mask_stride,
                                                            mp_size,
                                                            iterations,
                                                            reduce_width);
    else
        throw std::runtime_error("Unsupport Seq_Length!");
}

template void launch_attn_softmax_v2(float* vals,
                                     float* mask,
                                     float* alibi,
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
                                     cudaStream_t stream);
template void launch_attn_softmax_v2(__half* vals,
                                     __half* mask,
                                     __half* alibi,
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
                                     cudaStream_t stream);
