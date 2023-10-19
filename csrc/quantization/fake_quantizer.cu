// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <math.h>
#include "custom_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

__global__ void fake_quantize_kernel(__half* vals, int group_size, int num_bits)
{
#if __CUDA_ARCH__ >= 700 || defined(__HIP_PLATFORM_AMD__)

    cg::thread_block b = cg::this_thread_block();  // tb
    cg::thread_block_tile<32> g =
        cg::tiled_partition<32>(b);  // warp, 32 not optimal for AMD which should be 64.

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(__half);

    __half data[vals_per_access];

    int group_id = blockIdx.x;

    int thread_index = id * vals_per_access;
    int reg_count = 0;
    int offset = group_id * group_size;
    float max = -10000.0;
    for (int thread_index = id * vals_per_access; thread_index < group_size;
         thread_index += blockDim.x * vals_per_access) {
        mem_access::load_global<granularity>(data, vals + offset + thread_index);

#pragma unroll
        for (int i = 0; i < vals_per_access; i++) {
            if (abs((float)data[i]) > max) max = abs((float)data[i]);
        }
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (max < temp) max = temp;
    }
    __shared__ float partialMax[WARP_SIZE];

    if (lane == 0) partialMax[gid] = max;

    b.sync();

    if (lane < warp_num) max = partialMax[lane];

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_down(max, i);
        if (max < temp) max = temp;
    }

    max = g.shfl(max, 0);

    float q_scale = (float)(1 << num_bits) / (2 * max + 1e-5);
    float q_scale_inv = 1 / q_scale;
    int q_range_max = (1 << (num_bits - 1)) - 1;
    int q_range_min = -(1 << (num_bits - 1));

    for (int thread_index = id * vals_per_access; thread_index < group_size;
         thread_index += blockDim.x * vals_per_access) {
        mem_access::load_global<granularity>(data, vals + offset + thread_index);
#pragma unroll
        for (int j = 0; j < vals_per_access; j++) {
            float q_data;
            q_data = __half2float(data[j]);
            q_data = __float2int_rn(q_data * q_scale);
            q_data = q_data > (q_range_max) ? (q_range_max)
                                            : (q_data < (q_range_min) ? (q_range_min) : q_data);
            data[j] = __float2half_rn(q_data * q_scale_inv);
        }
        mem_access::store_global<granularity>(vals + offset + thread_index, data);
    }

#endif
}

__global__ void fake_quantize_kernel(float* vals, int group_size, int num_bits)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    constexpr int granularity = 16;
    constexpr int vals_per_access = granularity / sizeof(float);

    float data[vals_per_access];

    int bid = blockIdx.x;

    int thread_index = id * vals_per_access;

    int reg_count = 0;

    int offset = bid * group_size;

    float max = -10000.0;

    for (int thread_index = id * vals_per_access; thread_index < group_size;
         thread_index += blockDim.x * vals_per_access) {
        mem_access::load_global<granularity>(data, vals + offset + thread_index);

#pragma unroll
        for (int i = 0; i < vals_per_access; i++) {
            if (abs(data[i]) > max) max = abs(data[i]);
        }
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (max < temp) max = temp;
    }
    __shared__ float partialMax[WARP_SIZE];

    if (lane == 0) partialMax[gid] = max;

    b.sync();

    if (lane < warp_num) max = partialMax[lane];

    b.sync();

#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shfl_down(max, i);
        if (max < temp) max = temp;
    }

    max = g.shfl(max, 0);

    float q_scale = (1 << num_bits) / (2 * max + 1e-5);
    float q_scale_inv = 1 / q_scale;

    int q_range_max = (1 << (num_bits - 1)) - 1;
    int q_range_min = -(1 << (num_bits - 1));

    for (int thread_index = id * vals_per_access; thread_index < group_size;
         thread_index += blockDim.x * vals_per_access) {
        mem_access::load_global<granularity>(data, vals + offset + thread_index);
#pragma unroll
        for (int j = 0; j < vals_per_access; j++) {
            float q_data;
            q_data = __float2int_rn(data[j] * q_scale);
            q_data = q_data > (q_range_max) ? (q_range_max)
                                            : (q_data < (q_range_min) ? (q_range_min) : q_data);
            data[j] = roundf(q_data * q_scale_inv);
        }
        mem_access::store_global<granularity>(vals + offset + thread_index, data);
    }
}

template <typename T>
void launch_fake_quantize_kernel(T* vals,
                                 int total_count,
                                 int group_num,
                                 int num_bits,
                                 cudaStream_t stream)
{
    dim3 grid_dim(group_num);
    dim3 block_dim(1024);

    fake_quantize_kernel<<<grid_dim, block_dim, 0, stream>>>(
        vals, total_count / group_num, num_bits);
}

template void launch_fake_quantize_kernel(float* vals,
                                          int total_count,
                                          int group_num,
                                          int num_bits,
                                          cudaStream_t stream);
template void launch_fake_quantize_kernel(__half* vals,
                                          int total_count,
                                          int group_num,
                                          int num_bits,
                                          cudaStream_t stream);

__global__ void sr_fake_quantize_kernel(__half* vals,
                                        int token_size,
                                        int token_num,
                                        int num_bits,
                                        std::pair<uint64_t, uint64_t> seed)
{
#if __CUDA_ARCH__ >= 700 || defined(__HIP_PLATFORM_AMD__)

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float2* vals_cast = reinterpret_cast<float2*>(vals);

    __half2 data_low[128];
    __half2 data_high[128];

    int bid = blockIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);
    unsigned int tid = threadIdx.x;
    int reg_count = 0;
    int offset = bid * token_size;
    int group_index = bid * token_size + tid;

    int total_count = token_size * token_num;
    if (group_index < total_count) {
        // float min = 10000.0;
        float max = -10000.0;
        while (tid < token_size) {
            float2 data = vals_cast[offset + tid];
            __half2* data_h = reinterpret_cast<__half2*>(&data);
            data_low[reg_count] = data_h[0];
            data_high[reg_count] = data_h[1];

            float2 data_f[2];
            data_f[0] = __half22float2(data_h[0]);
            data_f[1] = __half22float2(data_h[1]);

            if (abs((float)data_f[0].x) > max) max = abs((float)data_f[0].x);
            if (abs((float)data_f[0].y) > max) max = abs((float)data_f[0].y);
            if (abs((float)data_f[1].x) > max) max = abs((float)data_f[1].x);
            if (abs((float)data_f[1].y) > max) max = abs((float)data_f[1].y);

            tid += blockDim.x;
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shfl_xor(max, i);
            if (max < temp) max = temp;
        }

        __shared__ float partialMax[WARP_SIZE];

        if (lane == 0) partialMax[gid] = max;

        b.sync();

        if (lane < warp_num) max = partialMax[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shfl_down(max, i);
            if (max < temp) max = temp;
        }

        max = g.shfl(max, 0);

        float q_scale_val = (float)(1 << num_bits) / (max * 2 + 1e-5);
        float high_q = (float)((1 << (num_bits - 1)) - 1);
        float low_q = (float)(-((1 << (num_bits - 1))));

        for (int i = 0; i < reg_count; i++) {
            int token_index = i * blockDim.x + threadIdx.x;
            if (token_index < token_size) {
                float2 data_f[2];
                data_f[0] = __half22float2(data_low[i]);
                data_f[1] = __half22float2(data_high[i]);

                float2 q_data_int[2];
                q_data_int[0].x = (float)((int)(data_f[0].x * q_scale_val));
                q_data_int[0].y = (float)((int)(data_f[0].y * q_scale_val));
                q_data_int[1].x = (float)((int)(data_f[1].x * q_scale_val));
                q_data_int[1].y = (float)((int)(data_f[1].y * q_scale_val));

                // Stochastic rounding
                float4 rand = curand_uniform4(&state);

                float q_error[4];
                q_error[0] = abs(data_f[0].x - (q_data_int[0].x / q_scale_val)) * q_scale_val;
                q_error[1] = abs(data_f[0].y - (q_data_int[0].y / q_scale_val)) * q_scale_val;
                q_error[2] = abs(data_f[1].x - (q_data_int[1].x / q_scale_val)) * q_scale_val;
                q_error[3] = abs(data_f[1].y - (q_data_int[1].y / q_scale_val)) * q_scale_val;

                q_data_int[0].x =
                    (rand.x < q_error[0] && q_data_int[0].x > low_q && q_data_int[0].x < high_q)
                        ? (q_data_int[0].x + (data_f[0].x > 0 ? 1 : -1))
                        : q_data_int[0].x;
                q_data_int[0].y =
                    (rand.y < q_error[1] && q_data_int[0].y > low_q && q_data_int[0].y < high_q)
                        ? (q_data_int[0].y + (data_f[0].y > 0 ? 1 : -1))
                        : q_data_int[0].y;
                q_data_int[1].x =
                    (rand.w < q_error[2] && q_data_int[1].x > low_q && q_data_int[1].x < high_q)
                        ? (q_data_int[1].x + (data_f[1].x > 0 ? 1 : -1))
                        : q_data_int[1].x;
                q_data_int[1].y =
                    (rand.z < q_error[3] && q_data_int[1].y > low_q && q_data_int[1].y < high_q)
                        ? (q_data_int[1].y + (data_f[1].y > 0 ? 1 : -1))
                        : q_data_int[1].y;

                data_f[0].x = q_data_int[0].x / q_scale_val;
                data_f[0].y = q_data_int[0].y / q_scale_val;
                data_f[1].x = q_data_int[1].x / q_scale_val;
                data_f[1].y = q_data_int[1].y / q_scale_val;

                float2 result;
                __half2* result_h = reinterpret_cast<__half2*>(&result);
                result_h[0] = __float22half2_rn(data_f[0]);
                result_h[1] = __float22half2_rn(data_f[1]);

                vals_cast[offset + token_index] = result;
            }
        }
    }
#endif
}

__global__ void sr_fake_quantize_kernel(float* vals,
                                        int token_size,
                                        int token_num,
                                        int num_bits,
                                        std::pair<uint64_t, uint64_t> seed)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    int idx = blockIdx.x * blockDim.x + id;

    float4* vals_cast = reinterpret_cast<float4*>(vals);

    float4 data[128];

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    int group_index = bid * token_size + threadIdx.x;
    int reg_count = 0;
    int total_count = token_size * token_num;
    if (group_index < total_count) {
        // float min = 10000.0;
        float max = -10000.0;

        while (tid < token_size) {
            data[reg_count] = vals_cast[group_index];

            if (abs(data[reg_count].x) > max) max = abs(data[reg_count].x);
            if (abs(data[reg_count].y) > max) max = abs(data[reg_count].y);
            if (abs(data[reg_count].z) > max) max = abs(data[reg_count].z);
            if (abs(data[reg_count].w) > max) max = abs(data[reg_count].w);

            group_index += blockDim.x;
            tid += blockDim.x;
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shfl_xor(max, i);
            if (max < temp) max = temp;
        }
        __shared__ float partialMax[WARP_SIZE];

        if (lane == 0) partialMax[gid] = max;

        b.sync();

        if (lane < warp_num) max = partialMax[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shfl_down(max, i);
            if (max < temp) max = temp;
        }

        max = g.shfl(max, 0);

        float q_scale_val = (float)(1 << num_bits) / (max * 2 + 1e-5);
        float high_q = (float)((1 << (num_bits - 1)) - 1);
        float low_q = (float)(-((1 << (num_bits - 1))));

        int offset = (bid)*token_size;
        for (int i = 0; i < reg_count; i++) {
            group_index = i * blockDim.x + threadIdx.x;
            if (group_index < token_size) {
                float4 q_data = data[i];

                float4 q_data_int;
                q_data_int.x = (float)((int)(q_data.x * q_scale_val));
                q_data_int.y = (float)((int)(q_data.y * q_scale_val));
                q_data_int.w = (float)((int)(q_data.w * q_scale_val));
                q_data_int.z = (float)((int)(q_data.z * q_scale_val));

                // Stochastic rounding
                float4 rand = curand_uniform4(&state);

                float q_error[4];
                q_error[0] = abs(q_data.x - (q_data_int.x / q_scale_val)) * q_scale_val;
                q_error[1] = abs(q_data.y - (q_data_int.y / q_scale_val)) * q_scale_val;
                q_error[2] = abs(q_data.w - (q_data_int.w / q_scale_val)) * q_scale_val;
                q_error[3] = abs(q_data.z - (q_data_int.z / q_scale_val)) * q_scale_val;

                q_data_int.x =
                    (rand.x < q_error[0] && q_data_int.x > low_q && q_data_int.x < high_q)
                        ? (q_data_int.x + (q_data.x > 0 ? 1 : -1))
                        : q_data_int.x;
                q_data_int.y =
                    (rand.y < q_error[1] && q_data_int.y > low_q && q_data_int.y < high_q)
                        ? (q_data_int.y + (q_data.y > 0 ? 1 : -1))
                        : q_data_int.y;
                q_data_int.w =
                    (rand.w < q_error[2] && q_data_int.w > low_q && q_data_int.w < high_q)
                        ? (q_data_int.w + (q_data.w > 0 ? 1 : -1))
                        : q_data_int.w;
                q_data_int.z =
                    (rand.z < q_error[3] && q_data_int.z > low_q && q_data_int.z < high_q)
                        ? (q_data_int.z + (q_data.z > 0 ? 1 : -1))
                        : q_data_int.z;

                q_data_int.x /= q_scale_val;
                q_data_int.y /= q_scale_val;
                q_data_int.w /= q_scale_val;
                q_data_int.z /= q_scale_val;

                vals_cast[group_index + offset] = q_data_int;
            }
        }
    }
}

template <typename T>
void launch_sr_fake_quantize_kernel(T* vals,
                                    int total_count,
                                    int group_num,
                                    int num_bits,
                                    cudaStream_t stream)
{
    dim3 block_dim(1024);
    dim3 grid_dim(group_num);

    uint64_t inc = total_count / grid_dim.x / block_dim.x;
    std::pair<uint64_t, uint64_t> seed = TrainingContext::Instance().IncrementOffset(inc);

    sr_fake_quantize_kernel<<<grid_dim, block_dim, 0, stream>>>(
        vals, (total_count / group_num) / 4, group_num, num_bits, seed);
}
template void launch_sr_fake_quantize_kernel(float* vals,
                                             int total_count,
                                             int group_num,
                                             int num_bits,
                                             cudaStream_t stream);
template void launch_sr_fake_quantize_kernel(__half* vals,
                                             int total_count,
                                             int group_num,
                                             int num_bits,
                                             cudaStream_t stream);

__global__ void fake_quantize_kernel_asym(__half* vals, int group_size, int num_bits)
{
#if __CUDA_ARCH__ >= 700 || defined(__HIP_PLATFORM_AMD__)

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    float2* vals_cast = reinterpret_cast<float2*>(vals);

    float2 data[MAX_REG];

    int group_id = blockIdx.x;

    {
        int group_index = id;
        int reg_count = 0;
        int offset = group_id * group_size;
        float max = -10000.0;
        float min = 10000.0;

        while (group_index < group_size && reg_count < MAX_REG) {
            data[reg_count] = vals_cast[offset + group_index];
            __half* data_h = reinterpret_cast<__half*>(&data[reg_count]);

            if (((float)data_h[0]) > max) max = (float)data_h[0];
            if (((float)data_h[1]) > max) max = (float)data_h[1];
            if (((float)data_h[2]) > max) max = (float)data_h[2];
            if (((float)data_h[3]) > max) max = (float)data_h[3];

            if (((float)data_h[0]) < min) min = (float)data_h[0];
            if (((float)data_h[1]) < min) min = (float)data_h[1];
            if (((float)data_h[2]) < min) min = (float)data_h[2];
            if (((float)data_h[3]) < min) min = (float)data_h[3];

            group_index += blockDim.x;
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shfl_xor(max, i);
            if (max < temp) max = temp;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shfl_xor(min, i);
            if (min > temp) min = temp;
        }

        __shared__ float partialMax[WARP_SIZE];
        __shared__ float partialMin[WARP_SIZE];

        if (lane == 0) partialMax[gid] = max;
        if (lane == 0) partialMin[gid] = min;

        b.sync();

        if (lane < warp_num) max = partialMax[lane];
        if (lane < warp_num) min = partialMin[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shfl_down(max, i);
            if (max < temp) max = temp;
        }
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shfl_down(min, i);
            if (min > temp) min = temp;
        }

        max = g.shfl(max, 0);
        min = g.shfl(min, 0);

        float q_scale = ((max - min) + 1e-5) / (float)(1 << num_bits);
        float q_scale_inv = 1 / q_scale;

        for (int i = 0; i < reg_count; i++) {
            group_index = i * blockDim.x + id;
            if (group_index < group_size) {
                __half2* data_h = reinterpret_cast<__half2*>(&data[i]);
                float2 q_data[2];
                q_data[0] = __half22float2(data_h[0]);
                q_data[1] = __half22float2(data_h[1]);

                float2 q_data_int[2];

                q_data_int[0].x = roundf((q_data[0].x - min) * q_scale_inv);
                q_data_int[0].y = roundf((q_data[0].y - min) * q_scale_inv);
                q_data_int[1].x = roundf((q_data[1].x - min) * q_scale_inv);
                q_data_int[1].y = roundf((q_data[1].y - min) * q_scale_inv);

                q_data_int[0].x = q_data_int[0].x * q_scale + min;
                q_data_int[0].y = q_data_int[0].y * q_scale + min;
                q_data_int[1].x = q_data_int[1].x * q_scale + min;
                q_data_int[1].y = q_data_int[1].y * q_scale + min;

                data_h[0] = __float22half2_rn(q_data_int[0]);
                data_h[1] = __float22half2_rn(q_data_int[1]);

                vals_cast[offset + group_index] = data[i];
            }
        }
    }
#endif
}

__global__ void fake_quantize_kernel_asym(float* vals, int group_size, int num_bits)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    float4* vals_cast = reinterpret_cast<float4*>(vals);

    float4 data[MAX_REG];

    int bid = blockIdx.x;

    int group_index = bid * group_size + id;
    int reg_count = 0;

    float max = -10000.0;
    float min = 10000.0;

    while (id < group_size && reg_count < MAX_REG) {
        float4 data_reg = vals_cast[group_index];
        data[reg_count] = data_reg;

        if (data_reg.x > max) max = data_reg.x;
        if (data_reg.y > max) max = data_reg.y;
        if (data_reg.w > max) max = data_reg.w;
        if (data_reg.z > max) max = data_reg.z;

        if (data_reg.x < min) min = data_reg.x;
        if (data_reg.y < min) min = data_reg.y;
        if (data_reg.w < min) min = data_reg.w;
        if (data_reg.z < min) min = data_reg.z;

        group_index += blockDim.x;
        id += blockDim.x;
        reg_count++;
    }
    id = threadIdx.x;

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (max < temp) max = temp;
    }

#pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(min, i);
        if (min > temp) min = temp;
    }

    __shared__ float partialMax[WARP_SIZE];
    __shared__ float partialMin[WARP_SIZE];

    if (lane == 0) partialMax[gid] = max;
    if (lane == 0) partialMin[gid] = min;

    b.sync();

    if (lane < warp_num) max = partialMax[lane];
    if (lane < warp_num) min = partialMin[lane];

#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shfl_down(max, i);
        if (max < temp) max = temp;
    }
#pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shfl_down(min, i);
        if (min > temp) min = temp;
    }

    max = g.shfl(max, 0);
    min = g.shfl(min, 0);

    float q_scale = ((max - min) + 1e-5) / (float)(1 << num_bits);
    float q_scale_inv = 1 / q_scale;
    for (int i = 0; i < reg_count; i++) {
        group_index = i * blockDim.x + id;
        if (group_index < group_size) {
            float4 q_data;
            q_data = data[i];

            float4 q_data_int;
            q_data_int.x = roundf((q_data.x - min) * q_scale_inv);
            q_data_int.y = roundf((q_data.y - min) * q_scale_inv);
            q_data_int.w = roundf((q_data.w - min) * q_scale_inv);
            q_data_int.z = roundf((q_data.z - min) * q_scale_inv);

            q_data.x = q_data_int.x * q_scale + min;
            q_data.y = q_data_int.y * q_scale + min;
            q_data.w = q_data_int.w * q_scale + min;
            q_data.z = q_data_int.z * q_scale + min;

            vals_cast[group_index + bid * group_size] = q_data;
        }
    }
}

template <typename T>
void launch_fake_quantize_kernel_asym(T* vals,
                                      int total_count,
                                      int group_num,
                                      int num_bits,
                                      cudaStream_t stream)
{
    dim3 grid_dim(group_num);
    dim3 block_dim(1024);

    fake_quantize_kernel_asym<<<grid_dim, block_dim, 0, stream>>>(
        vals, (total_count / group_num) / 4, num_bits);
}

template void launch_fake_quantize_kernel_asym(float* vals,
                                               int total_count,
                                               int group_num,
                                               int num_bits,
                                               cudaStream_t stream);
template void launch_fake_quantize_kernel_asym(__half* vals,
                                               int total_count,
                                               int group_num,
                                               int num_bits,
                                               cudaStream_t stream);

__global__ void sr_fake_quantize_kernel_asym(__half* vals,
                                             int token_size,
                                             int token_num,
                                             int num_bits,
                                             std::pair<uint64_t, uint64_t> seed)
{
#if __CUDA_ARCH__ >= 700 || defined(__HIP_PLATFORM_AMD__)

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float2* vals_cast = reinterpret_cast<float2*>(vals);

    __half2 data_low[128];
    __half2 data_high[128];

    int bid = blockIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);
    unsigned int tid = threadIdx.x;
    int reg_count = 0;
    int offset = bid * token_size;
    int group_index = bid * token_size + tid;

    int total_count = token_size * token_num;
    if (group_index < total_count) {
        float min = 10000.0;
        float max = -10000.0;
        while (tid < token_size) {
            float2 data = vals_cast[offset + tid];
            __half2* data_h = reinterpret_cast<__half2*>(&data);
            data_low[reg_count] = data_h[0];
            data_high[reg_count] = data_h[1];

            float2 data_f[2];
            data_f[0] = __half22float2(data_h[0]);
            data_f[1] = __half22float2(data_h[1]);

            if (((float)data_f[0].x) > max) max = (float)data_f[0].x;
            if (((float)data_f[0].y) > max) max = (float)data_f[0].y;
            if (((float)data_f[1].x) > max) max = (float)data_f[1].x;
            if (((float)data_f[1].y) > max) max = (float)data_f[1].y;

            if (((float)data_f[0].x) < min) min = (float)data_f[0].x;
            if (((float)data_f[0].y) < min) min = (float)data_f[0].y;
            if (((float)data_f[1].x) < min) min = (float)data_f[1].x;
            if (((float)data_f[1].y) < min) min = (float)data_f[1].y;

            tid += blockDim.x;
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shfl_xor(max, i);
            if (max < temp) max = temp;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shfl_xor(min, i);
            if (min > temp) min = temp;
        }

        __shared__ float partialMax[WARP_SIZE];
        __shared__ float partialMin[WARP_SIZE];

        if (lane == 0) partialMax[gid] = max;
        if (lane == 0) partialMin[gid] = min;

        b.sync();

        if (lane < warp_num) max = partialMax[lane];
        if (lane < warp_num) min = partialMin[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shfl_down(max, i);
            if (max < temp) max = temp;
        }
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shfl_down(min, i);
            if (min > temp) min = temp;
        }

        max = g.shfl(max, 0);
        min = g.shfl(min, 0);

        float q_scale_val = ((max - min) + 1e-5) / (float)(1 << num_bits);
        float q_scale_val_inv = 1 / q_scale_val;
        float high_q = (float)((1 << num_bits) - 1);

        for (int i = 0; i < reg_count; i++) {
            int token_index = i * blockDim.x + threadIdx.x;
            if (token_index < token_size) {
                float2 data_f[2];
                data_f[0] = __half22float2(data_low[i]);
                data_f[1] = __half22float2(data_high[i]);

                float2 q_data_int[2];
                q_data_int[0].x = (float)((unsigned int)((data_f[0].x - min) * q_scale_val_inv));
                q_data_int[0].y = (float)((unsigned int)((data_f[0].y - min) * q_scale_val_inv));
                q_data_int[1].x = (float)((unsigned int)((data_f[1].x - min) * q_scale_val_inv));
                q_data_int[1].y = (float)((unsigned int)((data_f[1].y - min) * q_scale_val_inv));

                // Stochastic rounding
                float4 rand = curand_uniform4(&state);

                float q_error[4];
                q_error[0] =
                    abs(data_f[0].x - ((q_data_int[0].x * q_scale_val) + min)) * q_scale_val_inv;
                q_error[1] =
                    abs(data_f[0].y - ((q_data_int[0].y * q_scale_val) + min)) * q_scale_val_inv;
                q_error[2] =
                    abs(data_f[1].x - ((q_data_int[1].x * q_scale_val) + min)) * q_scale_val_inv;
                q_error[3] =
                    abs(data_f[1].y - ((q_data_int[1].y * q_scale_val) + min)) * q_scale_val_inv;

                q_data_int[0].x = (rand.x < q_error[0] && q_data_int[0].x < high_q)
                                      ? (q_data_int[0].x + 1)
                                      : q_data_int[0].x;
                q_data_int[0].y = (rand.y < q_error[1] && q_data_int[0].y < high_q)
                                      ? (q_data_int[0].y + 1)
                                      : q_data_int[0].y;
                q_data_int[1].x = (rand.w < q_error[2] && q_data_int[1].x < high_q)
                                      ? (q_data_int[1].x + 1)
                                      : q_data_int[1].x;
                q_data_int[1].y = (rand.z < q_error[3] && q_data_int[1].y < high_q)
                                      ? (q_data_int[1].y + 1)
                                      : q_data_int[1].y;

                data_f[0].x = q_data_int[0].x * q_scale_val + min;
                data_f[0].y = q_data_int[0].y * q_scale_val + min;
                data_f[1].x = q_data_int[1].x * q_scale_val + min;
                data_f[1].y = q_data_int[1].y * q_scale_val + min;

                float2 result;
                __half2* result_h = reinterpret_cast<__half2*>(&result);
                result_h[0] = __float22half2_rn(data_f[0]);
                result_h[1] = __float22half2_rn(data_f[1]);

                vals_cast[offset + token_index] = result;
            }
        }
    }
#endif
}

__global__ void sr_fake_quantize_kernel_asym(float* vals,
                                             int token_size,
                                             int token_num,
                                             int num_bits,
                                             std::pair<uint64_t, uint64_t> seed)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    int idx = blockIdx.x * blockDim.x + id;

    float4* vals_cast = reinterpret_cast<float4*>(vals);

    float4 data[128];

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    int group_index = bid * token_size + threadIdx.x;
    int reg_count = 0;
    int total_count = token_size * token_num;
    if (group_index < total_count) {
        float min = 10000.0;
        float max = -10000.0;

        while (tid < token_size) {
            float4 data_reg = vals_cast[group_index];
            data[reg_count] = data_reg;
            if (data_reg.x > max) max = data_reg.x;
            if (data_reg.y > max) max = data_reg.y;
            if (data_reg.w > max) max = data_reg.w;
            if (data_reg.z > max) max = data_reg.z;

            if (data_reg.x < min) min = data_reg.x;
            if (data_reg.y < min) min = data_reg.y;
            if (data_reg.w < min) min = data_reg.w;
            if (data_reg.z < min) min = data_reg.z;

            group_index += blockDim.x;
            tid += blockDim.x;
            reg_count++;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shfl_xor(max, i);
            if (max < temp) max = temp;
        }

#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
            auto temp = g.shfl_xor(min, i);
            if (min > temp) min = temp;
        }

        __shared__ float partialMax[WARP_SIZE];
        __shared__ float partialMin[WARP_SIZE];

        if (lane == 0) partialMax[gid] = max;
        if (lane == 0) partialMin[gid] = min;

        b.sync();

        if (lane < warp_num) max = partialMax[lane];
        if (lane < warp_num) min = partialMin[lane];

#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shfl_down(max, i);
            if (max < temp) max = temp;
        }
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            auto temp = g.shfl_down(min, i);
            if (min > temp) min = temp;
        }

        max = g.shfl(max, 0);
        min = g.shfl(min, 0);

        float q_scale_val = ((max - min) + 1e-5) / (float)(1 << num_bits);
        float high_q = (float)((1 << num_bits) - 1);

        int offset = (bid)*token_size;
        for (int i = 0; i < reg_count; i++) {
            group_index = i * blockDim.x + threadIdx.x;
            if (group_index < token_size) {
                float4 q_data = data[i];

                float4 q_data_int;
                q_data_int.x = (float)((int)((q_data.x - min) / q_scale_val));
                q_data_int.y = (float)((int)((q_data.y - min) / q_scale_val));
                q_data_int.w = (float)((int)((q_data.w - min) / q_scale_val));
                q_data_int.z = (float)((int)((q_data.z - min) / q_scale_val));

                // Stochastic rounding
                float4 rand = curand_uniform4(&state);

                float q_error[4];
                q_error[0] = abs(q_data.x - ((q_data_int.x * q_scale_val) + min)) / q_scale_val;
                q_error[1] = abs(q_data.y - ((q_data_int.y * q_scale_val) + min)) / q_scale_val;
                q_error[2] = abs(q_data.w - ((q_data_int.w * q_scale_val) + min)) / q_scale_val;
                q_error[3] = abs(q_data.z - ((q_data_int.z * q_scale_val) + min)) / q_scale_val;

                q_data_int.x = (rand.x < q_error[0] && q_data_int.x < high_q) ? (q_data_int.x + 1)
                                                                              : q_data_int.x;
                q_data_int.y = (rand.y < q_error[1] && q_data_int.y < high_q) ? (q_data_int.y + 1)
                                                                              : q_data_int.y;
                q_data_int.w = (rand.w < q_error[2] && q_data_int.w < high_q) ? (q_data_int.w + 1)
                                                                              : q_data_int.w;
                q_data_int.z = (rand.z < q_error[3] && q_data_int.z < high_q) ? (q_data_int.z + 1)
                                                                              : q_data_int.z;

                q_data_int.x = q_data_int.x * q_scale_val + min;
                q_data_int.y = q_data_int.y * q_scale_val + min;
                q_data_int.w = q_data_int.w * q_scale_val + min;
                q_data_int.z = q_data_int.z * q_scale_val + min;

                vals_cast[group_index + offset] = q_data_int;
            }
        }
    }
}
template <typename T>
void launch_sr_fake_quantize_kernel_asym(T* vals,
                                         int total_count,
                                         int group_num,
                                         int num_bits,
                                         cudaStream_t stream)
{
    dim3 block_dim(1024);
    dim3 grid_dim(group_num);

    uint64_t inc = total_count / grid_dim.x / block_dim.x;
    std::pair<uint64_t, uint64_t> seed = TrainingContext::Instance().IncrementOffset(inc);

    sr_fake_quantize_kernel<<<grid_dim, block_dim, 0, stream>>>(
        vals, (total_count / group_num) / 4, group_num, num_bits, seed);
}
template void launch_sr_fake_quantize_kernel_asym(float* vals,
                                                  int total_count,
                                                  int group_num,
                                                  int num_bits,
                                                  cudaStream_t stream);
template void launch_sr_fake_quantize_kernel_asym(__half* vals,
                                                  int total_count,
                                                  int group_num,
                                                  int num_bits,
                                                  cudaStream_t stream);
