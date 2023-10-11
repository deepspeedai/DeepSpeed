// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "memory_access_utils.h"

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedArray {
    using Element = T;
    static const int kElements = N;

    __device__ __host__ AlignedArray() {}

    __device__ __host__ AlignedArray(const T& rhs)
    {
#pragma unroll
        for (int idx = 0; idx < kElements; ++idx) { this->at(idx) = rhs; }
    }

    __device__ __host__ T& operator[](int offset)
    {
        return reinterpret_cast<T&>(this->buffer[offset]);
    }

    __device__ __host__ const T& operator[](int offset) const
    {
        return reinterpret_cast<const T&>(this->buffer[offset]);
    }

    __device__ __host__ T& at(int offset) { return reinterpret_cast<T&>(this->buffer[offset]); }

    __device__ __host__ const T& at(int offset) const
    {
        return reinterpret_cast<const T&>(this->buffer[offset]);
    }

    __device__ __host__ AlignedArray<T, N> operator+(const AlignedArray<T, N>& rhs) const
    {
        AlignedArray<T, N> ret;

#pragma unroll
        for (int idx = 0; idx < kElements; ++idx) { ret[idx] = this->at(idx) + rhs.at(idx); }

        return ret;
    }

    __device__ __forceinline__ void clear()
    {
#pragma unroll
        for (int idx = 0; idx < kElements; ++idx) { this->at(idx) = Element(0); }
    }

    Element buffer[N];
};

template <typename T>
struct reduce_max {
    __device__ __forceinline__ T operator()(const T& lhs, const T& rhs)
    {
        return lhs > rhs ? lhs : rhs;
    }
};

template <typename T>
struct reduce_min {
    __device__ __forceinline__ T operator()(const T& lhs, const T& rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }
};

template <typename T, int N>
struct subtract {
    __device__ __forceinline__ AlignedArray<T, N> operator()(const AlignedArray<T, N>& lhs,
                                                             const T& rhs)
    {
        AlignedArray<T, N> ret;

#pragma unroll
        for (int idx = 0; idx < N; ++idx) { ret[idx] = lhs[idx] - rhs; }

        return ret;
    }
};

template <typename T, int N>
struct plus {
    __device__ __forceinline__ AlignedArray<T, N> operator()(const AlignedArray<T, N>& lhs,
                                                             const T& rhs)
    {
        AlignedArray<T, N> ret;

#pragma unroll
        for (int idx = 0; idx < N; ++idx) { ret[idx] = lhs[idx] + rhs; }

        return ret;
    }
};

template <typename T, int N>
struct multiply {
    __device__ __forceinline__ AlignedArray<T, N> operator()(const AlignedArray<T, N>& lhs,
                                                             const T& rhs)
    {
        AlignedArray<T, N> ret;

#pragma unroll
        for (int idx = 0; idx < N; ++idx) { ret[idx] = lhs[idx] * rhs; }

        return ret;
    }
};

template <typename T, int N>
struct clamp {
    __device__ __forceinline__ AlignedArray<T, N> operator()(const AlignedArray<T, N>& lhs,
                                                             const T& min_val,
                                                             const T& max_val)
    {
        AlignedArray<T, N> ret;

#pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = reduce_max<T>()(reduce_min<T>()(lhs[idx], max_val), min_val);
        }

        return ret;
    }
};

template <typename T, int N>
struct round_int;

template <int N>
struct round_int<half, N> {
    __device__ __forceinline__ AlignedArray<half, N> operator()(const AlignedArray<half, N>& lhs)
    {
        AlignedArray<half, N> ret;

#pragma unroll
        for (int idx = 0; idx < N; ++idx) { ret[idx] = hrint(lhs[idx]); }

        return ret;
    }
};

template <typename T, int N>
struct divide {
    __device__ __forceinline__ AlignedArray<T, N> operator()(const AlignedArray<T, N>& lhs,
                                                             const T& rhs)
    {
        AlignedArray<T, N> ret;

#pragma unroll
        for (int idx = 0; idx < N; ++idx) { ret[idx] = lhs[idx] / rhs; }

        return ret;
    }
};

template <typename T, int N, typename Reducer>
__device__ __forceinline__ T to_scalar(const AlignedArray<T, N>& data)
{
    Reducer re;
    T res = data[0];

#pragma unroll
    for (int idx = 1; idx < N; ++idx) { res = re(res, data[idx]); }

    return res;
}

template <int N>
__device__ __forceinline__ AlignedArray<half, N * 2> int4_to_half(
    const AlignedArray<uint8_t, N>& data)
{
    AlignedArray<half, N * 2> ret;

#pragma unroll
    for (int idx = 0; idx < N * 2; idx += 2) {
        ret[idx] = half(int(data[idx / 2] >> 4));
        ret[idx + 1] = half(int(data[idx / 2] & 0xf));
    }

    return ret;
}

__global__ void dequantize_int4_to_half(uint8_t* data_in,
                                        half* data_out,
                                        half* scale_buffer,
                                        half* min_val_buffer,
                                        int num_group,
                                        int group_size)
{
    using AccessType = AlignedArray<uint8_t, 4>;
    using AccessTypeOut = AlignedArray<half, 8>;

    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_group * group_size / 8;
         idx += blockDim.x * gridDim.x) {
        int id_group = idx / (group_size / 8);
        AccessType value = reinterpret_cast<AccessType*>(data_in)[idx];
        half scale = scale_buffer[id_group];
        half min_value = min_val_buffer[id_group];

        AccessTypeOut output = int4_to_half(value);
        output = divide<half, 8>()(output, scale);
        output = plus<half, 8>()(output, min_value);

        reinterpret_cast<AccessTypeOut*>(data_out)[idx] = output;
    }
}

void launch_dequantize_int4_to_half_experimental(uint8_t* data_in,
                                                 half* data_out,
                                                 half* scale_buffer,
                                                 half* min_val_buffer,
                                                 int num_group,
                                                 int group_size,
                                                 cudaStream_t stream)
{
    int num_warp = num_group / 4;
    int num_block = num_warp / 8;  // 256 trd / block

    dequantize_int4_to_half<<<num_block, 256, 0, stream>>>(
        data_in, data_out, scale_buffer, min_val_buffer, num_group, group_size);
}

template <int N>
__device__ __forceinline__ AlignedArray<half, N> int8_to_half(const AlignedArray<uint8_t, N>& data)
{
    AlignedArray<half, N> ret;

#pragma unroll
    for (int idx = 0; idx < N; idx += 1) { ret[idx] = half(int(data[idx])); }

    return ret;
}

__global__ void dequantize_int8_to_half(uint8_t* data_in,
                                        half* data_out,
                                        half* scale_buffer,
                                        half* min_val_buffer,
                                        int num_group,
                                        int group_size)
{
    using AccessType = AlignedArray<uint8_t, 8>;
    using AccessTypeOut = AlignedArray<half, 8>;

    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_group * group_size / 8;
         idx += blockDim.x * gridDim.x) {
        int id_group = idx / (group_size / 8);
        AccessType value = reinterpret_cast<AccessType*>(data_in)[idx];
        half scale = scale_buffer[id_group];
        half min_value = min_val_buffer[id_group];

        AccessTypeOut output = int8_to_half(value);
        output = divide<half, 8>()(output, scale);
        output = plus<half, 8>()(output, min_value);

        reinterpret_cast<AccessTypeOut*>(data_out)[idx] = output;
    }
}

void launch_dequantize_int8_to_half_experimental(uint8_t* data_in,
                                                 half* data_out,
                                                 half* scale_buffer,
                                                 half* min_val_buffer,
                                                 int num_group,
                                                 int group_size,
                                                 cudaStream_t stream)
{
    int num_warp = num_group / 4;
    int num_block = num_warp / 8;  // 256 trd / block

    dequantize_int8_to_half<<<num_block, 256, 0, stream>>>(
        data_in, data_out, scale_buffer, min_val_buffer, num_group, group_size);
}
