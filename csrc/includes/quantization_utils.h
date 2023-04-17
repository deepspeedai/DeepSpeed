// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cassert>
#include "conversion_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization.h"
#include "reduction_utils.h"

#pragma once

using rop = reduce::ROpType;

namespace quantize {
constexpr int granularity = 16;
constexpr int h_per_load = granularity / sizeof(__half);
constexpr int h2_per_load = granularity / sizeof(__half2);
constexpr int max_threads = 1024;

/*
Class to hold the quantization parameters for a given tensor.
Holds the implementation of the quantization operation.
*/
template <Type qType, int numBits>
class Params {
public:
    /*
    Quantization implementation, supports
    1) 4 Bit
    2) 8 Bit
    3) Symmetric
    4) Asymmetric
    Function Arguments :
        val : The __half value to quantize.
    */
    DS_D_INLINE int8_t quantize(__half val);

    template <typename T>
    DS_D_INLINE T dequantize(int8_t val);

    DS_D_INLINE void store(float* params, int group_index);

    // Initialize from memory
    DS_D_INLINE Params(const float* params, int group_index);
};

template <int numBits>
class Params<Type::Symmetric, numBits> {
public:
    float scale;

    DS_D_INLINE Params(float max)
    {
        if (max == 0) {
            scale = 1.0;
        } else {
            scale = (1 << numBits) / (2 * max);
        }
    }

    DS_D_INLINE int8_t quantize(__half val)
    {
        constexpr int32_t q_min = -(1 << (numBits - 1));
        constexpr int32_t q_max = (1 << (numBits - 1)) - 1;

        float val_f = conversion::to<float>(val) * scale;
        int32_t data_i32 = conversion::to<int32_t>(val_f);
        data_i32 = min(max(data_i32, q_min), q_max);
        return (int8_t)data_i32;
    }

    template <typename T>
    DS_D_INLINE T dequantize(int8_t val)
    {
        const float val_deq_f = conversion::to<float>(val) * scale;
        return conversion::to<T>(val_deq_f);
    }

    DS_D_INLINE void store(float* params, int group_index)
    {
        const float store_scale = 1 / scale;
        mem_access::store_global<sizeof(float)>(params + group_index, &store_scale);
    }

    DS_D_INLINE Params(const float* params, int group_index)
    {
        mem_access::load_global<sizeof(float)>(&scale, params + group_index);
    }
};

template <int numBits>
class Params<Type::Asymmetric, numBits> {
public:
    float scale;
    float offset;

    DS_D_INLINE Params(float max, float min)
    {
        if (max == min) {
            scale = 1.0;
        } else {
            scale = ((1 << numBits)) / (max - min);
        }
        offset = (max + min) / 2;
    }

    DS_D_INLINE int8_t quantize(__half val)
    {
        constexpr int32_t q_min = -(1 << (numBits - 1));
        constexpr int32_t q_max = (1 << (numBits - 1)) - 1;

        float val_f = (conversion::to<float>(val) - offset) * scale;
        int32_t data_i32 = conversion::to<int32_t>(val_f);
        data_i32 = min(max(data_i32, q_min), q_max);
        return (int8_t)data_i32;
    }

    template <typename T>
    DS_D_INLINE T dequantize(int8_t val)
    {
        const float val_deq_f = ((conversion::to<float>(val)) * scale) + offset;
        return conversion::to<__half>(val_deq_f);
    }

    DS_D_INLINE void store(float* params, int group_index)
    {
        // Codegen should turn this into stg.64
        const float store_scale = 1 / scale;
        mem_access::store_global<sizeof(float)>(params + 2 * group_index, &store_scale);
        mem_access::store_global<sizeof(float)>(params + 2 * group_index + 1, &offset);
    }

    DS_D_INLINE Params(const float* params, int group_index)
    {
        // Codegen should turn this into ldg.64
        mem_access::load_global<sizeof(float)>(&scale, params + 2 * group_index);
        mem_access::load_global<sizeof(float)>(&offset, params + 2 * group_index + 1);
    }
};

/*
Group stats tracks the necessary statistics about the quantized group
to abstract the particulars for the main loop.
*/
template <Type qType>
class GroupStats {
public:
    DS_D_INLINE void update(__half2 val);

    DS_D_INLINE void reduce(cg::thread_block& tb, cg::thread_block_tile<hw_warp_size>& warp);
};

template <>
class GroupStats<Type::Symmetric> {
public:
    // Symmetric quantization only tracks the maximum absolute value
    __half2 cur_max;
    float max;

    /*
    Technically, this would give bad results if there
    are 0 values to process since the reduction would
    give -inf instead of 0. We do not consider this
    to be a reasonable edge case.
    */
    DS_D_INLINE GroupStats() { cur_max = reduce::init<rop::Max, __half2>(); }

    /*
    Updated the running absmax used to calculate params.
    Function Arguments :
        val : The __half2 value to update the running min and max with.
    */
    DS_D_INLINE void update(__half2 val)
    {
        cur_max = reduce::element<rop::Max>(cur_max, __habs2(val));
    }

    /*
    Function to return calculated quantization params.
    Template Arguments :
        numBits -   Number of bits in quantized element.    int : 8 or 4
    Function Arguments :
        tb      -   Threadblock object. cg::thread_block
        warp    -   Warp object.        cg::thread_block_tile<hw_warp_size>
    */
    template <int numBits, int threads_per_group>
    DS_D_INLINE Params<Type::Symmetric, numBits> get_params(
        cg::thread_block& tb,
        cg::thread_block_tile<hw_warp_size>& warp)
    {
        const float2 partial_max = conversion::to<float2>(cur_max);
        float max = reduce::element<rop::Max>(partial_max.x, partial_max.y);

        reduce::partitioned_block<rop::Max, threads_per_group>(tb, warp, max);
        Params<Type::Symmetric, numBits> params(max);

        return params;
    }
};

template <>
class GroupStats<Type::Asymmetric> {
public:
    __half2 cur_max;
    __half2 cur_min;

    /*
    Initialize cur_max to -inf, cur_min to inf since
    we are doing a true range analysis.
    */
    DS_D_INLINE GroupStats()
    {
        cur_max = reduce::init<rop::Max, __half2>();
        cur_min = reduce::init<rop::Min, __half2>();
    }

    /*
    Updated the running min and max used to calculate params.
    Function Arguments :
        val : The __half2 value to update the running min and max with.
    */
    DS_D_INLINE void update(__half2 val)
    {
        cur_max = reduce::element<rop::Max>(cur_max, val);
        cur_min = reduce::element<rop::Min>(cur_min, val);
    }

    /*
    Function to return calculated quantization params.
    Template Arguments :
        numBits -   Number of bits in quantized element.    int : 8 or 4
    Function Arguments :
        tb      -   Threadblock object. cg::thread_block
        warp    -   Warp object.        cg::thread_block_tile<hw_warp_size>
    */
    template <int numBits, int threads_per_group>
    DS_D_INLINE Params<Type::Asymmetric, numBits> get_params(
        cg::thread_block& tb,
        cg::thread_block_tile<hw_warp_size>& warp)
    {
        const float2 partial_max = conversion::to<float2>(cur_max);
        float max = reduce::element<rop::Max>(partial_max.x, partial_max.y);

        const float2 partial_min = conversion::to<float2>(cur_min);
        float min = reduce::element<rop::Min>(partial_min.x, partial_min.y);

        reduce::partitioned_block<rop::Max, rop::Min, threads_per_group>(tb, warp, max, min);

        Params<Type::Asymmetric, numBits> params(max, min);

        return params;
    }
};

/*
Device function that quantizes 16 bytes of __half type input data.
Template Arguments :
    numBits -   Number of bits in quantized element.    int : 8 or 4
    qType   - Type of quantization to perform.          Type::Symmetric or Type::Asymmetric
Function Arguments :
    local_output -  Pointer to local memory to store quantized data.    int8_t*
    data         -  Pointer to input data.                              __half*
    Params       -  Parameters for quantization.                        Params<qType, numBits>
*/
template <int numBits, Type qType>
DS_D_INLINE void _chunk(int8_t* local_output, const __half* data, Params<qType, numBits> q_params);

/*
Device function that quantizes 16 bytes of __half2 type input data.
Template Arguments :
    numBits -   Number of bits in quantized element.    int : 8 or 4
    qType   -   Type of quantization to perform.        Type::Symmetric or Type::Asymmetric
Function Arguments :
    local_output -  Pointer to local memory to store quantized data.    int8_t*
    data         -  Pointer to input data.                              __half2*
    Params       -  Parameters for quantization.                        Params<qType, numBits>
*/
template <int numBits, Type qType>
DS_D_INLINE void _chunk(int8_t* local_output, const __half2* data, Params<qType, numBits> q_params);

/*
Helper function to do serial reduction on register-file arrays.
Template Arguments :
    qType       -   Type of quantization to perform.        Type::Symmetric or Type::Asymmetric
    numChunks   -   Number of bits in quantized element.    int : 8 or 4
Function Arguments :
    local_buffer    -   Pointer memory with input half2 data to be quantized.
*/
template <Type qType, int numChunks>
DS_D_INLINE GroupStats<qType> _local_serial_reduce(__half2* local_buffer);

/*
The main loop of the kernel that quantizes array in local memory of __half2 type input data, when
Quantization parameters are pre-computed.
Template Arguments :
    qType       -   Type of quantization to perform.            Type::Symmetric or Type::Asymmetric
    numBits     -   Number of bits in quantized element.        int : 8 or 4
    numChunks   -   Number of chunks(16 bytes of Input data).   int : 8 or 4
Function Arguments :
    local_buffer    -   Pointer memory with input half2 data to be quantized.
    scales          -   Pointer to output scales.
    offsets         -   Pointer to output offsets.
    output_data     -   Pointer to output data.
    elems_per_group -   Number of elements to quantize in a group.
    q_params        -   Quantization parameters.
*/
template <int numBits, Type qType, int numChunks, int threads_per_group, int max_threads>
DS_D_INLINE void local_array(cg::thread_block& tb,
                             cg::thread_block_tile<hw_warp_size>& warp,
                             __half2* local_buffer,
                             float* __restrict__ scales,
                             float* __restrict__ offsets,
                             int8_t* __restrict__ output_data,
                             const int& elems_per_group,
                             const int& groups,
                             Params<qType, numBits> q_params);

/*
The main loop of the kernel that quantizes array in local memory of __half2 type input data.
This function computes quantization parameters for each group.
Template Arguments :
    qType   -   Type of quantization to perform.                Type::Symmetric or Type::Asymmetric
    numBits     -   Number of bits in quantized element.        int : 8 or 4
    numChunks   -   Number of chunks(16 bytes of Input data).   int : 8 or 4
Function Arguments :
    local_buffer    -   Pointer memory with input half2 data to be quantized.
    scales          -   Pointer to output scales.
    offsets         -   Pointer to output offsets.
    output_data     -   Pointer to output data.
    elems_per_group -   Number of elements to quantize in a group.
*/
template <Type qType, int numBits, int numChunks, int threads_per_group, int max_threads>
__device__ void local_array(__half2* local_buffer,
                            float* __restrict__ scales,
                            float* __restrict__ offsets,
                            int8_t* __restrict__ output_data,
                            const int& elems_per_group,
                            const int& groups);

template <int numBits, Type qType>
DS_D_INLINE void _chunk(int8_t* local_output, const __half* data, Params<qType, numBits> q_params)
{
    constexpr int32_t elems = 16 / sizeof(__half);
    constexpr int32_t num_elems_packed = 8 / numBits;

#pragma unroll
    for (int i = 0, oi = 0; i < elems; i += num_elems_packed, oi++) {
        if (num_elems_packed == 1) {
            // TODO(cmikeh2): refactor to use conversion utils
            local_output[i] = q_params.quantize(data[i]);
        } else if (num_elems_packed == 2) {
            int8_t data_i8_1 = q_params.quantize(data[i]);
            int8_t data_i8_2 = q_params.quantize(data[i + 1]);
            auto data_i8 = PackedInt4{data_i8_2, data_i8_1};
            local_output[oi] = *((int8_t*)(&data_i8));
        }
    }
}

template <int numBits, Type qType>
DS_D_INLINE void _chunk(int8_t* local_output, const __half2* data, Params<qType, numBits> q_params)
{
    const __half* data_cast = reinterpret_cast<const __half*>(data);
    _chunk<numBits>(local_output, data_cast, q_params);
}

template <Type qType, int numChunks>
DS_D_INLINE GroupStats<qType> _local_serial_reduce(__half2* local_buffer)
{
    GroupStats<qType> stats;
#pragma unroll
    for (int i = 0; i < numChunks * h2_per_load; i++) { stats.update(local_buffer[i]); }

    return stats;
}

template <Type qType, int numBits, int numChunks, int threads_per_group, int max_threads>
DS_D_INLINE void local_array(cg::thread_block& tb,
                             cg::thread_block_tile<hw_warp_size>& warp,
                             __half2* local_buffer,
                             float* __restrict__ global_params,
                             int8_t* __restrict__ output_data,
                             const int& elems_per_group,
                             const int& groups,
                             Params<qType, numBits> q_params)
{
    constexpr int num_ele_int8 = 8 / numBits;
    constexpr int num_int8_out = quantize::h_per_load / num_ele_int8;

    // Indexing offsets
    const int block_num =
        (tb.group_index().x * max_threads / threads_per_group) + tb.thread_index().y;
    const int block_offset = block_num * elems_per_group;
    const int elem_offset = tb.thread_index().x * quantize::h_per_load;
    const int base_offset = (block_offset + elem_offset) / num_ele_int8;
    const int stride = tb.size() * quantize::h_per_load / num_ele_int8;

    int8_t local_output[num_int8_out];

    if (tb.thread_index().x == 0 && block_num < groups) {
        q_params.store(
            global_params,
            (tb.group_index().x * max_threads / threads_per_group) + tb.thread_index().y);
    }
#pragma unroll
    for (int i = 0; i < numChunks; i++) {
        if (elem_offset + i * stride * num_ele_int8 < elems_per_group && block_num < groups) {
            quantize::_chunk<numBits, qType>(
                local_output, local_buffer + i * quantize::h2_per_load, q_params);
            mem_access::store_global<num_int8_out>(output_data + (base_offset + i * stride),
                                                   local_output);
        }
    }
}

template <Type qType, int numBits, int numChunks, int threads_per_group, int max_threads>
DS_D_INLINE void local_array(cg::thread_block& tb,
                             cg::thread_block_tile<hw_warp_size>& warp,
                             __half* local_buffer,
                             float* __restrict__ global_params,
                             int8_t* __restrict__ output_data,
                             const int& elems_per_group,
                             const int& groups,
                             Params<qType, numBits> q_params)
{
    __half2* local_buffer_h2 = reinterpret_cast<__half2*>(local_buffer);

    quantize::local_array<qType, numBits, numChunks, threads_per_group, max_threads>(
        tb, warp, local_buffer, global_params, output_data, elems_per_group, groups, q_params);
}

template <Type qType,
          int numBits,
          int numChunks,
          int threads_per_group = max_threads,
          int max_threads = 256>
__device__ void local_array(__half2* local_buffer,
                            float* __restrict__ global_params,
                            int8_t* __restrict__ output_data,
                            const int& elems_per_group,
                            const int& groups)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    auto group_stats = _local_serial_reduce<qType, numChunks>(local_buffer);
    auto params = group_stats.template get_params<numBits, threads_per_group>(tb, warp);

    quantize::local_array<qType, numBits, numChunks, threads_per_group, max_threads>(
        tb, warp, local_buffer, global_params, output_data, elems_per_group, groups, params);
}

template <Type qType, int numBits, int numChunks, int threads_per_group, int max_threads>
__device__ void local_array(__half* local_buffer,
                            float* __restrict__ global_params,
                            int8_t* __restrict__ output_data,
                            const int& elems_per_group,
                            const int& groups)
{
    __half2* local_buffer_h2 = reinterpret_cast<__half2*>(local_buffer);
    quantize::local_array<qType, numBits, numChunks, threads_per_group, max_threads>(
        local_buffer_h2, global_params, output_data, elems_per_group, groups);
}

}  // namespace quantize
