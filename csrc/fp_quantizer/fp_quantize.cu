// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <stdexcept>
#include "context.h"
#include "fp_quantize.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"

#include <cuda.h>
#include <stdint.h>

#include <cuda_fp16.h>
#include <curand_kernel.h>

#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif
#include <cuda_runtime_api.h>

using ROp = reduce::ROpType;

namespace quantization {

constexpr int access_granularity = 16;
constexpr int quanitzed_access_granularity = 4;
constexpr int quanitzed_access_granularity_6bits = 2;
constexpr int threads = 256;
constexpr int warps = threads / 32;

}  // namespace quantization

template <int _mantisa_bits, int q_mantisa_bits, int stochastic_rounding>
__device__ void round(uint32_t& mantisa, uint32_t& dst_exponent, curandStatePhilox4_32_10_t* state)
{
    constexpr uint32_t mantisa_mask = (1 << (_mantisa_bits - q_mantisa_bits)) - 1;
    uint32_t offset = stochastic_rounding ? (curand_poisson(state, 10) & mantisa_mask)
                                          : 1 << (_mantisa_bits - q_mantisa_bits - 1);
    mantisa += offset;
    dst_exponent += (((mantisa & ~mantisa_mask) == (1 << _mantisa_bits)) ? 1 : 0);
}

template <int _mantisa_bits, int _exponent_bits, int q_mantisa_bits, int q_exponent_bits>
__device__ void clip(uint32_t& exponent, uint32_t& mantisa)
{
    constexpr uint32_t max_exponent = (1 << (q_exponent_bits - 1)) + (1 << (_exponent_bits - 1));
    constexpr uint32_t min_exponent =
        (1 << (_exponent_bits - 1)) - ((1 << (q_exponent_bits - 1)) - 1);
    if (exponent > max_exponent) {
        exponent = max_exponent;
        mantisa = (((uint32_t)-1) >> (32 - q_mantisa_bits)) << 1;  //.11 .. 10
    }
    if (exponent < min_exponent) {
        exponent = min_exponent;
        mantisa = 0;
    }
}

template <typename T,
          int unroll,
          int _mantisa_bits,
          int _exponent_bits,
          int total_q_bits = 8,
          int q_mantisa_bits = 3,
          int stochastic_rounding = 0>
__global__ void apply_quantization(T* val,
                                   uint8_t* q_val,
                                   int group_size,
                                   std::pair<uint64_t, uint64_t> seed,
                                   float q_range)
{
    int tidx = threadIdx.x;
    int wid = tidx >> 5;
    int lane = tidx & 0x1f;
    int gid = blockIdx.x * quantization::warps + wid;

    constexpr int q_exponent_bits = total_q_bits - q_mantisa_bits - 1;
    constexpr uint32_t _mantisa_mask = (1 << _mantisa_bits) - 1;
    constexpr uint32_t _exponent_mask = ((1 << _exponent_bits) - 1) << _mantisa_bits;
    constexpr uint32_t _sign_mask = 1 << (_mantisa_bits + _exponent_bits);
    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    constexpr uint32_t vector_size = quantization::access_granularity / sizeof(T);
    constexpr uint32_t load_stride = vector_size * hw_warp_size;
    constexpr uint32_t store_stride = (total_q_bits * vector_size / 8) * hw_warp_size;
    const uint32_t thread_offset = lane * vector_size;
    const uint32_t store_thread_offset = lane * (total_q_bits * vector_size / 8);
    const uint32_t base_load_offset = gid * group_size + thread_offset;
    const uint32_t base_store_offset =
        gid * ((group_size * total_q_bits / 8) + 4) +
        store_thread_offset;  // 4-byte for saving the scale per group
    const T* load_base_ptr = val + base_load_offset;
    T tmp_buf[unroll * vector_size];
    T cur_max;
    reduce::init<ROp::Max>(&cur_max);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

#pragma unroll
    for (int i = 0; i < unroll; i++) {
        if (i * load_stride + thread_offset < group_size) {
            mem_access::load_global<quantization::access_granularity>(
                &tmp_buf[vector_size * i], load_base_ptr + i * load_stride);
            for (int j = 0; j < vector_size; j++)
                cur_max = reduce::element<ROp::Max>(cur_max, __habs(tmp_buf[i * vector_size + j]));
        }
    }
    reduce::_block<T, 1, ROp::Max>(tb, warp, &cur_max);

    int mantisa_mask = ((1 << q_mantisa_bits) - 1);
    mantisa_mask <<= (_mantisa_bits - q_mantisa_bits);

    uint8_t* store_base_ptr = q_val + base_store_offset;
    float scale = (float)q_range / conversion::to<float>(cur_max);
#pragma unroll
    for (int i = 0; i < unroll; i++) {
        if (i * load_stride + thread_offset < group_size) {
            uint64_t q_buf = 0;
            uint64_t q_buf1 = 0;
#pragma unroll
            for (int j = 0; j < vector_size; j++) {
                float val_f = conversion::to<float>(tmp_buf[i * vector_size + j]) * scale;
                uint32_t* data = reinterpret_cast<uint32_t*>(&val_f);
                uint32_t sign = (data[0] & _sign_mask) >> (_mantisa_bits + _exponent_bits);
                uint32_t cur_exponent = (data[0] & _exponent_mask) >> _mantisa_bits;
                uint32_t dst_mantisa = (data[0] & _mantisa_mask);

                uint32_t dst_exponent = cur_exponent;

                round<_mantisa_bits, q_mantisa_bits, stochastic_rounding>(
                    dst_mantisa, dst_exponent, &state);
                if (cur_exponent != 0)
                    clip<_mantisa_bits, _exponent_bits, q_mantisa_bits, q_exponent_bits>(
                        dst_exponent, dst_mantisa);

                dst_mantisa = (dst_mantisa & mantisa_mask) >> (_mantisa_bits - q_mantisa_bits);

                if (dst_exponent != (1 << q_exponent_bits) - 1)
                    dst_exponent = (dst_exponent - ((1 << (_exponent_bits - 1)) - 1)) +
                                   (1 << (q_exponent_bits - 1)) - 1;
                if (total_q_bits == 8 || total_q_bits == 4 || total_q_bits == 6)
                    q_buf = q_buf |
                            ((uint64_t)((uint8_t)(sign << (q_exponent_bits + q_mantisa_bits) |
                                                  (dst_exponent << q_mantisa_bits) | dst_mantisa))
                             << j * total_q_bits);
                else if (total_q_bits == 12) {
                    if (j < 5)
                        q_buf =
                            q_buf |
                            ((uint64_t)((uint16_t)(sign << (q_exponent_bits + q_mantisa_bits) |
                                                   (dst_exponent << q_mantisa_bits) | dst_mantisa))
                             << j * total_q_bits);
                    else
                        q_buf1 =
                            q_buf1 |
                            ((uint64_t)((uint16_t)(sign << (q_exponent_bits + q_mantisa_bits) |
                                                   (dst_exponent << q_mantisa_bits) | dst_mantisa))
                             << (j - 5) * total_q_bits);
                }
            }
            if (total_q_bits == 12) {
                uint64_t last_nibble_mask = 0xf;
                last_nibble_mask = q_buf1 & last_nibble_mask;
                q_buf = (last_nibble_mask << 60) | q_buf;
                q_buf1 >>= 4;
            }
            uint8_t* int8_data = reinterpret_cast<uint8_t*>(&q_buf);
            uint8_t* int8_data1 = reinterpret_cast<uint8_t*>(&q_buf1);
            if (total_q_bits == 6) {
                mem_access::store_global<quantization::quanitzed_access_granularity_6bits>(
                    store_base_ptr + i * store_stride, int8_data);
                mem_access::store_global<quantization::quanitzed_access_granularity_6bits>(
                    store_base_ptr + i * store_stride +
                        quantization::quanitzed_access_granularity_6bits,
                    int8_data + quantization::quanitzed_access_granularity_6bits);
                mem_access::store_global<quantization::quanitzed_access_granularity_6bits>(
                    store_base_ptr + i * store_stride +
                        quantization::quanitzed_access_granularity_6bits * 2,
                    int8_data + 2 * quantization::quanitzed_access_granularity_6bits);
            } else {
                mem_access::store_global<quantization::quanitzed_access_granularity>(
                    store_base_ptr + i * store_stride, int8_data);

                if (total_q_bits > 4) {
                    mem_access::store_global<quantization::quanitzed_access_granularity>(
                        store_base_ptr + i * store_stride +
                            quantization::quanitzed_access_granularity,
                        int8_data + quantization::quanitzed_access_granularity);
                    if (total_q_bits == 12) {
                        mem_access::store_global<quantization::quanitzed_access_granularity>(
                            store_base_ptr + i * store_stride +
                                quantization::quanitzed_access_granularity * 2,
                            int8_data1);
                    }
                }
            }
        }
    }
    if (lane == 0) {
        float q_scale = conversion::to<float>(cur_max) / (float)q_range;
        uint8_t* scale_as_int8 = reinterpret_cast<uint8_t*>(&q_scale);
        uint32_t scale_offset =
            gid * ((group_size * total_q_bits / 8) + 4) + (group_size * total_q_bits / 8);
        if (total_q_bits != 6)
            mem_access::store_global<quantization::quanitzed_access_granularity>(
                q_val + scale_offset, scale_as_int8);
        else {
            mem_access::store_global<quantization::quanitzed_access_granularity_6bits>(
                q_val + scale_offset, scale_as_int8);
            mem_access::store_global<quantization::quanitzed_access_granularity_6bits>(
                q_val + scale_offset + quantization::quanitzed_access_granularity_6bits,
                scale_as_int8 + quantization::quanitzed_access_granularity_6bits);
        }
    }
}

template <typename T,
          int q_mantisa_bits,
          int total_q_bits = 16,
          int _mantisa_bits = 3,
          int _exponent_bits = 4>
__global__ void apply_dequantization(uint8_t* val, T* q_val, int group_size, int total_num_elements)
{
    constexpr uint32_t vector_size = quantization::access_granularity / sizeof(T);
    int tidx = (blockIdx.x * blockDim.x + threadIdx.x) * vector_size;

    constexpr int quantized_bits = _mantisa_bits + _exponent_bits + 1;
    constexpr int q_exponent_bits = total_q_bits - q_mantisa_bits - 1;
    constexpr uint16_t _mantisa_mask = (1 << _mantisa_bits) - 1;
    constexpr uint16_t _exponent_mask = ((1 << _exponent_bits) - 1) << _mantisa_bits;
    constexpr uint16_t _sign_mask = 1 << (_mantisa_bits + _exponent_bits);
    const uint32_t g_index = (tidx / group_size);
    const uint32_t group_size_bytes = (group_size * quantized_bits / 8);
    const uint8_t* load_base_ptr =
        val + g_index * (group_size_bytes + 4) + (tidx % group_size) * quantized_bits / 8;

    int mantisa_mask = ((1 << q_mantisa_bits) - 1);
    mantisa_mask <<= (_mantisa_bits - q_mantisa_bits);

    T* store_base_ptr = q_val + tidx;
    float scale;

    uint8_t* scale_as_int8 = reinterpret_cast<uint8_t*>(&scale);
    if (quantized_bits == 6) {
        mem_access::load_global<quantization::quanitzed_access_granularity>(
            scale_as_int8, val + g_index * (group_size_bytes + 4) + group_size_bytes);
        mem_access::load_global<quantization::quanitzed_access_granularity_6bits>(
            scale_as_int8 + quantization::quanitzed_access_granularity_6bits,
            val + g_index * (group_size_bytes + 4) + group_size_bytes +
                quantization::quanitzed_access_granularity_6bits);
    } else
        mem_access::load_global<quantization::quanitzed_access_granularity>(
            scale_as_int8, val + g_index * (group_size_bytes + 4) + group_size_bytes);

    if (tidx < total_num_elements) {
        uint64_t q_buf_in;
        uint64_t q_buf_in1;
        uint8_t* int8_data = reinterpret_cast<uint8_t*>(&q_buf_in);
        uint8_t* int8_data1 = reinterpret_cast<uint8_t*>(&q_buf_in1);
        if (quantized_bits == 6) {
            mem_access::load_global<quantization::quanitzed_access_granularity_6bits>(
                int8_data, load_base_ptr);
            mem_access::load_global<quantization::quanitzed_access_granularity_6bits>(
                int8_data + quantization::quanitzed_access_granularity_6bits,
                load_base_ptr + quantization::quanitzed_access_granularity_6bits);
            mem_access::load_global<quantization::quanitzed_access_granularity_6bits>(
                int8_data + quantization::quanitzed_access_granularity_6bits * 2,
                load_base_ptr + quantization::quanitzed_access_granularity_6bits * 2);

        } else {
            mem_access::load_global<quantization::quanitzed_access_granularity>(int8_data,
                                                                                load_base_ptr);
            if (quantized_bits > 4) {
                mem_access::load_global<quantization::quanitzed_access_granularity>(
                    int8_data + quantization::quanitzed_access_granularity,
                    load_base_ptr + quantization::quanitzed_access_granularity);
                if (quantized_bits == 12) {
                    mem_access::load_global<quantization::quanitzed_access_granularity>(
                        int8_data1, load_base_ptr + quantization::quanitzed_access_granularity * 2);
                }
            }
        }
        T store_buf[vector_size];
        uint16_t* q_buf = reinterpret_cast<uint16_t*>(store_buf);
#pragma unroll
        for (int j = 0; j < vector_size; j++) {
            uint16_t new_data;
            if (j < 5 || quantized_bits != 12) {
                new_data = (uint16_t)(q_buf_in >> (j * quantized_bits));
            } else {
                if (j == 5) {
                    new_data = (uint16_t)(q_buf_in1);
                    new_data = (uint16_t)((new_data << 4) | (q_buf_in >> 60));
                } else
                    new_data = (uint16_t)(q_buf_in1 >> ((j - 6) * quantized_bits + 8));
            }

            uint16_t sign = (new_data & _sign_mask) >> (_mantisa_bits + _exponent_bits);
            uint16_t dst_exponent = (new_data & _exponent_mask) >> _mantisa_bits;
            uint16_t dst_mantisa = (new_data & _mantisa_mask);

            if (dst_exponent != (1 << q_exponent_bits) - 1)
                dst_exponent = (dst_exponent - ((1 << (_exponent_bits - 1)) - 1)) +
                               (1 << (q_exponent_bits - 1)) - 1;

            q_buf[j] =
                ((sign << (q_exponent_bits + q_mantisa_bits)) | (dst_exponent << q_mantisa_bits) |
                 (dst_mantisa << (q_mantisa_bits - _mantisa_bits)));
            float up_cast = conversion::to<float>(store_buf[j]);
            store_buf[j] = conversion::to<T>(up_cast * scale);
        }
        mem_access::store_global<quantization::access_granularity>(store_base_ptr, store_buf);
    }
}

#define LAUNCH_FOR_QUANTIZATION_UNROLL(COUNT)                                    \
    case COUNT:                                                                  \
        apply_quantization<T,                                                    \
                           COUNT,                                                \
                           mantisa,                                              \
                           exponent,                                             \
                           CONST_Q_BITS,                                         \
                           CONST_Q_MANTISA_BITS,                                 \
                           CONST_STOCHASTIC_ROUNDING>                            \
            <<<grid, block, 0, stream>>>(val, q_val, group_size, seed, q_range); \
        break;

template <typename T, int mantisa, int exponent>
void launch_quantization(T* val,
                         uint8_t* q_val,
                         int num_groups,
                         int group_size,
                         cudaStream_t stream,
                         float q_range,
                         int q_bits,
                         int q_mantisa_bits,
                         int stochastic_rounding)
{
    const dim3 grid((num_groups + quantization::warps - 1) / quantization::warps);
    const dim3 block(quantization::threads);

    std::pair<uint64_t, uint64_t> seed = FPContext::Instance().IncrementOffset(16);

    constexpr int vals_per_unroll = hw_warp_size * quantization::access_granularity / sizeof(T);

    const int copy_unroll = (group_size + vals_per_unroll - 1) / vals_per_unroll;
    QUANT_SWITCH((q_bits - q_mantisa_bits - 1) * q_mantisa_bits + stochastic_rounding, [&] {
        switch (copy_unroll) {
            LAUNCH_FOR_QUANTIZATION_UNROLL(1)
            LAUNCH_FOR_QUANTIZATION_UNROLL(2)
            LAUNCH_FOR_QUANTIZATION_UNROLL(3)
            LAUNCH_FOR_QUANTIZATION_UNROLL(4)
            LAUNCH_FOR_QUANTIZATION_UNROLL(5)
            LAUNCH_FOR_QUANTIZATION_UNROLL(6)
        }
    });
}
#define INSTANTIATE_LAUNCH_QUANTIZATION(T, mantisa, exponent) \
    template void launch_quantization<T, mantisa, exponent>(  \
        T*, uint8_t*, int, int, cudaStream_t, float q_range, int, int, int);
// fp8(E4M3), nearest-rounding
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_QUANTIZATION(__nv_bfloat16, 23, 8);
#endif
INSTANTIATE_LAUNCH_QUANTIZATION(__half, 23, 8);

template <typename T, int mantisa>
void launch_dequantization(uint8_t* val,
                           T* q_val,
                           int num_groups,
                           int group_size,
                           int q_mantisa_bits,
                           int q_exponent_bits,
                           cudaStream_t stream)
{
    int blocks = ((num_groups * group_size) - 1) /
                     (quantization::threads * (quantization::access_granularity / sizeof(T))) +
                 1;
    const dim3 grid(blocks);
    const dim3 block(quantization::threads);
    DEQUANT_SWITCH(q_mantisa_bits * q_exponent_bits, [&] {
        apply_dequantization<T, mantisa, 16, CONST_Q_MANTISA_BITS, CONST_Q_EXPONENT_BITS>
            <<<grid, block, 0, stream>>>(val, q_val, group_size, (num_groups * group_size));
    });
}
#define INSTANTIATE_LAUNCH_DEQUANTIZATION(T, mantisa) \
    template void launch_dequantization<T, mantisa>(uint8_t*, T*, int, int, int, int, cudaStream_t);
// fp8(E4M3)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_DEQUANTIZATION(__nv_bfloat16, 7);
#endif
INSTANTIATE_LAUNCH_DEQUANTIZATION(__half, 10);

template <typename T,
          int q_mantisa_bits,
          int total_q_bits = 16,
          int _mantisa_bits = 3,
          int _exponent_bits = 4>
__global__ void apply_selective_dequantization(uint8_t* val,
                                               T* q_val,
                                               int32_t* indexes,
                                               int group_size,
                                               int total_num_elements)
{
    int index = indexes[blockIdx.x];
    constexpr uint32_t vector_size = quantization::access_granularity / sizeof(T);
    int tidx = (blockIdx.y * blockDim.x + threadIdx.x) * vector_size;
    int input_index = index * total_num_elements + tidx;
    constexpr int quantized_bits = _mantisa_bits + _exponent_bits + 1;
    constexpr int q_exponent_bits = total_q_bits - q_mantisa_bits - 1;
    constexpr uint16_t _mantisa_mask = (1 << _mantisa_bits) - 1;
    constexpr uint16_t _exponent_mask = ((1 << _exponent_bits) - 1) << _mantisa_bits;
    constexpr uint16_t _sign_mask = 1 << (_mantisa_bits + _exponent_bits);
    const uint32_t g_index = (input_index / group_size);
    const uint32_t group_size_bytes = (group_size * quantized_bits / 8);
    const uint8_t* load_base_ptr =
        val + g_index * (group_size_bytes + 4) + (input_index % group_size) * quantized_bits / 8;

    int mantisa_mask = ((1 << q_mantisa_bits) - 1);
    mantisa_mask <<= (_mantisa_bits - q_mantisa_bits);

    T* store_base_ptr = q_val + tidx + blockIdx.x * total_num_elements;
    float scale;

    uint8_t* scale_as_int8 = reinterpret_cast<uint8_t*>(&scale);
    if (quantized_bits == 6) {
        mem_access::load_global<quantization::quanitzed_access_granularity>(
            scale_as_int8, val + g_index * (group_size_bytes + 4) + group_size_bytes);
        mem_access::load_global<quantization::quanitzed_access_granularity_6bits>(
            scale_as_int8 + quantization::quanitzed_access_granularity_6bits,
            val + g_index * (group_size_bytes + 4) + group_size_bytes +
                quantization::quanitzed_access_granularity_6bits);
    } else
        mem_access::load_global<quantization::quanitzed_access_granularity>(
            scale_as_int8, val + g_index * (group_size_bytes + 4) + group_size_bytes);

    if (tidx < total_num_elements) {
        uint64_t q_buf_in;
        uint64_t q_buf_in1;
        uint8_t* int8_data = reinterpret_cast<uint8_t*>(&q_buf_in);
        uint8_t* int8_data1 = reinterpret_cast<uint8_t*>(&q_buf_in1);
        if (quantized_bits == 6) {
            mem_access::load_global<quantization::quanitzed_access_granularity_6bits>(
                int8_data, load_base_ptr);
            mem_access::load_global<quantization::quanitzed_access_granularity_6bits>(
                int8_data + quantization::quanitzed_access_granularity_6bits,
                load_base_ptr + quantization::quanitzed_access_granularity_6bits);
            mem_access::load_global<quantization::quanitzed_access_granularity_6bits>(
                int8_data + quantization::quanitzed_access_granularity_6bits * 2,
                load_base_ptr + quantization::quanitzed_access_granularity_6bits * 2);
        } else {
            mem_access::load_global<quantization::quanitzed_access_granularity>(int8_data,
                                                                                load_base_ptr);
            if (quantized_bits > 4) {
                mem_access::load_global<quantization::quanitzed_access_granularity>(
                    int8_data + quantization::quanitzed_access_granularity,
                    load_base_ptr + quantization::quanitzed_access_granularity);
                if (quantized_bits == 12) {
                    mem_access::load_global<quantization::quanitzed_access_granularity>(
                        int8_data1, load_base_ptr + quantization::quanitzed_access_granularity * 2);
                }
            }
        }
        T store_buf[vector_size];
        uint16_t* q_buf = reinterpret_cast<uint16_t*>(store_buf);
#pragma unroll
        for (int j = 0; j < vector_size; j++) {
            uint16_t new_data;
            if (j < 5 || quantized_bits != 12) {
                new_data = (uint16_t)(q_buf_in >> (j * quantized_bits));
            } else {
                if (j == 5) {
                    new_data = (uint16_t)(q_buf_in1);
                    new_data = (uint16_t)((new_data << 4) | (q_buf_in >> 60));
                } else
                    new_data = (uint16_t)(q_buf_in1 >> ((j - 6) * quantized_bits + 8));
            }

            uint16_t sign = (new_data & _sign_mask) >> (_mantisa_bits + _exponent_bits);
            uint16_t dst_exponent = (new_data & _exponent_mask) >> _mantisa_bits;
            uint16_t dst_mantisa = (new_data & _mantisa_mask);

            if (dst_exponent != (1 << q_exponent_bits) - 1)
                dst_exponent = (dst_exponent - ((1 << (_exponent_bits - 1)) - 1)) +
                               (1 << (q_exponent_bits - 1)) - 1;

            q_buf[j] =
                ((sign << (q_exponent_bits + q_mantisa_bits)) | (dst_exponent << q_mantisa_bits) |
                 (dst_mantisa << (q_mantisa_bits - _mantisa_bits)));
            float up_cast = conversion::to<float>(store_buf[j]);
            store_buf[j] = conversion::to<T>(up_cast * scale);
        }
        mem_access::store_global<quantization::access_granularity>(store_base_ptr, store_buf);
    }
}

template <typename T, int mantisa>
void launch_selective_dequantization(uint8_t* val,
                                     T* q_val,
                                     int32_t* indexes,
                                     int num_groups,
                                     int group_size,
                                     int num_indexes,
                                     int q_mantisa_bits,
                                     int q_exponent_bits,
                                     cudaStream_t stream)
{
    int total_elements_per_index = (num_groups / num_indexes) * group_size;
    int blocks = (total_elements_per_index - 1) /
                     (quantization::threads * (quantization::access_granularity / sizeof(T))) +
                 1;
    const dim3 grid(num_indexes, blocks);
    const dim3 block(quantization::threads);
    DEQUANT_SWITCH(q_mantisa_bits * q_exponent_bits, [&] {
        apply_selective_dequantization<T, mantisa, 16, CONST_Q_MANTISA_BITS, CONST_Q_EXPONENT_BITS>
            <<<grid, block, 0, stream>>>(val, q_val, indexes, group_size, total_elements_per_index);
    });
}
#define INSTANTIATE_LAUNCH_SELECTIVE_DEQUANTIZATION(T, mantisa) \
    template void launch_selective_dequantization<T, mantisa>(  \
        uint8_t*, T*, int32_t*, int, int, int, int, int, cudaStream_t);
// fp8(E4M3)
#ifdef BF16_AVAILABLE
INSTANTIATE_LAUNCH_SELECTIVE_DEQUANTIZATION(__nv_bfloat16, 7);
#endif
INSTANTIATE_LAUNCH_SELECTIVE_DEQUANTIZATION(__half, 10);
