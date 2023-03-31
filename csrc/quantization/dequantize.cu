// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "dequantization_utils.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

template <typename T, int numBits, dequantize::Type qType, int unroll, int threads>
__global__ void dequantize_kernel(T* __restrict__ dequant_data,
                                  const int8_t* __restrict__ q_data,
                                  const float* __restrict__ q_params,
                                  int elems_per_group,
                                  int total_elems)
{
    dequantize::to_global<T, numBits, qType, unroll, threads>(
        dequant_data, q_data, q_params, elems_per_group, total_elems);
}

#define LAUNCH_DEQUANT_KERNEL(num_bits, q_type)                                          \
    dequantize_kernel<T, num_bits, q_type, unroll, threads><<<grid, block, 0, stream>>>( \
        dequant_data, q_data, q_params, elems_per_group, total_elems);

template <typename T>
void launch_dequantize_kernel(T* dequant_data,
                              const int8_t* q_data,
                              const float* q_params,
                              quantize::Type q_type,
                              int num_bits,
                              int elems_per_group,
                              int total_elems,
                              cudaStream_t stream)
{
    constexpr int unroll = 8;
    constexpr int threads = 512;
    constexpr int elems_per_block = unroll * threads * dequantize::granularity / (sizeof(T));

    const dim3 block(threads);
    const dim3 grid((total_elems + elems_per_block - 1) / elems_per_block);

    // TODO(cmikeh2): It may make sense to tune unroll, there is perf benefit for large
    // problem sizes with this large unroll value.
    if (num_bits == 8 && q_type == quantize::Type::Symmetric) {
        LAUNCH_DEQUANT_KERNEL(8, quantize::Type::Symmetric);
    } else if (num_bits == 8 && q_type == quantize::Type::Asymmetric) {
        LAUNCH_DEQUANT_KERNEL(8, quantize::Type::Asymmetric);
    } else if (num_bits == 4 && q_type == quantize::Type::Symmetric) {
        LAUNCH_DEQUANT_KERNEL(4, quantize::Type::Symmetric);
    } else if (num_bits == 4 && q_type == quantize::Type::Asymmetric) {
        LAUNCH_DEQUANT_KERNEL(4, quantize::Type::Asymmetric);
    }
}

template void launch_dequantize_kernel(__half* dequant_data,
                                       const int8_t* q_data,
                                       const float* q_params,
                                       quantize::Type q_type,
                                       int num_bits,
                                       int elems_per_group,
                                       int total_elems,
                                       cudaStream_t stream);

template void launch_dequantize_kernel(float* dequant_data,
                                       const int8_t* q_data,
                                       const float* q_params,
                                       quantize::Type q_type,
                                       int num_bits,
                                       int elems_per_group,
                                       int total_elems,
                                       cudaStream_t stream);
