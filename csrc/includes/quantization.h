// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cuda_fp16.h>
#include "ds_kernel_utils.h"

namespace quantize {

enum class Type { Symmetric, Asymmetric };

struct PackedInt4 {
    int8_t high : 4;
    int8_t low : 4;
};

DS_HD_INLINE bool requires_offset(Type qType) { return qType == Type::Asymmetric; }

}  // namespace quantize

void launch_quant(int8_t* output_data,
                  float* params,
                  const __half* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  cudaStream_t stream);

template <typename T>
void launch_dequantize_kernel(T* dequant_data,
                              const int8_t* q_data,
                              const float* q_params,
                              quantize::Type q_type,
                              int num_bits,
                              int elems_per_group,
                              int total_elems,
                              cudaStream_t stream);

template <typename T>
void launch_fake_quantize_kernel(T* vals,
                                 int total_count,
                                 int group_num,
                                 int num_bits,
                                 cudaStream_t stream);
template <typename T>
void launch_sr_fake_quantize_kernel(T* vals,
                                    int total_count,
                                    int group_num,
                                    int num_bits,
                                    cudaStream_t stream);
template <typename T>
void launch_fake_quantize_kernel_asym(T* vals,
                                      int total_count,
                                      int group_num,
                                      int num_bits,
                                      cudaStream_t stream);
template <typename T>
void launch_sr_fake_quantize_kernel_asym(T* vals,
                                         int total_count,
                                         int group_num,
                                         int num_bits,
                                         cudaStream_t stream);
