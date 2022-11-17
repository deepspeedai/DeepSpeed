
#pragma once

#include <cuda_fp16.h>
#include "ds_kernel_utils.h"

namespace quantize {

enum class Type { Symmetric, Asymmetric, IntegerSymmetric };

struct PackedInt4 {
    int8_t high : 4;
    int8_t low : 4;
};

DS_HD_INLINE bool requires_offset(Type qType) { return qType == Type::Asymmetric; }

}  // namespace quantize

template <int numBits, quantize::Type qType>
void launch_quant(int8_t* output_data,
                  float* params,
                  const __half* input_data,
                  int groups,
                  int elems_per_group,
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
