#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

#define MAX_WARP_NUM 32
#define WARP_SIZE 32
#define SMs 80

#define MAX_REGISTERS 256
template <typename T>
void launch_attn_softmax_v2(T* vals,
                            T* mask,
                            bool triangular,
                            bool recompute,
                            bool local_attention,
                            int window_size,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            float scale,
                            cudaStream_t stream);

// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream);
template <typename T>
void launch_bias_add(T* input, const T* bias, int hidden_size, int batch_size, cudaStream_t stream);

template <typename T>
void launch_bias_residual(T* input,
                          const T* residual,
                          const T* bias,
                          int size,
                          int intermediate_size,
                          cudaStream_t stream);

template <typename T>
void launch_layer_norm(T* out,
                       T* vals,
                       const T* gamma,
                       const T* beta,
                       float epsilon,
                       int batch_size,
                       int hidden_dim,
                       cudaStream_t stream);

template <typename T>
void launch_residual_layer_norm(T* norm,
                                T* res_add,
                                T* vals,
                                T* residual,
                                const T* bias,
                                const T* gamma,
                                const T* beta,
                                float epsilon,
                                int batch_size,
                                int hidden_dim,
                                bool preLN,
                                cudaStream_t stream);
template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       unsigned merge_count,
                       cudaStream_t stream);
