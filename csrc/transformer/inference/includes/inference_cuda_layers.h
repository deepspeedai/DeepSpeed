// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"

#include <cuda.h>
#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

#define MAX_WARP_NUM 32
#define WARP_SIZE 32

#define MAX_THREADS 1024
#define SMs 80

#define MAX_REGISTERS 256

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
                            int offset,
                            int mask_stride,
                            int mp_size,
                            cudaStream_t stream);

// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream);

template <typename T>
void launch_gated_activation(T* output,
                             const T* activation,
                             const T* bias,
                             int rows,
                             int output_stride,
                             int elems_per_row,
                             bool use_gelu,
                             cudaStream_t stream);

// Fused bias add with relu activation
template <typename T>
void launch_bias_relu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream);

template <typename T>
void launch_bias_add(T* input, const T* bias, int hidden_size, int batch_size, cudaStream_t stream);

template <typename T>
void launch_bias_residual(T* input,
                          T* output,
                          T* attn,
                          T* bias,
                          T* attn_bias,
                          int batch,
                          int hidden_dim,
                          int mp_size,
                          bool preln,
                          cudaStream_t stream);

template <typename T>
void launch_fused_ln(T* output,
                     const T* vals,
                     const T* gamma,
                     const T* beta,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     cudaStream_t stream);

template <typename T>
void launch_fused_residual_ln(T* output,
                              const T* vals,
                              const T* residual,
                              const T* bias,
                              const T* gamma,
                              const T* beta,
                              float epsilon,
                              int rows,
                              int elems_per_row,
                              cudaStream_t stream);

template <typename T>
void launch_fused_residual_ln_store_pre_ln_res(T* norm_output,
                                               T* res_output,
                                               const T* vals,
                                               const T* residual,
                                               const T* bias,
                                               const T* gamma,
                                               const T* beta,
                                               float epsilon,
                                               int rows,
                                               int elems_per_row,
                                               cudaStream_t stream);

template <typename T>
void launch_rms_norm(T* norm_output,
                     T* res_output,
                     const T* vals,
                     const T* residual,
                     const T* gamma,
                     float epsilon,
                     int rows,
                     int elems_per_row,
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

template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       cudaStream_t stream);
template <typename T>
void launch_gptj_residual_add(T* input,
                              T* output,
                              T* attn,
                              T* bias,
                              T* attn_bias,
                              int batch,
                              int head_size,
                              int mp_size,
                              cudaStream_t stream);

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 cudaStream_t stream,
                                 int max_out_tokens);

template <typename T>
void launch_moe_res_matmul(T* residual,
                           T* coef,
                           T* mlp_out,
                           int seq_len,
                           int hidden_dim,
                           cudaStream_t stream);

// 4D transform [0, 1, 2, 3] -> [0, 2, 1, 3]
template <typename T>
void launch_transform4d_0213(T* out,
                             const T* in,
                             int batch_size,
                             int heads,
                             int seq_length,
                             int hidden_dim,
                             cudaStream_t stream,
                             int trans_count);
template <typename T>
void launch_bias_add_transform_0213(T* outputs,
                                    T* vals,
                                    T* vals1,
                                    const T* vals2,
                                    const T* bias,
                                    int batch_size,
                                    int seq_length,
                                    unsigned seq_offset,
                                    int seq_length1,
                                    int hidden_dim,
                                    int heads,
                                    int rotary_dim,
                                    bool rotate_half,
                                    bool rotate_every_two,
                                    cudaStream_t stream,
                                    int trans_count,
                                    int max_out_tokens);
template <typename T>
void pad_data(T* padded_output,
              T* output,
              int bsz,
              int head_size,
              int padded_head_size,
              cudaStream_t stream);

template <typename T>
void pad_head_seq(T* padded_output,
                  T* output,
                  int bsz,
                  int seq_len,
                  int padded_seq_len,
                  int head_size,
                  int padded_head_size,
                  cudaStream_t stream);

template <typename T>
void launch_pad_add_transform_0213(T* output,
                                   const T* vals,
                                   int batch_size,
                                   int hidden_dim,
                                   int seq_length,
                                   int padded_seq_len,
                                   int heads,
                                   int padded_head_size,
                                   cudaStream_t stream);

template <typename T>
void launch_vector_add(T* out,
                       const T* a,
                       const T* b,
                       float gamma,
                       int num_elems,
                       cudaStream_t stream);
